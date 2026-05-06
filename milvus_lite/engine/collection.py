"""Collection — engine entry point.

Phase 4 scope:
    - insert(records, partition_name="_default")
    - get(pks, partition_names=None) — reads MemTable + Segments
    - search(query_vectors, top_k, metric_type, partition_names=None)
    - Synchronous flush triggered when MemTable.size() >= MEMTABLE_SIZE_LIMIT
    - Crash recovery on construction (replays WAL, rebuilds delta_index,
      loads all manifest segments)
    - WAL + MemTable + Manifest + DeltaIndex + Segment cache

NOT yet:
    - delete (Phase 5) — Collection.delete is not exposed, but the
      plumbing (DeleteOp dispatch in _apply, MemTable.apply_delete,
      delta_index, bitmap pipeline) is all in place. Phase 5 just adds
      the public method.
    - compaction (Phase 6)
    - partition CRUD (Phase 7)

Layering: Collection sits at the top of the engine layer. It is the
only place that knows about Operation dispatch — storage/wal.py and
storage/memtable.py both still take raw RecordBatches. This keeps the
storage layer free of engine-layer types.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

from milvus_lite.constants import (
    ALL_PARTITIONS,
    DEFAULT_NUM_PARTITIONS,
    DEFAULT_PARTITION,
    FILTER_CACHE_SIZE,
    MEMTABLE_SIZE_LIMIT,
    PARTITION_KEY_BUCKET_PREFIX,
)
from milvus_lite.engine.compaction import CompactionManager
from milvus_lite.engine.flush import execute_flush
from milvus_lite.engine.operation import DeleteOp, InsertOp, Operation
from milvus_lite.engine.recovery import execute_recovery
from milvus_lite.exceptions import (
    CollectionNotLoadedError,
    IndexAlreadyExistsError,
    IndexNotFoundError,
    PartitionNotFoundError,
    SchemaValidationError,
)
from milvus_lite.index.brute_force import BruteForceIndex
from milvus_lite.index.spec import IndexSpec
from milvus_lite.schema.types import DataType, FunctionType
from milvus_lite.schema.arrow_builder import (
    build_wal_data_schema,
    build_wal_delta_schema,
    get_primary_field,
    get_vector_field,
)
from milvus_lite.schema.types import CollectionSchema
from milvus_lite.schema.validation import (
    separate_dynamic_fields,
    validate_record,
    validate_schema,
)
from milvus_lite.search.assembler import assemble_candidates
from milvus_lite.search.executor import execute_search
from milvus_lite.search.executor_indexed import execute_search_with_index
from milvus_lite.storage.manifest import Manifest
from milvus_lite.storage.memtable import MemTable
from milvus_lite.storage.segment import Segment
from milvus_lite.storage.wal import WAL

if False:  # TYPE_CHECKING
    from milvus_lite.search.filter.semantic import CompiledExpr  # noqa: F401


# Segment cache key: (partition, relative_path) — relative_path is what
# the manifest stores so two segments cannot collide on the same name.
_SegmentKey = Tuple[str, str]


def _row_matches_filter(record: dict, compiled_filter) -> bool:
    """Evaluate a CompiledExpr against a single dict row.

    Used by Collection.get() after a successful pk lookup, when we need
    to filter the single hit row by the user's expression. Builds a
    1-row pa.Table on the fly so we can reuse the existing evaluator.
    For Phase F1 we always go through the python_backend (cheaper than
    constructing an Arrow table for one row).
    """
    from milvus_lite.search.filter.eval.python_backend import _eval_row
    result = _eval_row(compiled_filter.ast, record)
    return bool(result) if result is not None else False


def _strip_injected_output_fields(
    results: List[List[dict]],
    output_fields: Optional[List[str]],
) -> List[List[dict]]:
    """Restore the user's projection after internal field injection."""
    if output_fields is None:
        return results

    keep = set(output_fields)
    stripped: List[List[dict]] = []
    for hits in results:
        out_hits = []
        for hit in hits:
            new_hit = dict(hit)
            entity = hit.get("entity") or {}
            new_hit["entity"] = {
                k: v for k, v in entity.items()
                if k in keep
            }
            out_hits.append(new_hit)
        stripped.append(out_hits)
    return stripped


class Collection:
    """A single Collection — schema + WAL + MemTable + Manifest + DeltaIndex.

    Construction is non-destructive and crash-tolerant:
        1. Load Manifest (with .prev fallback if current is corrupted).
        2. Run recovery — replay any WAL files, rebuild DeltaIndex,
           clean orphan Parquet files.
        3. Allocate a fresh WAL number = max(found WALs + 1, manifest's
           active_wal_number).

    insert() validates fail-fast (no partial state on failure), allocates
    one _seq per row, writes to WAL then MemTable, and triggers a
    synchronous flush if the MemTable hit the size limit.
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        schema: CollectionSchema,
    ) -> None:
        validate_schema(schema)

        self._name = name
        self._data_dir = data_dir
        self._schema = schema
        self._pk_name = get_primary_field(schema).name
        _vf = get_vector_field(schema)
        self._vector_name: Optional[str] = _vf.name if _vf is not None else None

        self._wal_data_schema = build_wal_data_schema(schema)
        self._wal_delta_schema = build_wal_delta_schema(schema)

        os.makedirs(data_dir, exist_ok=True)

        # ── 1. load manifest ────────────────────────────────────
        self._manifest = Manifest.load(data_dir)

        # ── 2. recovery ─────────────────────────────────────────
        self._memtable, self._delta_index, next_wal_number = execute_recovery(
            data_dir=data_dir,
            schema=schema,
            manifest=self._manifest,
        )

        # ── 3. fresh WAL ────────────────────────────────────────
        # next_seq must clear both manifest's recorded seq AND any seq
        # we just learned from WAL replay.
        self._next_seq = max(
            self._manifest.current_seq, self._memtable.max_seq
        ) + 1

        wal_dir = os.path.join(data_dir, "wal")
        self._wal = WAL(
            wal_dir=wal_dir,
            wal_data_schema=self._wal_data_schema,
            wal_delta_schema=self._wal_delta_schema,
            wal_number=next_wal_number,
        )

        # ── 4. segment cache ────────────────────────────────────
        # Loaded from every data file referenced by the manifest. The
        # cache is keyed by (partition, relative_path) and is refreshed
        # after each flush so the search path always sees the latest
        # set of immutable segments.
        self._segment_cache: Dict[_SegmentKey, Segment] = {}
        self._refresh_segment_cache()

        # ── 5. compaction manager ───────────────────────────────
        self._compaction_mgr = CompactionManager(data_dir, schema)

        # Maintenance lock serializes mutations to manifest, segment
        # cache, and delta_index between the user thread (flush) and
        # the background worker (compaction + index build).
        self._maintenance_lock: threading.RLock = threading.RLock()
        # Single-threaded executor for background compaction + index
        # build. Insert returns as soon as data is persisted; heavy
        # work happens off the user thread.
        self._bg_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"milvus_lite-{name}",
        )
        self._bg_closed: bool = False

        # ── 6. filter expression cache (Phase F2c) ──────────────
        # LRU on (expr_string → CompiledExpr) — schema is implicit since
        # the cache is per-Collection. Bounded by FILTER_CACHE_SIZE so
        # adversarial / heavy expression diversity can't OOM.
        from milvus_lite.search.filter.cache import LRUCache
        self._filter_cache: LRUCache = LRUCache(maxsize=FILTER_CACHE_SIZE)

        # ── 7. index state machine (Phase 9.3) ──────────────────
        # _index_specs mirrors manifest's per-field IndexSpec dict.
        # _load_state mirrors Milvus's loaded/released semantics:
        #   - Collections WITHOUT any IndexSpec auto-load on construction
        #   - Collections WITH IndexSpecs start as released; user must load()
        self._index_specs: Dict[str, IndexSpec] = dict(self._manifest.index_specs)
        self._load_state: str = "loaded" if not self._index_specs else "released"

        # ── 8. auto_id support (Phase 15) ──────────────────────────
        pk_field = get_primary_field(schema)
        self._auto_id: bool = pk_field.auto_id
        # _next_auto_id tracks the next ID to assign. We initialize it
        # from the manifest's current_seq to ensure monotonic growth
        # across restarts. This is safe because _seq is always >= any
        # previously assigned auto_id.
        self._next_auto_id: int = self._next_seq

        # ── 9. Function chain + search-time helpers ─────────────────
        from milvus_lite.function.builder import build_ingestion_chain

        field_by_name = {f.name: f for f in schema.fields}
        self._ingestion_chain = build_ingestion_chain(schema, field_by_name)

        # Search-time query helpers (not part of the chain — they transform
        # *queries* before vector search, not records or results).
        self._bm25_analyzers: List[Tuple[str, str, Any]] = []
        self._embedding_providers: List[Tuple[str, str, Any]] = []
        if schema.functions:
            from milvus_lite.analyzer.factory import create_analyzer
            for func in schema.functions:
                if func.function_type == FunctionType.BM25:
                    in_name = func.input_field_names[0]
                    out_name = func.output_field_names[0]
                    in_field = field_by_name[in_name]
                    analyzer = create_analyzer(in_field.analyzer_params)
                    self._bm25_analyzers.append((in_name, out_name, analyzer))
                elif func.function_type == FunctionType.TEXT_EMBEDDING:
                    from milvus_lite.embedding.factory import create_embedding_provider
                    in_name = func.input_field_names[0]
                    out_name = func.output_field_names[0]
                    provider = create_embedding_provider(func.params)
                    self._embedding_providers.append((in_name, out_name, provider))

        # ── 10. partition key (auto-bucket partitions) ──────────────
        pk_key_field = next(
            (f for f in schema.fields if f.is_partition_key), None
        )
        self._partition_key_field: Optional[str] = (
            pk_key_field.name if pk_key_field else None
        )
        self._num_partition_buckets: int = DEFAULT_NUM_PARTITIONS
        if self._partition_key_field is not None:
            for i in range(self._num_partition_buckets):
                bucket = f"{PARTITION_KEY_BUCKET_PREFIX}{i}"
                if not self._manifest.has_partition(bucket):
                    self._manifest.add_partition(bucket)

    # ── public API ──────────────────────────────────────────────

    def insert(
        self,
        records: List[dict],
        partition_name: str = DEFAULT_PARTITION,
    ) -> List[Any]:
        """Insert records into the collection. Returns the list of pks.

        Each record is validated up-front (fail fast — no partial state
        on validation error). After WAL+MemTable apply, if the MemTable
        has hit MEMTABLE_SIZE_LIMIT, a synchronous flush runs before
        returning.

        When ``is_partition_key`` is set on a schema field, records are
        automatically routed to bucket partitions based on the hash of
        that field's value. The caller's ``partition_name`` is ignored.
        """
        if not isinstance(records, list):
            raise TypeError(f"records must be a list, got {type(records).__name__}")
        if not records:
            return []

        # 1. auto-generate primary key IDs if auto_id is enabled
        if self._auto_id:
            id_start = self._next_auto_id
            self._next_auto_id += len(records)
            for i, r in enumerate(records):
                if self._pk_name not in r or r[self._pk_name] is None:
                    r[self._pk_name] = id_start + i

        # 2. auto-generate function output fields via ingestion chain
        if self._ingestion_chain:
            from milvus_lite.function.dataframe import DataFrame
            self._ingestion_chain.execute(DataFrame.from_records(records))

        # 3. validate every record up-front
        for r in records:
            validate_record(r, self._schema)

        # 4. partition key routing
        if self._partition_key_field is not None:
            return self._insert_with_partition_key(records)

        if not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)

        # 5. allocate seqs
        seq_start = self._next_seq
        self._next_seq += len(records)
        seqs = list(range(seq_start, seq_start + len(records)))

        # 6. build wal_data RecordBatch
        batch = self._build_wal_data_batch(records, partition_name, seqs)

        # 7. construct Operation and dispatch
        op = InsertOp(partition=partition_name, batch=batch)
        self._apply(op)

        # 8. trigger flush if we hit the size limit
        if self._memtable.size() >= MEMTABLE_SIZE_LIMIT:
            self._trigger_flush()

        return [r[self._pk_name] for r in records]

    def _insert_with_partition_key(self, records: List[dict]) -> List[Any]:
        """Route records to bucket partitions by hashing the partition key."""
        import hashlib
        field = self._partition_key_field
        n = self._num_partition_buckets

        # Group records by bucket, tracking original indices for ordered PK return
        buckets: Dict[str, List[tuple]] = {}  # bucket → [(orig_idx, record)]
        for orig_idx, r in enumerate(records):
            val = r.get(field)
            # Hash: consistent bucket assignment
            h = int(hashlib.md5(str(val).encode()).hexdigest(), 16) % n
            bucket_name = f"{PARTITION_KEY_BUCKET_PREFIX}{h}"
            buckets.setdefault(bucket_name, []).append((orig_idx, r))

        # Insert each bucket's records
        ordered_pks: List[tuple] = []  # [(orig_idx, pk)]
        for bucket_name, bucket_entries in buckets.items():
            bucket_records = [r for _, r in bucket_entries]
            seq_start = self._next_seq
            self._next_seq += len(bucket_records)
            seqs = list(range(seq_start, seq_start + len(bucket_records)))
            batch = self._build_wal_data_batch(bucket_records, bucket_name, seqs)
            op = InsertOp(partition=bucket_name, batch=batch)
            self._apply(op)
            for (orig_idx, r) in bucket_entries:
                ordered_pks.append((orig_idx, r[self._pk_name]))

        if self._memtable.size() >= MEMTABLE_SIZE_LIMIT:
            self._trigger_flush()

        # Return PKs in original input order
        ordered_pks.sort(key=lambda x: x[0])
        return [pk for _, pk in ordered_pks]

    def upsert(
        self,
        records: List[dict],
        partition_name: str = DEFAULT_PARTITION,
    ) -> List[Any]:
        """Upsert with partial update support.

        For each record whose pk already exists, reads the old record
        and merges new fields onto it so callers don't need to provide
        every field.  Records with new pks are inserted as-is (all
        required fields must be present).

        Returns the list of pks, same as insert().
        """
        if not isinstance(records, list):
            raise TypeError(f"records must be a list, got {type(records).__name__}")
        if not records:
            return []

        # Build merged records: for existing pks, fill missing fields
        # from the old record.
        merged: List[dict] = []
        for rec in records:
            pk = rec.get(self._pk_name)
            if pk is None:
                # No pk or auto_id — treat as a fresh insert
                merged.append(rec)
                continue

            old = self._get_raw(pk)
            if old is None:
                merged.append(rec)
                continue

            # Merge: old record is the base, new record overrides
            combined = dict(old)
            combined.update(rec)
            merged.append(combined)

        return self.insert(merged, partition_name=partition_name)

    def _get_raw(self, pk: Any) -> Optional[dict]:
        """Internal point read for a single pk, bypassing load-state check.

        Returns the record dict with dynamic fields unpacked and
        internal columns (_seq, _partition, $meta) stripped, or None.
        Used by upsert() to read old records for merging.
        """
        # 1. MemTable
        rec = self._memtable.get(pk)
        if rec is not None:
            rec.pop("$meta", None)
            return rec
        if self._memtable.is_locally_deleted(pk):
            return None

        # 2. Segments
        seg_snap, delta_snap = self._read_snapshot()
        best_seq = -1
        best_segment = None
        best_row_idx = -1
        for seg in seg_snap:
            row_idx = seg.find_row(pk)
            if row_idx is None:
                continue
            seq = int(seg.seqs[row_idx])
            if seq > best_seq:
                best_seq = seq
                best_segment = seg
                best_row_idx = row_idx

        if best_segment is not None:
            if not delta_snap.is_deleted(pk, best_seq):
                rec = best_segment.row_to_dict(best_row_idx)
                rec.pop("$meta", None)
                return rec

        return None

    def delete(
        self,
        pks: List[Any],
        partition_name: Optional[str] = None,
    ) -> int:
        """Delete a batch of pks. Returns the number of pks scheduled.

        ``partition_name=None`` is a cross-partition delete: the
        tombstone applies to whichever partition the pk currently lives
        in, and at flush time it is replicated into the delta files of
        every existing partition.

        Phase-5 semantics:
            - The whole batch shares ONE _seq (architectural invariant:
              batch delete is one logical event).
            - Deleting a non-existent pk is NOT an error — it just
              writes a tombstone that will never match anything.
            - This method does not return whether each pk actually
              existed; it returns ``len(pks)`` so the caller can
              distinguish "called with N" from "called with 0".
        """
        if not isinstance(pks, list):
            raise TypeError(f"pks must be a list, got {type(pks).__name__}")
        if not pks:
            return 0

        target_partition = partition_name if partition_name is not None else ALL_PARTITIONS

        # Validate the explicit partition exists. ALL_PARTITIONS is a
        # sentinel and is always valid.
        if partition_name is not None and not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)

        # Allocate ONE seq for the whole batch.
        seq = self._next_seq
        self._next_seq += 1

        batch = self._build_wal_delta_batch(pks, target_partition, seq)
        op = DeleteOp(partition=target_partition, batch=batch)
        self._apply(op)

        if self._memtable.size() >= MEMTABLE_SIZE_LIMIT:
            self._trigger_flush()

        return len(pks)

    def get(
        self,
        pks: List[Any],
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[dict]:
        """Point read across MemTable + segments.

        Lookup order per pk:
            1. MemTable._pk_index → live insert (newest possible state)
            2. MemTable._delete_index → live tombstone shadows everything
            3. Segments → scan for the largest seq across all segments
               in the requested partitions; check delta_index for an
               on-disk tombstone with a larger seq.

        If ``expr`` is provided, hit records are additionally filtered
        by the compiled scalar expression.

        Returns records in input pk order; missing pks are skipped
        (NOT padded with None).
        """
        if not isinstance(pks, list):
            raise TypeError(f"pks must be a list, got {type(pks).__name__}")

        self._require_loaded()

        partition_filter = set(partition_names) if partition_names else None
        compiled_filter = self._compile_filter(expr) if expr else None
        # Snapshot tombstones so the bg worker's gc_below doesn't
        # invalidate entries we rely on during this read.
        seg_snap, delta_snap = self._read_snapshot()

        out: List[dict] = []

        for pk in pks:
            rec: Optional[dict] = None

            # Step 1: live insert in MemTable.
            mt_rec = self._memtable.get(pk, partition_filter=partition_filter)
            if mt_rec is not None:
                rec = mt_rec
            elif self._memtable.is_locally_deleted(pk):
                # Step 2: live tombstone shadows any segment hit.
                continue
            else:
                # Step 3: scan segments for the latest version of pk.
                best_seq = -1
                best_segment: Optional[Segment] = None
                best_row_idx: int = -1
                for segment in seg_snap:
                    if partition_filter is not None and segment.partition not in partition_filter:
                        continue
                    row_idx = segment.find_row(pk)
                    if row_idx is None:
                        continue
                    seq = int(segment.seqs[row_idx])
                    if seq > best_seq:
                        best_seq = seq
                        best_segment = segment
                        best_row_idx = row_idx

                if best_segment is not None:
                    if not delta_snap.is_deleted(pk, best_seq):
                        rec = best_segment.row_to_dict(best_row_idx)

            if rec is None:
                continue

            # Apply optional scalar filter to the single hit row.
            if compiled_filter is not None and not _row_matches_filter(rec, compiled_filter):
                continue

            out.append(self._project_record(rec, output_fields))

        return out

    def search(
        self,
        query_vectors: List[list],
        top_k: int = 10,
        metric_type: str = "COSINE",
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        group_by_field: Optional[str] = None,
        group_size: int = 1,
        strict_group_size: bool = False,
        radius: Optional[float] = None,
        range_filter: Optional[float] = None,
        offset: int = 0,
        ranker: Optional[dict] = None,
    ) -> List[List[dict]]:
        """Vector top-k search.

        Args:
            query_vectors: list of length nq, each item a list of length dim
                (for FLOAT_VECTOR) or list of dict (for SPARSE_FLOAT_VECTOR).
            top_k: requested k (number of groups when group_by_field is set).
            metric_type: "COSINE" / "L2" / "IP" / "BM25".
            partition_names: optional partition filter.
            expr: optional Milvus-style scalar filter expression.
            output_fields: optional whitelist of fields to include in entity.
            anns_field: name of the vector field to search on.
            group_by_field: optional scalar field to group results by.
            group_size: number of results per group (default 1).
            strict_group_size: if True, discard groups with fewer than
                group_size results.
            radius: optional distance lower bound (exclusive).
            range_filter: optional distance upper bound (inclusive).
            offset: number of results to skip before returning (default 0).
            ranker: optional request-level Boost Ranker spec.

        Returns:
            List of length nq. Each inner list has dicts of shape
            ``{"id": pk, "distance": float, "entity": {field: value, ...}}``.
            When group_by_field is set, results are grouped and flattened
            (up to top_k * group_size total hits per query).
        """
        if not isinstance(query_vectors, list):
            raise TypeError(
                f"query_vectors must be a list, got {type(query_vectors).__name__}"
            )
        if not query_vectors:
            return []

        self._require_loaded()
        _user_output_fields = list(output_fields) if output_fields is not None else None

        # Validate group_by_field
        if group_by_field is not None:
            gf = next((f for f in self._schema.fields if f.name == group_by_field), None)
            if gf is None:
                raise SchemaValidationError(
                    f"group_by_field {group_by_field!r} not found in schema"
                )
            _GROUP_BY_ALLOWED = (
                DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
                DataType.BOOL, DataType.VARCHAR,
            )
            if gf.dtype not in _GROUP_BY_ALLOWED:
                raise SchemaValidationError(
                    f"group_by_field {group_by_field!r} has type {gf.dtype.name} "
                    f"which is not supported for group_by"
                )

        # Over-fetch when group_by, range search, or offset is active
        effective_top_k = top_k + offset
        if group_by_field is not None:
            effective_top_k = max((top_k + offset) * group_size * 3, (top_k + offset) * 10)
        if radius is not None or range_filter is not None:
            effective_top_k = max(effective_top_k, (top_k + offset) * 5)
        if ranker is not None:
            effective_top_k = max(effective_top_k, (top_k + offset) * 10)

        # Resolve the target vector field
        vector_field = self._resolve_anns_field(anns_field)
        field_schema = next(f for f in self._schema.fields if f.name == vector_field)

        _boost_field_injected = False
        if ranker is not None and output_fields is not None:
            requested = set(output_fields)
            output_fields = [
                f.name for f in self._schema.fields
                if f.name == self._pk_name or f.name in requested or not f.is_primary
            ]
            _boost_field_injected = True

        # If grouping is active, ensure the group key is available for
        # post-processing even when the user did not request it.
        _group_by_field_injected = False
        if group_by_field is not None and output_fields is not None:
            if group_by_field not in output_fields:
                output_fields = list(output_fields) + [group_by_field]
                _group_by_field_injected = True

        if field_schema.dtype == DataType.SPARSE_FLOAT_VECTOR:
            raw_results = self._search_sparse(
                query_vectors=query_vectors,
                vector_field=vector_field,
                top_k=effective_top_k,
                metric_type=metric_type,
                partition_names=partition_names,
                expr=expr,
                output_fields=output_fields,
            )
        else:
            # Dense float vector search — auto-embed text queries if needed
            query_vectors = self._maybe_embed_queries(query_vectors, vector_field)
            q_arr = np.asarray(query_vectors, dtype=np.float32)
            if q_arr.ndim != 2:
                raise ValueError(
                    f"query_vectors must be a 2-D list, got shape {q_arr.shape}"
                )
            compiled_filter = self._compile_filter(expr) if expr else None
            seg_snap, delta_snap = self._read_snapshot()
            raw_results = execute_search_with_index(
                query_vectors=q_arr,
                segments=seg_snap,
                memtable=self._memtable,
                delta_index=delta_snap,
                top_k=effective_top_k,
                metric_type=metric_type,
                pk_field=self._pk_name,
                vector_field=vector_field,
                partition_names=partition_names,
                compiled_filter=compiled_filter,
                output_fields=output_fields,
            )

        # Apply range filter (before group_by)
        if radius is not None or range_filter is not None:
            range_limit = effective_top_k if ranker is not None else top_k + offset
            raw_results = _apply_range_filter(
                raw_results, radius, range_filter, range_limit,
                metric_type=metric_type,
            )

        if ranker is not None:
            from milvus_lite.rerank.boost import apply_boost_ranker

            raw_results = apply_boost_ranker(
                raw_results,
                ranker,
                metric_type=metric_type,
                pk_name=self._pk_name,
                compile_filter=self._compile_filter,
                row_matches_filter=_row_matches_filter,
            )

        if group_by_field is not None:
            raw_results = _apply_group_by(
                raw_results, group_by_field, top_k + offset,
                group_size, strict_group_size,
            )
        if offset > 0:
            raw_results = [hits[offset:offset + top_k] for hits in raw_results]
        elif ranker is not None:
            raw_results = [hits[:top_k] for hits in raw_results]
        # Convert IP distances to Milvus convention
        if metric_type == "IP":
            for hits in raw_results:
                for hit in hits:
                    hit["distance"] = -hit["distance"]

        if _boost_field_injected or _group_by_field_injected:
            raw_results = _strip_injected_output_fields(raw_results, _user_output_fields)

        return raw_results

    def _resolve_anns_field(self, anns_field: Optional[str]) -> str:
        """Resolve the anns_field parameter to a concrete field name.

        Returns the first FLOAT_VECTOR field if anns_field is None.
        Validates that the field exists and is a vector type.
        """
        if anns_field is None:
            if self._vector_name is None:
                # Sparse-only collection — caller must specify anns_field
                # explicitly (e.g. the sparse vector field name).
                raise SchemaValidationError(
                    "collection has no FLOAT_VECTOR field; "
                    "specify anns_field explicitly for sparse search"
                )
            return self._vector_name

        field = next((f for f in self._schema.fields if f.name == anns_field), None)
        if field is None:
            raise SchemaValidationError(
                f"anns_field {anns_field!r} not found in schema"
            )
        if field.dtype not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR):
            raise SchemaValidationError(
                f"anns_field {anns_field!r} is not a vector field "
                f"(dtype={field.dtype.name})"
            )
        return anns_field

    def _search_sparse(
        self,
        query_vectors: List,
        vector_field: str,
        top_k: int,
        metric_type: str,
        partition_names: Optional[List[str]],
        expr: Optional[str],
        output_fields: Optional[List[str]],
    ) -> List[List[dict]]:
        """Sparse vector search using per-segment cached BM25 indexes.

        Architecture (Perf-3):
        - Each immutable segment gets a cached SparseInvertedIndex
          (built once, reused across searches).
        - The mutable memtable's index is rebuilt each search (small).
        - Per-source top-k results are merged globally.

        TODO: IDF accuracy — each segment currently uses its own IDF
        statistics, so BM25 scores from different segments have different
        baselines. Fix: aggregate global statistics (doc_count/avgdl/df
        summed across segments) at search time and use global IDF for
        scoring. Similar to Elasticsearch's DFS_QUERY_THEN_FETCH strategy.
        """
        from milvus_lite.analyzer.sparse import bytes_to_sparse
        from milvus_lite.index.sparse_inverted import SparseInvertedIndex

        partition_filter = set(partition_names) if partition_names else None
        _exclude_fields = {f.name for f in self._schema.fields
                          if f.is_primary or f.dtype in (
                              DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR)}

        # BM25 params
        bm25_k1 = 1.5
        bm25_b = 0.75
        sparse_spec = self._index_specs.get(vector_field)
        if sparse_spec and sparse_spec.index_type == "SPARSE_INVERTED_INDEX":
            bm25_k1 = sparse_spec.build_params.get("bm25_k1", 1.5)
            bm25_b = sparse_spec.build_params.get("bm25_b", 0.75)

        # Convert query vectors upfront
        query_sparse = self._prepare_sparse_queries(query_vectors)
        nq = len(query_sparse)

        # Per-source candidates: (distance, global_pk, source_ref)
        Candidate = Tuple[float, Any, Any]  # (dist, pk, (tbl, row_idx))
        per_query_candidates: List[List[Candidate]] = [[] for _ in range(nq)]

        # Read snapshot — keeps segment cache and tombstones from crossing
        # compaction / tombstone-GC generations.
        seg_snapshot, delta_snap = self._read_snapshot()

        # ── Build global pk→best_seq map for cross-segment dedup ──
        global_pk_seq: Dict[Any, int] = {}
        for seg in seg_snapshot:
            if partition_filter is not None:
                if seg.partition not in partition_filter:
                    continue
            for i, pk in enumerate(seg.pks):
                seq = int(seg.seqs[i])
                if pk not in global_pk_seq or seq > global_pk_seq[pk]:
                    global_pk_seq[pk] = seq
        # Memtable pks always win (highest seq)
        for pk, (_, _, seq) in self._memtable.pk_index_snapshot():
            global_pk_seq[pk] = max(seq, global_pk_seq.get(pk, -1))

        # ── Per-segment search (cached indexes) ──────────────────
        for seg in seg_snapshot:
            if partition_filter is not None:
                if seg.partition not in partition_filter:
                    continue
            table = seg.table
            if table is None or len(table) == 0:
                continue

            # Build or reuse cached sparse index for this segment
            cache_key = f"_sparse_{vector_field}"
            cached_idx = seg.indexes.get(cache_key)
            if cached_idx is None:
                sparse_batch = table.column(vector_field).to_pylist()
                sparse_vecs = [
                    bytes_to_sparse(r) if isinstance(r, bytes) else (r or {})
                    for r in sparse_batch
                ]
                cached_idx = SparseInvertedIndex(k1=bm25_k1, b=bm25_b)
                cached_idx.build(sparse_vecs)  # no valid_mask — full segment
                seg.attach_index(cached_idx, field_name=cache_key)

            # Build valid_mask for this segment (dedup + tombstone + filter)
            pks = seg.pks
            seqs = seg.seqs
            n = len(pks)
            valid_mask = np.ones(n, dtype=bool)
            for i in range(n):
                pk, seq = pks[i], int(seqs[i])
                if delta_snap.is_deleted(pk, seq):
                    valid_mask[i] = False
                elif global_pk_seq.get(pk, -1) > seq:
                    # Stale version — a newer version exists in another segment or memtable
                    valid_mask[i] = False

            # Apply scalar filter
            if expr:
                compiled = self._compile_filter(expr)
                from milvus_lite.search.filter.eval import evaluate as filter_evaluate
                fmask = filter_evaluate(compiled, table).to_numpy(zero_copy_only=False)
                valid_mask = valid_mask & fmask

            if not valid_mask.any():
                continue

            # Search this segment's cached index
            local_ids, dists = cached_idx.search(query_sparse, top_k, valid_mask=valid_mask)
            for qi in range(nq):
                for j in range(top_k):
                    lid = int(local_ids[qi, j])
                    if lid < 0:
                        break
                    per_query_candidates[qi].append(
                        (float(dists[qi, j]), pks[lid], (table, lid))
                    )

        # ── Memtable search (rebuilt each time — small + mutable) ─
        mt = self._memtable
        mt_pks: list = []
        mt_sparse: list = []
        mt_refs: list = []
        for pk, seq, record in mt.active_record_snapshots(
            partition_names=partition_names
        ):
            raw = record.get(vector_field)
            mt_pks.append(pk)
            mt_sparse.append(
                bytes_to_sparse(raw) if isinstance(raw, bytes) else (raw or {})
            )
            mt_refs.append((record, seq))

        if mt_pks:
            mt_valid = np.ones(len(mt_pks), dtype=bool)

            if expr:
                compiled = self._compile_filter(expr)
                from milvus_lite.search.filter.eval.python_backend import _eval_row
                for i in range(len(mt_pks)):
                    if mt_valid[i]:
                        record, _seq = mt_refs[i]
                        if not _eval_row(compiled.ast, record):
                            mt_valid[i] = False

            mt_idx = SparseInvertedIndex(k1=bm25_k1, b=bm25_b)
            mt_idx.build(mt_sparse, valid_mask=mt_valid)
            local_ids, dists = mt_idx.search(query_sparse, top_k)
            for qi in range(nq):
                for j in range(top_k):
                    lid = int(local_ids[qi, j])
                    if lid < 0:
                        break
                    per_query_candidates[qi].append(
                        (float(dists[qi, j]), mt_pks[lid], mt_refs[lid])
                    )

        # ── Global merge + dedup + materialize ────────────────────
        # Dedup: if same pk appears from segment + memtable, keep
        # the one with the highest seq (latest version).
        def _is_memtable_sparse_ref(source: Any) -> bool:
            return isinstance(source, tuple) and bool(source) and isinstance(source[0], dict)

        results: List[List[dict]] = []
        for qi in range(nq):
            candidates = per_query_candidates[qi]
            # First, deduplicate by pk keeping the latest version (highest seq).
            pk_best: dict = {}  # pk → (dist, pk, (tbl, row_i))
            for cand in candidates:
                dist, pk, source = cand
                if _is_memtable_sparse_ref(source):
                    _record, seq = source
                    seq = int(seq)
                else:
                    tbl, row_i = source
                    seq = int(tbl.column("_seq")[row_i].as_py())
                if pk not in pk_best or seq > pk_best[pk][0]:
                    pk_best[pk] = (seq, cand)
            deduped = [v[1] for v in pk_best.values()]
            # Sort by distance ascending (smaller = better)
            deduped.sort(key=lambda c: c[0])
            hits: list = []
            for dist, pk, source in deduped:
                # Deferred materialization
                entity = {}
                if _is_memtable_sparse_ref(source):
                    source, _seq = source
                    if output_fields is None:
                        entity = {
                            k: v for k, v in source.items()
                            if k not in ("_seq", "_partition") and k not in _exclude_fields
                        }
                    elif output_fields:
                        entity = {
                            fname: source[fname]
                            for fname in output_fields
                            if fname != self._pk_name and fname in source
                        }
                else:
                    tbl, row_i = source
                    if output_fields is None:
                        for col in tbl.column_names:
                            if col in ("_seq", "_partition") or col in _exclude_fields:
                                continue
                            entity[col] = tbl.column(col)[row_i].as_py()
                    elif output_fields:
                        for fname in output_fields:
                            if fname == self._pk_name:
                                continue
                            entity[fname] = tbl.column(fname)[row_i].as_py()
                hits.append({"id": pk, "distance": dist, "entity": entity})
                if len(hits) >= top_k:
                    break
            results.append(hits)

        return results

    def _maybe_embed_queries(self, query_vectors: List, vector_field: str) -> List:
        """If query_vectors contains strings and this field has a TEXT_EMBEDDING
        function, auto-embed them. Otherwise return as-is."""
        if not query_vectors or not isinstance(query_vectors[0], str):
            return query_vectors

        # Find the embedding provider for this vector field
        provider = None
        for _in, out, prov in self._embedding_providers:
            if out == vector_field:
                provider = prov
                break

        if provider is None:
            raise SchemaValidationError(
                f"Text query on field {vector_field!r} requires a "
                f"TEXT_EMBEDDING function targeting that field"
            )

        embedded = []
        for qv in query_vectors:
            if isinstance(qv, str):
                embedded.append(provider.embed_query(qv))
            else:
                embedded.append(qv)
        return embedded

    def _prepare_sparse_queries(self, query_vectors: List) -> List[Dict[int, float]]:
        """Convert query vectors to sparse dicts (text → tokenize → TF)."""
        query_sparse: List[Dict[int, float]] = []
        for qv in query_vectors:
            if isinstance(qv, dict):
                query_sparse.append(qv)
            elif isinstance(qv, str):
                analyzer = self._bm25_analyzers[0][2] if self._bm25_analyzers else None
                if analyzer is None:
                    raise SchemaValidationError(
                        "Text query requires a BM25 function with an analyzer"
                    )
                from milvus_lite.analyzer.sparse import compute_tf
                query_sparse.append(compute_tf(analyzer.analyze(qv)))
            else:
                raise SchemaValidationError(
                    f"Sparse search query must be a dict or string, "
                    f"got {type(qv).__name__}"
                )
        return query_sparse

    def query(
        self,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[dict]:
        """Pure scalar query — no vector, no distance.

        Returns all records matching the filter expression. Used for
        Milvus-style query() workflows where you just want to find rows
        by their scalar attributes.

        Args:
            expr: Milvus-style filter expression. None or empty string
                means "return all records" (used by query_iterator).
            output_fields: subset of fields to include in returned dicts.
                None means all schema fields (with _seq / _partition stripped).
                The pk field is always included.
            partition_names: optional partition filter
            limit: max number of rows to return; None = unbounded

        Returns:
            List of dicts (each a record matching the filter). Order is
            "segments first, then MemTable" — within each source, the
            order is the underlying iteration order. No top-k sort.
        """
        if expr is not None and not isinstance(expr, str):
            raise TypeError("query() expr must be a string or None")

        self._require_loaded()

        compiled_filter = self._compile_filter(expr) if expr else None

        seg_snap, delta_snap = self._read_snapshot()

        all_pks, all_seqs, _all_vectors, all_rec_sources, filter_mask = assemble_candidates(
            segments=seg_snap,
            memtable=self._memtable,
            vector_field=self._vector_name,
            partition_names=partition_names,
            filter_compiled=compiled_filter,
        )

        if not all_pks:
            return []

        # Combine bitmap (dedup + tombstone) with filter_mask via build_valid_mask.
        from milvus_lite.search.bitmap import build_valid_mask
        from milvus_lite.search.assembler import materialize_record
        # Use the DeltaIndex captured with seg_snap above; bg GC may
        # mutate the live DeltaIndex after candidate assembly starts.
        mask = build_valid_mask(
            all_pks, all_seqs, delta_snap,
            filter_mask=filter_mask, memtable=self._memtable,
        )

        # Deferred materialization: only materialize records that pass the mask.
        effective_limit = (offset + limit) if limit is not None else None
        live_indices = np.flatnonzero(mask)
        out: List[dict] = []
        for i in live_indices:
            rec = materialize_record(all_rec_sources[int(i)])
            out.append(self._project_record(rec, output_fields))
            if effective_limit is not None and len(out) >= effective_limit:
                break
        return out[offset:]

    def _index_dir(self, partition: str) -> str:
        """Phase 9.4: canonical path for a partition's index sidecar dir.

        Layout: ``data_dir/partitions/<partition>/indexes/``

        The directory is created on demand by build_or_load_index when
        the first .idx is written.
        """
        return os.path.join(self._data_dir, "partitions", partition, "indexes")

    def _require_loaded(self) -> None:
        """Phase 9.3 guard: search/get/query require loaded state.

        Collections without an IndexSpec are auto-loaded on construction
        (see __init__), so this only fires after explicit create_index +
        no load(), or after explicit release().
        """
        if self._load_state != "loaded":
            raise CollectionNotLoadedError(
                f"Collection {self._name!r} is in state {self._load_state!r}; "
                f"call load() before search/get/query"
            )

    def _compile_filter(self, expr_str: str) -> "CompiledExpr":
        """Parse + compile a filter expression, with LRU caching.

        The cache is keyed only on the expression string because the
        schema is implicit (this Collection's). Schema is immutable for
        the lifetime of a Collection, so cached entries never go stale.
        """
        cached = self._filter_cache.get(expr_str)
        if cached is not None:
            return cached
        from milvus_lite.search.filter import compile_filter
        compiled = compile_filter(expr_str, self._schema)
        self._filter_cache.put(expr_str, compiled)
        return compiled

    def _project_record(
        self,
        record: dict,
        output_fields: Optional[List[str]],
    ) -> dict:
        """Apply output_fields projection to a record dict.

        - None → return all fields (stripping internal $meta key)
        - list → keep only the named fields, plus the pk field
        """
        if output_fields is None:
            return {k: v for k, v in record.items() if k != "$meta"}
        keep = set(output_fields)
        keep.add(self._pk_name)
        keep.discard("$meta")
        return {k: v for k, v in record.items() if k in keep}

    def flush(self) -> None:
        """Force a synchronous flush of the current MemTable.

        No-op if the MemTable is empty.
        """
        if self._memtable.size() == 0:
            return
        self._trigger_flush()

    # ── partition CRUD (Phase 9.1) ──────────────────────────────

    def create_partition(self, partition_name: str) -> None:
        """Create a new partition.

        - Registers the partition on the manifest (raises
          PartitionAlreadyExistsError if already there).
        - Persists the manifest atomically.
        - Creates the on-disk partition directory so flush can write
          into it later. The dir is empty at this point.
        """
        if self._partition_key_field is not None:
            raise SchemaValidationError(
                "cannot create manual partitions when partition key is set "
                f"(partition_key field: {self._partition_key_field!r})"
            )
        with self._maintenance_lock:
            self._manifest.add_partition(partition_name)
            self._manifest.save()
        partition_dir = os.path.join(
            self._data_dir, "partitions", partition_name
        )
        os.makedirs(partition_dir, exist_ok=True)

    def drop_partition(self, partition_name: str) -> None:
        """Drop a partition and remove all its on-disk files.

        - Forbidden for the default partition (raises
          DefaultPartitionError via manifest).
        - Raises PartitionNotFoundError if the partition doesn't exist.
        - Auto-flushes any pending MemTable rows first so we don't
          lose live writes that target this partition.
        - Removes the partition from the manifest, then deletes the
          on-disk partition directory (which contains data + delta +
          future indexes/).
        - Drops any cached Segments belonging to this partition.

        Tombstones in delta_index for the dropped partition's pks are
        left intact — they will be GC'd by the regular tombstone GC
        once min_active_data_seq advances past them. This is safe
        because dropping a partition means there is no longer any
        live data row those tombstones could shadow.
        """
        if self._partition_key_field is not None:
            raise SchemaValidationError(
                "cannot drop partitions when partition key is set "
                f"(partition_key field: {self._partition_key_field!r})"
            )
        # Validate first so we don't trigger an unnecessary flush.
        if not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)

        # Flush any pending writes so we don't drop in-flight rows
        # silently (the user's "insert then drop" should not lose
        # the inserts).  Flush is outside the lock — _trigger_flush
        # acquires _maintenance_lock internally (RLock, re-entrant).
        if self._memtable.size() > 0:
            self._trigger_flush()

        with self._maintenance_lock:
            # remove_partition raises DefaultPartitionError or
            # PartitionNotFoundError as appropriate.  Re-validates
            # existence under lock (guards against concurrent drop).
            self._manifest.remove_partition(partition_name)
            self._manifest.save()

            # Drop in-memory segment cache entries for this partition.
            for key in list(self._segment_cache.keys()):
                if key[0] == partition_name:
                    del self._segment_cache[key]

        # Remove on-disk partition directory.  Safe outside the lock:
        # the partition is already gone from the manifest, so bg
        # compaction (which iterates manifest.list_partitions under
        # the lock) will never touch this directory.
        partition_dir = os.path.join(
            self._data_dir, "partitions", partition_name
        )
        if os.path.exists(partition_dir):
            import shutil
            shutil.rmtree(partition_dir, ignore_errors=False)

    def list_partitions(self) -> List[str]:
        """Return all partition names, sorted."""
        return self._manifest.list_partitions()

    def has_partition(self, partition_name: str) -> bool:
        """Check whether a partition exists."""
        return self._manifest.has_partition(partition_name)

    # ── index lifecycle (Phase 9.3) ─────────────────────────────

    def create_index(
        self,
        field_name: str,
        index_params: dict,
    ) -> None:
        """Persist an IndexSpec on the manifest. Does NOT build any
        index here — that happens at load() time, mirroring Milvus.

        Args:
            field_name: must be a vector field declared in the schema.
            index_params: dict containing at minimum::

                {
                    "index_type":  "HNSW" | "BRUTE_FORCE" | ...,
                    "metric_type": "COSINE" | "L2" | "IP",
                    "params":      {...},   # optional, build_params
                    "search_params": {...}, # optional, search defaults
                }

        Raises:
            IndexAlreadyExistsError: an index already exists
            SchemaValidationError:   field_name doesn't exist or isn't
                                     a vector field
            ValueError:              metric_type / index_type missing or invalid

        Side effect: collection moves to ``released`` state. The user
        must call ``load()`` to actually build segment indexes and
        re-enable search.
        """
        with self._maintenance_lock:
            if field_name in self._index_specs:
                raise IndexAlreadyExistsError(
                    f"index already exists for field {field_name!r}; "
                    f"call drop_index first"
                )

            # Validate the field is in the schema and is a vector type.
            target = next((f for f in self._schema.fields if f.name == field_name), None)
            if target is None:
                raise SchemaValidationError(
                    f"unknown field {field_name!r} for create_index"
                )
            if target.dtype not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR):
                raise SchemaValidationError(
                    f"field {field_name!r} has type {target.dtype.name}; "
                    f"create_index only supports vector fields"
                )

            spec = IndexSpec(
                field_name=field_name,
                index_type=index_params["index_type"],
                metric_type=index_params["metric_type"],
                build_params=dict(index_params.get("params") or {}),
                search_params=dict(index_params.get("search_params") or {}),
            )

            self._index_specs[field_name] = spec
            self._manifest.set_index_spec(spec)
            self._manifest.save()

            # Milvus semantics: create_index preserves load state. If
            # loaded, build indexes inline for existing segments so search
            # works immediately. If released, the build is lazy on load().
            if self._load_state == "loaded":
                for seg in self._segment_cache.values():
                    if seg.num_rows > 0:
                        seg.build_or_load_index(spec, self._index_dir(seg.partition))

    def drop_index(self, field_name: Optional[str] = None) -> None:
        """Remove the IndexSpec, release in-memory indexes, and delete
        on-disk .idx files.

        Args:
            field_name: optional; if given, must match the existing
                spec's field_name. None means "drop whatever index is
                there" (matches Milvus's drop_index without args).

        Raises:
            IndexNotFoundError: no index has been created

        Phase 9.4: also walks every partition's ``indexes/`` directory
        and deletes the .idx files matching the dropped index_type.
        Other index_type files (if any — currently impossible since we
        only support one index per Collection) are left alone.
        """
        with self._maintenance_lock:
            if not self._index_specs:
                raise IndexNotFoundError("no index to drop")
            if field_name is not None and field_name not in self._index_specs:
                raise IndexNotFoundError(
                    f"no index on field {field_name!r}; "
                    f"indexed fields: {list(self._index_specs.keys())}"
                )
            # Milvus semantics: drop_index is blocked when the collection
            # is loaded. Caller must release() first.
            if self._load_state == "loaded":
                raise SchemaValidationError(
                    "vector index cannot be dropped on loaded collection; "
                    "call release() first"
                )

            # Determine which spec(s) to drop
            if field_name is not None:
                drop_specs = [self._index_specs[field_name]]
            else:
                drop_specs = list(self._index_specs.values())

            # Release in-memory indexes for the affected fields.
            for spec in drop_specs:
                for seg in self._segment_cache.values():
                    seg.release_index(field_name=spec.field_name)

            # Delete on-disk .idx files matching the dropped (field, type)
            # pair. File format: <stem>.<field>.<type>.idx
            for spec in drop_specs:
                suffix = f".{spec.field_name}.{spec.index_type.lower()}.idx"
                for partition in self._manifest.list_partitions():
                    index_dir = self._index_dir(partition)
                    if not os.path.exists(index_dir):
                        continue
                    for entry in os.listdir(index_dir):
                        if entry.endswith(suffix):
                            try:
                                os.remove(os.path.join(index_dir, entry))
                            except OSError:
                                pass

            # Remove from specs
            for spec in drop_specs:
                del self._index_specs[spec.field_name]
                self._manifest.remove_index_spec(spec.field_name)
            self._manifest.save()

            # If no indexes remain, auto-load (backward compat).
            if not self._index_specs:
                self._load_state = "loaded"

    def has_index(self, field_name: Optional[str] = None) -> bool:
        """True iff create_index has been called (and not dropped).
        If field_name is given, checks that specific field."""
        if field_name is not None:
            return field_name in self._index_specs
        return bool(self._index_specs)

    def list_indexes(self) -> List[str]:
        """Return index names for this collection.

        pymilvus's search_iterator v1 matches ``index_name`` against
        ``anns_field``, so MilvusLite reports the field name as the
        public index name.
        """
        return sorted(self._index_specs)

    def get_index_info(self, field_name: Optional[str] = None) -> Optional[dict]:
        """Return IndexSpec as dict. If field_name is None, returns first."""
        if field_name is not None:
            spec = self._index_specs.get(field_name)
            return spec.to_dict() if spec else None
        if not self._index_specs:
            return None
        return next(iter(self._index_specs.values())).to_dict()

    def load(self) -> None:
        """Move to the loaded state. Build or load a VectorIndex per
        segment if an IndexSpec exists; idempotent if already loaded.

        Phase 9.4: indexes are persisted to disk. The first load() after
        a fresh create_index builds them and writes .idx sidecars; every
        subsequent load() (including after process restart) reads them
        back via Segment.build_or_load_index, so cold-start is fast.

        Phase 9.3-9.4 only routes to BruteForceIndex; Phase 9.5 will
        plug in the factory + FaissHnswIndex.

        Raises any exception encountered during build, with the state
        machine rolled back to released.
        """
        with self._maintenance_lock:
            if self._load_state == "loaded":
                return
            self._load_state = "loading"
            try:
                for spec in self._index_specs.values():
                    for seg in self._segment_cache.values():
                        if seg.num_rows == 0:
                            continue
                        seg.build_or_load_index(
                            spec, self._index_dir(seg.partition)
                        )
                self._load_state = "loaded"
            except Exception:
                self._load_state = "released"
                raise

    def release(self) -> None:
        """Drop all in-memory segment indexes; subsequent search() raises
        ``CollectionNotLoadedError`` until load() is called again.

        No-op if there's no IndexSpec (such collections never enter
        the released state — see Collection.__init__ for the rationale).
        """
        with self._maintenance_lock:
            if not self._index_specs:
                return
            for seg in self._segment_cache.values():
                seg.release_index()
            self._load_state = "released"

    @property
    def load_state(self) -> str:
        """Current load state: 'released' | 'loading' | 'loaded'.

        Mirrors Milvus's GetLoadState response. The Phase 10 gRPC
        adapter maps this directly to milvus.LoadState enum.
        """
        return self._load_state

    # ── statistics & introspection (Phase 9.1) ──────────────────

    @property
    def name(self) -> str:
        """Collection name (read-only)."""
        return self._name

    @property
    def schema(self) -> CollectionSchema:
        """Collection schema (read-only)."""
        return self._schema

    @property
    def num_entities(self) -> int:
        """Approximate live row count across MemTable + segments.

        Walks pks + seqs only (no record materialization), then runs
        the same bitmap pipeline as search to dedup upserts and apply
        tombstones. O(N) where N is the total candidate row count.

        This is the value pymilvus's get_collection_stats reports as
        ``row_count``.
        """
        return self._num_entities(partition_names=None)

    def partition_num_entities(self, partition_name: str) -> int:
        """Approximate live row count for one partition."""
        if not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)
        return self._num_entities(partition_names=[partition_name])

    def _num_entities(self, partition_names: Optional[List[str]]) -> int:
        pk_chunks: List[List[Any]] = []
        seq_chunks: List[np.ndarray] = []

        seg_snap, delta_snap = self._read_snapshot()

        for seg in seg_snap:
            if partition_names is not None and seg.partition not in partition_names:
                continue
            if seg.num_rows == 0:
                continue
            pk_chunks.append(list(seg.pks))
            seq_chunks.append(seg.seqs)

        mt_pks, mt_seqs, _vecs, _records = self._memtable.to_search_arrays(
            vector_field=self._vector_name,
            partition_names=partition_names,
        )
        if mt_pks:
            pk_chunks.append(mt_pks)
            seq_chunks.append(mt_seqs)

        if not pk_chunks:
            return 0

        all_pks: List[Any] = []
        for c in pk_chunks:
            all_pks.extend(c)
        all_seqs = np.concatenate(seq_chunks)

        from milvus_lite.search.bitmap import build_valid_mask
        mask = build_valid_mask(
            all_pks, all_seqs, delta_snap,
            memtable=self._memtable,
        )
        return int(mask.sum())

    def describe(self) -> dict:
        """Return a dict summarizing the Collection.

        Used by pymilvus's describe_collection mapping in Phase 10.
        Mirrors the shape Milvus returns: collection name + schema +
        partition list + row count + index info + load state.

        """
        return {
            "name": self._name,
            "schema": {
                "fields": [
                    {
                        "name": f.name,
                        "dtype": f.dtype.name,
                        "is_primary": f.is_primary,
                        "nullable": f.nullable,
                        "dim": f.dim,
                        "max_length": f.max_length,
                    }
                    for f in self._schema.fields
                ],
                "enable_dynamic_field": self._schema.enable_dynamic_field,
            },
            "partitions": self.list_partitions(),
            "num_entities": self.num_entities,
            "load_state": self._load_state,
            "index_specs": {
                k: v.to_dict() for k, v in self._index_specs.items()
            },
        }

    # ── orchestration ───────────────────────────────────────────

    def _apply(self, op: Operation) -> None:
        """Single write entry point.

        Dispatches Operation to WAL and MemTable. Storage layer methods
        take raw batches (no Operation knowledge), so the dispatch is
        explicit here.
        """
        if isinstance(op, InsertOp):
            self._wal.write_insert(op.batch)
            self._memtable.apply_insert(op.batch)
        else:  # DeleteOp
            self._wal.write_delete(op.batch)
            self._memtable.apply_delete(op.batch)

    def _trigger_flush(self) -> None:
        """Flush pipeline — synchronous up to data persistence, then
        compaction + index build are offloaded to the background worker.

        Synchronous (user thread blocks):
            1. Freeze MemTable + WAL, swap in fresh ones
            2. Write data/delta parquet files
            3. Atomic manifest commit
            4. Refresh segment cache for newly flushed segments
            5. Build index for newly flushed segments (fast, small segments)

        Asynchronous (background thread):
            6. Run compaction (can merge up to 200K rows, minutes of IO)
            7. Build index on merged segments (can be slow for HNSW_SQ)

        Insert throughput is bounded by the sync path only. The bg queue
        is single-threaded so compaction tasks serialize naturally.
        """
        # ── Step 1: freeze ──────────────────────────────────────
        frozen_memtable = self._memtable
        frozen_wal = self._wal
        new_wal_number = frozen_wal.number + 1

        self._memtable = MemTable(self._schema)
        wal_dir = os.path.join(self._data_dir, "wal")
        self._wal = WAL(
            wal_dir=wal_dir,
            wal_data_schema=self._wal_data_schema,
            wal_delta_schema=self._wal_delta_schema,
            wal_number=new_wal_number,
        )

        # ── Steps 2-4: sync execute_flush under the maintenance lock.
        # Only data persistence + cache swap happen here. Index building
        # (even for small newly-flushed segments) is deferred to the bg
        # worker so insert latency is bounded by parquet/manifest IO only.
        with self._maintenance_lock:
            execute_flush(
                frozen_memtable=frozen_memtable,
                frozen_wal=frozen_wal,
                data_dir=self._data_dir,
                schema=self._schema,
                manifest=self._manifest,
                delta_index=self._delta_index,
                new_wal_number=new_wal_number,
            )
            self._refresh_segment_cache()

        # ── Steps 5-7: async compaction + index build
        self._schedule_bg_maintenance()

    def _schedule_bg_maintenance(self) -> None:
        """Submit compaction + post-compaction index build to bg worker.

        Single-threaded executor serializes tasks so compaction steps
        across partitions run in order. Exceptions are logged but not
        propagated to the user thread.
        """
        if self._bg_closed:
            return
        self._bg_executor.submit(self._bg_compact_and_index)

    def _wait_for_bg(self, timeout: Optional[float] = None) -> None:
        """Block until all pending background tasks complete.

        Intended for tests and for explicit drain points (e.g. before
        a manifest snapshot). Does not prevent new tasks from being
        scheduled after it returns.
        """
        # Submitting a no-op future and waiting on it forces serial
        # draining through the single-threaded executor.
        if self._bg_closed:
            return
        fut = self._bg_executor.submit(lambda: None)
        fut.result(timeout=timeout)

    def _bg_compact_and_index(self) -> None:
        """Background worker body. Serialized by the single-worker pool.

        Lock scope is tight: compaction (manifest mutation + cache swap)
        holds the lock, then releases it during the slow index-build
        step. This keeps concurrent user-thread flushes short-blocked
        even when HNSW/HNSW_SQ builds take minutes.
        """
        try:
            # Phase A: compaction (manifest + cache mutation) — under lock.
            with self._maintenance_lock:
                for partition in self._manifest.list_partitions():
                    self._compaction_mgr.maybe_compact(
                        partition, self._manifest, self._delta_index
                    )
                self._refresh_segment_cache()
                self._cleanup_orphan_index_files()

            # Phase B: index build — outside the lock. build_or_load_index
            # operates on an immutable Segment and writes a dedicated .idx
            # file, so concurrent user-thread flushes are safe.
            import time as _time
            t0 = _time.monotonic()
            built = self._ensure_loaded_segments_indexed()
            if built > 0:
                logger.info(
                    "bg: built %d segment indexes in %.2fs",
                    built, _time.monotonic() - t0,
                )
        except Exception:
            logger.exception(
                "background compaction/index build failed; will retry "
                "on next flush"
            )

    def _ensure_loaded_segments_indexed(self) -> int:
        """Phase 9.4: post-flush / post-compaction index hook.

        For every segment in the cache that lacks an attached index,
        build/load it. No-op when:
            - Collection has no IndexSpec (nothing to build)
            - Collection is not in 'loaded' state (the user explicitly
              released, so we don't bring it back)
        Already-attached segments are skipped by build_or_load_index.

        Returns number of indexes built/loaded.
        """
        if self._load_state != "loaded" or not self._index_specs:
            return 0
        # Snapshot both collections — bg index build runs without the
        # maintenance lock, so the main thread may concurrently add
        # new segments to _segment_cache or touch _index_specs.
        specs = tuple(self._index_specs.values())
        segs = self._segments_snapshot()
        built = 0
        for spec in specs:
            for seg in segs:
                if spec.field_name not in seg.indexes and seg.num_rows > 0:
                    seg.build_or_load_index(
                        spec, self._index_dir(seg.partition)
                    )
                    built += 1
        return built

    def _cleanup_orphan_index_files(self) -> None:
        """Phase 9.4: delete .idx files whose source segment is gone.

        Called from _trigger_flush after _refresh_segment_cache (which
        evicts compaction-removed segments). The cleanup compares the
        on-disk indexes/ directories against the manifest's data file
        list and removes any .idx whose stem doesn't match a current
        data file.

        This is the architectural safety net for invariant §11
        (index 1:1 bound to data; lifecycles strictly aligned).
        """
        if not self._index_specs:
            return

        # File format: <data_stem>.<field>.<index_type>.idx
        # Build {field → index_type_lower} so we can validate both parts.
        expected: Dict[str, str] = {
            spec.field_name: spec.index_type.lower()
            for spec in self._index_specs.values()
        }
        for partition, data_files in self._manifest.get_all_data_files().items():
            index_dir = self._index_dir(partition)
            if not os.path.exists(index_dir):
                continue
            valid_stems = {
                os.path.splitext(os.path.basename(df))[0] for df in data_files
            }
            for entry in os.listdir(index_dir):
                if not entry.endswith(".idx"):
                    continue
                base = entry[: -len(".idx")]
                stem_field, _, idx_type = base.rpartition(".")
                stem, _, field = stem_field.rpartition(".")
                # Drop if (a) source data gone, or (b) field/type no
                # longer in the active index_specs (dropped index).
                if (
                    not stem
                    or stem not in valid_stems
                    or expected.get(field) != idx_type
                ):
                    try:
                        os.remove(os.path.join(index_dir, entry))
                    except OSError:
                        pass

    def _segments_snapshot(self) -> Tuple["Segment", ...]:
        """Thread-safe snapshot of the segment cache values.

        ``dict.values()`` is a live view — iterating it while the bg
        worker mutates ``_segment_cache`` raises RuntimeError. Readers
        (search/query/get/num_entities) take a snapshot once so they
        aren't perturbed by concurrent compaction.

        Segment objects themselves are immutable w.r.t. their data
        (pks/seqs/vectors/table are frozen after load), so operating on
        an evicted Segment is still correct — just a slightly stale
        view of the dataset.
        """
        with self._maintenance_lock:
            return tuple(self._segment_cache.values())

    def _read_snapshot(self) -> Tuple[Tuple["Segment", ...], Any]:
        """Return a consistent read snapshot of segments and tombstones.

        Compaction can replace segment files and then GC tombstones that
        are no longer needed by the new segment generation. Readers must
        not combine an old segment snapshot with a post-GC DeltaIndex,
        or deleted rows in the old segments can become visible again.
        """
        with self._maintenance_lock:
            return (
                tuple(self._segment_cache.values()),
                self._delta_index.frozen_copy(),
            )

    def _refresh_segment_cache(self) -> None:
        """Reconcile self._segment_cache with the manifest's data files.

        - Adds segments for any newly-written files.
        - Drops segments for files no longer referenced (e.g. after
          compaction in Phase 6).
        - Existing segments stay loaded (the underlying Parquet is
          immutable, so no need to reload).
        """
        current_keys: set = set()
        for partition, rels in self._manifest.get_all_data_files().items():
            for rel in rels:
                key = (partition, rel)
                current_keys.add(key)
                if key in self._segment_cache:
                    continue
                abs_path = os.path.join(
                    self._data_dir, "partitions", partition, rel
                )
                if not os.path.exists(abs_path):
                    # Should have been caught by recovery, but be defensive.
                    continue
                self._segment_cache[key] = Segment.load(
                    file_path=abs_path,
                    partition=partition,
                    pk_field=self._pk_name,
                    vector_field=self._vector_name,
                )

        # Evict segments for files that are no longer in the manifest.
        for key in list(self._segment_cache.keys()):
            if key not in current_keys:
                del self._segment_cache[key]

    # ── batch builders ──────────────────────────────────────────

    def _build_wal_data_batch(
        self,
        records: List[dict],
        partition_name: str,
        seqs: List[int],
    ) -> pa.RecordBatch:
        """Build a RecordBatch matching wal_data_schema.

        Splits dynamic fields into $meta if enable_dynamic_field is set.
        """
        n = len(records)
        cols: dict[str, list] = {
            "_seq": seqs,
            "_partition": [partition_name] * n,
        }

        for f in self._schema.fields:
            cols[f.name] = []

        meta_col: Optional[List[Optional[str]]] = None
        if self._schema.enable_dynamic_field:
            meta_col = []

        for r in records:
            schema_part, meta_json = separate_dynamic_fields(r, self._schema)
            for f in self._schema.fields:
                cols[f.name].append(schema_part.get(f.name))
            if meta_col is not None:
                meta_col.append(meta_json)

        if meta_col is not None:
            cols["$meta"] = meta_col

        # Serialize SPARSE_FLOAT_VECTOR columns: dict → packed bytes
        for f in self._schema.fields:
            if f.dtype == DataType.SPARSE_FLOAT_VECTOR:
                from milvus_lite.analyzer.sparse import sparse_to_bytes
                cols[f.name] = [
                    sparse_to_bytes(v) if isinstance(v, dict) else (v or b"")
                    for v in cols[f.name]
                ]

        # Serialize JSON columns: dict/list → JSON string
        import json as _json
        for f in self._schema.fields:
            if f.dtype == DataType.JSON:
                cols[f.name] = [
                    _json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list)) else v
                    for v in cols[f.name]
                ]

        # Replace None with zero vectors for nullable FLOAT_VECTOR fields.
        # Arrow Parquet doesn't support null in FixedSizeList, so we store
        # zeros and rely on the null info being tracked at read time by
        # checking if the vector is all-zeros (or via valid_data on gRPC).
        for f in self._schema.fields:
            if f.dtype == DataType.FLOAT_VECTOR and f.nullable and f.dim:
                zero = [0.0] * f.dim
                cols[f.name] = [
                    v if v is not None else zero
                    for v in cols[f.name]
                ]

        return pa.RecordBatch.from_pydict(cols, schema=self._wal_data_schema)

    def _build_wal_delta_batch(
        self,
        pks: List[Any],
        partition_name: str,
        seq: int,
    ) -> pa.RecordBatch:
        """Build a wal_delta RecordBatch for a delete operation.

        All rows share the same _seq and _partition (architectural
        invariant: batch delete is one logical event).
        """
        return pa.RecordBatch.from_pydict(
            {
                self._pk_name: pks,
                "_seq": [seq] * len(pks),
                "_partition": [partition_name] * len(pks),
            },
            schema=self._wal_delta_schema,
        )

    # ── lifecycle ───────────────────────────────────────────────

    def close(self) -> None:
        """Flush any pending state and shut down the WAL.

        Runs a final flush if the MemTable has data, waits for any
        in-flight background compaction/index tasks, then shuts down
        the background executor.
        """
        if self._memtable.size() > 0:
            self._trigger_flush()
            # _trigger_flush creates a new empty WAL; clean it up too
            self._wal.close_and_delete()
        else:
            # Even an empty MemTable needs WAL cleanup so we don't leave
            # an empty wal file behind.
            self._wal.close_and_delete()

        # Drain background tasks so manifest / cache / index state
        # are all committed before the Collection is considered closed.
        self._bg_closed = True
        self._bg_executor.shutdown(wait=True)

    # ── introspection ───────────────────────────────────────────

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def pk_field(self) -> str:
        return self._pk_name

    def count(self) -> int:
        """Number of live records in the MemTable.

        NOTE: this is the in-memory count only. Phase 4 will add a
        full collection count that includes flushed Parquet files.
        """
        return self._memtable.size()


# ── Module-level helpers ──────────────────────────────────────────────

def _apply_group_by(
    results: List[List[dict]],
    group_by_field: str,
    limit: int,
    group_size: int,
    strict_group_size: bool,
) -> List[List[dict]]:
    """Post-process search results to group by a scalar field.

    For each query:
    1. Iterate hits in distance order (already sorted).
    2. Group by group_by_field value.
    3. Each group keeps up to group_size hits.
    4. If strict_group_size, discard groups with < group_size hits.
    5. Take the first `limit` groups.
    6. Flatten groups back into a single list (groups ordered by their
       best hit's distance).

    Returns results in the same format as input but filtered/reordered.
    Each hit gets an extra key ``_group_by_value`` for the gRPC layer
    to build the group_by_field_value FieldData.
    """
    out: List[List[dict]] = []

    for query_hits in results:
        # group_key → list of hits (in distance order)
        groups: dict = {}
        group_order: list = []  # track first-seen order (= best distance)

        for hit in query_hits:
            gval = hit.get("entity", {}).get(group_by_field)
            if gval is None:
                # Try top-level (some code paths put fields at top level)
                gval = hit.get(group_by_field)

            if gval not in groups:
                groups[gval] = []
                group_order.append(gval)

            if len(groups[gval]) < group_size:
                # Attach group value to hit for gRPC layer
                hit_copy = dict(hit)
                hit_copy["_group_by_value"] = gval
                groups[gval].append(hit_copy)

        # Filter by strict_group_size
        if strict_group_size:
            group_order = [g for g in group_order if len(groups[g]) == group_size]

        # Take first `limit` groups, flatten
        selected_groups = group_order[:limit]
        flattened: list = []
        for gval in selected_groups:
            flattened.extend(groups[gval])

        out.append(flattened)

    return out


def _apply_range_filter(
    results: List[List[dict]],
    radius: Optional[float],
    range_filter: Optional[float],
    limit: int,
    metric_type: str = "COSINE",
) -> List[List[dict]]:
    """Filter search results by distance range.

    Milvus range search semantics:
        L2/COSINE: radius = max distance (outer), range_filter = min distance (inner)
            Keep: range_filter <= distance <= radius
        IP: radius = min score (inner), range_filter = max score (outer)
            Keep: radius <= distance <= range_filter
            (note: at this point IP distances are still internal -dot form)

    Either bound can be None (no bound on that side).
    After filtering, truncates to *limit* hits per query.
    """
    out: List[List[dict]] = []
    for query_hits in results:
        filtered = []
        for hit in query_hits:
            d = hit["distance"]
            if metric_type == "IP":
                # IP internal convention: -dot (smaller = more similar)
                # radius/range_filter are user-facing (positive dot values)
                # but _apply_range_filter runs BEFORE IP sign flip, so
                # negate the bounds for comparison
                if radius is not None and not (d <= -radius):
                    continue
                if range_filter is not None and not (d >= -range_filter):
                    continue
            else:
                # L2/COSINE: smaller distance = closer
                # radius = outer bound (max), range_filter = inner bound (min)
                if radius is not None and not (d <= radius):
                    continue
                if range_filter is not None and not (d >= range_filter):
                    continue
            filtered.append(hit)
        out.append(filtered[:limit])
    return out
