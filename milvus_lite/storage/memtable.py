"""Collection-level in-memory write buffer.

Internal representation (per modules.md §9.7):
    _insert_batches: list[pa.RecordBatch]              append-only
    _pk_index:       dict[pk → (batch_idx, row_idx, seq)]
    _delete_index:   dict[pk → (max_delete_seq, partition)]

The list is append-only so we never copy / mutate Arrow buffers on the
write path. Stale rows in older batches are reachable only via _pk_index;
flush() dedups by walking _pk_index.values().

The partition stored alongside each delete is the routing target for
the delta Parquet file at flush time. ``ALL_PARTITIONS`` means "apply
to every known partition" — those are replicated at flush time.

**Architectural invariant §1-2**: every cross-buffer decision is keyed on
``_seq``. ``apply_insert`` and ``apply_delete`` are order-independent —
calling them in any order with the same set of (pk, _seq) operations
yields the same final state. This is what makes recovery safe even if it
replays operations out of physical order.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa

from milvus_lite.constants import ALL_PARTITIONS
from milvus_lite.schema.arrow_builder import (
    build_data_schema,
    build_delta_schema,
    get_primary_field,
)
from milvus_lite.schema.types import CollectionSchema


# (batch_idx, row_idx, seq)
_PkPos = Tuple[int, int, int]
# (delete_seq, partition_name)
_DeleteEntry = Tuple[int, str]


class MemTable:
    """Collection-level in-memory write buffer.

    Phase 2 supports:
        - apply_insert / apply_delete (seq-aware, order-independent)
        - get(pk) for point reads
        - size() for flush triggering
        - get_active_records() for search/get during the read path

    Phase 3 will add flush() which dedups via _pk_index and emits Arrow
    Tables ready to write to Parquet.
    """

    def __init__(self, schema: CollectionSchema) -> None:
        self._schema = schema
        self._pk_name = get_primary_field(schema).name
        self._data_schema = build_data_schema(schema)
        self._delta_schema = build_delta_schema(schema)

        self._insert_batches: List[pa.RecordBatch] = []
        self._pk_index: Dict[Any, _PkPos] = {}
        self._delete_index: Dict[Any, _DeleteEntry] = {}
        self._lock = threading.RLock()

    # ── write path ──────────────────────────────────────────────

    def apply_insert(self, batch: pa.RecordBatch) -> None:
        """Apply a wal_data RecordBatch.

        Each row is checked against _delete_index and the existing
        _pk_index entry — only rows whose seq is the new maximum take
        effect. Filtered-out rows still live physically in the appended
        batch but are not reachable via the index.
        """
        if batch.num_rows == 0:
            return
        self._validate_insert_schema(batch)

        with self._lock:
            batch_idx = len(self._insert_batches)
            pk_col = batch.column(self._pk_name)
            seq_col = batch.column("_seq")

            any_kept = False
            for row_idx in range(batch.num_rows):
                pk = pk_col[row_idx].as_py()
                seq = seq_col[row_idx].as_py()

                # seq-aware: a newer delete blocks this insert
                existing_delete = self._delete_index.get(pk)
                if existing_delete is not None and existing_delete[0] >= seq:
                    continue

                # seq-aware: a newer insert blocks this insert
                existing_pos = self._pk_index.get(pk)
                if existing_pos is not None and existing_pos[2] >= seq:
                    continue

                # take effect
                self._pk_index[pk] = (batch_idx, row_idx, seq)
                # our seq > existing_delete (we passed the check above)
                self._delete_index.pop(pk, None)
                any_kept = True

            if any_kept:
                self._insert_batches.append(batch)

    def apply_delete(self, batch: pa.RecordBatch) -> None:
        """Apply a wal_delta RecordBatch.

        All rows in a delta batch share the same ``_seq`` AND the same
        ``_partition`` (a single delete call's batch). The batch itself
        is not retained — only the (pk, seq, partition) triples are
        folded into _delete_index.
        """
        if batch.num_rows == 0:
            return
        self._validate_delete_schema(batch)

        with self._lock:
            pk_col = batch.column(self._pk_name)
            seq_col = batch.column("_seq")
            partition_col = batch.column("_partition")
            # delete batches share one seq and one partition; read from row 0.
            seq = seq_col[0].as_py()
            partition = partition_col[0].as_py()

            for row_idx in range(batch.num_rows):
                pk = pk_col[row_idx].as_py()

                # seq-aware: a newer insert blocks this delete
                existing_pos = self._pk_index.get(pk)
                if existing_pos is not None and existing_pos[2] >= seq:
                    continue

                # update delete watermark to the larger seq
                existing_delete = self._delete_index.get(pk)
                if existing_delete is None or seq > existing_delete[0]:
                    self._delete_index[pk] = (seq, partition)

                # if pk_index entry exists with smaller seq, evict it
                if existing_pos is not None and existing_pos[2] < seq:
                    self._pk_index.pop(pk, None)

    # ── read path ───────────────────────────────────────────────

    def get(self, pk_value: Any, partition_filter: Optional[set] = None) -> Optional[dict]:
        """Point read for a single pk.

        Returns the live record dict (without _seq / _partition) or None
        if the pk is unknown, has been deleted, or is not in partition_filter.
        """
        with self._lock:
            pos = self._pk_index.get(pk_value)
            if pos is None:
                return None
            # _pk_index entries are guaranteed to NOT be shadowed by a delete:
            # apply_delete pops the entry, apply_insert pops the delete entry.
            # So we can return directly without re-checking _delete_index.
            batch_idx, row_idx, _ = pos
            batch = self._insert_batches[batch_idx]
            if partition_filter is not None:
                partition = batch.column("_partition")[row_idx].as_py()
                if partition not in partition_filter:
                    return None
            return self._row_to_dict(batch, row_idx)

    def get_active_records(
        self, partition_names: Optional[List[str]] = None
    ) -> List[dict]:
        """Return all live records, optionally filtered by partition.

        Used by Collection.search / Collection.get during the read path.
        Returned dicts are clean (no _seq, no _partition).
        """
        out: List[dict] = []
        partition_filter: Optional[set] = None
        if partition_names is not None:
            partition_filter = set(partition_names)

        with self._lock:
            for pk, (batch_idx, row_idx, _seq) in self._pk_index.items():
                batch = self._insert_batches[batch_idx]
                if partition_filter is not None:
                    partition = batch.column("_partition")[row_idx].as_py()
                    if partition not in partition_filter:
                        continue
                out.append(self._row_to_dict(batch, row_idx))
        return out

    def to_search_arrays(
        self,
        vector_field: Optional[str],
        partition_names: Optional[List[str]] = None,
    ) -> Tuple[List[Any], "np.ndarray", "np.ndarray", List[Tuple[Any, int]]]:
        """Walk _pk_index and emit (pks, seqs, vectors, row_refs) for search.

        Records are NOT materialized here — callers should use
        materialize_row(ref) to get the dict only for top-k winners.

        Returns:
            pks:      list of pk values (length M)
            seqs:     np.ndarray[uint64], shape (M,)
            vectors:  np.ndarray[float32], shape (M, dim)
            row_refs: list of (batch_idx, row_idx) tuples for deferred
                      materialization via materialize_row()
        """
        pks, seqs_arr, vectors_arr, row_refs, _table = self.to_search_snapshot(
            vector_field=vector_field,
            partition_names=partition_names,
            include_table=False,
        )
        return pks, seqs_arr, vectors_arr, row_refs

    def to_search_snapshot(
        self,
        vector_field: Optional[str],
        partition_names: Optional[List[str]] = None,
        include_table: bool = False,
    ) -> Tuple[
        List[Any],
        "np.ndarray",
        "np.ndarray",
        List[Tuple[Any, int]],
        Optional[pa.Table],
    ]:
        """Return an aligned snapshot for search/query candidates.

        ``to_search_arrays`` and ``to_arrow_table`` are individually
        thread-safe, but calling them separately can observe different
        MemTable states while a writer is active. This method captures
        candidate arrays and the optional filter-evaluation table under
        one lock, so their row order and lengths are identical.
        """
        import numpy as np

        pks: List[Any] = []
        seqs: List[int] = []
        vecs: List[list] = []
        row_refs: List[Tuple[int, int]] = []
        table_indices: List[int] = []
        batches: Tuple[pa.RecordBatch, ...] = ()
        offsets: List[int] = []
        partition_filter: Optional[set] = None
        if partition_names is not None:
            partition_filter = set(partition_names)

        with self._lock:
            batches = tuple(self._insert_batches)
            if include_table and batches:
                offsets = [0]
                for batch in batches:
                    offsets.append(offsets[-1] + batch.num_rows)

            for pk, (batch_idx, row_idx, seq) in self._pk_index.items():
                batch = self._insert_batches[batch_idx]
                if partition_filter is not None:
                    partition = batch.column("_partition")[row_idx].as_py()
                    if partition not in partition_filter:
                        continue
                pks.append(pk)
                seqs.append(seq)
                if vector_field is not None:
                    vecs.append(batch.column(vector_field)[row_idx].as_py())
                row_refs.append((batch_idx, row_idx))
                if include_table:
                    table_indices.append(offsets[batch_idx] + row_idx)

        seqs_arr = np.asarray(seqs, dtype=np.uint64)
        if vecs:
            # Handle nullable vectors: replace None with zero vectors
            has_none = any(v is None for v in vecs)
            if has_none:
                dim = next((len(v) for v in vecs if v is not None), 0)
                zero = [0.0] * dim
                vecs = [v if v is not None else zero for v in vecs]
            vectors_arr = np.asarray(vecs, dtype=np.float32)
        else:
            vectors_arr = np.zeros((0, 0), dtype=np.float32)

        table: Optional[pa.Table] = None
        if include_table:
            if not table_indices or not batches:
                table = self._data_schema.empty_table()
            else:
                table = pa.Table.from_batches(batches).take(
                    pa.array(table_indices, type=pa.int64())
                )
                # Drop WAL-only columns to match _data_schema layout.
                for col_name in ("_partition",):
                    if col_name in table.column_names:
                        table = table.drop(col_name)

        return pks, seqs_arr, vectors_arr, row_refs, table

    def materialize_row(self, batch_idx: int, row_idx: int) -> dict:
        """Materialize a single row dict from batch references.

        Used by executor for deferred materialization — only called
        for top-k winners, not all memtable rows.
        """
        with self._lock:
            return self._row_to_dict(self._insert_batches[batch_idx], row_idx)

    def is_locally_deleted(self, pk_value: Any) -> bool:
        """True iff *pk_value* has a tombstone in this MemTable's local
        _delete_index — i.e., a delete has been applied here that has not
        yet been flushed.

        Used by Collection.get to short-circuit a segment scan when the
        in-memory state already says the pk is deleted.
        """
        with self._lock:
            return pk_value in self._delete_index

    def to_arrow_table(
        self,
        partition_names: Optional[List[str]] = None,
    ) -> pa.Table:
        """Materialize the active live rows as a read-only pa.Table.

        Used by search/filter to evaluate scalar predicates against the
        in-memory data. The row order matches that of to_search_arrays
        (both iterate _pk_index in insertion order), so a filter mask
        produced by evaluating against the returned Table aligns with
        the candidate arrays from to_search_arrays.

        Returns an empty Table (with the data_schema layout) if there
        are no live rows in scope.
        """
        with self._lock:
            if not self._insert_batches:
                return self._data_schema.empty_table()

            batches = tuple(self._insert_batches)
            pk_items = tuple(self._pk_index.items())

            # Concatenate all stored batches into one full Table.
            full = pa.Table.from_batches(batches)

            # Compute global row offsets so we can convert (batch_idx, row_idx)
            # → flat index for pa.Table.take.
            offsets = [0]
            for b in batches:
                offsets.append(offsets[-1] + b.num_rows)

            partition_filter: Optional[set] = None
            if partition_names is not None:
                partition_filter = set(partition_names)

            indices: list[int] = []
            for pk, (batch_idx, row_idx, _seq) in pk_items:
                if partition_filter is not None:
                    batch = batches[batch_idx]
                    partition = batch.column("_partition")[row_idx].as_py()
                    if partition not in partition_filter:
                        continue
                indices.append(offsets[batch_idx] + row_idx)

        if not indices:
            return self._data_schema.empty_table()

        result = full.take(pa.array(indices, type=pa.int64()))
        # Drop WAL-only columns to match _data_schema layout
        for col_name in ("_partition",):
            if col_name in result.column_names:
                result = result.drop(col_name)
        return result

    def pk_index_snapshot(self) -> Tuple[Tuple[Any, _PkPos], ...]:
        """Return a stable snapshot of the live pk index."""
        with self._lock:
            return tuple(self._pk_index.items())

    def delete_index_snapshot(self) -> Dict[Any, _DeleteEntry]:
        """Return a stable snapshot of the live delete index."""
        with self._lock:
            return dict(self._delete_index)

    def active_record_snapshots(
        self,
        partition_names: Optional[List[str]] = None,
    ) -> List[Tuple[Any, int, dict]]:
        """Return stable ``(pk, seq, record)`` snapshots for live rows."""
        out: List[Tuple[Any, int, dict]] = []
        partition_filter: Optional[set] = None
        if partition_names is not None:
            partition_filter = set(partition_names)

        with self._lock:
            for pk, (batch_idx, row_idx, seq) in self._pk_index.items():
                batch = self._insert_batches[batch_idx]
                if partition_filter is not None:
                    partition = batch.column("_partition")[row_idx].as_py()
                    if partition not in partition_filter:
                        continue
                out.append((pk, seq, self._row_to_dict(batch, row_idx)))
        return out

    def size(self) -> int:
        """Active pk count + tombstone count.

        Used by flush triggering. NOT the physical row count of
        _insert_batches — that includes shadowed rows that no longer
        contribute to the visible state.
        """
        with self._lock:
            return len(self._pk_index) + len(self._delete_index)

    # ── flush ───────────────────────────────────────────────────

    def flush(
        self,
        known_partitions: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Optional[pa.Table], Optional[pa.Table]]]:
        """Materialize the live state as per-partition Arrow Tables.

        Returns:
            ``{partition_name: (data_table, delta_table)}``
            - ``data_table`` uses ``data_schema`` (no _partition column),
              or None if the partition has no inserts.
            - ``delta_table`` uses ``delta_schema`` (no _partition),
              or None if the partition has no deletes.

        Args:
            known_partitions: list of all partitions in the manifest at
                flush time. Cross-partition deletes (``ALL_PARTITIONS``)
                are replicated into each one. If None, ``ALL_PARTITIONS``
                deletes are NOT replicated and live only under the
                ``ALL_PARTITIONS`` key in the result — this is useful
                for unit tests but flush.execute_flush should always
                pass the manifest's partition list.

        The MemTable's internal state is NOT cleared. The caller (the
        flush pipeline) is expected to discard the frozen MemTable
        after consuming the result.
        """
        with self._lock:
            # ── 1. data tables ──────────────────────────────────────
            # Walk _pk_index to find live rows, group by their _partition.
            live_rows_per_partition: Dict[str, List[Tuple[int, int]]] = {}
            for pk, (batch_idx, row_idx, _seq) in self._pk_index.items():
                batch = self._insert_batches[batch_idx]
                partition = batch.column("_partition")[row_idx].as_py()
                live_rows_per_partition.setdefault(partition, []).append((batch_idx, row_idx))

            data_tables: Dict[str, pa.Table] = {}
            for partition, rows in live_rows_per_partition.items():
                data_tables[partition] = self._build_data_table(rows)

            # ── 2. delta tables ─────────────────────────────────────
            # Group _delete_index entries by their target partition.
            deletes_per_partition: Dict[str, List[Tuple[Any, int]]] = {}
            for pk, (delete_seq, partition) in self._delete_index.items():
                deletes_per_partition.setdefault(partition, []).append((pk, delete_seq))

            # Cross-partition deletes: replicate into every known partition.
            all_part_deletes = deletes_per_partition.pop(ALL_PARTITIONS, None)
            if all_part_deletes:
                if known_partitions is None:
                    # Caller did not pass partition list — preserve the _all
                    # bucket as-is so the test/caller can see it.
                    deletes_per_partition[ALL_PARTITIONS] = all_part_deletes
                else:
                    for p in known_partitions:
                        deletes_per_partition.setdefault(p, []).extend(all_part_deletes)

            delta_tables: Dict[str, pa.Table] = {}
            for partition, entries in deletes_per_partition.items():
                delta_tables[partition] = self._build_delta_table(entries)

        # ── 3. merge per-partition results ──────────────────────
        result: Dict[str, Tuple[Optional[pa.Table], Optional[pa.Table]]] = {}
        all_partitions = set(data_tables) | set(delta_tables)
        for p in all_partitions:
            result[p] = (data_tables.get(p), delta_tables.get(p))
        return result

    def _build_data_table(self, rows: List[Tuple[int, int]]) -> pa.Table:
        """Take a list of (batch_idx, row_idx) tuples and emit a data Table.

        The output uses ``data_schema`` (no _partition column).
        """
        # Strategy: build column-wise. For each column in data_schema,
        # walk the rows and pull values from the source batch.
        cols: Dict[str, list] = {name: [] for name in self._data_schema.names}
        for batch_idx, row_idx in rows:
            batch = self._insert_batches[batch_idx]
            for name in self._data_schema.names:
                cols[name].append(batch.column(name)[row_idx].as_py())
        return pa.Table.from_pydict(cols, schema=self._data_schema)

    def _build_delta_table(self, entries: List[Tuple[Any, int]]) -> pa.Table:
        """Build a delta Table from (pk, seq) entries.

        Output uses ``delta_schema`` (pk + _seq, no _partition).
        """
        pks = [e[0] for e in entries]
        seqs = [e[1] for e in entries]
        return pa.Table.from_pydict(
            {self._pk_name: pks, "_seq": seqs},
            schema=self._delta_schema,
        )

    @property
    def max_seq(self) -> int:
        """Largest _seq seen across inserts and deletes. -1 if empty.

        Used by flush.execute_flush to bump manifest.current_seq.
        """
        with self._lock:
            max_s = -1
            for _, _, seq in self._pk_index.values():
                if seq > max_s:
                    max_s = seq
            for delete_seq, _ in self._delete_index.values():
                if delete_seq > max_s:
                    max_s = delete_seq
            return max_s

    # ── introspection (test/debug) ──────────────────────────────

    @property
    def pk_name(self) -> str:
        return self._pk_name

    def num_physical_rows(self) -> int:
        """Total rows physically retained in _insert_batches.
        Includes shadowed rows. For test/debug only."""
        with self._lock:
            return sum(b.num_rows for b in self._insert_batches)

    # ── internal helpers ────────────────────────────────────────

    def _row_to_dict(self, batch: pa.RecordBatch, row_idx: int) -> dict:
        """Extract one row as a dict, stripping _seq and _partition.

        Dynamic fields stored in `$meta` are unpacked into the result.
        """
        result = {}
        meta_raw = None
        for name in batch.schema.names:
            if name in ("_seq", "_partition"):
                continue
            val = batch.column(name)[row_idx].as_py()
            if name == "$meta":
                meta_raw = val
            result[name] = val
        # Unpack $meta JSON into top-level keys (keeps $meta for filters)
        if meta_raw is not None:
            import json
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                    if isinstance(meta, dict):
                        result.update(meta)
                except (json.JSONDecodeError, ValueError):
                    pass
            elif isinstance(meta_raw, dict):
                result.update(meta_raw)
        return result

    def _validate_insert_schema(self, batch: pa.RecordBatch) -> None:
        names = set(batch.schema.names)
        required = {"_seq", "_partition", self._pk_name}
        missing = required - names
        if missing:
            raise ValueError(
                f"insert batch missing required columns: {sorted(missing)}"
            )

    def _validate_delete_schema(self, batch: pa.RecordBatch) -> None:
        names = set(batch.schema.names)
        required = {"_seq", "_partition", self._pk_name}
        missing = required - names
        if missing:
            raise ValueError(
                f"delete batch missing required columns: {sorted(missing)}"
            )
