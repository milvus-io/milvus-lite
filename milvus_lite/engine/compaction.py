"""Size-Tiered Compaction + Tombstone GC.

Per-partition compaction. Trigger conditions (either suffices):
    1. Some size bucket holds >= COMPACTION_MIN_FILES_PER_BUCKET files.
    2. The partition's total data file count exceeds MAX_DATA_FILES.

Compaction flow:
    1. Bucket the partition's data files by size.
    2. Pick a target set:
       - first bucket with >= MIN_FILES_PER_BUCKET, OR
       - all files (if total > MAX_DATA_FILES, force-compact)
    3. Read input files into Arrow tables.
    4. Concat → dedup by pk (keep max _seq) → filter delete tombstones
       via delta_index.is_deleted.
    5. Write the merged table to a new data file (skipped if 0 rows).
    6. Atomic Manifest update: remove old files, add new file.
    7. Delete the old files from disk.
    8. Tombstone GC: delta_index.gc_below(min_active_data_seq).

Delta files are NOT consumed during compaction. A delta entry survives
as long as any data file with seq_min <= delete_seq might still contain
its pk — that condition can only be checked globally (across all
partitions), so it lives in step 8's gc_below call. Phase-6+ optimization
can also delete fully-obsolete delta files from disk.

Crash safety:
    Crash before Step 6 (manifest commit) → orphan new file, manifest
        unchanged → recovery's _cleanup_orphan_files removes it.
    Crash during Step 6 → atomic rename, either old or new manifest.
    Crash after Step 6 mid-Step-7 → some old files orphaned (in disk
        but not in manifest); recovery cleans them.
    Crash in Step 8 → delta_index reset to in-memory rebuild on next
        start, no on-disk impact.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from milvus_lite.constants import (
    COMPACTION_BUCKET_BOUNDARIES,
    COMPACTION_MIN_FILES_PER_BUCKET,
    MAX_DATA_FILES,
    MAX_SEGMENT_ROWS,
)
from milvus_lite.schema.arrow_builder import (
    build_data_schema,
    get_primary_field,
)
from milvus_lite.constants import DATA_FILE_TEMPLATE, SEQ_FORMAT_WIDTH
from milvus_lite.storage.data_file import (
    parse_seq_range,
    read_data_file,
    write_data_file,
)

if TYPE_CHECKING:
    from milvus_lite.schema.types import CollectionSchema
    from milvus_lite.storage.delta_index import DeltaIndex
    from milvus_lite.storage.manifest import Manifest

logger = logging.getLogger(__name__)


class CompactionManager:
    """Per-Collection compaction driver.

    Stateless across calls — every maybe_compact() call inspects the
    current Manifest from scratch. The Collection holds one instance.
    """

    def __init__(self, data_dir: str, schema: "CollectionSchema") -> None:
        self._data_dir = data_dir
        self._schema = schema
        self._pk_name = get_primary_field(schema).name
        self._data_schema = build_data_schema(schema)

    # ── public API ──────────────────────────────────────────────

    def maybe_compact(
        self,
        partition: str,
        manifest: "Manifest",
        delta_index: "DeltaIndex",
    ) -> bool:
        """Check if *partition* needs compaction; if so, run it.

        Returns True if a compaction was actually performed.
        """
        files = manifest.get_data_files(partition)
        if len(files) < COMPACTION_MIN_FILES_PER_BUCKET:
            return False

        partition_dir = os.path.join(self._data_dir, "partitions", partition)
        buckets = self._bucket_files(partition_dir, files)
        target = self._select_target(buckets, len(files), partition_dir)
        if target is None:
            return False

        logger.info(
            "compaction: partition=%s, merging %d files",
            partition, len(target),
        )
        self._compact_files(partition, partition_dir, target, manifest, delta_index)

        # Tombstone GC: safe under concurrent reads because all read
        # paths call DeltaIndex.snapshot() at the start of a request
        # (see Collection.get/search/query/num_entities). gc_below
        # mutates the live DeltaIndex; in-flight readers still hold
        # their own frozen dict copies so they correctly filter
        # deleted rows from the segment snapshots they're iterating.
        self._gc_tombstones(manifest, delta_index)
        return True

    # ── bucketing + selection ───────────────────────────────────

    def _bucket_files(
        self,
        partition_dir: str,
        files: List[str],
    ) -> List[List[Tuple[str, int]]]:
        """Bucket *files* by size. Returns one list per size bucket;
        each entry is (filename, byte_size)."""
        n_buckets = len(COMPACTION_BUCKET_BOUNDARIES) + 1
        buckets: List[List[Tuple[str, int]]] = [[] for _ in range(n_buckets)]
        for fn in files:
            abs_path = os.path.join(partition_dir, fn)
            if not os.path.exists(abs_path):
                # Defensive — shouldn't happen if recovery is correct.
                continue
            size = os.path.getsize(abs_path)
            buckets[self._bucket_index(size)].append((fn, size))
        return buckets

    @staticmethod
    def _bucket_index(size: int) -> int:
        for i, boundary in enumerate(COMPACTION_BUCKET_BOUNDARIES):
            if size < boundary:
                return i
        return len(COMPACTION_BUCKET_BOUNDARIES)

    @staticmethod
    def _select_target(
        buckets: List[List[Tuple[str, int]]],
        total_files: int,
        partition_dir: str,
    ) -> Optional[List[str]]:
        """Pick the set of files to compact.

        Strategy:
            - First, look for a bucket with >= MIN_FILES_PER_BUCKET.
              Take as many files as fit under MAX_SEGMENT_ROWS.
            - Else, if total file count > MAX_DATA_FILES, greedily pack
              smallest-first across buckets, capped at MAX_SEGMENT_ROWS.
            - Else None.
        """
        for bucket in buckets:
            if len(bucket) >= COMPACTION_MIN_FILES_PER_BUCKET:
                picked = CompactionManager._cap_by_row_limit(
                    [fn for fn, _ in bucket], partition_dir,
                )
                if len(picked) >= COMPACTION_MIN_FILES_PER_BUCKET:
                    return picked
        if total_files > MAX_DATA_FILES:
            all_files: List[str] = []
            for bucket in buckets:
                all_files.extend(fn for fn, _ in bucket)
            picked = CompactionManager._cap_by_row_limit(all_files, partition_dir)
            if len(picked) >= 2:
                return picked
        return None

    @staticmethod
    def _cap_by_row_limit(
        files: List[str], partition_dir: str,
    ) -> List[str]:
        """Select a prefix of *files* whose combined row count fits
        under 2×MAX_SEGMENT_ROWS.

        The 2x budget gives compaction room for shrinkage from dedup
        and tombstone filtering — if the merged live rows exceed
        MAX_SEGMENT_ROWS, _compact_files splits the output into
        multiple segments, each under the cap. The 2x budget also
        bounds per-compaction memory/IO.
        """
        import pyarrow.parquet as pq
        budget = 2 * MAX_SEGMENT_ROWS
        out: List[str] = []
        total = 0
        for fn in files:
            abs_path = os.path.join(partition_dir, fn)
            try:
                n = pq.ParquetFile(abs_path).metadata.num_rows
            except Exception:
                continue
            if n >= budget:
                continue  # single file already too large — terminal, skip
            if total + n > budget:
                break
            out.append(fn)
            total += n
        return out

    # ── core compaction ─────────────────────────────────────────

    def _compact_files(
        self,
        partition: str,
        partition_dir: str,
        files_to_compact: List[str],
        manifest: "Manifest",
        delta_index: "DeltaIndex",
    ) -> None:
        import time as _time
        t0 = _time.monotonic()

        # 1. Read all input files.
        tables: List[pa.Table] = []
        for fn in files_to_compact:
            abs_path = os.path.join(partition_dir, fn)
            tables.append(read_data_file(abs_path))
        if not tables:
            return
        combined = pa.concat_tables(tables)
        input_rows = combined.num_rows

        # 2. Dedup by pk (keep max _seq).
        deduped = self._dedup_max_seq(combined)

        # 3. Filter rows that have a tombstone with strictly larger seq.
        filtered = self._filter_deleted(deduped, delta_index)

        # 4. Write merged file(s). If the live-row count exceeds
        # MAX_SEGMENT_ROWS (after dedup + tombstone filtering), split
        # into multiple segments under the cap. The split is in seq
        # order so each output file has a contiguous seq range.
        new_rels: List[str] = []
        if filtered.num_rows > 0:
            # Sort by _seq so slices have monotonic seq ranges.
            sort_idx = pc.sort_indices(filtered, sort_keys=[("_seq", "ascending")])
            filtered = filtered.take(sort_idx)

            chunk_size = MAX_SEGMENT_ROWS
            total_rows = filtered.num_rows
            if total_rows <= chunk_size:
                chunks = [filtered]
            else:
                chunks = [
                    filtered.slice(i, min(chunk_size, total_rows - i))
                    for i in range(0, total_rows, chunk_size)
                ]

            for chunk in chunks:
                seqs = chunk.column("_seq").to_pylist()
                content_seq_min = min(seqs)
                content_seq_max = max(seqs)
                unique_min, unique_max = self._pick_unique_seq_range(
                    partition_dir, content_seq_min, content_seq_max,
                )
                rel = write_data_file(
                    chunk, partition_dir, seq_min=unique_min, seq_max=unique_max,
                )
                new_rels.append(rel)

        # 5. Atomic manifest update.
        manifest.remove_data_files(partition, files_to_compact)
        for rel in new_rels:
            manifest.add_data_file(partition, rel)
        manifest.save()

        # 6. Delete old files from disk. Past this point a crash leaves
        #    orphan files, which recovery's _cleanup_orphan_files handles.
        for fn in files_to_compact:
            abs_path = os.path.join(partition_dir, fn)
            if os.path.exists(abs_path):
                try:
                    os.remove(abs_path)
                except OSError as e:
                    logger.warning("compaction: failed to remove %s: %s", abs_path, e)

        elapsed = _time.monotonic() - t0
        output_rows = filtered.num_rows
        logger.info(
            "compaction: partition=%s done in %.2fs — "
            "%d input files (%d rows) → %d output files (%d rows, %d removed)",
            partition, elapsed,
            len(files_to_compact), input_rows,
            len(new_rels), output_rows, input_rows - output_rows,
        )

    _MAX_SEQ_BUMP_ATTEMPTS = 10_000

    @staticmethod
    def _pick_unique_seq_range(
        partition_dir: str,
        seq_min: int,
        seq_max: int,
    ) -> Tuple[int, int]:
        """Return a (seq_min, seq_max) pair whose corresponding filename
        does not yet exist on disk.

        Edge case: a single input file already covers the merged range
        (e.g. inputs [1,10] + [3,5] → union [1,10] which collides with
        the [1,10] input). In that case we keep seq_min and bump seq_max
        until the filename is free. The seq_max stored in the filename is
        an UPPER BOUND on actual content seqs (the file may contain
        smaller seqs only), so this is always safe.

        Raises RuntimeError if no free filename is found within
        _MAX_SEQ_BUMP_ATTEMPTS (guards against infinite loop on
        pathological directory state).
        """
        rel_dir = "data"
        candidate_max = seq_max
        for _ in range(CompactionManager._MAX_SEQ_BUMP_ATTEMPTS):
            filename = DATA_FILE_TEMPLATE.format(
                min=seq_min, max=candidate_max, w=SEQ_FORMAT_WIDTH
            )
            abs_path = os.path.join(partition_dir, rel_dir, filename)
            if not os.path.exists(abs_path):
                return seq_min, candidate_max
            candidate_max += 1
        raise RuntimeError(
            f"failed to find unique seq range after "
            f"{CompactionManager._MAX_SEQ_BUMP_ATTEMPTS} attempts "
            f"(partition_dir={partition_dir}, seq_min={seq_min})"
        )

    def _dedup_max_seq(self, table: pa.Table) -> pa.Table:
        """For each pk, keep only the row with the largest _seq.

        Uses Arrow sort + shift-compare to stay in the C++ layer:
        sort by (pk ASC, _seq DESC), then keep only the first row
        of each pk group (where pk differs from the previous row).
        """
        if table.num_rows <= 1:
            return table
        sort_idx = pc.sort_indices(table, sort_keys=[
            (self._pk_name, "ascending"), ("_seq", "descending"),
        ])
        sorted_t = table.take(sort_idx)
        pk_col = sorted_t.column(self._pk_name)
        n = pk_col.length()
        # mask[0] = True (always keep first row); mask[i] = pk[i] != pk[i-1]
        changed = pc.not_equal(pk_col.slice(0, n - 1), pk_col.slice(1))
        # pc.not_equal may return ChunkedArray; flatten for concat_arrays.
        if isinstance(changed, pa.ChunkedArray):
            changed = changed.combine_chunks()
        mask = pa.concat_arrays([pa.array([True]), changed])
        return sorted_t.filter(mask)

    def _filter_deleted(
        self,
        table: pa.Table,
        delta_index: "DeltaIndex",
    ) -> pa.Table:
        """Drop rows whose pk has a tombstone with strictly larger seq."""
        if table.num_rows == 0 or len(delta_index) == 0:
            return table
        pks = table.column(self._pk_name).to_pylist()
        seqs = table.column("_seq").to_pylist()
        keep_indices: List[int] = []
        for i, pk in enumerate(pks):
            if not delta_index.is_deleted(pk, seqs[i]):
                keep_indices.append(i)
        if len(keep_indices) == len(pks):
            return table  # nothing filtered
        if not keep_indices:
            # All filtered — return an empty table with the same schema.
            return table.slice(0, 0)
        return table.take(pa.array(keep_indices, type=pa.int64()))

    # ── tombstone GC ────────────────────────────────────────────

    def _gc_tombstones(
        self,
        manifest: "Manifest",
        delta_index: "DeltaIndex",
    ) -> int:
        """Drop delta_index entries AND delta parquet files below the
        global min_active_data_seq.

        In-memory GC: delta_index.gc_below removes tombstones from the
        live dict (safe because readers hold frozen_copy snapshots).

        On-disk GC: delegated to ``_gc_delta_files`` — delta parquet
        files whose seq_max < threshold are fully obsolete.

        Returns number of in-memory tombstone entries removed.
        """
        global_min = self._global_min_active_data_seq(manifest)
        removed = delta_index.gc_below(global_min)
        delta_files_removed = self._gc_delta_files(manifest, global_min)

        if removed > 0 or delta_files_removed:
            logger.info(
                "tombstone GC: threshold=%d, %d in-memory entries removed, "
                "delta files purged=%s",
                global_min, removed, delta_files_removed,
            )

        return removed

    def _gc_delta_files(
        self,
        manifest: "Manifest",
        global_min: int,
    ) -> bool:
        """Remove obsolete delta parquet files from manifest and disk.

        A delta file is obsolete when its seq_max < global_min — every
        tombstone in it has been superseded.

        Crash safety: manifest.save() persists the removal *before*
        physical file deletion. A crash after save but before delete
        leaves orphan files on disk (harmless — recovery cleans them).
        The previous ordering (delete first, save later) could leave
        the manifest referencing deleted files, breaking recovery.

        Returns True if any delta files were removed.
        """
        gc_plan: List[Tuple[str, List[str]]] = []
        for partition in manifest.list_partitions():
            delta_files = manifest.get_delta_files(partition)
            if not delta_files:
                continue
            obsolete: List[str] = []
            for rel in delta_files:
                try:
                    _seq_min, seq_max = parse_seq_range(rel)
                except ValueError:
                    continue
                if seq_max < global_min:
                    obsolete.append(rel)
            if obsolete:
                gc_plan.append((partition, obsolete))

        if not gc_plan:
            return False

        # Persist removal in manifest BEFORE deleting physical files.
        for partition, files in gc_plan:
            manifest.remove_delta_files(partition, files)
        manifest.save()

        # Now safe to delete files from disk. Failures are logged but
        # non-fatal — orphan files are cleaned by recovery.
        for partition, files in gc_plan:
            partition_dir = os.path.join(
                self._data_dir, "partitions", partition
            )
            for fn in files:
                abs_path = os.path.join(partition_dir, fn)
                if os.path.exists(abs_path):
                    try:
                        os.remove(abs_path)
                    except OSError as e:
                        logger.warning(
                            "delta GC: failed to remove %s: %s", abs_path, e
                        )

        return True

    @staticmethod
    def _global_min_active_data_seq(manifest: "Manifest") -> int:
        """Smallest seq_min across every data file in every partition.

        If there are no data files, returns sys.maxsize so the entire
        delta_index can be drained safely.
        """
        min_seq = sys.maxsize
        for _partition, files in manifest.get_all_data_files().items():
            for rel in files:
                try:
                    seq_min, _seq_max = parse_seq_range(rel)
                except ValueError:
                    continue
                if seq_min < min_seq:
                    min_seq = seq_min
        return min_seq
