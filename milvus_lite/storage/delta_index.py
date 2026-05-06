"""In-memory delete index — pk → max_delete_seq.

Built at startup by reading all delta Parquet files of a Collection,
maintained incrementally during normal flush, and queried by the
search path's bitmap pipeline (Phase 4).

Tombstone GC (architectural invariant §3) is implemented in
``gc_below`` and called from compaction (Phase 6).

This module has no IO. ``rebuild_from`` takes already-resolved file
paths and uses ``read_delta_file`` to load them — IO concern is in
delta_file.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pyarrow as pa

from milvus_lite.storage.delta_file import read_delta_file


class DeltaIndex:
    """Collection-level in-memory delete watermark map.

    Semantics:
        ``is_deleted(pk, data_seq)``  ⟺  ``_map.get(pk, -1) > data_seq``

    A pk is "deleted" w.r.t. a data row iff there exists a delete
    operation with a strictly larger _seq than the data row's _seq.
    Same-seq is NOT a delete (the data row IS the latest state in
    that case — actually impossible since seqs are unique per
    Collection, but worth being explicit).
    """

    def __init__(self, pk_name: str) -> None:
        self._pk_name = pk_name
        self._map: Dict[Any, int] = {}

    # ── update ──────────────────────────────────────────────────

    def add_batch(self, batch: pa.RecordBatch) -> None:
        """Fold a delta RecordBatch into the index.

        Each row contributes (pk, _seq); the index keeps the max seq
        per pk.

        Accepts both wal_delta batches (which include _partition) and
        delta_schema batches (which don't), as long as both ``{pk}`` and
        ``_seq`` columns are present.
        """
        if batch.num_rows == 0:
            return
        names = set(batch.schema.names)
        if self._pk_name not in names or "_seq" not in names:
            raise ValueError(
                f"delta batch missing required columns: need {self._pk_name!r} and '_seq', "
                f"got {sorted(names)}"
            )

        pk_col = batch.column(self._pk_name)
        seq_col = batch.column("_seq")
        for i in range(batch.num_rows):
            pk = pk_col[i].as_py()
            seq = seq_col[i].as_py()
            existing = self._map.get(pk, -1)
            if seq > existing:
                self._map[pk] = seq

    def add_table(self, table: pa.Table) -> None:
        """Convenience: fold every RecordBatch in a Table."""
        for batch in table.to_batches():
            self.add_batch(batch)

    # ── query ───────────────────────────────────────────────────

    def is_deleted(self, pk_value: Any, data_seq: int) -> bool:
        """True iff a delete with seq > data_seq exists for *pk_value*."""
        existing = self._map.get(pk_value)
        if existing is None:
            return False
        return existing > data_seq

    def frozen_copy(self) -> "DeltaIndex":
        """Return a frozen copy suitable for concurrent read-only use.

        The search/query/num_entities paths call this once per request,
        then use the returned DeltaIndex for tombstone checks. That
        insulates them from the live DeltaIndex being mutated
        concurrently by the background compaction worker (which runs
        ``add_batch`` during delta absorption and ``gc_below`` after
        compaction).

        The cost is an O(N tombstones) dict copy, which is sub-ms for
        realistic tombstone volumes.
        """
        copy = DeltaIndex(self._pk_name)
        copy._map = dict(self._map)
        return copy

    # ── GC ──────────────────────────────────────────────────────

    def gc_below(self, min_active_data_seq: int) -> int:
        """Drop tombstones whose delete_seq < min_active_data_seq.

        Correctness (architectural invariant §3):
            delete_seq < min_active_data_seq
            ⟹ no data file has seq_min ≤ delete_seq containing this pk
            ⟹ no live data row needs this tombstone to be filtered
            ⟹ safe to drop.

        Args:
            min_active_data_seq: smallest seq_min among all data files
                currently referenced by the Manifest, across all
                partitions. If there are no data files, the caller may
                pass ``sys.maxsize`` to drain the entire index.

        Returns:
            Number of tombstones removed.
        """
        to_remove = [
            pk for pk, seq in self._map.items() if seq < min_active_data_seq
        ]
        for pk in to_remove:
            del self._map[pk]
        return len(to_remove)

    # ── lifecycle ───────────────────────────────────────────────

    @classmethod
    def rebuild_from(
        cls,
        pk_name: str,
        partition_delta_files: Dict[str, List[str]],
    ) -> "DeltaIndex":
        """Build a fresh DeltaIndex from on-disk delta Parquet files.

        Args:
            pk_name: primary key field name (used to extract the pk
                column from delta files).
            partition_delta_files: ``{partition_name: [absolute paths]}``.
                Cross-partition note: a single tombstone may live in
                multiple partitions' delta files (cross-partition delete
                replication at flush time). The max-seq aggregation
                naturally dedups them.

        Returns:
            A populated DeltaIndex.
        """
        idx = cls(pk_name)
        for _partition, paths in partition_delta_files.items():
            for path in paths:
                table = read_delta_file(path)
                idx.add_table(table)
        return idx

    # ── introspection ───────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._map)

    @property
    def snapshot(self) -> Dict[Any, int]:
        """Return a copy of the internal map."""
        return dict(self._map)
