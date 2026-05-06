"""Flush pipeline — synchronous, 7 steps.

Per architectural invariant §6, flush blocks the writer until done.

Pre-condition: the caller has already done Step 1 — frozen the active
MemTable + WAL pair, and replaced them with fresh ones on the
Collection. ``execute_flush`` receives the frozen pair and walks
Steps 2-7.

Step 2: frozen_memtable.flush() → {partition → (data_table, delta_table)}
Step 3: write data + delta Parquet files into per-partition dirs
Step 4: fold delta tables into delta_index (in-memory commit)
Step 5: atomic Manifest update (new files + current_seq + active_wal)
Step 6: frozen_wal.close_and_delete() + clean any orphan WAL files
Step 7: compaction trigger (Phase 6 — placeholder for now)

Crash safety:
    Step 3 crash → orphan Parquet on disk, manifest unchanged →
                   recovery cleans orphans
    Step 5 mid-write → manifest is atomic (rename), either old or new
                       version is loaded; recovery WAL replay handles
                       both cases
    Step 5 done, Step 6 not yet → manifest points to new files; WAL
                                  still on disk → recovery replays WAL
                                  but _seq dedup makes it idempotent
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from milvus_lite.constants import ALL_PARTITIONS
from milvus_lite.storage.data_file import write_data_file
from milvus_lite.storage.delta_file import write_delta_file

if TYPE_CHECKING:
    from milvus_lite.schema.types import CollectionSchema
    from milvus_lite.storage.delta_index import DeltaIndex
    from milvus_lite.storage.manifest import Manifest
    from milvus_lite.storage.memtable import MemTable
    from milvus_lite.storage.wal import WAL


def execute_flush(
    frozen_memtable: "MemTable",
    frozen_wal: "WAL",
    data_dir: str,
    schema: "CollectionSchema",
    manifest: "Manifest",
    delta_index: "DeltaIndex",
    new_wal_number: int,
) -> None:
    """Execute steps 2-7 of the flush pipeline.

    Args:
        frozen_memtable: the MemTable that was just frozen by Collection.
        frozen_wal: the WAL paired with frozen_memtable; will be closed
            and deleted at Step 6.
        data_dir: Collection's root data directory.
        schema: Collection's schema.
        manifest: live Manifest object — will be mutated and saved.
        delta_index: live DeltaIndex — will be mutated in Step 4.
        new_wal_number: the WAL number that the Collection has already
            switched to (Step 1). This goes into manifest.active_wal_number
            so recovery knows where to pick up.
    """
    # ── Step 2: materialize the frozen MemTable ─────────────────
    known_partitions = manifest.list_partitions()
    flushed = frozen_memtable.flush(known_partitions=known_partitions)

    if not flushed:
        # Empty MemTable. Still update active_wal_number and clean up the
        # frozen WAL — otherwise we'd leave it orphaned forever.
        manifest.active_wal_number = new_wal_number
        manifest.save()
        frozen_wal.close_and_delete()
        _cleanup_old_wals(os.path.join(data_dir, "wal"), frozen_wal.number)
        return

    max_seq = frozen_memtable.max_seq

    # Track what we wrote so we can update the manifest atomically.
    written_data: List[tuple] = []   # (partition, rel_path)
    written_delta: List[tuple] = []  # (partition, rel_path)

    partitions_root = os.path.join(data_dir, "partitions")

    # ── Step 3: write Parquet files ─────────────────────────────
    for partition, (data_table, delta_table) in flushed.items():
        # Defensive: cross-partition deletes should already be expanded
        # by MemTable.flush(known_partitions=...). If we still see "_all"
        # here it means the partition list was empty — skip silently.
        if partition == ALL_PARTITIONS:
            continue

        partition_dir = os.path.join(partitions_root, partition)

        if data_table is not None and data_table.num_rows > 0:
            seq_min, seq_max = _seq_range(data_table)
            rel = write_data_file(data_table, partition_dir, seq_min, seq_max)
            written_data.append((partition, rel))

        if delta_table is not None and delta_table.num_rows > 0:
            seq_min, seq_max = _seq_range(delta_table)
            rel = write_delta_file(delta_table, partition_dir, seq_min, seq_max)
            written_delta.append((partition, rel))

    # ── Step 4: in-memory commit (delta_index) ──────────────────
    # We do this BEFORE manifest.save() so that any reader that opens
    # the manifest at the new version will also see the matching delta
    # entries in the in-memory index. (In Phase 3 there's only one
    # writer/reader, but the ordering matters once concurrent search
    # is added.)
    for partition, (_data_table, delta_table) in flushed.items():
        if delta_table is not None and delta_table.num_rows > 0:
            delta_index.add_table(delta_table)

    # ── Step 5: manifest commit ─────────────────────────────────
    for partition, rel in written_data:
        manifest.add_data_file(partition, rel)
    for partition, rel in written_delta:
        manifest.add_delta_file(partition, rel)

    if max_seq >= 0:
        manifest.current_seq = max_seq
    manifest.active_wal_number = new_wal_number

    manifest.save()

    # ── Step 6: WAL cleanup ─────────────────────────────────────
    frozen_wal.close_and_delete()
    _cleanup_old_wals(os.path.join(data_dir, "wal"), frozen_wal.number)

    # ── Step 7: compaction (Phase 6) ────────────────────────────
    # placeholder — no-op until compaction lands.


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _seq_range(table) -> tuple:
    seq_col = table.column("_seq")
    pylist = seq_col.to_pylist()
    return min(pylist), max(pylist)


def _cleanup_old_wals(wal_dir: str, up_to_number: int) -> None:
    """Remove any WAL pairs whose number is <= *up_to_number*.

    Used after flush Step 6 to mop up any earlier WAL files that
    survived a previous flush crash.
    """
    # Local import to avoid the static import cycle warning at module load.
    from milvus_lite.storage.wal import _cleanup_old_wals as _cleanup
    _cleanup(wal_dir, up_to_number)
