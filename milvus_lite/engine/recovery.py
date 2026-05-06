"""Crash recovery — 5 steps.

Pre-condition: caller has already loaded the Manifest (Step 1).

Step 2: Replay any WAL files found on disk into a fresh MemTable, in
        _seq order via replay_wal_operations().
Step 3: Verify Manifest's data/delta files exist on disk; warn on
        mismatches but accept them as a sign of a compaction-mid-crash.
Step 4: Clean orphan Parquet files (on disk but not in Manifest).
Step 5: Rebuild DeltaIndex from all delta files in the Manifest.

Returns the recovered (memtable, delta_index, next_wal_number).

The recovery contract is order-independent: replay can hit operations
in any physical order because MemTable is seq-aware (architectural
invariant §2). replay_wal_operations sorts by _seq purely for
next-seq derivation cleanliness, not correctness.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Iterator, List, Tuple

from milvus_lite.engine.operation import DeleteOp, InsertOp, Operation
from milvus_lite.storage.delta_index import DeltaIndex
from milvus_lite.storage.memtable import MemTable
from milvus_lite.storage.wal import WAL

if TYPE_CHECKING:
    from milvus_lite.schema.types import CollectionSchema
    from milvus_lite.storage.manifest import Manifest

logger = logging.getLogger(__name__)


def execute_recovery(
    data_dir: str,
    schema: "CollectionSchema",
    manifest: "Manifest",
) -> Tuple[MemTable, DeltaIndex, int]:
    """Run the 5-step recovery and return restored state.

    Returns:
        (memtable, delta_index, next_wal_number)
        - memtable: replayed-from-WAL MemTable, possibly empty
        - delta_index: rebuilt from manifest's delta files
        - next_wal_number: the WAL number a fresh Collection should
          start writing to (max existing + 1, or manifest's
          active_wal_number, whichever is greater)
    """
    from milvus_lite.schema.arrow_builder import get_primary_field

    pk_name = get_primary_field(schema).name
    wal_dir = os.path.join(data_dir, "wal")

    memtable = MemTable(schema)

    # ── Step 2: replay WAL ──────────────────────────────────────
    found_wal_numbers = WAL.find_wal_files(wal_dir)
    for n in found_wal_numbers:
        for op in replay_wal_operations(wal_dir, n, pk_name):
            if isinstance(op, InsertOp):
                memtable.apply_insert(op.batch)
            else:
                memtable.apply_delete(op.batch)

    # ── Step 3: verify manifest files exist ─────────────────────
    # Phase 3 doesn't have compaction yet, so missing files would be
    # a real bug. We log loudly rather than crash so manual recovery
    # is possible.
    for partition, files in manifest.get_all_data_files().items():
        for rel in files:
            abs_path = _abs_partition_file(data_dir, partition, rel)
            if not os.path.exists(abs_path):
                logger.warning(
                    "manifest references missing data file: %s/%s",
                    partition, rel,
                )
    for partition, files in manifest.get_all_delta_files().items():
        for rel in files:
            abs_path = _abs_partition_file(data_dir, partition, rel)
            if not os.path.exists(abs_path):
                logger.warning(
                    "manifest references missing delta file: %s/%s",
                    partition, rel,
                )

    # ── Step 4: orphan file cleanup ─────────────────────────────
    _cleanup_orphan_files(data_dir, manifest)

    # ── Step 5: rebuild delta_index ─────────────────────────────
    delta_files_per_partition: dict[str, List[str]] = {}
    for partition, rels in manifest.get_all_delta_files().items():
        abs_paths = [
            _abs_partition_file(data_dir, partition, rel) for rel in rels
        ]
        # Filter out missing files (already warned in Step 3).
        delta_files_per_partition[partition] = [
            p for p in abs_paths if os.path.exists(p)
        ]
    delta_index = DeltaIndex.rebuild_from(pk_name, delta_files_per_partition)

    # ── next_wal_number ─────────────────────────────────────────
    candidates = []
    if found_wal_numbers:
        candidates.append(max(found_wal_numbers) + 1)
    if manifest.active_wal_number is not None:
        candidates.append(manifest.active_wal_number)
    next_wal_number = max(candidates) if candidates else 1

    return memtable, delta_index, next_wal_number


# ---------------------------------------------------------------------------
# replay_wal_operations
# ---------------------------------------------------------------------------

def replay_wal_operations(
    wal_dir: str,
    wal_number: int,
    pk_field: str,
) -> Iterator[Operation]:
    """Read a WAL pair and yield Operations in _seq order.

    This is the engine-layer wrapper around the storage-layer
    ``WAL.recover``. Storage stays raw-batch only (see modules.md
    §9.16.5 for the dependency-layering rationale); this function
    is the adapter that knows about Operation types.

    Yields a stream sorted by the starting _seq of each batch — so
    a delta batch with seq=6 comes between two insert batches with
    seqs 5 and 7. This is purely for "the last yielded op.seq is
    the new max seq" cleanliness; correctness does NOT depend on
    order (MemTable is seq-aware).

    The wal_data file's batches preserve their _partition column from
    write time, so the InsertOp's partition is recovered correctly.
    """
    data_batches, delta_batches = WAL.recover(wal_dir, wal_number)

    items: list[tuple[int, Operation]] = []

    for b in data_batches:
        if b.num_rows == 0:
            continue
        partition = b.column("_partition")[0].as_py()
        seq_min = b.column("_seq")[0].as_py()
        items.append((seq_min, InsertOp(partition=partition, batch=b)))

    for b in delta_batches:
        if b.num_rows == 0:
            continue
        partition = b.column("_partition")[0].as_py()
        seq = b.column("_seq")[0].as_py()
        items.append((seq, DeleteOp(partition=partition, batch=b)))

    items.sort(key=lambda x: x[0])
    for _, op in items:
        yield op


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _abs_partition_file(data_dir: str, partition: str, rel_path: str) -> str:
    return os.path.join(data_dir, "partitions", partition, rel_path)


def _cleanup_orphan_files(data_dir: str, manifest: "Manifest") -> None:
    """Remove Parquet files on disk that are NOT referenced by the manifest.

    Walks every partition's data/ and delta/ subdirectories.

    Phase 9.4: also walks ``indexes/`` and removes any .idx whose
    source data file (matched by filename stem) is no longer in the
    manifest. This handles the case where a crash happens after
    compaction wrote new segments but before the old indexes were
    cleaned up.
    """
    partitions_root = os.path.join(data_dir, "partitions")
    if not os.path.exists(partitions_root):
        return

    referenced_data: dict[str, set[str]] = {
        p: set(files) for p, files in manifest.get_all_data_files().items()
    }
    referenced_delta: dict[str, set[str]] = {
        p: set(files) for p, files in manifest.get_all_delta_files().items()
    }
    # For index orphan detection we need the bare stems (without the
    # "data/" prefix and ".parquet" suffix) so we can compare against
    # .idx file stems.
    referenced_data_stems: dict[str, set[str]] = {}
    for p, files in referenced_data.items():
        stems: set[str] = set()
        for rel in files:
            stem = os.path.splitext(os.path.basename(rel))[0]
            stems.add(stem)
        referenced_data_stems[p] = stems

    for partition in os.listdir(partitions_root):
        partition_dir = os.path.join(partitions_root, partition)
        if not os.path.isdir(partition_dir):
            continue

        # data subdir
        data_subdir = os.path.join(partition_dir, "data")
        if os.path.isdir(data_subdir):
            ref = referenced_data.get(partition, set())
            for fn in os.listdir(data_subdir):
                rel = os.path.join("data", fn)
                if rel not in ref:
                    abs_path = os.path.join(partition_dir, rel)
                    try:
                        os.remove(abs_path)
                        logger.info("recovery: removed orphan data file %s", abs_path)
                    except OSError:
                        pass

        # delta subdir
        delta_subdir = os.path.join(partition_dir, "delta")
        if os.path.isdir(delta_subdir):
            ref = referenced_delta.get(partition, set())
            for fn in os.listdir(delta_subdir):
                rel = os.path.join("delta", fn)
                if rel not in ref:
                    abs_path = os.path.join(partition_dir, rel)
                    try:
                        os.remove(abs_path)
                        logger.info("recovery: removed orphan delta file %s", abs_path)
                    except OSError:
                        pass

        # Phase 9.4: indexes subdir
        index_subdir = os.path.join(partition_dir, "indexes")
        if os.path.isdir(index_subdir):
            valid_stems = referenced_data_stems.get(partition, set())
            for fn in os.listdir(index_subdir):
                # Index filename convention:
                #   <data_stem>.<field_name>.<index_type_lower>.idx
                # Strip .idx, then rpartition twice to peel off index_type
                # and field_name, leaving the data_stem.
                # E.g. "data_000001_000050.dense.hnsw.idx"
                #  → base = "data_000001_000050.dense.hnsw"
                #  → after first rpartition: ("data_000001_000050.dense", "hnsw")
                #  → after second rpartition: ("data_000001_000050", "dense")
                if not fn.endswith(".idx"):
                    continue
                base = fn[:-len(".idx")]
                stem_field, _, _index_type = base.rpartition(".")
                stem, _, _field_name = stem_field.rpartition(".")
                if not stem:
                    # Malformed name; remove defensively.
                    stem = stem_field or base
                if stem not in valid_stems:
                    abs_path = os.path.join(index_subdir, fn)
                    try:
                        os.remove(abs_path)
                        logger.info("recovery: removed orphan index file %s", abs_path)
                    except OSError:
                        pass
