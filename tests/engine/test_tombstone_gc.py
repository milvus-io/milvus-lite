"""Tombstone GC tests — verifies that compaction triggers correct
gc_below behaviour and that subsequent queries remain consistent.

Architectural invariant §3 (proof in modules.md §9.16):
    A delete tombstone (pk, delete_seq) is unreachable iff every active
    data file has seq_min > delete_seq. The conservative form drops
    tombstones whose delete_seq is strictly less than the global
    min_active_data_seq.
"""

import os
import sys

import pyarrow as pa
import pytest

from milvus_lite.constants import COMPACTION_MIN_FILES_PER_BUCKET, DEFAULT_PARTITION
from milvus_lite.engine.compaction import CompactionManager
from milvus_lite.schema.arrow_builder import build_data_schema, build_delta_schema
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.storage.data_file import write_data_file
from milvus_lite.storage.delta_index import DeltaIndex
from milvus_lite.storage.manifest import Manifest


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


@pytest.fixture
def harness(tmp_path, schema):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    return {
        "data_dir": data_dir,
        "schema": schema,
        "manifest": Manifest(data_dir),
        "delta_index": DeltaIndex("id"),
        "mgr": CompactionManager(data_dir, schema),
    }


def _write_file(harness, seq_min, seq_max, pks_with_seqs):
    table = pa.Table.from_pydict(
        {
            "_seq": [s for _, s in pks_with_seqs],
            "id": [pk for pk, _ in pks_with_seqs],
            "vec": [[0.5, 0.25] for _ in pks_with_seqs],
            "title": ["t" for _ in pks_with_seqs],
        },
        schema=build_data_schema(harness["schema"]),
    )
    partition_dir = os.path.join(harness["data_dir"], "partitions", DEFAULT_PARTITION)
    os.makedirs(partition_dir, exist_ok=True)
    rel = write_data_file(table, partition_dir, seq_min, seq_max)
    harness["manifest"].add_data_file(DEFAULT_PARTITION, rel)
    return rel


# ---------------------------------------------------------------------------
# _global_min_active_data_seq
# ---------------------------------------------------------------------------

def test_global_min_no_files():
    """No data files → return sys.maxsize so the entire delta_index drains."""
    m = Manifest(data_dir="/dummy")
    assert CompactionManager._global_min_active_data_seq(m) == sys.maxsize


def test_global_min_single_file(harness):
    _write_file(harness, seq_min=10, seq_max=20, pks_with_seqs=[("a", 10), ("b", 20)])
    assert CompactionManager._global_min_active_data_seq(harness["manifest"]) == 10


def test_global_min_across_partitions(tmp_path, schema):
    data_dir = str(tmp_path / "data")
    manifest = Manifest(data_dir)
    manifest.add_partition("p1")
    manifest.add_partition("p2")
    # File names encode the seq range; the function parses them.
    manifest.add_data_file("p1", "data/data_000050_000060.parquet")
    manifest.add_data_file("p2", "data/data_000010_000020.parquet")
    manifest.add_data_file("p2", "data/data_000100_000200.parquet")
    assert CompactionManager._global_min_active_data_seq(manifest) == 10


# ---------------------------------------------------------------------------
# gc_below correctness via compaction trigger
# ---------------------------------------------------------------------------

def test_compaction_runs_gc_below(harness, schema):
    """Set up: 4 data files at seq 100..103, plus tombstones at seqs
    50, 80, 200. After compaction (min_active_data_seq = 100),
    tombstones < 100 (i.e. 50 and 80) must be GC'd, but the one at
    200 must remain."""
    for i in range(4):
        _write_file(harness, seq_min=100 + i, seq_max=100 + i,
                    pks_with_seqs=[(f"doc_{i}", 100 + i)])

    harness["delta_index"].add_batch(pa.RecordBatch.from_pydict(
        {"id": ["x", "y", "z"], "_seq": [50, 80, 200]},
        schema=build_delta_schema(schema),
    ))
    assert len(harness["delta_index"]) == 3

    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )

    # After compaction, the merged file has seq_min = 100. Tombstones
    # 50 and 80 are < 100 → dropped. 200 stays.
    snap = harness["delta_index"].snapshot
    assert "x" not in snap
    assert "y" not in snap
    assert snap.get("z") == 200


def test_gc_does_not_drop_tombstones_in_same_seq_range(harness, schema):
    """A tombstone whose delete_seq equals the new file's seq_min must
    NOT be dropped (gc_below uses strict <)."""
    for i in range(4):
        _write_file(harness, seq_min=100 + i, seq_max=100 + i,
                    pks_with_seqs=[(f"doc_{i}", 100 + i)])

    harness["delta_index"].add_batch(pa.RecordBatch.from_pydict(
        {"id": ["edge"], "_seq": [100]},  # exactly equal to min_active
        schema=build_delta_schema(schema),
    ))
    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )
    # Edge tombstone must survive — could still apply to a future
    # data file that doesn't exist yet but might have seq_min == 100
    # in some recovery scenarios.
    assert harness["delta_index"].snapshot.get("edge") == 100


def test_gc_after_compaction_does_not_break_query(harness, schema):
    """Even after gc_below removes obsolete tombstones, every still-live
    data row must remain queryable / unaffected."""
    # Initial state: 4 files with rows doc_0..doc_3 at seqs 100..103.
    # Tombstone for doc_0 at seq 999 (definitely active).
    # After compaction, doc_0 should be filtered out, doc_1..doc_3 stay.
    for i in range(4):
        _write_file(harness, seq_min=100 + i, seq_max=100 + i,
                    pks_with_seqs=[(f"doc_{i}", 100 + i)])
    harness["delta_index"].add_batch(pa.RecordBatch.from_pydict(
        {"id": ["doc_0"], "_seq": [999]},
        schema=build_delta_schema(schema),
    ))

    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )

    # The merged file should contain doc_1, doc_2, doc_3.
    [rel] = harness["manifest"].get_data_files(DEFAULT_PARTITION)
    abs_path = os.path.join(harness["data_dir"], "partitions", DEFAULT_PARTITION, rel)
    from milvus_lite.storage.data_file import read_data_file
    table = read_data_file(abs_path)
    pks = set(table.column("id").to_pylist())
    assert pks == {"doc_1", "doc_2", "doc_3"}

    # The doc_0 tombstone (seq=999) survives because it's > min_active(101)
    # — wait, after compaction the merged file has seq_min = 100..103, so
    # min_active = 100. 999 > 100 → tombstone survives.
    assert harness["delta_index"].snapshot.get("doc_0") == 999


def test_gc_drops_everything_when_no_data_files(harness, schema):
    """If compaction empties the partition (all rows filtered + no other
    data files), and we re-run compaction, the delta_index can drop
    every entry — there's no data row left to filter against.

    But the gc only triggers from maybe_compact, which requires
    >= MIN_FILES_PER_BUCKET files. So we test gc_below directly here.
    """
    # No data files in manifest. _global_min returns sys.maxsize.
    harness["delta_index"].add_batch(pa.RecordBatch.from_pydict(
        {"id": ["a", "b", "c"], "_seq": [10, 20, 30]},
        schema=build_delta_schema(schema),
    ))
    assert len(harness["delta_index"]) == 3

    # Manually invoke the GC (the manager exposes _gc_tombstones via
    # maybe_compact, but with no files there's nothing to compact).
    removed = harness["mgr"]._gc_tombstones(harness["manifest"], harness["delta_index"])
    assert removed == 3
    assert len(harness["delta_index"]) == 0


# ---------------------------------------------------------------------------
# Delta file GC — on-disk parquet removal
# ---------------------------------------------------------------------------

def _write_delta_file(harness, schema, seq_min, seq_max, pks):
    """Write a delta Parquet file and register it in the manifest."""
    table = pa.Table.from_pydict(
        {"id": pks, "_seq": [seq_min + i for i in range(len(pks))]},
        schema=build_delta_schema(schema),
    )
    partition_dir = os.path.join(harness["data_dir"], "partitions", DEFAULT_PARTITION)
    from milvus_lite.storage.delta_file import write_delta_file
    rel = write_delta_file(table, partition_dir, seq_min, seq_max)
    harness["manifest"].add_delta_file(DEFAULT_PARTITION, rel)
    return rel


def test_gc_removes_obsolete_delta_files(harness, schema):
    """Delta parquet files whose seq_max < global_min_active_data_seq
    should be deleted from disk and removed from the manifest."""
    # Data files at seq 100..103 → global_min_active_data_seq = 100.
    for i in range(4):
        _write_file(harness, seq_min=100 + i, seq_max=100 + i,
                    pks_with_seqs=[(f"doc_{i}", 100 + i)])

    # Delta files:
    #   delta_050_080 → seq_max=80 < 100 → should be GC'd
    #   delta_090_200 → seq_max=200 ≥ 100 → should survive
    rel_old = _write_delta_file(harness, schema, 50, 80, ["x", "y"])
    rel_new = _write_delta_file(harness, schema, 90, 200, ["z"])

    # Add tombstones to delta_index too.
    harness["delta_index"].add_batch(pa.RecordBatch.from_pydict(
        {"id": ["x", "y", "z"], "_seq": [50, 80, 200]},
        schema=build_delta_schema(schema),
    ))

    partition_dir = os.path.join(harness["data_dir"], "partitions", DEFAULT_PARTITION)

    # Verify files exist before GC.
    old_path = os.path.join(partition_dir, rel_old)
    new_path = os.path.join(partition_dir, rel_new)
    assert os.path.exists(old_path)
    assert os.path.exists(new_path)

    # Trigger compaction → gc runs.
    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )

    # Old delta file gone from disk and manifest.
    assert not os.path.exists(old_path)
    delta_files = harness["manifest"].get_delta_files(DEFAULT_PARTITION)
    assert rel_old not in delta_files

    # New delta file survives.
    assert os.path.exists(new_path)
    assert rel_new in delta_files


def test_gc_keeps_delta_files_when_data_seqs_are_low(harness, schema):
    """If data file seqs are low, delta files should not be removed."""
    # Data at seq 10..13 → global_min = 10.
    for i in range(4):
        _write_file(harness, seq_min=10 + i, seq_max=10 + i,
                    pks_with_seqs=[(f"doc_{i}", 10 + i)])

    # Delta at seq 50 → seq_max=50 ≥ 10 → should survive.
    rel = _write_delta_file(harness, schema, 50, 50, ["x"])
    harness["delta_index"].add_batch(pa.RecordBatch.from_pydict(
        {"id": ["x"], "_seq": [50]},
        schema=build_delta_schema(schema),
    ))

    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )

    delta_files = harness["manifest"].get_delta_files(DEFAULT_PARTITION)
    assert rel in delta_files
