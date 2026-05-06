"""Tests for storage/delta_index.py"""

import os
import sys

import pyarrow as pa
import pytest

from milvus_lite.schema.arrow_builder import build_delta_schema, build_wal_delta_schema
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.storage.delta_file import write_delta_file
from milvus_lite.storage.delta_index import DeltaIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ])


def _delta_batch(schema, pks, seqs):
    return pa.RecordBatch.from_pydict(
        {"id": pks, "_seq": seqs},
        schema=build_delta_schema(schema),
    )


def _wal_delta_batch(schema, pks, seq, partition="_default"):
    return pa.RecordBatch.from_pydict(
        {
            "id": pks,
            "_seq": [seq] * len(pks),
            "_partition": [partition] * len(pks),
        },
        schema=build_wal_delta_schema(schema),
    )


# ---------------------------------------------------------------------------
# Construction / basic add
# ---------------------------------------------------------------------------

def test_empty_index():
    idx = DeltaIndex("id")
    assert len(idx) == 0
    assert not idx.is_deleted("anything", 1)


def test_add_batch_simple(schema):
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a", "b"], [10, 10]))
    assert len(idx) == 2
    assert idx.is_deleted("a", 5)
    assert idx.is_deleted("b", 5)


def test_add_empty_batch_noop(schema):
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, [], []))
    assert len(idx) == 0


def test_add_batch_takes_max_seq(schema):
    """Same pk added twice — index keeps the larger seq."""
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a"], [5]))
    idx.add_batch(_delta_batch(schema, ["a"], [10]))
    idx.add_batch(_delta_batch(schema, ["a"], [3]))  # smaller — must NOT overwrite
    assert idx.snapshot["a"] == 10


def test_add_batch_accepts_wal_delta_schema(schema):
    """add_batch should accept both delta_schema and wal_delta_schema
    (the wal version has an extra _partition column)."""
    idx = DeltaIndex("id")
    idx.add_batch(_wal_delta_batch(schema, ["a"], seq=7))
    assert idx.is_deleted("a", 5)


def test_add_batch_missing_pk_column_raises():
    idx = DeltaIndex("id")
    bad = pa.RecordBatch.from_pydict(
        {"_seq": [1]},
        schema=pa.schema([pa.field("_seq", pa.uint64())]),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        idx.add_batch(bad)


# ---------------------------------------------------------------------------
# is_deleted semantics
# ---------------------------------------------------------------------------

def test_is_deleted_strictly_greater(schema):
    """is_deleted is `delete_seq > data_seq`, not `>=`."""
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a"], [10]))
    # data_seq=5 < delete_seq=10 → deleted
    assert idx.is_deleted("a", 5)
    # data_seq=10 == delete_seq=10 → NOT deleted (data is the same op as delete in seq order)
    assert not idx.is_deleted("a", 10)
    # data_seq=11 > delete_seq=10 → NOT deleted (data is newer)
    assert not idx.is_deleted("a", 11)


def test_is_deleted_unknown_pk(schema):
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a"], [10]))
    assert not idx.is_deleted("ghost", 5)


def test_add_table_folds_all_batches(schema):
    """add_table should iterate over all RecordBatches in the Table."""
    idx = DeltaIndex("id")
    table = pa.Table.from_batches([
        _delta_batch(schema, ["a"], [5]),
        _delta_batch(schema, ["b"], [10]),
    ])
    idx.add_table(table)
    assert len(idx) == 2


# ---------------------------------------------------------------------------
# gc_below — tombstone GC
# ---------------------------------------------------------------------------

def test_gc_below_drops_old_tombstones(schema):
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a", "b", "c"], [10, 50, 100]))
    removed = idx.gc_below(60)
    # a (10) and b (50) are < 60 → dropped; c (100) stays
    assert removed == 2
    assert len(idx) == 1
    assert "c" in idx.snapshot


def test_gc_below_keeps_equal_seq(schema):
    """gc_below uses strict <, not ≤. delete_seq == min_active_data_seq stays."""
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a"], [60]))
    removed = idx.gc_below(60)
    assert removed == 0
    assert "a" in idx.snapshot


def test_gc_below_drains_with_maxsize(schema):
    """Passing sys.maxsize drops everything (used when no data files exist)."""
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a", "b"], [10, 1000000]))
    removed = idx.gc_below(sys.maxsize)
    assert removed == 2
    assert len(idx) == 0


def test_gc_below_zero_keeps_all(schema):
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a", "b"], [10, 50]))
    removed = idx.gc_below(0)
    assert removed == 0
    assert len(idx) == 2


def test_gc_below_correctness_invariant(schema):
    """The architectural invariant: a tombstone (pk, delete_seq) is safe
    to drop iff no data row with seq <= delete_seq exists. Phase 3 test
    only verifies the conservative form: drop tombstones whose delete_seq
    is strictly below the global min_active_data_seq."""
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a"], [5]))
    idx.add_batch(_delta_batch(schema, ["b"], [15]))

    # Imagine min_active_data_seq = 10 (smallest seq_min in current data files)
    # — any data row containing 'a' must have seq >= 10. But 'a's tombstone
    # has delete_seq=5, so a delete_seq of 5 cannot supersede any data row
    # with seq >= 10 (5 is not > 10). The tombstone is unreachable. Safe to drop.
    idx.gc_below(10)
    assert "a" not in idx.snapshot
    assert "b" in idx.snapshot


# ---------------------------------------------------------------------------
# rebuild_from — startup path
# ---------------------------------------------------------------------------

def test_rebuild_from_no_files():
    idx = DeltaIndex.rebuild_from("id", partition_delta_files={})
    assert len(idx) == 0


def test_rebuild_from_single_file(tmp_path, schema):
    partition_dir = tmp_path / "partitions" / "_default"
    partition_dir.mkdir(parents=True)
    table = pa.Table.from_batches([_delta_batch(schema, ["a", "b"], [10, 10])])
    rel = write_delta_file(table, str(partition_dir), seq_min=10, seq_max=10)
    abs_path = str(partition_dir / rel)

    idx = DeltaIndex.rebuild_from("id", {"_default": [abs_path]})
    assert len(idx) == 2
    assert idx.is_deleted("a", 5)
    assert idx.is_deleted("b", 5)


def test_rebuild_from_multiple_files_max_seq_wins(tmp_path, schema):
    partition_dir = tmp_path / "partitions" / "_default"
    partition_dir.mkdir(parents=True)

    # Two files, same pk with different seqs.
    t1 = pa.Table.from_batches([_delta_batch(schema, ["a"], [10])])
    t2 = pa.Table.from_batches([_delta_batch(schema, ["a"], [20])])
    p1 = str(partition_dir / write_delta_file(t1, str(partition_dir), 10, 10))
    p2 = str(partition_dir / write_delta_file(t2, str(partition_dir), 20, 20))

    idx = DeltaIndex.rebuild_from("id", {"_default": [p1, p2]})
    assert idx.snapshot["a"] == 20


def test_rebuild_from_multiple_partitions(tmp_path, schema):
    p1_dir = tmp_path / "partitions" / "p1"
    p2_dir = tmp_path / "partitions" / "p2"
    p1_dir.mkdir(parents=True)
    p2_dir.mkdir(parents=True)

    t1 = pa.Table.from_batches([_delta_batch(schema, ["a"], [5])])
    t2 = pa.Table.from_batches([_delta_batch(schema, ["b"], [7])])
    abs1 = str(p1_dir / write_delta_file(t1, str(p1_dir), 5, 5))
    abs2 = str(p2_dir / write_delta_file(t2, str(p2_dir), 7, 7))

    idx = DeltaIndex.rebuild_from("id", {"p1": [abs1], "p2": [abs2]})
    assert len(idx) == 2
    assert idx.is_deleted("a", 1)
    assert idx.is_deleted("b", 1)


# ---------------------------------------------------------------------------
# snapshot is a copy
# ---------------------------------------------------------------------------

def test_snapshot_is_a_copy(schema):
    idx = DeltaIndex("id")
    idx.add_batch(_delta_batch(schema, ["a"], [10]))
    snap = idx.snapshot
    snap["b"] = 99  # mutate the snapshot
    assert "b" not in idx.snapshot
    assert idx.snapshot["a"] == 10
