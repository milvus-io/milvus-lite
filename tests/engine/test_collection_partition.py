"""Phase 9.1.1 — Collection partition CRUD tests.

Validates Collection.create_partition / drop_partition / list_partitions /
has_partition. Manifest already implements the partition CRUD primitives;
this layer wraps them with on-disk directory management and pre-flush
behavior.
"""

import os

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.exceptions import (
    DefaultPartitionError,
    PartitionAlreadyExistsError,
    PartitionNotFoundError,
)
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ])


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# list / has — fresh state
# ---------------------------------------------------------------------------

def test_fresh_collection_has_only_default(col):
    assert col.list_partitions() == ["_default"]
    assert col.has_partition("_default") is True
    assert col.has_partition("nope") is False


# ---------------------------------------------------------------------------
# create_partition
# ---------------------------------------------------------------------------

def test_create_partition_basic(col):
    col.create_partition("p1")
    assert col.has_partition("p1") is True
    assert "p1" in col.list_partitions()
    # list_partitions returns sorted, so the new one should be in there.
    assert col.list_partitions() == sorted(["_default", "p1"])


def test_create_partition_creates_dir_on_disk(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    c.create_partition("zone_a")
    expected = tmp_path / "data" / "partitions" / "zone_a"
    assert expected.exists() and expected.is_dir()
    c.close()


def test_create_partition_persists_across_restart(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    c.create_partition("p1")
    c.create_partition("p2")
    c.close()

    c2 = Collection("test", str(tmp_path / "data"), schema)
    try:
        assert sorted(c2.list_partitions()) == ["_default", "p1", "p2"]
    finally:
        c2.close()


def test_create_partition_duplicate_raises(col):
    col.create_partition("p1")
    with pytest.raises(PartitionAlreadyExistsError):
        col.create_partition("p1")


def test_create_partition_duplicate_default_raises(col):
    with pytest.raises(PartitionAlreadyExistsError):
        col.create_partition("_default")


# ---------------------------------------------------------------------------
# drop_partition
# ---------------------------------------------------------------------------

def test_drop_partition_basic(col):
    col.create_partition("p1")
    col.drop_partition("p1")
    assert col.has_partition("p1") is False
    assert "p1" not in col.list_partitions()


def test_drop_partition_removes_dir_on_disk(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    c.create_partition("p1")
    partition_dir = tmp_path / "data" / "partitions" / "p1"
    assert partition_dir.exists()
    c.drop_partition("p1")
    assert not partition_dir.exists()
    c.close()


def test_drop_default_partition_raises(col):
    with pytest.raises(DefaultPartitionError):
        col.drop_partition("_default")


def test_drop_nonexistent_partition_raises(col):
    with pytest.raises(PartitionNotFoundError):
        col.drop_partition("ghost")


def test_drop_partition_persists_across_restart(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    c.create_partition("p1")
    c.create_partition("p2")
    c.drop_partition("p1")
    c.close()

    c2 = Collection("test", str(tmp_path / "data"), schema)
    try:
        assert sorted(c2.list_partitions()) == ["_default", "p2"]
    finally:
        c2.close()


def test_drop_partition_with_data(tmp_path, schema):
    """Drop a partition that has flushed data files. The data files +
    partition dir should disappear; segment cache should drop them."""
    c = Collection("test", str(tmp_path / "data"), schema)
    c.create_partition("p1")
    c.insert(
        [{"id": f"x{i}", "vec": [0.1, 0.2, 0.3, 0.4], "score": 0.5}
         for i in range(5)],
        partition_name="p1",
    )
    c.flush()

    # Sanity: there is at least one segment in p1.
    p1_segments = [k for k in c._segment_cache if k[0] == "p1"]
    assert len(p1_segments) >= 1

    c.drop_partition("p1")

    # Segments for p1 should be evicted from the cache.
    p1_segments_after = [k for k in c._segment_cache if k[0] == "p1"]
    assert p1_segments_after == []

    partition_dir = tmp_path / "data" / "partitions" / "p1"
    assert not partition_dir.exists()

    c.close()


def test_drop_partition_with_pending_memtable_flushes_first(tmp_path, schema):
    """If the memtable has rows targeting an OTHER partition when we
    drop p1, we should not lose them. The drop should auto-flush first."""
    c = Collection("test", str(tmp_path / "data"), schema)
    c.create_partition("p1")
    c.create_partition("p2")
    c.insert(
        [{"id": "a", "vec": [1.0, 2.0, 3.0, 4.0], "score": 1.0}],
        partition_name="p2",
    )
    # No flush — there is a pending row in the memtable for p2.
    assert c._memtable.size() > 0

    c.drop_partition("p1")
    # Memtable should be empty (auto-flushed).
    assert c._memtable.size() == 0
    # The p2 row should still be retrievable after the flush.
    rec = c.get(["a"])
    assert len(rec) == 1
    assert rec[0]["id"] == "a"
    c.close()
