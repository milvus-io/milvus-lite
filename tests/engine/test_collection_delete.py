"""Phase-5 Collection.delete tests.

Covers:
    - basic delete (single + batch)
    - delete unknown pk (no error)
    - delete shadows in MemTable
    - delete after flush (segment) is filtered by delta_index
    - cross-partition delete (partition_name=None → ALL_PARTITIONS)
    - delete after restart still hides the record
    - search/get both honor deletes
"""

import os

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.exceptions import PartitionNotFoundError, SchemaValidationError
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


def _make_record(i, prefix="doc"):
    return {
        "id": f"{prefix}_{i:04d}",
        "vec": [0.5, 0.25, 0.125, 0.75],
        "title": f"t{i}",
    }


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("c", str(tmp_path / "d"), schema)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_delete_pks_must_be_list(col):
    with pytest.raises(TypeError, match="must be a list"):
        col.delete("not a list")


def test_delete_empty_list_returns_zero(col):
    assert col.delete([]) == 0


def test_delete_unknown_partition_raises(col):
    with pytest.raises(PartitionNotFoundError):
        col.delete(["x"], partition_name="ghost")


# ---------------------------------------------------------------------------
# Basic delete in MemTable
# ---------------------------------------------------------------------------

def test_delete_single_pk_in_memtable(col):
    col.insert([_make_record(0)])
    col.delete(["doc_0000"])
    assert col.get(["doc_0000"]) == []


def test_delete_batch(col):
    col.insert([_make_record(i) for i in range(5)])
    n = col.delete([f"doc_{i:04d}" for i in (0, 2, 4)])
    assert n == 3
    # 1, 3 still alive
    survivors = col.get([f"doc_{i:04d}" for i in range(5)])
    surviving_ids = {r["id"] for r in survivors}
    assert surviving_ids == {"doc_0001", "doc_0003"}


def test_delete_unknown_pk_no_error(col):
    """Deleting a pk that was never inserted is OK — writes a tombstone
    that never matches anything."""
    n = col.delete(["ghost"])
    assert n == 1
    # No insert ever happened, so no record either way.
    assert col.get(["ghost"]) == []


def test_delete_then_get_returns_empty(col):
    col.insert([_make_record(0)])
    assert len(col.get(["doc_0000"])) == 1
    col.delete(["doc_0000"])
    assert col.get(["doc_0000"]) == []


# ---------------------------------------------------------------------------
# Delete after flush (segment + delta_index)
# ---------------------------------------------------------------------------

def test_delete_after_flush_hides_segment(col):
    """Insert, flush (X is in a segment), delete X, get(X) returns nothing
    because delta_index has the tombstone."""
    col.insert([_make_record(0)])
    col.flush()
    assert col.count() == 0
    col.delete(["doc_0000"])
    assert col.get(["doc_0000"]) == []


def test_delete_then_flush_persists_delta(col):
    col.insert([_make_record(0)])
    col.flush()
    col.delete(["doc_0000"])
    col.flush()
    # Manifest should now have a delta file.
    delta_files = col._manifest.get_delta_files("_default")
    assert len(delta_files) == 1
    assert col.get(["doc_0000"]) == []


def test_delete_does_not_resurrect_via_search(col):
    """search() must also honor deletes (via the bitmap pipeline)."""
    col.insert([
        {"id": "near", "vec": [1.0, 0.0, 0.0, 0.0], "title": "n"},
        {"id": "far",  "vec": [0.0, 1.0, 0.0, 0.0], "title": "f"},
    ])
    col.delete(["near"])
    results = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=10, metric_type="COSINE")
    [hits] = results
    ids = {h["id"] for h in hits}
    assert "near" not in ids
    assert ids == {"far"}


def test_delete_after_flush_search_filters(col):
    col.insert([
        {"id": "near", "vec": [1.0, 0.0, 0.0, 0.0], "title": "n"},
        {"id": "far",  "vec": [0.0, 1.0, 0.0, 0.0], "title": "f"},
    ])
    col.flush()
    col.delete(["near"])
    col.flush()
    results = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=10, metric_type="COSINE")
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"far"}


# ---------------------------------------------------------------------------
# Cross-partition delete (_all)
# ---------------------------------------------------------------------------

def test_cross_partition_delete_in_memtable(tmp_path, schema):
    col = Collection("c", str(tmp_path / "d"), schema)
    col._manifest.add_partition("p1")
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "p1"}], partition_name="p1")
    col.insert([{"id": "y", "vec": [0.5, 0.25, 0.125, 0.75], "title": "default"}], partition_name="_default")

    # Cross-partition delete (partition_name=None)
    col.delete(["x", "y"])
    assert col.get(["x"]) == []
    assert col.get(["y"]) == []
    col.close()


def test_cross_partition_delete_after_flush(tmp_path, schema):
    """Insert into p1 + p2, flush both, then cross-partition delete.
    After a second flush, both partitions should have a delta file with
    the tombstone."""
    col = Collection("c", str(tmp_path / "d"), schema)
    col._manifest.add_partition("p1")
    col._manifest.add_partition("p2")
    col.insert([{"id": "z", "vec": [0.5, 0.25, 0.125, 0.75], "title": "p1"}], partition_name="p1")
    col.insert([{"id": "w", "vec": [0.5, 0.25, 0.125, 0.75], "title": "p2"}], partition_name="p2")
    col.flush()

    col.delete(["z", "w"])  # cross-partition
    col.flush()

    # Each partition should now have at least one delta file.
    p1_delta = col._manifest.get_delta_files("p1")
    p2_delta = col._manifest.get_delta_files("p2")
    assert len(p1_delta) >= 1
    assert len(p2_delta) >= 1

    assert col.get(["z"]) == []
    assert col.get(["w"]) == []
    col.close()


# ---------------------------------------------------------------------------
# Delete + restart (delta_index rebuild from disk)
# ---------------------------------------------------------------------------

def test_delete_persists_across_restart(tmp_path, schema):
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([_make_record(0)])
    col1.delete(["doc_0000"])
    col1.close()  # flushes both insert + delete

    col2 = Collection("c", data_dir, schema)
    # delta_index rebuilt from disk
    assert col2.get(["doc_0000"]) == []
    # Search also honors it
    results = col2.search([[0.5, 0.25, 0.125, 0.75]], top_k=10, metric_type="COSINE")
    ids = {h["id"] for r in results for h in r}
    assert "doc_0000" not in ids
    col2.close()


def test_delete_via_wal_replay(tmp_path, schema):
    """Delete in MemTable + crash (no flush). Recovery replays the
    DeleteOp from the wal_delta file."""
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([_make_record(0)])
    col1.delete(["doc_0000"])
    # Crash — no close().
    del col1

    col2 = Collection("c", data_dir, schema)
    assert col2.get(["doc_0000"]) == []
    col2.close()


def test_delete_then_insert_then_restart(tmp_path, schema):
    """Insert(X) → delete(X) → insert(X) → close → restart.
    The new X should be visible after restart."""
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([{"id": "X", "vec": [0.5, 0.25, 0.125, 0.75], "title": "v1"}])
    col1.delete(["X"])
    col1.insert([{"id": "X", "vec": [0.5, 0.25, 0.125, 0.75], "title": "v2"}])
    col1.close()

    col2 = Collection("c", data_dir, schema)
    [rec] = col2.get(["X"])
    assert rec["title"] == "v2"
    col2.close()


# ---------------------------------------------------------------------------
# Delete that triggers flush
# ---------------------------------------------------------------------------

def test_delete_triggers_flush(tmp_path, schema, monkeypatch):
    """A delete that pushes the MemTable over the size limit must
    trigger a synchronous flush."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([_make_record(i) for i in range(4)])
    # Memtable size = 4. The next delete bumps it past 5.
    col.delete(["ghost1", "ghost2"])
    # Should have flushed.
    assert col.count() < 6
    col.close()
