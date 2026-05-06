"""Tests for milvus_lite.db.MilvusLite — multi-Collection lifecycle + LOCK."""

import multiprocessing
import os
import time

import pytest

from milvus_lite.db import MilvusLite, LOCK_FILENAME, SCHEMA_FILENAME
from milvus_lite.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DataDirLockedError,
    SchemaValidationError,
)
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


@pytest.fixture
def db(tmp_path):
    d = MilvusLite(str(tmp_path / "data"))
    yield d
    d.close()


def _make_record(i, prefix="doc"):
    return {
        "id": f"{prefix}_{i:04d}",
        "vec": [0.5, 0.25, 0.125, 0.75],
        "title": f"t{i}",
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_creates_data_dir(tmp_path):
    data_dir = tmp_path / "fresh"
    assert not data_dir.exists()
    db = MilvusLite(str(data_dir))
    try:
        assert data_dir.exists()
        assert (data_dir / "collections").exists()
        assert (data_dir / LOCK_FILENAME).exists()
    finally:
        db.close()


def test_construction_idempotent_on_existing(tmp_path):
    data_dir = str(tmp_path / "data")
    db1 = MilvusLite(data_dir)
    db1.close()
    db2 = MilvusLite(data_dir)
    db2.close()  # no error


# ---------------------------------------------------------------------------
# Collection CRUD
# ---------------------------------------------------------------------------

def test_create_collection(db, schema):
    col = db.create_collection("docs", schema)
    assert col.name == "docs"
    assert db.has_collection("docs")
    assert "docs" in db.list_collections()


def test_create_duplicate_raises(db, schema):
    db.create_collection("docs", schema)
    with pytest.raises(CollectionAlreadyExistsError):
        db.create_collection("docs", schema)


def test_create_with_invalid_schema_raises(db):
    bad_schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        # missing FLOAT_VECTOR
    ])
    with pytest.raises(SchemaValidationError):
        db.create_collection("docs", bad_schema)
    # No directory should have been created.
    assert not db.has_collection("docs")


def test_get_collection_returns_cached_instance(db, schema):
    col1 = db.create_collection("docs", schema)
    col2 = db.get_collection("docs")
    assert col1 is col2


def test_get_collection_not_found(db):
    with pytest.raises(CollectionNotFoundError):
        db.get_collection("ghost")


def test_drop_collection(db, schema):
    db.create_collection("docs", schema)
    db.drop_collection("docs")
    assert not db.has_collection("docs")
    assert "docs" not in db.list_collections()


def test_drop_nonexistent_is_idempotent(db):
    """Dropping a non-existent collection should silently succeed (Milvus compat)."""
    db.drop_collection("ghost")  # no error


def test_drop_removes_data_dir(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "data"))
    db.create_collection("docs", schema)
    col_dir = os.path.join(db.data_dir, "collections", "docs")
    assert os.path.exists(col_dir)
    db.drop_collection("docs")
    assert not os.path.exists(col_dir)
    db.close()


def test_list_collections_sorted(db, schema):
    db.create_collection("zebra", schema)
    db.create_collection("alpha", schema)
    db.create_collection("middle", schema)
    assert db.list_collections() == ["alpha", "middle", "zebra"]


def test_list_collections_empty(db):
    assert db.list_collections() == []


# ---------------------------------------------------------------------------
# rename_collection (Issue #11)
# ---------------------------------------------------------------------------

def test_rename_collection(db, schema):
    col = db.create_collection("old_name", schema)
    col.insert([{"id": "a", "vec": [1, 0, 0, 0]}])
    db.rename_collection("old_name", "new_name")
    assert not db.has_collection("old_name")
    assert db.has_collection("new_name")
    # Data survives rename
    col2 = db.get_collection("new_name")
    col2.load()
    assert col2.num_entities == 1


def test_rename_nonexistent_raises(db):
    with pytest.raises(CollectionNotFoundError):
        db.rename_collection("ghost", "new")


def test_rename_to_existing_raises(db, schema):
    db.create_collection("a", schema)
    db.create_collection("b", schema)
    with pytest.raises(CollectionAlreadyExistsError):
        db.rename_collection("a", "b")


def test_rename_invalid_name_raises(db, schema):
    db.create_collection("src", schema)
    with pytest.raises(ValueError):
        db.rename_collection("src", "")


# ---------------------------------------------------------------------------
# get_collection_stats (Phase 9.1.4)
# ---------------------------------------------------------------------------

def test_get_collection_stats_empty(db, schema):
    db.create_collection("docs", schema)
    stats = db.get_collection_stats("docs")
    assert stats == {"row_count": 0}


def test_get_collection_stats_with_data(db, schema):
    col = db.create_collection("docs", schema)
    col.insert([_make_record(i) for i in range(7)])
    stats = db.get_collection_stats("docs")
    assert stats["row_count"] == 7


def test_get_collection_stats_after_flush_and_delete(db, schema):
    col = db.create_collection("docs", schema)
    col.insert([_make_record(i) for i in range(5)])
    col.flush()
    col.delete(pks=[_make_record(0)["id"], _make_record(1)["id"]])
    col.flush()
    stats = db.get_collection_stats("docs")
    assert stats["row_count"] == 3


def test_get_collection_stats_unknown_raises(db):
    with pytest.raises(CollectionNotFoundError):
        db.get_collection_stats("ghost")


# ---------------------------------------------------------------------------
# Alias + truncate (P0 compatibility)
# ---------------------------------------------------------------------------

def test_alias_persists_across_reopen(tmp_path, schema):
    data_dir = str(tmp_path / "data")
    db = MilvusLite(data_dir)
    db.create_collection("docs", schema)
    db.create_alias("docs", "current")
    db.close()

    db2 = MilvusLite(data_dir)
    try:
        assert db2.describe_alias("current") == {
            "alias": "current",
            "collection": "docs",
        }
        assert db2.get_collection("current").name == "docs"
    finally:
        db2.close()


def test_truncate_collection_preserves_schema_and_alias(db, schema):
    col = db.create_collection("docs", schema)
    db.create_alias("docs", "current")
    col.insert([_make_record(i) for i in range(3)])
    assert db.get_collection_stats("docs")["row_count"] == 3

    db.truncate_collection("current")
    assert db.get_collection_stats("docs")["row_count"] == 0
    assert db.describe_alias("current")["collection"] == "docs"

    fresh = db.get_collection("docs")
    fresh.insert([_make_record(9)])
    assert db.get_collection_stats("current")["row_count"] == 1


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["", "../escape", "foo/bar", "foo\\bar", ".", ".."])
def test_invalid_names_rejected(db, schema, name):
    with pytest.raises((ValueError, TypeError)):
        db.create_collection(name, schema)


def test_non_string_name_rejected(db, schema):
    with pytest.raises(TypeError):
        db.create_collection(123, schema)


# ---------------------------------------------------------------------------
# Persistence across reopens
# ---------------------------------------------------------------------------

def test_collection_persists_across_reopen(tmp_path, schema):
    data_dir = str(tmp_path / "data")
    db1 = MilvusLite(data_dir)
    col1 = db1.create_collection("docs", schema)
    col1.insert([_make_record(0), _make_record(1)])
    col1.close()
    db1.close()

    db2 = MilvusLite(data_dir)
    assert db2.has_collection("docs")
    col2 = db2.get_collection("docs")
    rec = col2.get(["doc_0000"])
    assert len(rec) == 1
    db2.close()


def test_get_loads_schema_from_disk(tmp_path, schema):
    data_dir = str(tmp_path / "data")
    db1 = MilvusLite(data_dir)
    db1.create_collection("docs", schema)
    db1.close()

    db2 = MilvusLite(data_dir)
    col = db2.get_collection("docs")
    # The schema loaded from disk should match the original.
    assert col.schema.fields[0].name == "id"
    assert col.schema.fields[1].dtype == DataType.FLOAT_VECTOR
    assert col.schema.fields[1].dim == 4
    db2.close()


# ---------------------------------------------------------------------------
# Multi-collection isolation
# ---------------------------------------------------------------------------

def test_two_collections_independent(db, schema):
    col_a = db.create_collection("a", schema)
    col_b = db.create_collection("b", schema)

    col_a.insert([_make_record(0, prefix="a")])
    col_b.insert([_make_record(0, prefix="b")])

    # Each collection only sees its own data.
    assert col_a.get(["a_0000"])[0]["id"] == "a_0000"
    assert col_a.get(["b_0000"]) == []
    assert col_b.get(["b_0000"])[0]["id"] == "b_0000"
    assert col_b.get(["a_0000"]) == []


def test_drop_one_keeps_other(db, schema):
    db.create_collection("a", schema)
    col_b = db.create_collection("b", schema)
    col_b.insert([_make_record(0)])

    db.drop_collection("a")
    assert db.list_collections() == ["b"]
    # b is still functional
    assert len(col_b.get(["doc_0000"])) == 1


# ---------------------------------------------------------------------------
# close() + closed state
# ---------------------------------------------------------------------------

def test_close_is_idempotent(db):
    db.close()
    db.close()  # no error


def test_operations_after_close_raise(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "data"))
    db.close()
    with pytest.raises(RuntimeError, match="closed"):
        db.create_collection("docs", schema)
    with pytest.raises(RuntimeError):
        db.get_collection("docs")
    with pytest.raises(RuntimeError):
        db.drop_collection("docs")


def test_close_releases_lock(tmp_path):
    """After close(), another MilvusLite instance should be able to open
    the same data_dir."""
    data_dir = str(tmp_path / "data")
    db1 = MilvusLite(data_dir)
    db1.close()
    db2 = MilvusLite(data_dir)  # should not raise
    db2.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager(tmp_path, schema):
    data_dir = str(tmp_path / "data")
    with MilvusLite(data_dir) as db:
        db.create_collection("docs", schema)
        assert db.has_collection("docs")
    # After exit, the lock should be released.
    with MilvusLite(data_dir) as db2:
        assert db2.has_collection("docs")


# ---------------------------------------------------------------------------
# LOCK file — multi-process exclusion
# ---------------------------------------------------------------------------

def test_double_open_in_same_process_raises(tmp_path):
    data_dir = str(tmp_path / "data")
    db1 = MilvusLite(data_dir)
    try:
        with pytest.raises(DataDirLockedError):
            MilvusLite(data_dir)
    finally:
        db1.close()


def _try_open_in_subprocess(data_dir, result_queue):
    """Helper for the subprocess multi-process test."""
    try:
        db = MilvusLite(data_dir)
        db.close()
        result_queue.put("opened")
    except DataDirLockedError:
        result_queue.put("locked")
    except Exception as e:
        result_queue.put(f"error: {type(e).__name__}: {e}")


def test_lock_blocks_subprocess(tmp_path):
    """A separate process trying to open the same data_dir while we
    hold the lock should fail with DataDirLockedError."""
    data_dir = str(tmp_path / "data")
    db = MilvusLite(data_dir)
    try:
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_try_open_in_subprocess, args=(data_dir, q))
        p.start()
        p.join(timeout=10)
        assert not p.is_alive(), "subprocess hung"
        result = q.get_nowait()
        assert result == "locked", f"expected 'locked', got {result!r}"
    finally:
        db.close()


def test_subprocess_can_open_after_close(tmp_path):
    """After we close, a subprocess should be able to take the lock."""
    data_dir = str(tmp_path / "data")
    db = MilvusLite(data_dir)
    db.close()

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_try_open_in_subprocess, args=(data_dir, q))
    p.start()
    p.join(timeout=10)
    assert not p.is_alive()
    result = q.get_nowait()
    assert result == "opened", f"expected 'opened', got {result!r}"
