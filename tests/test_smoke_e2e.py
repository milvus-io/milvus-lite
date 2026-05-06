"""End-to-end smoke test — exercise the public API surface as a user would.

This is the closest thing to a documented "quickstart" — if this test
ever stops passing, the README quickstart is broken too.
"""

import numpy as np
import pytest

from milvus_lite import (
    Collection,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    CollectionSchema,
    DataDirLockedError,
    DataType,
    FieldSchema,
    MilvusLite,
    MilvusLiteError,
    SchemaValidationError,
)


def _make_schema(dim=8):
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


def test_quickstart(tmp_path):
    """The minimal flow a new user would write."""
    data_dir = str(tmp_path / "data")
    schema = _make_schema(dim=4)

    with MilvusLite(data_dir) as db:
        col = db.create_collection("docs", schema)
        col.insert([
            {"id": "doc_1", "vec": [1.0, 0.0, 0.0, 0.0], "title": "first"},
            {"id": "doc_2", "vec": [0.0, 1.0, 0.0, 0.0], "title": "second"},
            {"id": "doc_3", "vec": [0.0, 0.0, 1.0, 0.0], "title": "third"},
        ])

        # Point read
        rec = col.get(["doc_1"])
        assert len(rec) == 1
        assert rec[0]["title"] == "first"

        # Vector search
        results = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=2, metric_type="COSINE")
        assert len(results) == 1
        hits = results[0]
        assert len(hits) == 2
        assert hits[0]["id"] == "doc_1"

        # Delete
        col.delete(["doc_1"])
        assert col.get(["doc_1"]) == []


def test_full_lifecycle_with_restart(tmp_path):
    """Create → insert → flush → close → reopen → query → delete →
    reopen → verify delete persisted."""
    data_dir = str(tmp_path / "data")
    schema = _make_schema()
    rng = np.random.default_rng(0)
    n = 50
    vectors = rng.standard_normal((n, 8)).astype(np.float32)
    records = [
        {"id": f"doc_{i:03d}", "vec": vectors[i].tolist(), "title": f"t{i}"}
        for i in range(n)
    ]

    # ── 1. Create + insert + flush + close ─────────────────────
    with MilvusLite(data_dir) as db:
        col = db.create_collection("docs", schema)
        col.insert(records)
        col.flush()

    # ── 2. Reopen + query everything ───────────────────────────
    with MilvusLite(data_dir) as db:
        assert "docs" in db.list_collections()
        col = db.get_collection("docs")
        for i in range(n):
            rec = col.get([f"doc_{i:03d}"])
            assert len(rec) == 1, f"missing doc_{i:03d} after restart"

        # Search consistency
        q = vectors[0].tolist()
        hits = col.search([q], top_k=5, metric_type="L2")[0]
        assert hits[0]["id"] == "doc_000"

    # ── 3. Reopen + delete half ────────────────────────────────
    deleted = [f"doc_{i:03d}" for i in range(0, n, 2)]
    with MilvusLite(data_dir) as db:
        col = db.get_collection("docs")
        col.delete(deleted)

    # ── 4. Reopen + verify deletes persisted ───────────────────
    with MilvusLite(data_dir) as db:
        col = db.get_collection("docs")
        for pk in deleted:
            assert col.get([pk]) == [], f"{pk} resurrected"
        # The other half is still there
        for i in range(1, n, 2):
            assert len(col.get([f"doc_{i:03d}"])) == 1


def test_multi_collection_isolation(tmp_path):
    """Two collections under the same DB don't see each other's data,
    even after restart + flush."""
    data_dir = str(tmp_path / "data")

    with MilvusLite(data_dir) as db:
        schema_a = _make_schema(dim=4)
        schema_b = _make_schema(dim=8)
        col_a = db.create_collection("a", schema_a)
        col_b = db.create_collection("b", schema_b)

        col_a.insert([{"id": "x", "vec": [1.0, 0.0, 0.0, 0.0], "title": "a-only"}])
        col_b.insert([{"id": "x", "vec": [0.0] * 8, "title": "b-only"}])

        # Same pk, different collections — independent
        assert col_a.get(["x"])[0]["title"] == "a-only"
        assert col_b.get(["x"])[0]["title"] == "b-only"

    # After restart
    with MilvusLite(data_dir) as db:
        assert set(db.list_collections()) == {"a", "b"}
        col_a = db.get_collection("a")
        col_b = db.get_collection("b")
        assert col_a.get(["x"])[0]["title"] == "a-only"
        assert col_b.get(["x"])[0]["title"] == "b-only"
        # Verify schema dim differs
        assert col_a.schema.fields[1].dim == 4
        assert col_b.schema.fields[1].dim == 8


def test_drop_then_recreate(tmp_path):
    """Drop a collection, then create one with the same name and a
    different schema — should not see any old data."""
    data_dir = str(tmp_path / "data")
    with MilvusLite(data_dir) as db:
        schema_v1 = _make_schema(dim=4)
        col = db.create_collection("docs", schema_v1)
        col.insert([{"id": "doc_old", "vec": [1.0, 0.0, 0.0, 0.0], "title": "v1"}])
        col.flush()
        db.drop_collection("docs")
        assert not db.has_collection("docs")

        schema_v2 = _make_schema(dim=8)
        col2 = db.create_collection("docs", schema_v2)
        # Old data must be gone
        assert col2.get(["doc_old"]) == []
        # Schema is the new one
        assert col2.schema.fields[1].dim == 8


def test_search_brute_force_match_through_db(tmp_path):
    """Larger end-to-end search check, opened via the public MilvusLite API."""
    rng = np.random.default_rng(7)
    n = 200
    dim = 16
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    records = [
        {"id": f"doc_{i:04d}", "vec": vectors[i].tolist(), "title": f"t{i}"}
        for i in range(n)
    ]

    with MilvusLite(str(tmp_path / "data")) as db:
        col = db.create_collection("docs", _make_schema(dim=dim))
        col.insert(records)
        col.flush()
        col.insert(records[:50])  # mix flushed segment + memtable

        q = rng.standard_normal((1, dim)).astype(np.float32)
        results = col.search(q.tolist(), top_k=10, metric_type="L2")
        actual = [h["id"] for h in results[0]]

        dists = np.linalg.norm(vectors - q[0], axis=1)
        expected = [f"doc_{i:04d}" for i in np.argsort(dists)[:10]]
        assert actual == expected


def test_exception_hierarchy_is_importable():
    """All public exceptions are subclasses of MilvusLiteError."""
    assert issubclass(SchemaValidationError, MilvusLiteError)
    assert issubclass(CollectionNotFoundError, MilvusLiteError)
    assert issubclass(CollectionAlreadyExistsError, MilvusLiteError)
    assert issubclass(DataDirLockedError, MilvusLiteError)
