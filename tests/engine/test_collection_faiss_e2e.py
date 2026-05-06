"""Phase 9.5 — Collection-level FAISS HNSW end-to-end tests.

These integrate Phase 9.1 (partition CRUD), 9.2 (executor_indexed),
9.3 (state machine), 9.4 (persistence), and 9.5 (FAISS) — the full
Phase 9 stack from create_index → load → search through the actual
HNSW index.

Skipped automatically when faiss-cpu is not installed.
"""

import os

import numpy as np
import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.index.factory import is_faiss_available
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema

pytestmark = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss-cpu is not installed"
)


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64, nullable=True),
        FieldSchema(name="active", dtype=DataType.BOOL),
    ])


def _vec(seed: int):
    return np.random.RandomState(seed).randn(8).astype(np.float32).tolist()


HNSW_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
    "search_params": {"ef": 128},
}


# ---------------------------------------------------------------------------
# create_index → load → search end-to-end
# ---------------------------------------------------------------------------

def test_create_index_load_search_full_flow(tmp_path, schema):
    c = Collection("e2e", str(tmp_path / "data"), schema)
    try:
        c.insert([
            {"id": i, "vec": _vec(i), "title": f"t{i}", "active": (i % 2 == 0)}
            for i in range(50)
        ])
        c.flush()

        c.create_index("vec", HNSW_PARAMS)
        assert c.has_index() is True
        # create_index preserves load state — already loaded since
        # Collection auto-loads when no prior index spec exists.
        assert c.load_state == "loaded"

        # Every segment should now have a FaissHnswIndex attached
        from milvus_lite.index.faiss_hnsw import FaissHnswIndex
        for seg in c._segment_cache.values():
            assert isinstance(seg.index, FaissHnswIndex)

        res = c.search([_vec(900)], top_k=10)
        assert len(res[0]) == 10
        assert all("id" in r for r in res[0])
    finally:
        c.close()


def test_hnsw_idx_files_persisted_on_disk(tmp_path, schema):
    c = Collection("persist", str(tmp_path / "data"), schema)
    try:
        c.insert([
            {"id": i, "vec": _vec(i), "title": "x", "active": True}
            for i in range(20)
        ])
        c.flush()
        c.create_index("vec", HNSW_PARAMS)
        c.load()

        idx_dir = os.path.join(str(tmp_path / "data"), "partitions", "_default", "indexes")
        files = sorted(os.listdir(idx_dir))
        assert all(f.endswith(".vec.hnsw.idx") for f in files)
        assert len(files) >= 1
    finally:
        c.close()


def test_restart_loads_persisted_hnsw(tmp_path, schema):
    """Round-trip through process restart: build HNSW, close, reopen,
    load, search must return the same top-k as before close."""
    c = Collection("restart", str(tmp_path / "data"), schema)
    c.insert([
        {"id": i, "vec": _vec(i), "title": "x", "active": True}
        for i in range(30)
    ])
    c.flush()
    c.create_index("vec", HNSW_PARAMS)
    c.load()
    expected = c.search([_vec(0)], top_k=5)
    c.close()

    c2 = Collection("restart", str(tmp_path / "data"), schema)
    try:
        assert c2.has_index() is True
        assert c2.load_state == "released"
        c2.load()
        actual = c2.search([_vec(0)], top_k=5)
        # Top-k IDs must agree (recall@5 ~= 1.0 for n=30 + low ef)
        assert [r["id"] for r in actual[0]] == [r["id"] for r in expected[0]]
    finally:
        c2.close()


# ---------------------------------------------------------------------------
# HNSW + scalar filter combined (the Phase 8 ↔ Phase 9 bridge)
# ---------------------------------------------------------------------------

def test_hnsw_search_with_scalar_filter(tmp_path, schema):
    """The whole reason segment-level FAISS was the right call: the
    Phase-8 bitmap pipeline becomes a pre-filter for HNSW via the
    valid_mask path. This test confirms scalar filters still work
    after the FAISS swap."""
    c = Collection("filter", str(tmp_path / "data"), schema)
    try:
        c.insert([
            {"id": i, "vec": _vec(i), "title": "x", "active": (i % 3 == 0)}
            for i in range(40)
        ])
        c.flush()
        c.create_index("vec", HNSW_PARAMS)
        c.load()

        # Only 0, 3, 6, 9, 12, ... should be returned
        res = c.search([_vec(0)], top_k=20, expr="active == true")
        ids = [r["id"] for r in res[0]]
        assert all(i % 3 == 0 for i in ids)
        # Top-1 should still be id=0 (self) — recall guarantee for HNSW
        assert ids[0] == 0
    finally:
        c.close()


def test_hnsw_search_partition_filter(tmp_path, schema):
    c = Collection("part", str(tmp_path / "data"), schema)
    try:
        c.create_partition("p1")
        c.create_partition("p2")
        c.insert(
            [{"id": i, "vec": _vec(i), "title": "x", "active": True} for i in range(20)],
            partition_name="p1",
        )
        c.insert(
            [{"id": 100 + i, "vec": _vec(100 + i), "title": "x", "active": True} for i in range(20)],
            partition_name="p2",
        )
        c.flush()

        c.create_index("vec", HNSW_PARAMS)
        c.load()

        res = c.search([_vec(0)], top_k=10, partition_names=["p1"])
        ids = [r["id"] for r in res[0]]
        assert all(i < 100 for i in ids)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Compaction + HNSW interplay
# ---------------------------------------------------------------------------

def test_hnsw_survives_compaction(tmp_path, schema, monkeypatch):
    """After compaction merges segments, the new merged segment should
    have a freshly built HNSW; the old segments' .idx files should be
    gone; search should return correct top-k."""
    monkeypatch.setattr("milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 2)

    c = Collection("compact", str(tmp_path / "data"), schema)
    try:
        c.create_index("vec", HNSW_PARAMS)
        c.load()

        for batch in range(3):
            c.insert([
                {"id": batch * 100 + i, "vec": _vec(batch * 100 + i),
                 "title": "x", "active": True}
                for i in range(10)
            ])
            c.flush()
        c._wait_for_bg()  # compaction + index rebuild on bg worker

        # Verify .idx files match data files 1:1
        # New format: <stem>.<field>.<type>.idx → strip .vec.hnsw.idx
        idx_dir = os.path.join(str(tmp_path / "data"), "partitions", "_default", "indexes")
        data_dir = os.path.join(str(tmp_path / "data"), "partitions", "_default", "data")
        idx_stems = {f[: -len(".vec.hnsw.idx")] for f in os.listdir(idx_dir)}
        data_stems = {os.path.splitext(f)[0] for f in os.listdir(data_dir)}
        assert idx_stems == data_stems

        # Search still works
        res = c.search([_vec(50)], top_k=5)
        assert len(res[0]) == 5
    finally:
        c.close()


# ---------------------------------------------------------------------------
# drop_index removes HNSW .idx files
# ---------------------------------------------------------------------------

def test_drop_index_removes_hnsw_files(tmp_path, schema):
    c = Collection("drop", str(tmp_path / "data"), schema)
    try:
        c.insert([
            {"id": i, "vec": _vec(i), "title": "x", "active": True}
            for i in range(15)
        ])
        c.flush()
        c.create_index("vec", HNSW_PARAMS)

        idx_dir = os.path.join(str(tmp_path / "data"), "partitions", "_default", "indexes")
        assert any(f.endswith(".vec.hnsw.idx") for f in os.listdir(idx_dir))

        c.release()  # drop_index requires released state
        c.drop_index("vec")
        assert os.listdir(idx_dir) == []
    finally:
        c.close()
