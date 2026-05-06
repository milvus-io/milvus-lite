"""Phase 9.6 — long-running stress test for the vector index lifecycle.

The architectural invariant we're stress-testing here:
    "the .idx file set is always 1:1 with the data parquet set, no
     orphans, no missing entries, even after thousands of insert /
     flush / compaction cycles"

Skipped by default (`@pytest.mark.slow`); run with::

    pytest -m slow tests/engine/test_index_longrun.py

These tests don't require faiss-cpu — they exercise the lifecycle on
BruteForceIndex, which is faster to build and equally effective at
catching .idx ↔ data file invariant violations. A separate test does
the same with HNSW when faiss is available.
"""

import os

import numpy as np
import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.index.factory import is_faiss_available
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=32, nullable=True),
    ])


def _vec(seed: int) -> list:
    return np.random.RandomState(seed).randn(8).astype(np.float32).tolist()


def _index_files(data_dir: str, partition: str, suffix: str) -> set:
    p = os.path.join(data_dir, "partitions", partition, "indexes")
    if not os.path.exists(p):
        return set()
    return {f for f in os.listdir(p) if f.endswith(suffix)}


def _data_stems(data_dir: str, partition: str) -> set:
    p = os.path.join(data_dir, "partitions", partition, "data")
    if not os.path.exists(p):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(p)}


def _assert_one_to_one(data_dir: str, partition: str, suffix: str,
                       col: "Collection" = None) -> None:
    """Architectural invariant §11: every data file has an .idx, every
    .idx has a data file, and the stems match exactly.

    Waits for pending background index builds before asserting, since
    flush is now async and .idx files appear after some delay.
    """
    if col is not None:
        col._wait_for_bg()
    idx_files = _index_files(data_dir, partition, suffix)
    idx_stems = {f[: -len(suffix)] for f in idx_files}
    data_stems = _data_stems(data_dir, partition)
    assert idx_stems == data_stems, (
        f"index/data 1:1 violated:\n"
        f"  data only: {sorted(data_stems - idx_stems)}\n"
        f"  idx only:  {sorted(idx_stems - data_stems)}"
    )


# ---------------------------------------------------------------------------
# BruteForce stress
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_index_lifecycle_long_run_brute_force(tmp_path, schema, monkeypatch):
    """500 inserts, frequent flushes, frequent compaction. After every
    flush the .idx ↔ data invariant must hold.

    Targets the same regime as test_long_run_file_count_bounded but
    adds the index hook to the picture so we catch any drift in
    flush/compaction's _ensure_loaded_segments_indexed +
    _cleanup_orphan_index_files coordination.
    """
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)
    monkeypatch.setattr(
        "milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 3
    )

    data_dir = str(tmp_path / "data")
    c = Collection("longrun", data_dir, schema)
    try:
        c.create_index("vec", {
            "index_type": "BRUTE_FORCE",
            "metric_type": "COSINE",
            "params": {},
        })
        c.load()

        n_records = 500
        for i in range(n_records):
            c.insert([{"id": i, "vec": _vec(i), "title": f"t{i}"}])
            # Insert triggers a flush at MEMTABLE_SIZE_LIMIT, which
            # in turn may trigger compaction. After each insert
            # the invariant must hold.
            _assert_one_to_one(data_dir, "_default", ".vec.brute_force.idx", c)

        # Final state: every record still queryable.
        for i in range(0, n_records, 50):
            res = c.search([_vec(i)], top_k=1)
            assert res[0][0]["id"] == i, f"id={i} not at top-1 after long-run"
    finally:
        c.close()


@pytest.mark.slow
def test_index_lifecycle_with_deletes_brute_force(tmp_path, schema, monkeypatch):
    """Insert + delete + flush + compaction interleaved. The .idx ↔ data
    invariant must hold throughout, AND deletes must remain effective."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)
    monkeypatch.setattr(
        "milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 3
    )

    data_dir = str(tmp_path / "data")
    c = Collection("longrun_del", data_dir, schema)
    try:
        c.create_index("vec", {
            "index_type": "BRUTE_FORCE",
            "metric_type": "L2",
            "params": {},
        })
        c.load()

        deleted = set()
        n = 200
        for i in range(n):
            c.insert([{"id": i, "vec": _vec(i), "title": "x"}])
            if i % 4 == 0:
                c.delete([i])
                deleted.add(i)
            _assert_one_to_one(data_dir, "_default", ".vec.brute_force.idx", c)

        # Force a final flush so any in-memory state goes to disk.
        c.flush()
        _assert_one_to_one(data_dir, "_default", ".vec.brute_force.idx", c)

        # Tombstones must remain effective.
        for i in sorted(deleted):
            assert c.get([i]) == [], f"deleted id={i} resurrected"

        # Survivors must still be searchable.
        for i in range(0, n, 17):
            if i in deleted:
                continue
            res = c.search([_vec(i)], top_k=1)
            assert len(res[0]) >= 1
    finally:
        c.close()


@pytest.mark.slow
def test_restart_after_long_run_brute_force(tmp_path, schema, monkeypatch):
    """Long run → close → reopen → load (should read all .idx from disk)
    → search results match what we had before close."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)
    monkeypatch.setattr(
        "milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 3
    )

    data_dir = str(tmp_path / "data")
    c = Collection("rstrt", data_dir, schema)
    c.create_index("vec", {
        "index_type": "BRUTE_FORCE",
        "metric_type": "COSINE",
        "params": {},
    })
    c.load()

    n = 300
    for i in range(n):
        c.insert([{"id": i, "vec": _vec(i), "title": "x"}])
    c.flush()
    _assert_one_to_one(data_dir, "_default", ".vec.brute_force.idx", c)

    expected = c.search([_vec(150)], top_k=5)
    expected_ids = [r["id"] for r in expected[0]]
    c.close()

    c2 = Collection("rstrt", data_dir, schema)
    try:
        assert c2.load_state == "released"
        c2.load()
        # Invariant still holds after recovery
        _assert_one_to_one(data_dir, "_default", ".vec.brute_force.idx", c2)
        actual = c2.search([_vec(150)], top_k=5)
        assert [r["id"] for r in actual[0]] == expected_ids
    finally:
        c2.close()


# ---------------------------------------------------------------------------
# FAISS HNSW stress (skipped without faiss)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not is_faiss_available(), reason="faiss-cpu is not installed")
def test_index_lifecycle_long_run_hnsw(tmp_path, schema, monkeypatch):
    """Same workload as the brute-force long run, but with FAISS HNSW.
    Catches any FAISS-specific failure mode in the flush/compaction
    hooks (e.g. .idx file format mismatch, stem-parsing edge cases)."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)
    monkeypatch.setattr(
        "milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 3
    )

    data_dir = str(tmp_path / "data")
    c = Collection("hnsw_long", data_dir, schema)
    try:
        c.create_index("vec", {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 32},
        })
        c.load()

        n_records = 200
        for i in range(n_records):
            c.insert([{"id": i, "vec": _vec(i), "title": "x"}])
            _assert_one_to_one(data_dir, "_default", ".vec.hnsw.idx", c)

        c.flush()
        _assert_one_to_one(data_dir, "_default", ".vec.hnsw.idx", c)

        # Sample a few queries — HNSW recall@1 ≥ 0.95 typically; we
        # check that the self-query returns SOME hit (not empty).
        for i in range(0, n_records, 30):
            res = c.search([_vec(i)], top_k=5)
            assert len(res[0]) >= 1
    finally:
        c.close()
