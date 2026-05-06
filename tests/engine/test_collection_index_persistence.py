"""Phase 9.4 — Index file persistence + flush/compaction/recovery hooks.

The end-to-end story we're validating:

1. create_index → load → .idx files written under indexes/
2. close → reopen → load → fast (loads existing .idx, no rebuild)
3. flush after load → new segment gets a new .idx automatically
4. compaction → old .idx files cleaned, new merged segment gets a new .idx
5. drop_partition → all .idx files in that partition are removed with the dir
6. drop_index → all .idx files matching the index_type are removed
7. recovery cleans orphan .idx whose source data file is gone
"""

import os

import numpy as np
import pytest

from milvus_lite.constants import MEMTABLE_SIZE_LIMIT
from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64, nullable=True),
    ])


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    yield c
    c.close()


def _vec(i):
    return [float(i), float(i + 1), float(i + 2), float(i + 3)]


BRUTE_PARAMS = {"index_type": "BRUTE_FORCE", "metric_type": "L2", "params": {}}


def _index_files_for_partition(data_dir: str, partition: str) -> list:
    p = os.path.join(data_dir, "partitions", partition, "indexes")
    if not os.path.exists(p):
        return []
    return sorted(os.listdir(p))


# ---------------------------------------------------------------------------
# Persistence basics
# ---------------------------------------------------------------------------

def test_load_writes_idx_files(col, tmp_path):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    col.load()

    files = _index_files_for_partition(str(tmp_path / "data"), "_default")
    assert len(files) >= 1
    assert all(f.endswith(".vec.brute_force.idx") for f in files)


def test_load_writes_one_idx_per_segment(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    try:
        # Create three separate segments via flush boundaries.
        for batch in range(3):
            c.insert([
                {"id": batch * 100 + i, "vec": _vec(batch * 100 + i), "title": "x"}
                for i in range(5)
            ])
            c.flush()

        c.create_index("vec", BRUTE_PARAMS)
        c.load()

        n_segments = sum(1 for s in c._segment_cache.values() if s.partition == "_default")
        files = _index_files_for_partition(str(tmp_path / "data"), "_default")
        assert len(files) == n_segments
    finally:
        c.close()


def test_idx_filename_matches_data_stem(col, tmp_path):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    col.load()

    data_dir = str(tmp_path / "data")
    data_files = sorted(os.listdir(os.path.join(data_dir, "partitions", "_default", "data")))
    idx_files = _index_files_for_partition(data_dir, "_default")

    data_stems = {os.path.splitext(f)[0] for f in data_files}
    for idx in idx_files:
        # New format: <stem>.<field>.<type>.idx → ".vec.brute_force.idx"
        assert idx.endswith(".vec.brute_force.idx")
        stem = idx[: -len(".vec.brute_force.idx")]
        assert stem in data_stems


# ---------------------------------------------------------------------------
# Restart fast-path (load reads existing .idx instead of rebuilding)
# ---------------------------------------------------------------------------

def test_restart_load_uses_persisted_idx(tmp_path, schema):
    """After restart, load() should pick up the existing .idx files
    rather than rebuild from scratch. We can't easily measure speed
    here, but we can verify that the .idx files are not regenerated
    (mtime is preserved)."""
    c = Collection("t", str(tmp_path / "data"), schema)
    c.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    c.flush()
    c.create_index("vec", BRUTE_PARAMS)
    c.load()
    c.close()

    data_dir = str(tmp_path / "data")
    idx_files = _index_files_for_partition(data_dir, "_default")
    assert len(idx_files) >= 1
    idx_path = os.path.join(data_dir, "partitions", "_default", "indexes", idx_files[0])
    mtime_before = os.path.getmtime(idx_path)

    # Tick the clock so mtime resolution can detect a rewrite.
    import time
    time.sleep(0.05)

    c2 = Collection("t", str(tmp_path / "data"), schema)
    try:
        assert c2.load_state == "released"
        c2.load()
        assert c2.load_state == "loaded"
        # The .idx file should not have been rewritten.
        mtime_after = os.path.getmtime(idx_path)
        assert mtime_after == mtime_before
        # Search should still work end-to-end.
        res = c2.search([_vec(2)], top_k=3)
        assert len(res[0]) == 3
    finally:
        c2.close()


def test_restart_load_then_search_results_correct(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    c.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(10)])
    c.flush()
    c.create_index("vec", BRUTE_PARAMS)
    c.load()
    expected = c.search([_vec(0)], top_k=5)
    c.close()

    c2 = Collection("t", str(tmp_path / "data"), schema)
    try:
        c2.load()
        actual = c2.search([_vec(0)], top_k=5)
        assert [r["id"] for r in actual[0]] == [r["id"] for r in expected[0]]
    finally:
        c2.close()


# ---------------------------------------------------------------------------
# Flush hook — new segment gets new .idx automatically
# ---------------------------------------------------------------------------

def test_flush_after_load_indexes_new_segments(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    try:
        c.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
        c.flush()
        c.create_index("vec", BRUTE_PARAMS)
        c.load()
        files_before = set(_index_files_for_partition(str(tmp_path / "data"), "_default"))

        # Insert + flush again → new segment.
        c.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(100, 110)])
        c.flush()
        c._wait_for_bg()  # index build runs on bg worker

        files_after = set(_index_files_for_partition(str(tmp_path / "data"), "_default"))
        assert files_after - files_before, "no new .idx file appeared after flush"

        # All cached segments should have indexes attached.
        for seg in c._segment_cache.values():
            if seg.num_rows > 0:
                assert seg.index is not None
    finally:
        c.close()


def test_flush_in_released_state_does_not_build_index(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    try:
        c.create_index("vec", BRUTE_PARAMS)
        c.release()  # Now explicitly in released state.
        c.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
        c.flush()
        # No .idx files should exist yet — we never called load().
        files = _index_files_for_partition(str(tmp_path / "data"), "_default")
        assert files == []
        # Segments should also lack in-memory indexes.
        for seg in c._segment_cache.values():
            assert seg.index is None
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Compaction hook — old .idx removed, new .idx built
# ---------------------------------------------------------------------------

def test_compaction_replaces_idx_files(tmp_path, schema, monkeypatch):
    """Force enough flushes to trigger compaction; verify old .idx files
    are gone and a new one matches the merged segment."""
    # Make compaction trigger more aggressively for testing.
    from milvus_lite import constants
    monkeypatch.setattr(constants, "COMPACTION_MIN_FILES_PER_BUCKET", 2)
    monkeypatch.setattr("milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 2)

    c = Collection("t", str(tmp_path / "data"), schema)
    try:
        c.create_index("vec", BRUTE_PARAMS)
        c.load()  # state = loaded; subsequent flushes will build .idx

        # Two flushes → at least 2 small segments → compaction merges them.
        for batch in range(3):
            c.insert([
                {"id": batch * 100 + i, "vec": _vec(batch * 100 + i), "title": "x"}
                for i in range(5)
            ])
            c.flush()
        c._wait_for_bg()  # compaction + index rebuild on bg worker

        files = _index_files_for_partition(str(tmp_path / "data"), "_default")

        data_dir = str(tmp_path / "data")
        data_files = sorted(os.listdir(os.path.join(data_dir, "partitions", "_default", "data")))

        # 1:1 — every data file has exactly one .idx and vice versa.
        data_stems = {os.path.splitext(f)[0] for f in data_files}
        idx_stems = {f[: -len(".vec.brute_force.idx")] for f in files}
        assert data_stems == idx_stems
    finally:
        c.close()


# ---------------------------------------------------------------------------
# drop_partition / drop_index cleanup
# ---------------------------------------------------------------------------

def test_drop_partition_removes_idx_files(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    try:
        c.create_partition("p1")
        c.insert(
            [{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)],
            partition_name="p1",
        )
        c.flush()
        c.create_index("vec", BRUTE_PARAMS)
        c.load()

        files = _index_files_for_partition(str(tmp_path / "data"), "p1")
        assert len(files) >= 1

        c.drop_partition("p1")
        # Whole partition dir gone → indexes/ gone.
        assert not os.path.exists(
            os.path.join(str(tmp_path / "data"), "partitions", "p1")
        )
    finally:
        c.close()


def test_drop_index_removes_idx_files(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    try:
        c.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
        c.flush()
        c.create_index("vec", BRUTE_PARAMS)

        files = _index_files_for_partition(str(tmp_path / "data"), "_default")
        assert len(files) >= 1

        c.release()  # drop_index requires released state
        c.drop_index("vec")
        files_after = _index_files_for_partition(str(tmp_path / "data"), "_default")
        assert files_after == []
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Recovery orphan cleanup
# ---------------------------------------------------------------------------

def test_recovery_cleans_orphan_idx(tmp_path, schema):
    """Plant a stray .idx file with no matching data parquet, then
    reopen the Collection — recovery should sweep it away."""
    c = Collection("t", str(tmp_path / "data"), schema)
    c.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    c.flush()
    c.create_index("vec", BRUTE_PARAMS)
    c.load()
    c.close()

    # Plant an orphan .idx in the indexes/ dir.
    indexes_dir = os.path.join(str(tmp_path / "data"), "partitions", "_default", "indexes")
    orphan = os.path.join(indexes_dir, "data_999999_999999.vec.brute_force.idx")
    with open(orphan, "wb") as f:
        np.save(f, np.zeros((1, 4), dtype=np.float32), allow_pickle=False)
    assert os.path.exists(orphan)

    # Reopen — recovery should delete the orphan.
    c2 = Collection("t", str(tmp_path / "data"), schema)
    try:
        assert not os.path.exists(orphan)
        # The legitimate .idx should still be there.
        remaining = _index_files_for_partition(str(tmp_path / "data"), "_default")
        assert len(remaining) >= 1
    finally:
        c2.close()


def test_recovery_keeps_valid_idx(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    c.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    c.flush()
    c.create_index("vec", BRUTE_PARAMS)
    c.load()
    files_before = set(_index_files_for_partition(str(tmp_path / "data"), "_default"))
    c.close()

    c2 = Collection("t", str(tmp_path / "data"), schema)
    try:
        files_after = set(_index_files_for_partition(str(tmp_path / "data"), "_default"))
        assert files_before == files_after
    finally:
        c2.close()
