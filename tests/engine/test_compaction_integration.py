"""Collection-level compaction integration tests + crash safety + long run.

These tests run compaction through the full Collection write path
(insert → flush → compaction → segment cache refresh) and verify both
correctness and resource boundedness.
"""

import os

import pytest

from milvus_lite.constants import (
    COMPACTION_MIN_FILES_PER_BUCKET,
    DEFAULT_PARTITION,
    MAX_DATA_FILES,
)
from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


def _rec(i, prefix="doc"):
    return {
        "id": f"{prefix}_{i:05d}",
        "vec": [0.5, 0.25],
        "title": f"t{i}",
    }


# ---------------------------------------------------------------------------
# Compaction triggers from a sequence of flushes
# ---------------------------------------------------------------------------

def test_repeated_flush_triggers_compaction(tmp_path, schema, monkeypatch):
    """N small flushes accumulate N data files; once N hits MIN_FILES_PER_BUCKET,
    compaction merges them into one."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 2)

    col = Collection("c", str(tmp_path / "d"), schema)
    # Each insert(2) triggers a flush. After COMPACTION_MIN_FILES_PER_BUCKET
    # flushes (4), compaction kicks in.
    for i in range(0, 2 * COMPACTION_MIN_FILES_PER_BUCKET, 2):
        col.insert([_rec(i), _rec(i + 1)])

    # Compaction runs on the background worker — wait for it to drain.
    col._wait_for_bg()

    files = col._manifest.get_data_files(DEFAULT_PARTITION)
    # After compaction, the 4 small files should have been merged.
    assert len(files) == 1, f"expected 1 merged file, got {len(files)}: {files}"

    # All records still readable.
    for i in range(2 * COMPACTION_MIN_FILES_PER_BUCKET):
        rec = col.get([f"doc_{i:05d}"])
        assert len(rec) == 1, f"missing doc_{i:05d}"
    col.close()


def test_compaction_preserves_search_results(tmp_path, schema, monkeypatch):
    """Search must return the same results before and after compaction."""
    import numpy as np
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 3)

    col = Collection("c", str(tmp_path / "d"), schema)
    rng = np.random.default_rng(99)
    n = 30
    vectors = rng.standard_normal((n, 2)).astype(np.float32)
    records = [
        {"id": f"doc_{i:05d}", "vec": vectors[i].tolist(), "title": f"t{i}"}
        for i in range(n)
    ]
    col.insert(records)

    # By now, multiple flushes + compactions have happened. Verify a
    # query returns the brute-force expected top-5.
    q = vectors[0].tolist()
    results = col.search([q], top_k=5, metric_type="L2")
    [hits] = results
    actual = [h["id"] for h in hits]

    dists = np.linalg.norm(vectors - vectors[0], axis=1)
    expected = [f"doc_{i:05d}" for i in np.argsort(dists)[:5]]
    assert actual == expected
    col.close()


def test_compaction_after_delete_drops_deleted_rows(tmp_path, schema, monkeypatch):
    """Insert + flush, delete + flush, insert + flush + ... eventually
    compaction merges everything and the merged file should NOT contain
    the deleted pks."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 2)

    col = Collection("c", str(tmp_path / "d"), schema)
    # Round 1: insert 4 records, force flushes — they end up in segments.
    col.insert([_rec(0), _rec(1)])
    col.insert([_rec(2), _rec(3)])
    # Force-trigger compaction by adding 2 more flushes.
    col.delete(["doc_00000"])
    col.insert([_rec(4)])  # triggers flush at size 2
    col.delete(["doc_00001"])
    col.insert([_rec(5)])  # triggers flush, now we have 4+ files → compact

    # After compaction, doc_00000 and doc_00001 should be gone.
    # doc_00002..00005 should still be readable.
    assert col.get(["doc_00000"]) == []
    assert col.get(["doc_00001"]) == []
    for i in range(2, 6):
        assert len(col.get([f"doc_{i:05d}"])) == 1
    col.close()


# ---------------------------------------------------------------------------
# Compaction crash injection
# ---------------------------------------------------------------------------

def test_crash_during_compaction_manifest_save(tmp_path, schema, monkeypatch):
    """Crash inside the compaction's Manifest.save(). The new merged file
    becomes an orphan; recovery cleans it up. Old files are still in the
    manifest and on disk, so all records remain queryable."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 2)

    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    # Build up COMPACTION_MIN_FILES_PER_BUCKET - 1 files cleanly so the
    # next flush triggers compaction.
    for i in range(0, 2 * (COMPACTION_MIN_FILES_PER_BUCKET - 1), 2):
        col.insert([_rec(i), _rec(i + 1)])

    # Now patch Manifest.save to raise on the NEXT call.
    from milvus_lite.storage import manifest as manifest_mod
    real_save = manifest_mod.Manifest.save
    saves_remaining = [1]  # let one save (the flush's save) succeed,
                            # then crash on compaction's save

    def crashing_save(self):
        if saves_remaining[0] > 0:
            saves_remaining[0] -= 1
            return real_save(self)
        raise SystemExit("crash during compaction manifest save")

    monkeypatch.setattr(manifest_mod.Manifest, "save", crashing_save)

    # This insert triggers a flush (1 save) + compaction (raises on 2nd save).
    # Compaction runs in bg, so the SystemExit surfaces when we drain.
    col.insert([_rec(100), _rec(101)])
    # Wait for bg — the crashing save logs via exception handler in
    # _bg_compact_and_index (doesn't re-raise to the main thread, but
    # manifest.save failure does leave the new merged parquet as orphan).
    col._wait_for_bg()

    # Restore save for shutdown path.
    monkeypatch.setattr(manifest_mod.Manifest, "save", real_save)
    del col

    # Restart — recovery should clean up any orphan compaction file
    # and leave the system in a consistent state.
    col2 = Collection("c", data_dir, schema)
    # All originally-inserted records must still be readable.
    n_committed = 2 * (COMPACTION_MIN_FILES_PER_BUCKET - 1)
    for i in range(n_committed):
        rec = col2.get([f"doc_{i:05d}"])
        assert len(rec) == 1, f"committed doc_{i:05d} missing after recovery"
    col2.close()


# ---------------------------------------------------------------------------
# Long-run resource boundedness
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_long_run_file_count_bounded(tmp_path, schema, monkeypatch):
    """Insert a lot of records with frequent flushes. Compaction must
    keep the data file count bounded."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)
    n_records = 500
    max_files_seen = 0
    for i in range(n_records):
        col.insert([_rec(i)])
        files_now = len(col._manifest.get_data_files(DEFAULT_PARTITION))
        if files_now > max_files_seen:
            max_files_seen = files_now

    # The data file count should never grow unboundedly. With size-tiered
    # compaction at MIN_FILES_PER_BUCKET=4, the steady state is bounded by
    # roughly the number of size buckets times the bucket capacity.
    assert max_files_seen <= MAX_DATA_FILES, (
        f"data file count grew to {max_files_seen}, exceeds MAX_DATA_FILES={MAX_DATA_FILES}"
    )

    # Final state: every record should still be queryable.
    for i in range(0, n_records, 50):
        assert len(col.get([f"doc_{i:05d}"])) == 1
    col.close()


@pytest.mark.slow
def test_long_run_with_deletes_bounds_delta_index(tmp_path, schema, monkeypatch):
    """Insert many records, delete a fraction. delta_index size should
    never EXCEED the number of distinct deleted pks (it's an upper bound).

    Note: tombstone GC in Phase 6 is conservative — it uses the global
    min_active_data_seq, which means a tombstone is only reclaimed when
    NO live data row has seq ≤ delete_seq. With an interleaved pattern
    like "insert + delete immediately", the surviving 2/3 of the records
    keep the global min_active low forever, so almost no GC happens.
    Aggressive per-pk GC is a Phase 6+ optimization.
    """
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)
    n_records = 200
    max_delta_index = 0
    deleted_pks: set = set()
    for i in range(n_records):
        col.insert([_rec(i)])
        if i % 3 == 0:
            pk = f"doc_{i:05d}"
            col.delete([pk])
            deleted_pks.add(pk)
        cur = len(col._delta_index)
        if cur > max_delta_index:
            max_delta_index = cur

    # Bounded by the number of distinct deleted pks.
    assert max_delta_index <= len(deleted_pks), (
        f"delta_index grew to {max_delta_index} > {len(deleted_pks)} distinct deletes"
    )
    # The deletes themselves must remain effective.
    for pk in deleted_pks:
        assert col.get([pk]) == [], f"deleted {pk} resurrected"
    col.close()


@pytest.mark.slow
def test_gc_progress_when_old_data_compacted_out(tmp_path, schema, monkeypatch):
    """A workload that DOES let GC make progress: delete the same pks
    repeatedly so the surviving content seq advances past old delete seqs.

    Concretely: insert N records, delete a fraction, then re-insert
    them all so their seqs all advance, then delete some again. After
    enough churn the early tombstones become unreachable.
    """
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)
    # Round 1: insert 20 records, delete first 10 (their seqs are ~1..30).
    col.insert([_rec(i) for i in range(20)])
    col.delete([f"doc_{i:05d}" for i in range(10)])

    initial_delta_size = len(col._delta_index)

    # Round 2-N: re-insert ALL 20 records repeatedly, each cycle bumps
    # their seqs higher. Compaction merges files; the merged content
    # seq_min advances. Eventually tombstones with delete_seq below
    # the new min become GC-able.
    for cycle in range(20):
        col.insert([_rec(i) for i in range(20)])

    # The delta_index should have shrunk — at minimum, the very first
    # tombstones (with the smallest delete seqs) should be GC'd.
    final_delta = len(col._delta_index)
    assert final_delta < initial_delta_size, (
        f"GC made no progress: {initial_delta_size} → {final_delta}"
    )
    col.close()


@pytest.mark.slow
def test_long_run_search_consistency(tmp_path, schema, monkeypatch):
    """After many flushes + compactions, search results must still
    match a brute-force computation against the live state."""
    import numpy as np
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 4)

    col = Collection("c", str(tmp_path / "d"), schema)
    rng = np.random.default_rng(2026)
    n = 200
    vectors = rng.standard_normal((n, 2)).astype(np.float32)
    records = [
        {"id": f"doc_{i:05d}", "vec": vectors[i].tolist(), "title": f"t{i}"}
        for i in range(n)
    ]
    col.insert(records)

    # Delete a third of them.
    deleted_idx = list(range(0, n, 3))
    col.delete([f"doc_{i:05d}" for i in deleted_idx])

    # Compute the live set
    live_idx = [i for i in range(n) if i not in set(deleted_idx)]
    live_vectors = vectors[live_idx]

    # Random query.
    q = rng.standard_normal((1, 2)).astype(np.float32)
    results = col.search(q.tolist(), top_k=10, metric_type="L2")
    [hits] = results
    actual_ids = [h["id"] for h in hits]

    dists = np.linalg.norm(live_vectors - q[0], axis=1)
    expected_top = np.argsort(dists)[:10]
    expected_ids = [f"doc_{live_idx[i]:05d}" for i in expected_top]
    assert actual_ids == expected_ids
    col.close()
