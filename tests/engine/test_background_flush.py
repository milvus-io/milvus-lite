"""Verify insert doesn't block on compaction / index build.

Compaction + index build run on a background worker after flush. Insert
returns as soon as data is persisted (WAL + parquet + manifest).
"""

import threading
import time
import numpy as np
import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
    ])


def _records(start, n):
    return [{"id": start + i,
             "vec": np.random.default_rng(i).random(8).tolist()}
            for i in range(n)]


def test_insert_returns_before_bg_index_build(tmp_path, schema, monkeypatch):
    """Insert returns quickly even when index build is pending.

    Simulates the real-world case where HNSW_SQ index build on
    large segments takes many seconds. That work must run off the
    user thread.
    """
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)
    # Attach an index spec so the bg worker has index work to do.
    col.create_index("vec", {
        "index_type": "BRUTE_FORCE", "metric_type": "COSINE", "params": {},
    })

    # Make the per-segment index build artificially slow.
    from milvus_lite.storage.segment import Segment
    real_build = Segment.build_or_load_index

    def slow_build(self, spec, index_dir):
        time.sleep(0.3)  # 300 ms per segment
        return real_build(self, spec, index_dir)

    monkeypatch.setattr(Segment, "build_or_load_index", slow_build)

    # Trigger several flushes in quick succession. Since the index
    # build runs outside the maintenance lock, inserts must not
    # serialize behind each previous flush's bg build.
    t0 = time.time()
    for batch in range(4):
        col.insert(_records(batch * 5, 5))
    insert_elapsed = time.time() - t0

    # With sync indexing, this would be >= 4 * 0.3 = 1.2s.
    # With async, should be well under 1s (only the sync flush path).
    assert insert_elapsed < 1.0, (
        f"insert took {insert_elapsed:.2f}s — index build appears to block it"
    )

    col._wait_for_bg()
    col.close()


def test_search_concurrent_with_bg_compaction(tmp_path, schema, monkeypatch):
    """Searches from a different thread work while bg compaction runs."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)

    # Fill with enough data to have segments.
    for batch in range(4):
        col.insert(_records(batch * 5, 5))

    errors = []

    def search_worker():
        try:
            for _ in range(20):
                res = col.search(
                    [[0.1] * 8], top_k=3, metric_type="COSINE",
                )
                assert len(res) == 1
        except Exception as e:
            errors.append(e)

    def insert_worker():
        try:
            for batch in range(4, 8):
                col.insert(_records(batch * 5, 5))
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=search_worker)
    t2 = threading.Thread(target=insert_worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"concurrent ops raised: {errors}"
    col.close()


def test_close_drains_bg_tasks(tmp_path, schema, monkeypatch):
    """close() waits for pending bg tasks before returning."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)

    # Trigger several flushes → multiple bg tasks queued.
    for batch in range(4):
        col.insert(_records(batch * 5, 5))

    col.close()
    # After close, reopen should see committed state — no half-flushed
    # manifest, no missing files.
    col2 = Collection("c", str(tmp_path / "d"), schema)
    col2.load()
    assert col2.num_entities == 20
    col2.close()


def test_bg_index_build_survives_concurrent_flush(tmp_path, schema, monkeypatch):
    """Regression: `_ensure_loaded_segments_indexed` used to iterate
    `_segment_cache.values()` directly, which raises RuntimeError if
    the main thread adds a new segment mid-iteration. Must snapshot."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 3)

    col = Collection("c", str(tmp_path / "d"), schema)
    col.create_index("vec", {
        "index_type": "BRUTE_FORCE", "metric_type": "COSINE", "params": {},
    })

    # Slow per-segment index build forces the bg thread to spend time
    # in the iteration, during which the main thread flushes more
    # segments. Without a snapshot, `dict changed size during iteration`
    # raises and the bg task aborts, leaving segments unindexed.
    from milvus_lite.storage.segment import Segment
    real_build = Segment.build_or_load_index

    def slow_build(self, spec, index_dir):
        time.sleep(0.05)
        return real_build(self, spec, index_dir)

    monkeypatch.setattr(Segment, "build_or_load_index", slow_build)

    # Log capture: watch for the "background compaction/index build failed"
    # error that the bug produces.
    import logging
    bg_errors = []

    class BgErrorHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                bg_errors.append(record.getMessage())

    handler = BgErrorHandler()
    logging.getLogger("milvus_lite.engine.collection").addHandler(handler)
    try:
        # Fire many flushes in a tight loop — each enqueues a bg task,
        # and the bg task iterates the cache which is being mutated.
        for i in range(30):
            col.insert(_records(i * 3, 3))
        col._wait_for_bg()
    finally:
        logging.getLogger("milvus_lite.engine.collection").removeHandler(handler)

    assert not bg_errors, f"bg worker raised errors: {bg_errors}"

    # Every segment should have its index attached after draining.
    col.load()
    for seg in col._segment_cache.values():
        if seg.num_rows > 0:
            assert seg.index is not None, (
                f"segment {seg.file_path} missing index after drain"
            )
    col.close()


def test_wait_for_bg_drains_pending(tmp_path, schema, monkeypatch):
    """_wait_for_bg blocks until all queued tasks finish."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)

    for batch in range(6):
        col.insert(_records(batch * 5, 5))

    # After wait, manifest should reflect any compaction outcome.
    col._wait_for_bg()

    # Sanity: we can read everything back.
    col.load()
    for i in range(30):
        res = col.get([i])
        assert len(res) == 1

    col.close()
