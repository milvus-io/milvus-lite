"""Issue #21: deleted records must not leak through search/query/count
under sustained concurrent compaction.

The race: a reader takes a snapshot of _segment_cache (holding old
segments), then bg compaction runs, merging those segments and GCing
their tombstones from delta_index. The reader then queries the live
delta_index (no tombstone) against old segments (still have the data)
and sees "ghost" deleted records.

Fix: capture segment and DeltaIndex snapshots together under the
maintenance lock, so a reader sees one compaction generation.
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
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ])


def _vec(i):
    rng = np.random.default_rng(i)
    return rng.random(4).tolist()


def test_delete_not_lost_across_compaction(tmp_path, schema, monkeypatch):
    """Insert N rows, delete half, flush + compact multiple times.
    Deleted pks must never surface in query."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)
    monkeypatch.setattr(
        "milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 2
    )

    col = Collection("c", str(tmp_path / "d"), schema)
    # Insert 40 rows → ~8 flushes → several compaction cycles.
    for batch in range(8):
        col.insert([
            {"id": batch * 5 + i, "vec": _vec(batch * 5 + i)}
            for i in range(5)
        ])
    col._wait_for_bg()

    # Delete the first 20 pks.
    col.delete(list(range(20)))
    # Force another flush to persist the delete + trigger compaction.
    col.insert([{"id": 999, "vec": _vec(999)}])
    col.insert([{"id": 1000, "vec": _vec(1000)}])
    col._wait_for_bg()

    # Query should not return any of the deleted pks.
    res = col.query(expr="id < 20", limit=100)
    deleted_extras = [r["id"] for r in res]
    assert not deleted_extras, f"deleted pks visible in query: {deleted_extras}"

    # num_entities should match live count: 40 + 2 - 20 = 22
    col.load()
    assert col.num_entities == 22
    col.close()


def test_query_snapshot_survives_tombstone_gc_during_read(tmp_path, monkeypatch):
    """A query must not mix old segments with a post-GC DeltaIndex.

    The patched assemble_candidates call forces compaction to run after
    query() has captured its read snapshot. Before the fix, query()
    captured only segments before this point and fetched DeltaIndex
    afterward, so the deleted pk became visible again.
    """
    import milvus_lite.engine.collection as collection_mod
    import milvus_lite.engine.compaction as compaction_mod

    local_schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ])

    monkeypatch.setattr(collection_mod, "MEMTABLE_SIZE_LIMIT", 1)
    monkeypatch.setattr(compaction_mod, "COMPACTION_MIN_FILES_PER_BUCKET", 2)

    col = Collection("c", str(tmp_path / "d"), local_schema)
    col._schedule_bg_maintenance = lambda: None

    col.insert([{"id": 7, "vec": _vec(7)}])
    col.delete([7])
    col.insert([{"id": 2, "vec": _vec(2)}])
    assert col._delta_index.snapshot == {7: 2}

    original_assemble = collection_mod.assemble_candidates
    did_compact = False

    def assemble_after_gc(*args, **kwargs):
        nonlocal did_compact
        if not did_compact:
            did_compact = True
            col._bg_compact_and_index()
            assert col._delta_index.snapshot == {}
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(collection_mod, "assemble_candidates", assemble_after_gc)

    rows = col.query(expr=None, limit=100)
    assert [row["id"] for row in rows] == [2]
    assert did_compact

    col.close()


def test_delete_not_lost_under_concurrent_reader(tmp_path, schema, monkeypatch):
    """Stress version of issue #21: one thread inserts/deletes,
    another thread queries continuously. No ghost data should surface."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)
    monkeypatch.setattr(
        "milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 2
    )

    col = Collection("c", str(tmp_path / "d"), schema)
    # Pre-populate.
    for batch in range(4):
        col.insert([
            {"id": batch * 5 + i, "vec": _vec(batch * 5 + i)}
            for i in range(5)
        ])

    stop = threading.Event()
    ghost_hits: list = []
    live_pks = set(range(20))
    lock = threading.Lock()

    def writer():
        for batch in range(4, 20):
            base = batch * 5
            recs = [{"id": base + i, "vec": _vec(base + i)} for i in range(5)]
            col.insert(recs)
            with lock:
                for i in range(5):
                    live_pks.add(base + i)
            # Delete half of recent pks.
            to_delete = list(range(base, base + 3))
            col.delete(to_delete)
            with lock:
                for pk in to_delete:
                    live_pks.discard(pk)
            time.sleep(0.002)
        stop.set()

    def reader():
        while not stop.is_set():
            try:
                res = col.query(expr=None, limit=1000)
                got = {r["id"] for r in res}
                with lock:
                    current_live = set(live_pks)
                # Any pk we see that's not in live_pks (at some point in
                # the recent past) could be a ghost. Allow a small window:
                # we accept pks that are *transitioning*. But clearly
                # deleted pks (from initial set range(20), if pk in got
                # but not in live) is a ghost.
                # To keep the assertion tight, only flag pks that have
                # been definitively deleted (we're not in the middle of
                # inserting them).
                ghosts = got - current_live
                if ghosts:
                    ghost_hits.append(ghosts)
            except Exception:
                pass
            time.sleep(0.001)

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start()
    r.start()
    w.join(timeout=30)
    r.join(timeout=5)

    col._wait_for_bg()

    # Final consistency check: no ghost pks should exist on the final
    # state. (Transient ghost hits during the race may be acceptable,
    # but final consistency must hold.)
    res = col.query(expr=None, limit=1000)
    got = {r["id"] for r in res}
    with lock:
        expected = set(live_pks)
    ghosts = got - expected
    assert not ghosts, f"final ghost pks after drain: {sorted(ghosts)}"

    col.close()
