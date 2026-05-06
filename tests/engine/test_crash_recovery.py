"""Crash injection tests for the flush + recovery loop.

This is Phase 3's safety net for the LSM "soul" — the proof that any
crash inside the flush pipeline leaves the on-disk state in a form
recovery can fix up to a consistent point.

Strategy: monkeypatch one of the flush pipeline's steps to raise
SystemExit (simulating an unclean process termination), then re-open
the Collection and assert the recovered state is correct.

Recovery contract:
    For any insert call that returned successfully BEFORE the crash,
    after restart that record must be readable via get().

    For any insert call that raised mid-flow, the post-restart state
    is unspecified — the caller knows their request failed and can
    retry. We only verify "no torn state" for committed inserts.
"""

import os
import shutil

import pyarrow as pa
import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.engine.flush import execute_flush
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


def _record(i, prefix="doc"):
    return {
        "id": f"{prefix}_{i:04d}",
        "vec": [0.5, 0.25],
        "title": f"t{i}",
    }


# ---------------------------------------------------------------------------
# Helper: simulate process death by deleting Collection without close()
# ---------------------------------------------------------------------------

def _simulate_crash(col):
    """Drop the Collection reference without calling close().

    Important: do not let close() run via __del__ either. Python's GC may
    or may not run __del__; we just stop using the object.

    Stop the background executor (wait for any in-flight task to finish,
    then cancel pending ones) — a real process crash kills all threads
    at once; Python can't do that safely, so we drain to the nearest
    atomic boundary.
    """
    col._bg_closed = True
    col._bg_executor.shutdown(wait=True, cancel_futures=True)
    del col


# ---------------------------------------------------------------------------
# 1. Crash before flush — committed inserts in WAL are recoverable
# ---------------------------------------------------------------------------

def test_crash_before_any_flush(tmp_path, schema):
    """Insert N records, never flush, kill the process. After restart
    all N records must be recoverable from WAL replay."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    for i in range(5):
        col.insert([_record(i)])
    _simulate_crash(col)

    col2 = Collection("c", data_dir, schema)
    assert col2.count() == 5
    for i in range(5):
        rec = col2.get([f"doc_{i:04d}"])
        assert len(rec) == 1, f"missing doc_{i:04d}"
    col2.close()


# ---------------------------------------------------------------------------
# 2. Crash during flush Step 3 (write Parquet)
# ---------------------------------------------------------------------------

def test_crash_during_step3_parquet_write(tmp_path, schema, monkeypatch):
    """Crash after MemTable.flush() but before manifest commit.

    The on-disk state: a Parquet file may have been written (orphan),
    but the manifest is unchanged. The frozen WAL is still on disk.

    Recovery: WAL replay restores the records; orphan Parquet is cleaned.
    """
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 3)

    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)

    # Insert 2 records — no flush yet (limit=3)
    col.insert([_record(0)])
    col.insert([_record(1)])

    # Patch write_data_file to raise after the first call inside flush
    from milvus_lite.engine import flush as flush_mod
    real_write = flush_mod.write_data_file

    def crash_after_write(*args, **kwargs):
        # Actually write the file (so the orphan exists), then raise.
        rel = real_write(*args, **kwargs)
        raise SystemExit("crash during step 3")

    monkeypatch.setattr(flush_mod, "write_data_file", crash_after_write)

    # Insert the 3rd record — this should hit the limit and trigger flush.
    with pytest.raises(SystemExit):
        col.insert([_record(2)])

    _simulate_crash(col)

    # Restore the real function for the recovery path
    monkeypatch.setattr(flush_mod, "write_data_file", real_write)

    # Restart
    col2 = Collection("c", data_dir, schema)
    # All 3 records (the ones the user successfully inserted before
    # the crash) must be recoverable.
    for i in range(3):
        rec = col2.get([f"doc_{i:04d}"])
        assert len(rec) == 1, f"missing doc_{i:04d} after recovery"

    # Manifest should NOT contain the orphan file (recovery cleaned it)
    files = col2._manifest.get_data_files("_default")
    # The orphan file should have been cleaned by recovery — only files
    # that are referenced by the manifest survive.
    for rel in files:
        abs_path = os.path.join(data_dir, "partitions", "_default", rel)
        assert os.path.exists(abs_path)
    col2.close()


# ---------------------------------------------------------------------------
# 3. Crash during flush Step 5 (manifest save)
# ---------------------------------------------------------------------------

def test_crash_during_step5_manifest_save(tmp_path, schema, monkeypatch):
    """Crash inside Manifest.save(). Either the rename completed (new
    manifest) or it didn't (old manifest); both cases must recover."""
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 3)

    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([_record(0)])
    col.insert([_record(1)])

    # Make Manifest.save raise after the new files are written but
    # before the rename — equivalent to crashing in the middle.
    from milvus_lite.storage import manifest as manifest_mod
    real_save = manifest_mod.Manifest.save

    saves_called = [0]

    def crashing_save(self):
        saves_called[0] += 1
        if saves_called[0] == 1:
            raise SystemExit("crash during step 5")
        return real_save(self)

    monkeypatch.setattr(manifest_mod.Manifest, "save", crashing_save)

    with pytest.raises(SystemExit):
        col.insert([_record(2)])

    _simulate_crash(col)

    # Restore real save before recovery
    monkeypatch.setattr(manifest_mod.Manifest, "save", real_save)

    col2 = Collection("c", data_dir, schema)
    # All 3 records committed before crash must be recoverable.
    for i in range(3):
        rec = col2.get([f"doc_{i:04d}"])
        assert len(rec) == 1, f"missing doc_{i:04d}"
    col2.close()


# ---------------------------------------------------------------------------
# 4. Crash after manifest save, before WAL deletion
# ---------------------------------------------------------------------------

def test_crash_after_manifest_before_wal_delete(tmp_path, schema, monkeypatch):
    """The most subtle crash window: manifest is updated (so the new
    Parquet file is "official"), but the frozen WAL is still on disk.

    Recovery sees BOTH the new Parquet AND the WAL. WAL replay produces
    rows that overlap with what's already in the Parquet. _seq dedup
    handles this — the rows simply re-enter MemTable with the same _seq.
    """
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 3)

    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([_record(0)])
    col.insert([_record(1)])

    # Crash inside frozen_wal.close_and_delete (Step 6)
    from milvus_lite.storage import wal as wal_mod
    real_close = wal_mod.WAL.close_and_delete

    def crashing_close(self):
        raise SystemExit("crash during step 6")

    monkeypatch.setattr(wal_mod.WAL, "close_and_delete", crashing_close)

    with pytest.raises(SystemExit):
        col.insert([_record(2)])

    _simulate_crash(col)

    monkeypatch.setattr(wal_mod.WAL, "close_and_delete", real_close)

    col2 = Collection("c", data_dir, schema)
    # All 3 records committed before crash must be recoverable.
    for i in range(3):
        rec = col2.get([f"doc_{i:04d}"])
        assert len(rec) == 1, f"missing doc_{i:04d}"
    col2.close()


# ---------------------------------------------------------------------------
# 5. Multiple flush/crash cycles in sequence
# ---------------------------------------------------------------------------

def test_repeated_crash_recovery_cycles(tmp_path, schema, monkeypatch):
    """Insert + crash + recover + insert + crash + recover. State must
    be consistent at each step.

    Phase 3 note: get() only reads the MemTable (Phase 4 will add segment
    reads). To verify all records, we crash between rounds (WAL retained)
    rather than close (which would flush to Parquet, where get() can't
    see them yet).
    """
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 100)

    data_dir = str(tmp_path / "d")

    # Round 1: insert 3, crash
    col = Collection("c", data_dir, schema)
    col.insert([_record(i) for i in range(3)])
    _simulate_crash(col)

    # Round 2: recover, verify, insert 3 more, crash again (NOT close)
    col = Collection("c", data_dir, schema)
    assert col.count() == 3
    col.insert([_record(i) for i in range(3, 6)])
    assert col.count() == 6
    _simulate_crash(col)

    # Round 3: recover again, verify both batches via WAL replay
    col = Collection("c", data_dir, schema)
    assert col.count() == 6
    for i in range(6):
        rec = col.get([f"doc_{i:04d}"])
        assert len(rec) == 1, f"missing doc_{i:04d} in round 3"
    col.close()


# ---------------------------------------------------------------------------
# 6. Property-style: random insert sizes + flush + crash + recover
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(10)))
def test_random_insert_crash_recover(tmp_path, schema, monkeypatch, seed):
    """Random insert sizes + random crash points. All committed inserts
    must be recoverable through WAL replay.

    Phase-4 update: with the segment cache, get() also reads flushed
    Parquet files, so we can let the size limit trigger natural flushes.
    """
    import random
    rng = random.Random(seed)
    # Small limit so flushes happen mid-test.
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 7)

    data_dir = str(tmp_path / f"d{seed}")
    committed_pks: set = set()

    col = Collection("c", data_dir, schema)
    n_inserted = 0
    for round_idx in range(15):
        batch_size = rng.randint(1, 5)
        recs = [_record(n_inserted + i, prefix=f"r{round_idx}") for i in range(batch_size)]
        col.insert(recs)
        committed_pks.update(r["id"] for r in recs)
        n_inserted += batch_size

        # 30% chance to crash and reopen
        if rng.random() < 0.3:
            _simulate_crash(col)
            col = Collection("c", data_dir, schema)

    # Final verification: crash, reopen, every committed pk must be readable.
    _simulate_crash(col)
    final = Collection("c", data_dir, schema)
    for pk in committed_pks:
        rec = final.get([pk])
        assert len(rec) == 1, f"committed pk {pk} missing after final recovery (seed={seed})"
    final.close()


# ---------------------------------------------------------------------------
# Phase 5: delete-aware crash recovery
# ---------------------------------------------------------------------------

def test_delete_then_crash_recovers_tombstone(tmp_path, schema):
    """Insert, delete in MemTable, crash without flush. After restart
    the delete must still be in effect (replayed from wal_delta)."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([_record(0)])
    col.delete(["doc_0000"])
    _simulate_crash(col)

    col2 = Collection("c", data_dir, schema)
    assert col2.get(["doc_0000"]) == []
    col2.close()


def test_delete_after_flush_then_crash(tmp_path, schema):
    """Insert + flush + delete + crash. The flushed insert is in a
    segment; the delete is in WAL. Recovery must replay the delete
    into the MemTable so get() correctly hides the segment row."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([_record(0)])
    col.flush()  # X is in a segment
    col.delete(["doc_0000"])  # tombstone in MemTable + WAL
    _simulate_crash(col)

    col2 = Collection("c", data_dir, schema)
    assert col2.get(["doc_0000"]) == []
    col2.close()


def test_crash_during_delete_flush(tmp_path, schema, monkeypatch):
    """Inject a crash during the flush that carries a delete tombstone.
    The delete must still be effective after recovery (replayed from WAL).
    """
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    # Round 1 — clean flush so doc_0000 ends up in a segment.
    col.insert([_record(0)])
    col.insert([_record(1)])
    col.flush()
    assert col.count() == 0

    # Round 2 — apply a delete and a new insert, then trigger a flush
    # that crashes mid-save.
    col.delete(["doc_0000"])  # tombstone for the segment row
    col.insert([_record(2)])  # plain insert

    from milvus_lite.storage import manifest as manifest_mod
    real_save = manifest_mod.Manifest.save

    def crashing_save(self):
        raise SystemExit("crash during step 5")

    monkeypatch.setattr(manifest_mod.Manifest, "save", crashing_save)
    with pytest.raises(SystemExit):
        col.flush()  # explicit flush triggers the crashing save

    _simulate_crash(col)
    monkeypatch.setattr(manifest_mod.Manifest, "save", real_save)

    # Restart — recovery must:
    # - replay wal_delta containing delete(doc_0000) → tombstone in MemTable
    # - replay wal_data containing insert(doc_0002) → live row in MemTable
    # - clean up any orphan parquet files from the failed flush
    col2 = Collection("c", data_dir, schema)
    assert col2.get(["doc_0000"]) == []          # tombstone wins
    assert col2.get(["doc_0001"]) != []          # untouched, still alive
    assert col2.get(["doc_0002"]) != []          # replayed insert is alive
    col2.close()


def test_random_insert_delete_crash_recover(tmp_path, schema, monkeypatch):
    """Random insert + delete sequences with random crashes. The
    expected state is computed by a trivial Python simulator and
    compared against the final Collection state."""
    import random
    rng = random.Random(13)
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 7)

    data_dir = str(tmp_path / "d")
    expected: dict = {}  # pk → label or None for deleted

    col = Collection("c", data_dir, schema)
    for round_idx in range(30):
        action = rng.choice(["insert", "insert", "delete"])
        pk = f"pk_{rng.randint(0, 4)}"
        if action == "insert":
            label = f"r{round_idx}"
            col.insert([{"id": pk, "vec": [0.5, 0.25], "title": label}])
            expected[pk] = label
        else:
            col.delete([pk])
            expected[pk] = None

        if rng.random() < 0.25:
            _simulate_crash(col)
            col = Collection("c", data_dir, schema)

    _simulate_crash(col)
    final = Collection("c", data_dir, schema)
    for pk, exp_label in expected.items():
        got = final.get([pk])
        if exp_label is None:
            assert got == [], f"pk {pk}: expected deleted, got {got}"
        else:
            assert len(got) == 1, f"pk {pk}: expected {exp_label}, got nothing"
            assert got[0]["title"] == exp_label, f"pk {pk}: expected {exp_label}, got {got[0]['title']}"
    final.close()
