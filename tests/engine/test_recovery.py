"""Tests for engine/recovery.py — replay_wal_operations + execute_recovery.

Phase 3 recovery scenarios (per wal-design.md §7.1):
    A. No WAL files at all → fresh state
    B. Only WAL(N) where N == manifest.active_wal_number
    C. WAL(N) + WAL(N+1), manifest.active_wal_number == N+1
    D. WAL(N) + WAL(N+1), manifest.active_wal_number == N
    E. Only data file or only delta file (one half of a pair)
"""

import os

import pyarrow as pa
import pytest

from milvus_lite.constants import DEFAULT_PARTITION
from milvus_lite.engine.flush import execute_flush
from milvus_lite.engine.operation import DeleteOp, InsertOp
from milvus_lite.engine.recovery import execute_recovery, replay_wal_operations
from milvus_lite.schema.arrow_builder import (
    build_wal_data_schema,
    build_wal_delta_schema,
)
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.storage.delta_index import DeltaIndex
from milvus_lite.storage.manifest import Manifest
from milvus_lite.storage.memtable import MemTable
from milvus_lite.storage.wal import WAL


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


@pytest.fixture
def wal_data_schema(schema):
    return build_wal_data_schema(schema)


@pytest.fixture
def wal_delta_schema(schema):
    return build_wal_delta_schema(schema)


def _insert_batch(wal_data_schema, rows):
    return pa.RecordBatch.from_pydict(
        {
            "_seq": [r[0] for r in rows],
            "_partition": [r[1] for r in rows],
            "id": [r[2] for r in rows],
            "vec": [r[3] for r in rows],
            "title": [r[4] for r in rows],
        },
        schema=wal_data_schema,
    )


def _delete_batch(wal_delta_schema, pks, seq, partition=DEFAULT_PARTITION):
    return pa.RecordBatch.from_pydict(
        {
            "id": pks,
            "_seq": [seq] * len(pks),
            "_partition": [partition] * len(pks),
        },
        schema=wal_delta_schema,
    )


def _write_wal(tmp_path, wal_data_schema, wal_delta_schema, wal_number, ops):
    """ops = [("insert"|"delete", batch), ...] — write to a closed WAL pair."""
    wal_dir = str(tmp_path / "wal")
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=wal_number, sync_mode="none")
    for kind, batch in ops:
        if kind == "insert":
            wal.write_insert(batch)
        else:
            wal.write_delete(batch)
    # Close writers without deleting (simulate crash before flush ran).
    if wal._data_writer is not None:
        wal._data_writer.close()
        wal._data_sink.close()
    if wal._delta_writer is not None:
        wal._delta_writer.close()
        wal._delta_sink.close()
    wal._closed = True
    return wal_dir


# ---------------------------------------------------------------------------
# replay_wal_operations
# ---------------------------------------------------------------------------

def test_replay_empty_wal(tmp_path, wal_data_schema, wal_delta_schema):
    """No WAL files → empty stream."""
    wal_dir = str(tmp_path / "wal")
    os.makedirs(wal_dir, exist_ok=True)
    ops = list(replay_wal_operations(wal_dir, 1, "id"))
    assert ops == []


def test_replay_inserts_only(tmp_path, wal_data_schema, wal_delta_schema):
    wal_dir = _write_wal(tmp_path, wal_data_schema, wal_delta_schema, 1, [
        ("insert", _insert_batch(wal_data_schema, [
            (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
            (2, DEFAULT_PARTITION, "b", [0.75, 0.125], "y"),
        ])),
    ])
    ops = list(replay_wal_operations(wal_dir, 1, "id"))
    assert len(ops) == 1
    assert isinstance(ops[0], InsertOp)
    assert ops[0].partition == DEFAULT_PARTITION
    assert ops[0].num_rows == 2


def test_replay_mixed_ops_sorted_by_seq(tmp_path, wal_data_schema, wal_delta_schema):
    """data and delta batches must be merged sorted by seq."""
    wal_dir = _write_wal(tmp_path, wal_data_schema, wal_delta_schema, 1, [
        ("insert", _insert_batch(wal_data_schema, [
            (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
        ])),
        ("delete", _delete_batch(wal_delta_schema, ["a"], seq=2)),
        ("insert", _insert_batch(wal_data_schema, [
            (3, DEFAULT_PARTITION, "a", [0.0625, 1.5], "z"),
        ])),
    ])
    ops = list(replay_wal_operations(wal_dir, 1, "id"))
    assert len(ops) == 3
    # First should be InsertOp at seq 1, second DeleteOp at seq 2, third InsertOp at seq 3
    assert isinstance(ops[0], InsertOp)
    assert ops[0].seq_min == 1
    assert isinstance(ops[1], DeleteOp)
    assert ops[1].seq == 2
    assert isinstance(ops[2], InsertOp)
    assert ops[2].seq_min == 3


def test_replay_partition_recovered(tmp_path, wal_data_schema, wal_delta_schema):
    """The InsertOp's partition should match what was written."""
    wal_dir = _write_wal(tmp_path, wal_data_schema, wal_delta_schema, 1, [
        ("insert", _insert_batch(wal_data_schema, [
            (1, "p1", "a", [0.5, 0.25], "x"),
        ])),
    ])
    [op] = list(replay_wal_operations(wal_dir, 1, "id"))
    assert op.partition == "p1"


# ---------------------------------------------------------------------------
# execute_recovery — scenario A: fresh
# ---------------------------------------------------------------------------

def test_execute_recovery_fresh(tmp_path, schema):
    """No data dir, no manifest, no WAL → empty state."""
    data_dir = str(tmp_path / "data")
    manifest = Manifest.load(data_dir)
    mt, idx, next_n = execute_recovery(data_dir, schema, manifest)
    assert mt.size() == 0
    assert len(idx) == 0
    assert next_n == 1


# ---------------------------------------------------------------------------
# execute_recovery — scenario B: WAL(N) needs replay
# ---------------------------------------------------------------------------

def test_execute_recovery_wal_replay(tmp_path, schema, wal_data_schema, wal_delta_schema):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)

    # Write a WAL with some data — simulating a crash before flush.
    _write_wal(tmp_path / "data", wal_data_schema, wal_delta_schema, 1, [
        ("insert", _insert_batch(wal_data_schema, [
            (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
            (2, DEFAULT_PARTITION, "b", [0.75, 0.125], "y"),
        ])),
    ])

    manifest = Manifest.load(data_dir)
    mt, idx, next_n = execute_recovery(data_dir, schema, manifest)

    # Both records should be back in MemTable.
    assert mt.size() == 2
    assert mt.get("a") is not None
    assert mt.get("b") is not None
    assert mt.get("a")["title"] == "x"
    # next_wal_number should be > the existing one
    assert next_n == 2


def test_execute_recovery_replays_inserts_and_deletes(
    tmp_path, schema, wal_data_schema, wal_delta_schema
):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)

    _write_wal(tmp_path / "data", wal_data_schema, wal_delta_schema, 1, [
        ("insert", _insert_batch(wal_data_schema, [
            (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
            (2, DEFAULT_PARTITION, "b", [0.75, 0.125], "y"),
        ])),
        ("delete", _delete_batch(wal_delta_schema, ["a"], seq=3)),
    ])

    manifest = Manifest.load(data_dir)
    mt, idx, _ = execute_recovery(data_dir, schema, manifest)

    # 'a' was deleted, 'b' is alive
    assert mt.get("a") is None
    assert mt.get("b") is not None


# ---------------------------------------------------------------------------
# execute_recovery — scenario E: half-pair (only data file or only delta)
# ---------------------------------------------------------------------------

def test_execute_recovery_only_data_file(
    tmp_path, schema, wal_data_schema, wal_delta_schema
):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    _write_wal(tmp_path / "data", wal_data_schema, wal_delta_schema, 1, [
        ("insert", _insert_batch(wal_data_schema, [
            (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
        ])),
    ])

    manifest = Manifest.load(data_dir)
    mt, _, _ = execute_recovery(data_dir, schema, manifest)
    assert mt.get("a") is not None


def test_execute_recovery_only_delta_file(
    tmp_path, schema, wal_data_schema, wal_delta_schema
):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    _write_wal(tmp_path / "data", wal_data_schema, wal_delta_schema, 1, [
        ("delete", _delete_batch(wal_delta_schema, ["ghost"], seq=5)),
    ])

    manifest = Manifest.load(data_dir)
    mt, _, _ = execute_recovery(data_dir, schema, manifest)
    # Only the tombstone is recovered.
    assert mt.size() == 1
    assert mt.get("ghost") is None  # deleted


# ---------------------------------------------------------------------------
# execute_recovery — orphan file cleanup
# ---------------------------------------------------------------------------

def test_execute_recovery_cleans_orphan_data_files(tmp_path, schema):
    """A Parquet file on disk that's NOT in the manifest must be removed."""
    data_dir = str(tmp_path / "data")
    partition_dir = os.path.join(data_dir, "partitions", DEFAULT_PARTITION, "data")
    os.makedirs(partition_dir, exist_ok=True)
    orphan = os.path.join(partition_dir, "data_000001_000003.parquet")
    open(orphan, "wb").write(b"orphan")

    manifest = Manifest.load(data_dir)
    execute_recovery(data_dir, schema, manifest)

    assert not os.path.exists(orphan)


def test_execute_recovery_keeps_referenced_files(
    tmp_path, schema, wal_data_schema
):
    """A Parquet file referenced by the manifest must be kept."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)

    # Use a real flush to create a referenced file.
    manifest = Manifest(data_dir)
    memtable = MemTable(schema)
    wal = WAL(
        os.path.join(data_dir, "wal"),
        wal_data_schema,
        build_wal_delta_schema(schema),
        wal_number=1,
        sync_mode="none",
    )
    delta_index = DeltaIndex("id")
    insert = _insert_batch(wal_data_schema, [
        (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
    ])
    wal.write_insert(insert)
    memtable.apply_insert(insert)
    execute_flush(
        memtable, wal, data_dir, schema, manifest, delta_index, new_wal_number=2
    )
    manifest_files = manifest.get_data_files(DEFAULT_PARTITION)
    assert len(manifest_files) == 1
    abs_path = os.path.join(data_dir, "partitions", DEFAULT_PARTITION, manifest_files[0])
    assert os.path.exists(abs_path)

    # Now run recovery — the file must still be there.
    manifest2 = Manifest.load(data_dir)
    execute_recovery(data_dir, schema, manifest2)
    assert os.path.exists(abs_path)


# ---------------------------------------------------------------------------
# execute_recovery — delta_index rebuild
# ---------------------------------------------------------------------------

def test_execute_recovery_rebuilds_delta_index(
    tmp_path, schema, wal_data_schema, wal_delta_schema
):
    """After a flush+restart, delta_index must contain the persisted deletes."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)

    manifest = Manifest(data_dir)
    memtable = MemTable(schema)
    wal = WAL(
        os.path.join(data_dir, "wal"),
        wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="none",
    )
    delta_index = DeltaIndex("id")
    delete = _delete_batch(wal_delta_schema, ["doomed"], seq=5)
    wal.write_delete(delete)
    memtable.apply_delete(delete)
    execute_flush(
        memtable, wal, data_dir, schema, manifest, delta_index, new_wal_number=2
    )

    # New process: load manifest, run recovery.
    manifest2 = Manifest.load(data_dir)
    _, idx2, _ = execute_recovery(data_dir, schema, manifest2)
    assert idx2.is_deleted("doomed", 1)
    assert len(idx2) == 1
