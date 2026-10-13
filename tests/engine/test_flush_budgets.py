"""Byte-budget flush regressions for small, repeatedly rewritten key sets."""

import pyarrow as pa
import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.arrow_builder import build_wal_data_schema, build_wal_delta_schema
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.storage.wal import WAL


def _schema(partition_key=False):
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="version", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR),
        FieldSchema(name="tenant", dtype=DataType.VARCHAR, is_partition_key=partition_key),
    ])


def _rows(version):
    return [
        {"id": i, "vec": [1.0, 0.0, 0.0, 0.0], "version": version,
         "text": "x" * 1024, "tenant": f"tenant-{i}"}
        for i in range(2)
    ]


@pytest.mark.parametrize("partition_key", [False, True])
def test_repeated_upsert_flushes_by_retained_memory(tmp_path, monkeypatch, partition_key):
    schema = _schema(partition_key)
    monkeypatch.setattr("milvus_lite.engine.collection.WAL_SIZE_LIMIT", 1 << 60)
    col = Collection("c", str(tmp_path), schema)
    try:
        col.upsert(_rows(0))
        limit = col._memtable.estimated_memory_bytes * 3
        monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_MEMORY_LIMIT", limit)
        initial_wal_number = col._wal.number
        for version in range(1, 30):
            col.upsert(_rows(version))
            assert col._memtable.estimated_memory_bytes < limit
            assert col._memtable.num_physical_rows() <= 4
            assert col._memtable.size() <= 2
        assert col._wal.number >= initial_wal_number + 10
        assert col.num_entities == 2
        assert {r["version"] for r in col.query("id >= 0")} == {29}
        hits = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=2, expr="version == 29")[0]
        assert {hit["id"] for hit in hits} == {0, 1}
        assert col._memtable.estimated_memory_bytes == 0
        assert WAL.existing_size_bytes(str(tmp_path / "wal")) == 0
    finally:
        col.close()
    reopened = Collection("c", str(tmp_path), schema)
    try:
        assert reopened.num_entities == 2
        assert {r["version"] for r in reopened.get([0, 1])} == {29}
    finally:
        reopened.close()


def test_repeated_delete_flushes_by_wal_bytes(tmp_path, monkeypatch):
    schema = _schema()
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_MEMORY_LIMIT", 1 << 60)
    col = Collection("c", str(tmp_path), schema)
    try:
        col.insert(_rows(0))
        col.flush()
        col.delete([0])
        tombstone_bytes = col._memtable.estimated_memory_bytes
        limit = col._wal.size_bytes * 2
        monkeypatch.setattr("milvus_lite.engine.collection.WAL_SIZE_LIMIT", limit)
        initial_wal_number = col._wal.number
        for _ in range(30):
            col.delete([0])
            assert col._wal.size_bytes < limit
            assert col._memtable.estimated_memory_bytes <= tombstone_bytes
            assert col.get([0]) == []
        assert col._wal.number > initial_wal_number
    finally:
        col.close()
    reopened = Collection("c", str(tmp_path), schema)
    try:
        assert reopened.get([0]) == []
        assert len(reopened.get([1])) == 1
    finally:
        reopened.close()


def test_wal_budget_combines_insert_and_delete_streams(tmp_path, monkeypatch):
    col = Collection("c", str(tmp_path), _schema())
    try:
        col.insert(_rows(0))
        initial_wal_number = col._wal.number
        monkeypatch.setattr(
            "milvus_lite.engine.collection.WAL_SIZE_LIMIT", col._wal.size_bytes + 1,
        )
        col.delete([0])
        assert col._wal.number == initial_wal_number + 1
        assert col._wal.size_bytes == 0
        assert col.get([0]) == []
        assert len(col.get([1])) == 1
    finally:
        col.close()


def _leave_wal(data_dir, schema, number, version):
    """Leave an uncheckpointed WAL, closing handles without deleting its files."""
    data_schema = build_wal_data_schema(schema)
    wal = WAL(str(data_dir / "wal"), data_schema, build_wal_delta_schema(schema), number)
    batch = pa.RecordBatch.from_pylist([
        dict(row, _seq=version * 2 + i + 1, _partition="_default")
        for i, row in enumerate(_rows(version))
    ], schema=data_schema)
    wal.write_insert(batch)
    wal._data_writer.close()
    wal._data_sink.close()
    return batch.get_total_buffer_size()


@pytest.mark.parametrize("budget", ["memory", "wal", "wal_after_write"])
def test_recovery_counts_old_wals_and_checkpoints_over_budget(tmp_path, monkeypatch, budget):
    schema = _schema()
    memory_bytes = sum(_leave_wal(tmp_path, schema, n + 1, n) for n in range(3))
    wal_bytes = WAL.existing_size_bytes(str(tmp_path / "wal"))
    monkeypatch.setattr(
        "milvus_lite.engine.collection.MEMTABLE_MEMORY_LIMIT",
        memory_bytes if budget == "memory" else 1 << 60,
    )
    monkeypatch.setattr(
        "milvus_lite.engine.collection.WAL_SIZE_LIMIT",
        (1 << 60) if budget == "memory" else wal_bytes + int(budget == "wal_after_write"),
    )
    col = Collection("c", str(tmp_path), schema)
    try:
        assert col.num_entities == 2
        assert {r["version"] for r in col.get([0, 1])} == {2}
        if budget == "wal_after_write":
            assert col._wal.number == 4
            assert col._recovered_wal_bytes == wal_bytes
            # The new WAL alone is below the limit; old WALs must count too.
            col.delete([0])
            assert col.get([0]) == []
        assert col._wal.number == 5
        assert col._memtable.estimated_memory_bytes == 0
        assert col._recovered_wal_bytes == 0
        assert WAL.existing_size_bytes(str(tmp_path / "wal")) == 0
        col.upsert(_rows(3))
        assert {r["version"] for r in col.get([0, 1])} == {3}
    finally:
        col.close()


def test_large_single_row_flushes_without_a_key_count_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_MEMORY_LIMIT", 1024)
    col = Collection("c", str(tmp_path), _schema())
    try:
        col.insert(_rows(0)[:1])
        assert col._memtable.estimated_memory_bytes == 0
        assert col.num_entities == 1
        assert col.get([0])[0]["text"] == "x" * 1024
    finally:
        col.close()


def test_recovery_flushes_delete_only_wal(tmp_path, monkeypatch):
    schema = _schema()
    delta_schema = build_wal_delta_schema(schema)
    wal = WAL(str(tmp_path / "wal"), build_wal_data_schema(schema), delta_schema, 1)
    for seq in range(1, 10):
        wal.write_delete(pa.RecordBatch.from_pylist([
            {"id": 0, "_seq": seq, "_partition": "_default"},
        ], schema=delta_schema))
    wal._delta_writer.close()
    wal._delta_sink.close()
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_MEMORY_LIMIT", 1 << 60)
    monkeypatch.setattr("milvus_lite.engine.collection.WAL_SIZE_LIMIT", 1024)
    col = Collection("c", str(tmp_path), schema)
    try:
        assert col._memtable.size() == 0
        assert col._manifest.get_delta_files("_default")
        assert WAL.existing_size_bytes(str(tmp_path / "wal")) == 0
        assert col.get([0]) == []
        col.insert(_rows(0))  # Recovered delete sequence must not shadow new writes.
        assert col.num_entities == 2
    finally:
        col.close()
