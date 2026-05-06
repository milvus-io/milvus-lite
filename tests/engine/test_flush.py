"""Tests for engine/flush.py — the 7-step pipeline.

These are unit-level tests of execute_flush in isolation. The Collection
integration (insert → trigger flush automatically) is covered in
test_collection.py once Phase 3 lands.
"""

import os

import pyarrow as pa
import pytest

from milvus_lite.constants import DEFAULT_PARTITION
from milvus_lite.engine.flush import execute_flush
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


@pytest.fixture
def harness(tmp_path, schema, wal_data_schema, wal_delta_schema):
    """Build a 'frozen' set of (memtable, wal, manifest, delta_index)
    ready to be passed to execute_flush."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    wal_dir = os.path.join(data_dir, "wal")

    manifest = Manifest(data_dir)
    memtable = MemTable(schema)
    wal = WAL(
        wal_dir=wal_dir,
        wal_data_schema=wal_data_schema,
        wal_delta_schema=wal_delta_schema,
        wal_number=1,
        sync_mode="none",
    )
    delta_index = DeltaIndex(pk_name="id")

    return {
        "data_dir": data_dir,
        "schema": schema,
        "manifest": manifest,
        "memtable": memtable,
        "wal": wal,
        "delta_index": delta_index,
    }


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


# ---------------------------------------------------------------------------
# Empty flush
# ---------------------------------------------------------------------------

def test_flush_empty_memtable(harness):
    """Flushing an empty MemTable should still rotate WAL bookkeeping."""
    execute_flush(
        frozen_memtable=harness["memtable"],
        frozen_wal=harness["wal"],
        data_dir=harness["data_dir"],
        schema=harness["schema"],
        manifest=harness["manifest"],
        delta_index=harness["delta_index"],
        new_wal_number=2,
    )
    assert harness["manifest"].active_wal_number == 2
    assert harness["manifest"].get_data_files(DEFAULT_PARTITION) == []
    assert harness["manifest"].get_delta_files(DEFAULT_PARTITION) == []


# ---------------------------------------------------------------------------
# Simple insert flush
# ---------------------------------------------------------------------------

def test_flush_inserts_writes_parquet_and_updates_manifest(harness, wal_data_schema):
    mt = harness["memtable"]
    wal = harness["wal"]

    batch = _insert_batch(wal_data_schema, [
        (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
        (2, DEFAULT_PARTITION, "b", [0.75, 0.125], "y"),
    ])
    wal.write_insert(batch)
    mt.apply_insert(batch)

    execute_flush(
        frozen_memtable=mt,
        frozen_wal=wal,
        data_dir=harness["data_dir"],
        schema=harness["schema"],
        manifest=harness["manifest"],
        delta_index=harness["delta_index"],
        new_wal_number=2,
    )

    # 1. Manifest has the new file.
    files = harness["manifest"].get_data_files(DEFAULT_PARTITION)
    assert len(files) == 1
    assert files[0] == "data/data_000001_000002.parquet"

    # 2. The Parquet file exists on disk.
    abs_path = os.path.join(
        harness["data_dir"], "partitions", DEFAULT_PARTITION, files[0]
    )
    assert os.path.exists(abs_path)

    # 3. current_seq updated.
    assert harness["manifest"].current_seq == 2

    # 4. active_wal_number switched.
    assert harness["manifest"].active_wal_number == 2

    # 5. WAL files removed.
    wal_dir = os.path.join(harness["data_dir"], "wal")
    assert list(p for p in os.listdir(wal_dir) if p.startswith("wal_")) == []

    # 6. Manifest persisted.
    from milvus_lite.storage.manifest import Manifest
    reloaded = Manifest.load(harness["data_dir"])
    assert reloaded.get_data_files(DEFAULT_PARTITION) == files
    assert reloaded.current_seq == 2
    assert reloaded.active_wal_number == 2


def test_flush_inserts_parquet_round_trip(harness, wal_data_schema):
    """The written Parquet file must round-trip back to the live data."""
    mt = harness["memtable"]
    wal = harness["wal"]

    batch = _insert_batch(wal_data_schema, [
        (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
        (2, DEFAULT_PARTITION, "b", [0.75, 0.125], "y"),
        (3, DEFAULT_PARTITION, "c", [0.0625, 1.5], "z"),
    ])
    wal.write_insert(batch)
    mt.apply_insert(batch)

    execute_flush(
        frozen_memtable=mt,
        frozen_wal=wal,
        data_dir=harness["data_dir"],
        schema=harness["schema"],
        manifest=harness["manifest"],
        delta_index=harness["delta_index"],
        new_wal_number=2,
    )

    files = harness["manifest"].get_data_files(DEFAULT_PARTITION)
    abs_path = os.path.join(
        harness["data_dir"], "partitions", DEFAULT_PARTITION, files[0]
    )
    from milvus_lite.storage.data_file import read_data_file
    table = read_data_file(abs_path)
    assert table.num_rows == 3
    assert set(table.column("id").to_pylist()) == {"a", "b", "c"}
    assert set(table.column("_seq").to_pylist()) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Multi-partition
# ---------------------------------------------------------------------------

def test_flush_multi_partition(harness, wal_data_schema):
    mt = harness["memtable"]
    wal = harness["wal"]
    harness["manifest"].add_partition("p1")
    harness["manifest"].add_partition("p2")

    batch = _insert_batch(wal_data_schema, [
        (1, "p1", "a", [0.5, 0.25], "x"),
        (2, "p2", "b", [0.75, 0.125], "y"),
        (3, "p1", "c", [0.0625, 1.5], "z"),
    ])
    wal.write_insert(batch)
    mt.apply_insert(batch)

    execute_flush(
        frozen_memtable=mt,
        frozen_wal=wal,
        data_dir=harness["data_dir"],
        schema=harness["schema"],
        manifest=harness["manifest"],
        delta_index=harness["delta_index"],
        new_wal_number=2,
    )

    p1_files = harness["manifest"].get_data_files("p1")
    p2_files = harness["manifest"].get_data_files("p2")
    assert len(p1_files) == 1
    assert len(p2_files) == 1
    # p1 file has rows a, c → seq range 1..3
    assert "data_000001_000003.parquet" in p1_files[0]
    # p2 file has only row b → seq range 2..2
    assert "data_000002_000002.parquet" in p2_files[0]


# ---------------------------------------------------------------------------
# Delete flush
# ---------------------------------------------------------------------------

def test_flush_deletes_write_delta_file(harness, wal_data_schema, wal_delta_schema):
    mt = harness["memtable"]
    wal = harness["wal"]

    insert = _insert_batch(wal_data_schema, [
        (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
    ])
    wal.write_insert(insert)
    mt.apply_insert(insert)

    delete = _delete_batch(wal_delta_schema, ["a"], seq=2)
    wal.write_delete(delete)
    mt.apply_delete(delete)

    execute_flush(
        frozen_memtable=mt,
        frozen_wal=wal,
        data_dir=harness["data_dir"],
        schema=harness["schema"],
        manifest=harness["manifest"],
        delta_index=harness["delta_index"],
        new_wal_number=2,
    )

    delta_files = harness["manifest"].get_delta_files(DEFAULT_PARTITION)
    assert len(delta_files) == 1
    assert "delta_000002_000002.parquet" in delta_files[0]

    # delta_index updated in-memory
    assert harness["delta_index"].is_deleted("a", 1)


def test_flush_no_data_only_delete(harness, wal_delta_schema):
    """A flush with only deletes (no inserts) should still work."""
    mt = harness["memtable"]
    wal = harness["wal"]

    delete = _delete_batch(wal_delta_schema, ["x", "y"], seq=5)
    wal.write_delete(delete)
    mt.apply_delete(delete)

    execute_flush(
        frozen_memtable=mt,
        frozen_wal=wal,
        data_dir=harness["data_dir"],
        schema=harness["schema"],
        manifest=harness["manifest"],
        delta_index=harness["delta_index"],
        new_wal_number=2,
    )

    assert harness["manifest"].get_data_files(DEFAULT_PARTITION) == []
    assert len(harness["manifest"].get_delta_files(DEFAULT_PARTITION)) == 1


# ---------------------------------------------------------------------------
# WAL cleanup
# ---------------------------------------------------------------------------

def test_flush_cleans_up_orphan_wals(harness, wal_data_schema):
    """If earlier WAL files survived a previous failed flush, the next
    flush should also clean them up."""
    wal_dir = os.path.join(harness["data_dir"], "wal")
    os.makedirs(wal_dir, exist_ok=True)

    # Drop an orphan WAL file from a previous "failed" flush.
    orphan_path = os.path.join(wal_dir, "wal_data_000000.arrow")
    open(orphan_path, "wb").write(b"junk")

    mt = harness["memtable"]
    wal = harness["wal"]
    batch = _insert_batch(wal_data_schema, [
        (1, DEFAULT_PARTITION, "a", [0.5, 0.25], "x"),
    ])
    wal.write_insert(batch)
    mt.apply_insert(batch)

    execute_flush(
        frozen_memtable=mt,
        frozen_wal=wal,
        data_dir=harness["data_dir"],
        schema=harness["schema"],
        manifest=harness["manifest"],
        delta_index=harness["delta_index"],
        new_wal_number=2,
    )

    # The orphan WAL must have been cleaned up too (number 0 <= frozen 1).
    assert not os.path.exists(orphan_path)
    # And the just-flushed WAL is also gone.
    assert not os.path.exists(os.path.join(wal_dir, "wal_data_000001.arrow"))
