"""Snapshot adapter boundary tests.

These are adapted from Milvus python_client snapshot scenarios, with the
snapshot operation invoked through MilvusLite's current engine API because the
installed pymilvus/proto surface does not expose Snapshot RPCs yet.
"""

import pytest
from pymilvus import DataType, MilvusClient

from milvus_lite.exceptions import CollectionAlreadyExistsError


DIM = 4


def _schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field("score", DataType.FLOAT, nullable=True)
    return schema


def _schema_string_pk_with_json():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("info", DataType.JSON, nullable=True)
    return schema


def _row(i):
    return {
        "id": i,
        "vec": [float(i), float(i + 1), float(i + 2), float(i + 3)],
        "title": f"doc_{i}",
        "score": float(i) / 10,
    }


def _count(client, collection_name, expr="id >= 0", partition_names=None):
    rows = client.query(
        collection_name,
        filter=expr,
        partition_names=partition_names,
        output_fields=["count(*)"],
    )
    return rows[0]["count(*)"]


def _restore_and_load(db, client, collection_name, snapshot_name, restored_name):
    db.restore_snapshot(collection_name, snapshot_name, restored_name)
    client.load_collection(restored_name)


def test_snapshot_create_list_drop_adapter_boundary(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_lifecycle", schema=_schema())
    milvus_client.insert("snap_lifecycle", [_row(i) for i in range(3)])

    snap = db.create_snapshot(
        "snap_lifecycle",
        "snap_a",
        description="adapter snapshot lifecycle",
    )

    assert snap["name"] == "snap_a"
    assert snap["description"] == "adapter snapshot lifecycle"
    assert snap["collection_name"] == "snap_lifecycle"
    assert [s["name"] for s in db.list_snapshots("snap_lifecycle")] == ["snap_a"]

    db.drop_snapshot("snap_lifecycle", "snap_a")
    assert db.list_snapshots("snap_lifecycle") == []


def test_snapshot_duplicate_name_rejected_per_collection(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_duplicate", schema=_schema())

    db.create_snapshot("snap_duplicate", "snap_a")

    with pytest.raises(FileExistsError):
        db.create_snapshot("snap_duplicate", "snap_a")


def test_same_snapshot_name_allowed_across_adapter_collections(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_same_a", schema=_schema())
    milvus_client.create_collection("snap_same_b", schema=_schema())
    milvus_client.insert("snap_same_a", [_row(1)])
    milvus_client.insert("snap_same_b", [_row(2)])

    snap_a = db.create_snapshot("snap_same_a", "shared")
    snap_b = db.create_snapshot("snap_same_b", "shared")
    _restore_and_load(db, milvus_client, "snap_same_a", "shared", "snap_same_a_restored")
    _restore_and_load(db, milvus_client, "snap_same_b", "shared", "snap_same_b_restored")

    assert snap_a["collection_name"] == "snap_same_a"
    assert snap_b["collection_name"] == "snap_same_b"
    assert milvus_client.query("snap_same_a_restored", filter="id >= 0", output_fields=["id"]) == [{"id": 1}]
    assert milvus_client.query("snap_same_b_restored", filter="id >= 0", output_fields=["id"]) == [{"id": 2}]


def test_snapshot_restore_preserves_pymilvus_inserted_data(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_restore", schema=_schema())
    milvus_client.insert("snap_restore", [_row(i) for i in range(5)])

    db.create_snapshot("snap_restore", "snap_a")
    milvus_client.insert("snap_restore", [_row(99)])

    _restore_and_load(db, milvus_client, "snap_restore", "snap_a", "snap_restored")

    restored = milvus_client.query(
        "snap_restored",
        filter="id >= 0",
        output_fields=["id", "title", "vec", "score"],
        limit=10,
    )
    assert sorted(r["id"] for r in restored) == [0, 1, 2, 3, 4]
    row_2 = next(r for r in restored if r["id"] == 2)
    assert row_2["title"] == "doc_2"
    assert row_2["vec"] == [2.0, 3.0, 4.0, 5.0]
    assert abs(row_2["score"] - 0.2) < 1e-6
    assert _count(milvus_client, "snap_restore") == 6

    hits = milvus_client.search(
        "snap_restored",
        data=[[2.0, 3.0, 4.0, 5.0]],
        limit=1,
        output_fields=["title"],
    )
    assert hits[0][0]["id"] == 2
    assert hits[0][0]["entity"]["title"] == "doc_2"


def test_snapshot_restore_rejects_existing_adapter_collection(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_conflict", schema=_schema())
    milvus_client.create_collection("snap_conflict_restored", schema=_schema())
    milvus_client.insert("snap_conflict", [_row(0)])
    db.create_snapshot("snap_conflict", "snap_a")

    with pytest.raises(CollectionAlreadyExistsError):
        db.restore_snapshot("snap_conflict", "snap_a", "snap_conflict_restored")


def test_snapshot_preserves_delete_state_from_adapter(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_delete", schema=_schema())
    milvus_client.insert("snap_delete", [_row(i) for i in range(6)])
    milvus_client.delete("snap_delete", ids=[1, 3, 5])

    db.create_snapshot("snap_delete", "after_delete")
    _restore_and_load(db, milvus_client, "snap_delete", "after_delete", "snap_delete_restored")

    rows = milvus_client.query(
        "snap_delete_restored",
        filter="id >= 0",
        output_fields=["id"],
        limit=10,
    )
    assert sorted(r["id"] for r in rows) == [0, 2, 4]


def test_multiple_snapshots_restore_different_time_points(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_points", schema=_schema())

    milvus_client.insert("snap_points", [_row(0)])
    db.create_snapshot("snap_points", "snap_one")
    milvus_client.insert("snap_points", [_row(1)])
    db.create_snapshot("snap_points", "snap_two")
    milvus_client.insert("snap_points", [_row(2)])

    _restore_and_load(db, milvus_client, "snap_points", "snap_one", "snap_one_restored")
    _restore_and_load(db, milvus_client, "snap_points", "snap_two", "snap_two_restored")

    assert _count(milvus_client, "snap_one_restored") == 1
    assert _count(milvus_client, "snap_two_restored") == 2
    assert _count(milvus_client, "snap_points") == 3


def test_snapshot_restore_preserves_dropped_partition_data(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_partition", schema=_schema())
    milvus_client.create_partition("snap_partition", "archive")
    milvus_client.insert("snap_partition", [_row(0)])
    milvus_client.insert("snap_partition", [_row(10)], partition_name="archive")

    db.create_snapshot("snap_partition", "with_archive")
    milvus_client.drop_partition("snap_partition", "archive")

    _restore_and_load(db, milvus_client, "snap_partition", "with_archive", "snap_partition_restored")

    assert "archive" in milvus_client.list_partitions("snap_partition_restored")
    assert _count(milvus_client, "snap_partition_restored", partition_names=["archive"]) == 1
    rows = milvus_client.query(
        "snap_partition_restored",
        filter="id >= 0",
        partition_names=["archive"],
        output_fields=["id"],
    )
    assert rows == [{"id": 10}]


def test_snapshot_restore_preserves_string_pk_and_json(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_json", schema=_schema_string_pk_with_json())
    milvus_client.insert("snap_json", [
        {
            "pk": "doc_a",
            "vec": [1.0, 0.0, 0.0, 0.0],
            "info": {"name": "alice", "tags": ["snapshot", "json"]},
        },
        {
            "pk": "doc_b",
            "vec": [0.0, 1.0, 0.0, 0.0],
            "info": {"name": "bob", "tags": ["restore"]},
        },
    ])

    db.create_snapshot("snap_json", "snap_a")
    _restore_and_load(db, milvus_client, "snap_json", "snap_a", "snap_json_restored")

    rows = milvus_client.query(
        "snap_json_restored",
        filter='pk == "doc_a"',
        output_fields=["pk", "info"],
    )
    assert len(rows) == 1
    assert rows[0]["pk"] == "doc_a"
    assert rows[0]["info"] == {"name": "alice", "tags": ["snapshot", "json"]}


def test_snapshot_restore_missing_or_deleted_snapshot_raises(milvus_client, grpc_server):
    _, db = grpc_server
    milvus_client.create_collection("snap_negative", schema=_schema())
    milvus_client.insert("snap_negative", [_row(0)])

    with pytest.raises(FileNotFoundError):
        db.restore_snapshot("snap_negative", "missing", "missing_restored")

    db.create_snapshot("snap_negative", "snap_a")
    db.drop_snapshot("snap_negative", "snap_a")
    with pytest.raises(FileNotFoundError):
        db.restore_snapshot("snap_negative", "snap_a", "deleted_restored")
