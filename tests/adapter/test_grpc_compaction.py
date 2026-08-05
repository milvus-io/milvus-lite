"""gRPC coverage for the manual-compaction compatibility shim."""

from pymilvus import DataType, MilvusClient


def _create(client, name):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("value", DataType.INT64)
    client.create_collection(name, schema=schema)


def test_compact_empty_collection_reports_completed(milvus_client):
    _create(milvus_client, "compact_empty")

    job_id = milvus_client.compact("compact_empty")

    assert isinstance(job_id, int)
    assert milvus_client.get_compaction_state(job_id).lower() == "completed"


def test_compact_preserves_logical_rows_and_search(milvus_client):
    _create(milvus_client, "compact_data")
    milvus_client.insert(
        "compact_data",
        [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "value": 10},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "value": 20},
        ],
    )
    milvus_client.upsert(
        "compact_data",
        [
            {"id": 2, "vec": [0.0, 0.0, 1.0, 0.0], "value": 200},
            {"id": 3, "vec": [0.0, 0.0, 0.0, 1.0], "value": 30},
        ],
    )
    milvus_client.delete("compact_data", ids=[1])
    milvus_client.flush("compact_data")
    before = milvus_client.query(
        "compact_data",
        filter="id >= 0",
        output_fields=["id", "value"],
        limit=10,
    )

    job_id = milvus_client.compact("compact_data")

    assert milvus_client.get_compaction_state(job_id).lower() == "completed"
    after = milvus_client.query(
        "compact_data",
        filter="id >= 0",
        output_fields=["id", "value"],
        limit=10,
    )
    assert sorted(before, key=lambda row: row["id"]) == sorted(
        after, key=lambda row: row["id"]
    )
    hits = milvus_client.search(
        "compact_data",
        data=[[0.0, 0.0, 1.0, 0.0]],
        limit=1,
        output_fields=["value"],
    )
    assert hits[0][0]["id"] == 2
    assert hits[0][0]["entity"]["value"] == 200
