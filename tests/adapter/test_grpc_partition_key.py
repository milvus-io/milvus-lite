"""Partition key — gRPC adapter integration tests.

Tests pymilvus → gRPC → MilvusLite partition key round-trip:
create with is_partition_key, insert (auto-routed), search, query.
"""

import pytest

from pymilvus import DataType, MilvusClient


DIM = 16


def _create_pk_collection(client, name):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("tenant", DataType.VARCHAR, max_length=64,
                     is_partition_key=True)
    schema.add_field("score", DataType.FLOAT, nullable=True)

    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
    client.create_collection(name, schema=schema, index_params=idx)
    return name


def test_create_with_partition_key(milvus_client):
    _create_pk_collection(milvus_client, "pk_create")
    desc = milvus_client.describe_collection("pk_create")
    fields = {f["name"]: f for f in desc["fields"]}
    assert fields["tenant"].get("is_partition_key") is True


def test_insert_and_query_with_partition_key(milvus_client):
    _create_pk_collection(milvus_client, "pk_iq")
    milvus_client.insert("pk_iq", [
        {"id": 1, "vec": [1.0] + [0.0] * (DIM - 1), "tenant": "a", "score": 1.0},
        {"id": 2, "vec": [0.0, 1.0] + [0.0] * (DIM - 2), "tenant": "b", "score": 2.0},
        {"id": 3, "vec": [0.0, 0.0, 1.0] + [0.0] * (DIM - 3), "tenant": "a", "score": 3.0},
    ])
    milvus_client.load_collection("pk_iq")
    res = milvus_client.query("pk_iq", filter="id >= 0", limit=10)
    assert len(res) == 3


def test_search_with_partition_key(milvus_client):
    _create_pk_collection(milvus_client, "pk_search")
    milvus_client.insert("pk_search", [
        {"id": i, "vec": [float(i == j) for j in range(DIM)],
         "tenant": f"t_{i % 3}"}
        for i in range(10)
    ])
    milvus_client.load_collection("pk_search")
    res = milvus_client.search("pk_search",
                                data=[[1.0] + [0.0] * (DIM - 1)],
                                limit=5)
    assert len(res[0]) == 5


def test_query_filter_on_partition_key_field(milvus_client):
    _create_pk_collection(milvus_client, "pk_filter")
    milvus_client.insert("pk_filter", [
        {"id": i, "vec": [0.1] * DIM, "tenant": f"t_{i % 3}"}
        for i in range(30)
    ])
    milvus_client.load_collection("pk_filter")
    res = milvus_client.query("pk_filter", filter='tenant == "t_0"',
                              limit=100)
    assert len(res) == 10
    assert all(r["tenant"] == "t_0" for r in res)


def test_partition_key_with_int64(milvus_client):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("group_id", DataType.INT64, is_partition_key=True)

    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
    milvus_client.create_collection("pk_int", schema=schema, index_params=idx)
    milvus_client.insert("pk_int", [
        {"id": i, "vec": [0.1] * DIM, "group_id": i % 5}
        for i in range(20)
    ])
    milvus_client.load_collection("pk_int")
    res = milvus_client.query("pk_int", filter="group_id == 0", limit=100)
    assert len(res) == 4


def test_drop_collection_with_partition_key(milvus_client):
    _create_pk_collection(milvus_client, "pk_drop")
    milvus_client.insert("pk_drop", [
        {"id": 1, "vec": [0.1] * DIM, "tenant": "a"},
    ])
    milvus_client.drop_collection("pk_drop")
    assert not milvus_client.has_collection("pk_drop")
