"""Compatibility tests for ShowCollections/ShowPartitions InMemory responses."""

import grpc
import pytest
from pymilvus import DataType, MilvusClient
from pymilvus.grpc_gen import milvus_pb2, milvus_pb2_grpc


def _make_schema():
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    return schema


def _make_index_params(client):
    params = client.prepare_index_params()
    params.add_index(
        field_name="vec",
        index_type="BRUTE_FORCE",
        metric_type="L2",
        params={},
    )
    return params


@pytest.fixture
def raw_stub(grpc_server):
    port, _db = grpc_server
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield milvus_pb2_grpc.MilvusServiceStub(channel)
    finally:
        channel.close()


def test_show_collections_in_memory_reports_loaded_percentage(
    milvus_client, raw_stub
):
    milvus_client.create_collection("demo", schema=_make_schema())

    response = raw_stub.ShowCollections(
        milvus_pb2.ShowCollectionsRequest(
            type=milvus_pb2.ShowType.InMemory,
            collection_names=["demo"],
        )
    )

    assert list(response.collection_names) == ["demo"]
    assert list(response.inMemory_percentages) == [100]
    assert list(response.query_service_available) == [True]
    assert list(response.shards_num) == [1]


def test_show_collections_in_memory_reports_released_percentage(
    milvus_client, raw_stub
):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.create_index("demo", _make_index_params(milvus_client))
    milvus_client.release_collection("demo")

    response = raw_stub.ShowCollections(
        milvus_pb2.ShowCollectionsRequest(
            type=milvus_pb2.ShowType.InMemory,
            collection_names=["demo"],
        )
    )

    assert list(response.collection_names) == ["demo"]
    assert list(response.inMemory_percentages) == [0]
    assert list(response.query_service_available) == [False]
    assert list(response.shards_num) == [1]


def test_show_collections_in_memory_filters_requested_names(
    milvus_client, raw_stub
):
    milvus_client.create_collection("alpha", schema=_make_schema())
    milvus_client.create_collection("beta", schema=_make_schema())

    response = raw_stub.ShowCollections(
        milvus_pb2.ShowCollectionsRequest(
            type=milvus_pb2.ShowType.InMemory,
            collection_names=["beta"],
        )
    )

    assert list(response.collection_names) == ["beta"]
    assert list(response.inMemory_percentages) == [100]


def test_show_partitions_in_memory_reports_collection_load_percentage(
    milvus_client, raw_stub
):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.create_partition("demo", "p1")

    response = raw_stub.ShowPartitions(
        milvus_pb2.ShowPartitionsRequest(
            collection_name="demo",
            type=milvus_pb2.ShowType.InMemory,
        )
    )

    assert list(response.partition_names) == ["_default", "p1"]
    assert list(response.inMemory_percentages) == [100, 100]


def test_show_partitions_in_memory_reports_released_collection(
    milvus_client, raw_stub
):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.create_partition("demo", "p1")
    milvus_client.create_index("demo", _make_index_params(milvus_client))
    milvus_client.release_collection("demo")

    response = raw_stub.ShowPartitions(
        milvus_pb2.ShowPartitionsRequest(
            collection_name="demo",
            type=milvus_pb2.ShowType.InMemory,
        )
    )

    assert list(response.partition_names) == ["_default", "p1"]
    assert list(response.inMemory_percentages) == [0, 0]


def test_show_partitions_in_memory_filters_requested_names(
    milvus_client, raw_stub
):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.create_partition("demo", "p1")
    milvus_client.create_partition("demo", "p2")

    response = raw_stub.ShowPartitions(
        milvus_pb2.ShowPartitionsRequest(
            collection_name="demo",
            type=milvus_pb2.ShowType.InMemory,
            partition_names=["p2"],
        )
    )

    assert list(response.partition_names) == ["p2"]
    assert list(response.inMemory_percentages) == [100]
