"""Upstream L2 Function Chain compatibility tests through MilvusClient."""

from __future__ import annotations

import shutil
import tempfile
import uuid

import pytest

try:
    from pymilvus import (
        DataType,
        Function,
        FunctionChain,
        FunctionChainStage,
        FunctionScore,
        FunctionType,
        MilvusClient,
    )
    from pymilvus.exceptions import MilvusException
    from pymilvus.function_chain import col, fn
    from pymilvus.grpc_gen import milvus_pb2
except (ImportError, AttributeError) as exc:
    pytest.skip(
        f"PyMilvus Function Chain API is unavailable: {exc}",
        allow_module_level=True,
    )

if "function_chains" not in milvus_pb2.SearchRequest.DESCRIPTOR.fields_by_name:
    pytest.skip(
        "PyMilvus SearchRequest has no function_chains field",
        allow_module_level=True,
    )

from milvus_lite.adapter.grpc.server import start_server_in_thread


DIM = 2
VECTOR_FIELD = "vector"
SCALAR_FIELD = "ts"


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="function_chain_l2_")
    grpc_server, database, port = start_server_in_thread(data_dir)
    try:
        yield port
    finally:
        try:
            grpc_server.stop(grace=2)
        finally:
            try:
                database.close()
            finally:
                shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def client(server):
    milvus_client = MilvusClient(uri=f"http://127.0.0.1:{server}")
    try:
        yield milvus_client
    finally:
        try:
            for name in milvus_client.list_collections():
                milvus_client.drop_collection(name)
        finally:
            milvus_client.close()


def _collection_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _create_function_chain_collection(client: MilvusClient) -> str:
    name = _collection_name("function_chain_l2")
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field(SCALAR_FIELD, DataType.INT64)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="BRUTE_FORCE",
        metric_type="L2",
        params={},
    )
    client.create_collection(
        name,
        schema=schema,
        index_params=index_params,
    )
    client.insert(
        name,
        [
            {"id": 1, SCALAR_FIELD: 10, VECTOR_FIELD: [0.0, 0.0]},
            {"id": 2, SCALAR_FIELD: 20, VECTOR_FIELD: [0.01, 0.0]},
            {"id": 3, SCALAR_FIELD: 30, VECTOR_FIELD: [0.02, 0.0]},
        ],
    )
    client.load_collection(name)
    return name


def _score_plus_ts_chain():
    return (
        FunctionChain(FunctionChainStage.L2_RERANK, name="score_plus_ts")
        .map(
            "$score",
            fn.num_combine(col("$score"), col(SCALAR_FIELD), mode="sum"),
        )
        .sort(col("$score"), desc=True, tie_break_col=col("$id"))
    )


def _hit_field(hit: dict, field: str):
    if field in hit:
        return hit[field]
    return hit.get("entity", {}).get(field)


def _assert_field_not_returned(hit: dict, field: str) -> None:
    assert field not in hit
    assert field not in hit.get("entity", {})


def _assert_search_error(
    client: MilvusClient,
    collection_name: str,
    function_chains,
    message: str,
    **kwargs,
) -> None:
    with pytest.raises(MilvusException) as exc_info:
        client.search(
            collection_name=collection_name,
            data=[[0.0, 0.0]],
            anns_field=VECTOR_FIELD,
            search_params={"metric_type": "L2", "params": {}},
            limit=3,
            function_chains=function_chains,
            **kwargs,
        )

    normalized_expected = message.replace("'", '"')
    normalized_actual = str(exc_info.value).replace("'", '"')
    assert normalized_expected in normalized_actual


def test_search_with_l2_function_chain_sdk_reranks_by_scalar_field(client):
    collection_name = _create_function_chain_collection(client)

    result = client.search(
        collection_name=collection_name,
        data=[[0.0, 0.0]],
        anns_field=VECTOR_FIELD,
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=[SCALAR_FIELD],
        function_chains=_score_plus_ts_chain(),
    )

    assert [hit["id"] for hit in result[0]] == [3, 2, 1]
    assert [_hit_field(hit, SCALAR_FIELD) for hit in result[0]] == [30, 20, 10]


def test_search_with_l2_function_chain_sdk_uses_hidden_input_field(client):
    collection_name = _create_function_chain_collection(client)

    result = client.search(
        collection_name=collection_name,
        data=[[0.0, 0.0]],
        anns_field=VECTOR_FIELD,
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=["id"],
        function_chains=_score_plus_ts_chain(),
    )

    assert [hit["id"] for hit in result[0]] == [3, 2, 1]
    for hit in result[0]:
        _assert_field_not_returned(hit, SCALAR_FIELD)


def test_search_with_l2_function_chain_sdk_temp_column_not_returned(client):
    collection_name = _create_function_chain_collection(client)
    chain = (
        FunctionChain(FunctionChainStage.L2_RERANK, name="l2_temp_score")
        .map(
            "tmp_score",
            fn.num_combine(col("$score"), col(SCALAR_FIELD), mode="sum"),
        )
        .map(
            "$score",
            fn.num_combine(col("tmp_score"), col("$score"), mode="sum"),
        )
        .sort(col("$score"), desc=True, tie_break_col=col("$id"))
    )

    result = client.search(
        collection_name=collection_name,
        data=[[0.0, 0.0]],
        anns_field=VECTOR_FIELD,
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=[SCALAR_FIELD],
        function_chains=chain,
    )

    assert [hit["id"] for hit in result[0]] == [3, 2, 1]
    assert [_hit_field(hit, SCALAR_FIELD) for hit in result[0]] == [30, 20, 10]
    for hit in result[0]:
        _assert_field_not_returned(hit, "tmp_score")


def test_search_with_l2_function_chain_sdk_limit_op(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="l2_limit",
    ).limit(2)

    result = client.search(
        collection_name=collection_name,
        data=[[0.0, 0.0]],
        anns_field=VECTOR_FIELD,
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        function_chains=chain,
    )

    assert len(result[0]) == 2


def test_search_rejects_l2_function_chain_write_readonly_system_column(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="bad_l2_write_id",
    ).map(
        "$id",
        fn.num_combine(col("$score"), col(SCALAR_FIELD), mode="sum"),
    )

    _assert_search_error(
        client,
        collection_name,
        chain,
        'system output "$id" is not writable',
    )


def test_search_rejects_l2_function_chain_reserved_temp_output(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="bad_l2_reserved_temp_output",
    ).map(
        "$tmp_score",
        fn.num_combine(col("$score"), col(SCALAR_FIELD), mode="sum"),
    )

    _assert_search_error(
        client,
        collection_name,
        chain,
        'system output "$tmp_score" is not writable',
    )


def test_search_rejects_l2_function_chain_read_internal_system_input(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="bad_l2_seg_offset_input",
    ).map(
        "$score",
        fn.num_combine(col("$seg_offset"), col("$score"), mode="sum"),
    )

    _assert_search_error(
        client,
        collection_name,
        chain,
        'system input "$seg_offset" is not supported',
    )


def test_search_rejects_l2_function_chain_read_unknown_system_input(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="bad_l2_unknown_system_input",
    ).map(
        "$score",
        fn.num_combine(col("$tmp_score"), col("$score"), mode="sum"),
    )

    _assert_search_error(
        client,
        collection_name,
        chain,
        'system input "$tmp_score" is not supported',
    )


def test_search_rejects_l2_function_chain_with_function_score(client):
    collection_name = _create_function_chain_collection(client)
    function = Function(
        name="boost_ts",
        function_type=FunctionType.RERANK,
        input_field_names=[],
        output_field_names=[],
        params={"reranker": "boost", "weight": "1.5"},
    )
    function_score = FunctionScore(functions=[function])

    _assert_search_error(
        client,
        collection_name,
        _score_plus_ts_chain(),
        "function_chains and ranker cannot be used together",
        ranker=function_score,
    )


def test_search_rejects_l2_function_chain_with_order_by(client):
    collection_name = _create_function_chain_collection(client)

    _assert_search_error(
        client,
        collection_name,
        _score_plus_ts_chain(),
        "order_by and function rerank cannot be used together",
        order_by_fields=[{"field": SCALAR_FIELD, "order": "asc"}],
    )
