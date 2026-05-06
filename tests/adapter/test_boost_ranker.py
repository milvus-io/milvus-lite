import pytest

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionScore,
    FunctionType,
    MilvusClient,
)


def _make_boost_function(name="boost", weight=1.5, filter_expr=None):
    params = {"reranker": "boost", "weight": weight}
    if filter_expr is not None:
        params["filter"] = filter_expr
    return Function(
        name=name,
        input_field_names=[],
        output_field_names=[],
        function_type=FunctionType.RERANK,
        params=params,
    )


def _make_function_score(functions, params=None):
    if not isinstance(functions, list):
        functions = [functions]
    return FunctionScore(functions=functions, params=params)


def _create_boost_collection(client: MilvusClient, name: str):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=2)
    schema.add_field("doctype", DataType.VARCHAR, max_length=64)
    schema.add_field("bucket", DataType.INT64)
    schema.add_field("quality", DataType.FLOAT)

    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="L2", params={})
    client.create_collection(name, schema=schema, index_params=idx)
    client.insert(name, [
        {"id": 1, "vec": [0.00, 0.0], "doctype": "abstract", "bucket": 10, "quality": 1.0},
        {"id": 2, "vec": [0.10, 0.0], "doctype": "body", "bucket": -10, "quality": 1.0},
        {"id": 3, "vec": [0.20, 0.0], "doctype": "abstract", "bucket": 20, "quality": -1.0},
        {"id": 4, "vec": [0.30, 0.0], "doctype": "body", "bucket": -20, "quality": -1.0},
    ])
    client.load_collection(name)


def test_search_with_boost_ranker_filter(milvus_client: MilvusClient):
    name = "boost_ranker_basic"
    _create_boost_collection(milvus_client, name)

    ranker = _make_boost_function(
        filter_expr="doctype == 'abstract'",
        weight=0.1,
    )

    results = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=["doctype"],
        ranker=ranker,
    )

    assert [hit["id"] for hit in results[0]] == [1, 3, 2]
    assert [hit["entity"]["doctype"] for hit in results[0]] == [
        "abstract", "abstract", "body",
    ]

    milvus_client.drop_collection(name)


def test_search_with_boost_ranker_preserves_vector_output(milvus_client: MilvusClient):
    name = "boost_ranker_vector_output"
    _create_boost_collection(milvus_client, name)

    results = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=1,
        output_fields=["vec"],
        ranker=_make_boost_function(weight=1.0),
    )

    assert results[0][0]["id"] == 1
    assert results[0][0]["entity"]["vec"] == [0.0, 0.0]

    milvus_client.drop_collection(name)


def test_search_with_boost_ranker_filter_field_not_output(
    milvus_client: MilvusClient,
):
    name = "boost_ranker_hidden_filter"
    _create_boost_collection(milvus_client, name)

    results = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.08]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=["bucket"],
        ranker=_make_boost_function(
            filter_expr="doctype == 'abstract'",
            weight=0.1,
        ),
    )

    assert [hit["id"] for hit in results[0]] == [1, 3, 2]
    for hit in results[0]:
        assert hit["entity"]["bucket"] in {10, 20, -10}
        assert "doctype" not in hit["entity"]

    milvus_client.drop_collection(name)


def test_hybrid_top_level_boost_ranker_filter_field_not_output(
    milvus_client: MilvusClient,
):
    name = "boost_ranker_hybrid_top_l0"
    _create_boost_collection(milvus_client, name)

    req = AnnSearchRequest(
        data=[[0.0, 0.08]],
        anns_field="vec",
        param={"metric_type": "L2"},
        limit=4,
    )
    results = milvus_client.hybrid_search(
        collection_name=name,
        reqs=[req],
        ranker=_make_boost_function(
            filter_expr="doctype == 'abstract'",
            weight=0.01,
        ),
        limit=3,
        output_fields=["bucket"],
    )

    assert [hit["id"] for hit in results[0]] == [1, 3, 2]
    for hit in results[0]:
        assert hit["entity"]["bucket"] in {10, 20, -10}
        assert "doctype" not in hit["entity"]

    milvus_client.drop_collection(name)


def test_search_with_boost_ranker_function_score_modes(milvus_client: MilvusClient):
    name = "boost_ranker_modes"
    _create_boost_collection(milvus_client, name)

    ranker = _make_function_score(
        [
            _make_boost_function(
                name="abstract_boost",
                filter_expr="doctype == 'abstract'",
                weight=0.1,
            ),
            _make_boost_function(
                name="positive_bucket_boost",
                filter_expr="bucket > 0",
                weight=0.1,
            ),
        ],
        params={"boost_mode": "Multiply", "function_mode": "Sum"},
    )

    results = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=4,
        output_fields=["doctype", "bucket"],
        ranker=ranker,
    )

    assert [hit["id"] for hit in results[0]] == [1, 3, 2, 4]
    for hit in results[0]:
        assert "doctype" in hit["entity"]
        assert "bucket" in hit["entity"]

    milvus_client.drop_collection(name)


def test_search_boost_ranker_with_search_filter_and_limit(milvus_client: MilvusClient):
    name = "boost_ranker_filter_limit"
    _create_boost_collection(milvus_client, name)

    ranker = _make_function_score(
        _make_boost_function(filter_expr="doctype == 'abstract'", weight=0.1)
    )
    results = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        filter="bucket >= 0",
        limit=1,
        output_fields=["bucket", "doctype"],
        ranker=ranker,
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0]["id"] == 1
    assert results[0][0]["entity"]["bucket"] >= 0

    milvus_client.drop_collection(name)


def test_search_boost_ranker_score_affected(milvus_client: MilvusClient):
    name = "boost_ranker_score"
    _create_boost_collection(milvus_client, name)

    plain = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=4,
    )
    boosted = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=4,
        ranker=_make_function_score(
            _make_boost_function(weight=10.0),
            params={"boost_mode": "Multiply"},
        ),
    )

    assert [hit["id"] for hit in plain[0]] == [hit["id"] for hit in boosted[0]]
    assert [hit["distance"] for hit in plain[0]] != [
        hit["distance"] for hit in boosted[0]
    ]

    milvus_client.drop_collection(name)


@pytest.mark.parametrize(
    ("ranker", "message"),
    [
        (_make_function_score(_make_boost_function(weight="invalid_float")), "weight"),
        (
            _make_function_score(Function(
                name="missing_weight",
                function_type=FunctionType.RERANK,
                input_field_names=[],
                output_field_names=[],
                params={"reranker": "boost"},
            )),
            "weight",
        ),
        (
            _make_function_score(_make_boost_function(
                weight=1.5,
                filter_expr="invalid_field @@@ 123",
            )),
            "@",
        ),
    ],
)
def test_search_boost_ranker_invalid_params(
    milvus_client: MilvusClient,
    ranker,
    message: str,
):
    name = "boost_ranker_invalid"
    _create_boost_collection(milvus_client, name)

    with pytest.raises(Exception, match=message):
        milvus_client.search(
            collection_name=name,
            data=[[0.0, 0.0]],
            anns_field="vec",
            search_params={"metric_type": "L2", "params": {}},
            limit=4,
            ranker=ranker,
        )

    milvus_client.drop_collection(name)
