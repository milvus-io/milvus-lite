"""Comprehensive dynamic-field regression coverage through pymilvus."""

import pytest

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionScore,
    FunctionType,
    MilvusClient,
    RRFRanker,
)


def _create_dynamic_collection(client, name):
    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=True
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=128)
    schema.add_field("bucket", DataType.VARCHAR, max_length=32)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=2)
    client.create_collection(name, schema=schema)
    client.insert(name, [
        {
            "id": 1,
            "text": "one",
            "bucket": "a",
            "vec": [0.0, 0.0],
            "page": 1,
            "source": "first",
            "active": True,
            "ratio": 1.5,
            "tags": [1, 2, 3],
            "payload": {"items": [{"score": 2}]},
            "nullable": None,
            "array_field": [0, 1, 2],
        },
        {
            "id": 2,
            "text": "two",
            "bucket": "a",
            "vec": [1.0, 0.0],
            "page": 2,
            "source": "second",
            "active": False,
            "ratio": 2.5,
            "tags": [2, 3],
            "payload": {"items": [{"score": 0}]},
            "array_field": [1, 2],
        },
        {
            "id": 3,
            "text": "three",
            "bucket": "b",
            "vec": [0.0, 1.0],
            "page": 3,
            "source": "third",
            "active": True,
            "ratio": 3.5,
            "tags": [],
            "payload": {"items": []},
            "array_field": [],
        },
    ])
    index = client.prepare_index_params()
    index.add_index(
        field_name="vec",
        index_type="BRUTE_FORCE",
        metric_type="L2",
        params={},
    )
    client.create_index(name, index)
    client.load_collection(name)
    return name


@pytest.mark.parametrize(
    "fields, expected, absent",
    [
        (["*"], {"text", "bucket", "vec", "page", "source"}, set()),
        (["$meta"], {"page", "source", "tags", "payload"}, {"text", "vec"}),
        (["page"], {"page"}, {"source", "text", "vec"}),
        (["text", "$meta"], {"text", "page", "source"}, {"vec"}),
    ],
)
def test_dynamic_get_projection(milvus_client, fields, expected, absent):
    name = _create_dynamic_collection(milvus_client, "dynamic_get_projection")

    rows = milvus_client.get(name, ids=[1], output_fields=fields)

    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert expected <= set(rows[0])
    assert set(rows[0]).isdisjoint(absent)
    milvus_client.drop_collection(name)


@pytest.mark.parametrize("fields", [["*"], ["$meta"]])
def test_dynamic_wildcard_preserves_all_json_types(milvus_client, fields):
    name = _create_dynamic_collection(milvus_client, "dynamic_wildcard_types")

    row = milvus_client.query(
        name, filter="id == 1", output_fields=fields, limit=1
    )[0]

    assert row["page"] == 1 and isinstance(row["page"], int)
    assert row["source"] == "first"
    assert row["active"] is True
    assert row["ratio"] == pytest.approx(1.5)
    assert row["tags"] == [1, 2, 3]
    assert row["payload"] == {"items": [{"score": 2}]}
    assert row["nullable"] is None
    milvus_client.drop_collection(name)


@pytest.mark.parametrize("method", ["query", "search"])
@pytest.mark.parametrize(
    "expr, expected_ids",
    [
        ('payload["items"][0]["score"] > 0', {1}),
        ('payload["items"][99]["score"] > 0', set()),
        ('payload["missing"][0] > 0', set()),
        ('source[0] == "f"', set()),
        ("array_contains(array_field, 0)", {1}),
        ("array_contains_all(array_field, [1, 2])", {1, 2}),
        ("array_contains_any(array_field, [0, 9])", {1}),
        ("array_length(array_field) == 0", {3}),
    ],
)
def test_dynamic_path_and_array_filters_through_grpc(
    milvus_client, method, expr, expected_ids
):
    name = _create_dynamic_collection(
        milvus_client, "dynamic_path_array_filter"
    )

    if method == "query":
        rows = milvus_client.query(
            name, filter=expr, output_fields=["page"], limit=10
        )
        actual_ids = {row["id"] for row in rows}
    else:
        results = milvus_client.search(
            name,
            data=[[0.0, 0.0]],
            anns_field="vec",
            search_params={"metric_type": "L2", "params": {}},
            limit=10,
            filter=expr,
            output_fields=["page"],
        )
        actual_ids = {hit["id"] for hit in results[0]}

    assert actual_ids == expected_ids
    milvus_client.drop_collection(name)


def test_dynamic_multi_query_search_with_meta_projection(milvus_client):
    name = _create_dynamic_collection(milvus_client, "dynamic_multi_query")

    results = milvus_client.search(
        name,
        data=[[0.0, 0.0], [1.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=1,
        output_fields=["$meta"],
    )

    assert len(results) == 2
    assert [hits[0]["id"] for hits in results] == [1, 2]
    for hits in results:
        assert "page" in hits[0]["entity"]
        assert "source" in hits[0]["entity"]
    milvus_client.drop_collection(name)


def test_dynamic_zero_hit_search_with_meta_projection(milvus_client):
    name = _create_dynamic_collection(milvus_client, "dynamic_zero_hit")

    results = milvus_client.search(
        name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        filter='payload["items"][99]["score"] > 0',
        output_fields=["$meta"],
    )

    assert len(results) == 1
    assert list(results[0]) == []
    milvus_client.drop_collection(name)


def test_dynamic_star_survives_grpc_group_by(milvus_client):
    name = _create_dynamic_collection(milvus_client, "dynamic_group_by")

    results = milvus_client.search(
        name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=2,
        group_by_field="bucket",
        group_size=1,
        output_fields=["*"],
    )

    assert len({hit["entity"]["bucket"] for hit in results[0]}) == 2
    assert all("page" in hit["entity"] for hit in results[0])
    assert all("source" in hit["entity"] for hit in results[0])
    milvus_client.drop_collection(name)


def test_dynamic_star_survives_grpc_ranker(milvus_client):
    name = _create_dynamic_collection(milvus_client, "dynamic_ranker")
    ranker = FunctionScore(
        functions=[Function(
            name="boost_all",
            function_type=FunctionType.RERANK,
            input_field_names=[],
            output_field_names=[],
            params={"reranker": "boost", "weight": 1.0},
        )],
        params={"boost_mode": "Multiply", "function_mode": "Multiply"},
    )

    results = milvus_client.search(
        name,
        data=[[0.0, 0.0]],
        anns_field="vec",
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=["*"],
        ranker=ranker,
    )

    assert all("page" in hit["entity"] for hit in results[0])
    assert all("source" in hit["entity"] for hit in results[0])
    milvus_client.drop_collection(name)


def test_dynamic_star_survives_hybrid_search_group_by(milvus_client):
    name = _create_dynamic_collection(milvus_client, "dynamic_hybrid")
    request = AnnSearchRequest(
        data=[[0.0, 0.0]],
        anns_field="vec",
        param={"metric_type": "L2"},
        limit=3,
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[request, request],
        ranker=RRFRanker(),
        limit=2,
        group_by_field="bucket",
        group_size=1,
        output_fields=["*"],
    )

    assert len({hit["entity"]["bucket"] for hit in results[0]}) == 2
    assert all("page" in hit["entity"] for hit in results[0])
    assert all("source" in hit["entity"] for hit in results[0])
    milvus_client.drop_collection(name)
