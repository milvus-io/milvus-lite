import pytest

from pymilvus import Function, FunctionScore, FunctionType
from pymilvus.client.prepare import Prepare

from milvus_lite.adapter.grpc.translators.search import parse_search_request
from milvus_lite.rerank.boost import apply_boost_ranker


def _boost_function(**params):
    return Function(
        name=params.pop("name", "boost"),
        input_field_names=[],
        output_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "boost", **params},
    )


def test_parse_search_request_decodes_boost_ranker():
    ranker = _boost_function(
        filter="doctype == 'abstract'",
        weight=0.5,
        random_score={"seed": 126, "field": "id"},
    )
    req = Prepare.search_requests_with_expr(
        collection_name="c",
        anns_field="vec",
        param={},
        limit=5,
        data=[[1.0, 0.0]],
        ranker=ranker,
    )

    parsed = parse_search_request(req)

    assert parsed["ranker"]["functions"][0]["params"]["reranker"] == "boost"
    assert parsed["ranker"]["functions"][0]["params"]["weight"] == 0.5
    assert parsed["ranker"]["functions"][0]["params"]["random_score"] == {
        "seed": 126,
        "field": "id",
    }


def test_boost_ranker_filter_multiplies_lower_is_better_distance():
    results = [[
        {"id": 1, "distance": 0.0, "entity": {"doctype": "abstract"}},
        {"id": 2, "distance": 0.10, "entity": {"doctype": "body"}},
        {"id": 3, "distance": 0.20, "entity": {"doctype": "abstract"}},
    ]]
    ranker = {
        "functions": [{
            "name": "boost",
            "params": {
                "reranker": "boost",
                "filter": "doctype == 'abstract'",
                "weight": 0.1,
            },
        }],
        "params": {"boost_mode": "multiply", "function_mode": "multiply"},
    }

    from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
    from milvus_lite.search.filter import compile_filter
    from milvus_lite.search.filter.eval.python_backend import _eval_row

    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("doctype", DataType.VARCHAR, max_length=64),
    ])

    boosted = apply_boost_ranker(
        results,
        ranker,
        metric_type="L2",
        pk_name="id",
        compile_filter=lambda expr: compile_filter(expr, schema),
        row_matches_filter=lambda row, compiled: bool(_eval_row(compiled.ast, row)),
    )

    assert [h["id"] for h in boosted[0]] == [1, 3, 2]
    assert boosted[0][1]["distance"] == pytest.approx(0.02)


def test_boost_ranker_cosine_weight_boosts_higher_is_better_score():
    results = [[
        {"id": 1, "distance": 0.10, "entity": {"doctype": "body"}},
        {"id": 2, "distance": 0.40, "entity": {"doctype": "abstract"}},
    ]]
    ranker = {
        "functions": [{
            "name": "boost",
            "params": {
                "reranker": "boost",
                "filter": "doctype == 'abstract'",
                "weight": 2.0,
            },
        }],
        "params": {"boost_mode": "multiply", "function_mode": "multiply"},
    }

    from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
    from milvus_lite.search.filter import compile_filter
    from milvus_lite.search.filter.eval.python_backend import _eval_row

    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("doctype", DataType.VARCHAR, max_length=64),
    ])

    boosted = apply_boost_ranker(
        results,
        ranker,
        metric_type="COSINE",
        pk_name="id",
        compile_filter=lambda expr: compile_filter(expr, schema),
        row_matches_filter=lambda row, compiled: bool(_eval_row(compiled.ast, row)),
    )

    assert [h["id"] for h in boosted[0]] == [2, 1]
    assert boosted[0][0]["distance"] == pytest.approx(-0.2)


def test_boost_ranker_function_score_sum_mode_is_deterministic():
    results = [[
        {"id": 1, "distance": 1.0, "entity": {}},
        {"id": 2, "distance": 1.0, "entity": {}},
    ]]
    ranker = {
        "functions": [
            {"name": "fixed", "params": {"reranker": "boost", "weight": 0.8}},
            {
                "name": "random",
                "params": {
                    "reranker": "boost",
                    "weight": 0.4,
                    "random_score": {"seed": 126, "field": "id"},
                },
            },
        ],
        "params": {"boost_mode": "multiply", "function_mode": "sum"},
    }

    first = apply_boost_ranker(
        results, ranker, metric_type="L2", pk_name="id",
        compile_filter=lambda _expr: None,
        row_matches_filter=lambda _row, _compiled: True,
    )
    second = apply_boost_ranker(
        results, ranker, metric_type="L2", pk_name="id",
        compile_filter=lambda _expr: None,
        row_matches_filter=lambda _row, _compiled: True,
    )

    assert first == second
    for hit in first[0]:
        assert 0.8 <= hit["distance"] <= 1.2


def test_parse_search_request_decodes_function_score_modes():
    ranker = FunctionScore(
        functions=[
            _boost_function(name="fixed", weight=0.8),
            _boost_function(
                name="random",
                weight=0.4,
                random_score={"seed": 126},
            ),
        ],
        params={"boost_mode": "Multiply", "function_mode": "Sum"},
    )
    req = Prepare.search_requests_with_expr(
        collection_name="c",
        anns_field="vec",
        param={},
        limit=5,
        data=[[1.0, 0.0]],
        ranker=ranker,
    )

    parsed = parse_search_request(req)

    assert parsed["ranker"]["params"] == {
        "boost_mode": "multiply",
        "function_mode": "sum",
    }
    assert len(parsed["ranker"]["functions"]) == 2
