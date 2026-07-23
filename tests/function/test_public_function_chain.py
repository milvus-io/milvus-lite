"""Tests for public function-chain compilation and expressions."""

import math
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest
from pymilvus.grpc_gen import schema_pb2

from milvus_lite.adapter.grpc.function_chain import (
    SearchFunctionChainPlan,
    execute_search_function_chain,
    hit_score_for_chain,
    merge_internal_output_fields,
    prepare_search_function_chain,
)
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.function.compiler import compile_function_chain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.expr import NumCombineExpr
from milvus_lite.function.expr.decay_expr import DecayExpr
from milvus_lite.function.ops.limit_op import LimitOp
from milvus_lite.function.ops.map_op import (
    ColumnBinding,
    LiteralBinding,
    MapOp,
)
from milvus_lite.function.ops.sort_op import SortOp
from milvus_lite.function.repr import (
    ChainRepr,
    ColumnArg,
    ExprRepr,
    LiteralArg,
    OpRepr,
    build_chain_info,
)
from milvus_lite.function.types import STAGE_L2_RERANK, FuncContext
from milvus_lite.function.validator import ValidatedChain, validate_function_chain
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def _validated(*ops: OpRepr) -> ValidatedChain:
    chain = ChainRepr(
        name="public",
        stage="FunctionChainStageL2Rerank",
        ops=ops,
        info=build_chain_info(ops),
    )
    return ValidatedChain(chain, ())


def _map(expr: ExprRepr, output: str = "$score") -> OpRepr:
    read_names = tuple(
        arg.name for arg in expr.args if isinstance(arg, ColumnArg)
    )
    return OpRepr(
        op="map",
        expr=expr,
        inputs=(),
        outputs=(output,),
        params={},
        read_names=read_names,
        write_names=(output,),
    )


def _search_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
            FieldSchema("title", DataType.VARCHAR, max_length=512),
            FieldSchema("popularity", DataType.FLOAT),
            FieldSchema("category", DataType.VARCHAR, max_length=64),
        ]
    )


def _column(name: str) -> schema_pb2.FunctionChainExprArg:
    return schema_pb2.FunctionChainExprArg(
        column=schema_pb2.FunctionChainColumnArg(name=name)
    )


def _search_chain_proto(*ops) -> schema_pb2.FunctionChain:
    return schema_pb2.FunctionChain(
        name="search-rerank",
        stage=schema_pb2.FunctionChainStageL2Rerank,
        ops=ops,
    )


def _limit_proto(limit: int, offset: int = 0):
    params = {
        "limit": schema_pb2.FunctionParamValue(int64_value=limit),
    }
    if offset:
        params["offset"] = schema_pb2.FunctionParamValue(int64_value=offset)
    return schema_pb2.FunctionChainOp(op="limit", params=params)


def _hidden_field_chain_proto(*, limit: int | None = None):
    ops = [
        schema_pb2.FunctionChainOp(
            op="map",
            expr=schema_pb2.FunctionChainExpr(
                name="num_combine",
                args=[_column("$score"), _column("popularity")],
                params={
                    "mode": schema_pb2.FunctionParamValue(
                        string_value="sum"
                    )
                },
            ),
            outputs=["tmp_score"],
        ),
        schema_pb2.FunctionChainOp(
            op="map",
            expr=schema_pb2.FunctionChainExpr(
                name="round_decimal",
                args=[_column("tmp_score")],
                params={
                    "decimal": schema_pb2.FunctionParamValue(int64_value=2)
                },
            ),
            outputs=["$score"],
        ),
    ]
    if limit is not None:
        ops.append(_limit_proto(limit))
    return _search_chain_proto(*ops)


def _prepare_plan(
    *,
    requested_output_fields=("title",),
    limit: int | None = None,
) -> SearchFunctionChainPlan:
    return prepare_search_function_chain(
        function_chains=[_hidden_field_chain_proto(limit=limit)],
        has_function_score=False,
        schema=_search_schema(),
        num_queries=2,
        requested_output_fields=(
            None
            if requested_output_fields is None
            else list(requested_output_fields)
        ),
    )


def test_merge_internal_output_fields_preserves_user_order():
    assert merge_internal_output_fields(
        ["title", "category"],
        ("popularity", "title", "popularity"),
    ) == ["title", "category", "popularity"]
    assert merge_internal_output_fields(None, ("popularity",)) is None


def test_prepare_empty_function_chains_returns_no_plan():
    assert prepare_search_function_chain(
        function_chains=[],
        has_function_score=True,
        schema=_search_schema(),
        num_queries=1,
        requested_output_fields=["title"],
    ) is None


def test_prepare_rejects_function_score_conflict_with_exact_message():
    with pytest.raises(SchemaValidationError) as error:
        prepare_search_function_chain(
            function_chains=[_hidden_field_chain_proto()],
            has_function_score=True,
            schema=_search_schema(),
            num_queries=1,
            requested_output_fields=["title"],
        )

    assert str(error.value) == (
        "function_score and function_chains cannot be used together"
    )


def test_prepare_requires_exactly_one_function_chain():
    chain = _hidden_field_chain_proto()

    with pytest.raises(
        SchemaValidationError,
        match="ordinary search supports exactly one function chain",
    ):
        prepare_search_function_chain(
            function_chains=[chain, chain],
            has_function_score=False,
            schema=_search_schema(),
            num_queries=1,
            requested_output_fields=["title"],
        )


def test_prepare_decodes_validates_compiles_and_freezes_plan():
    requested = ["title"]

    plan = prepare_search_function_chain(
        function_chains=[_hidden_field_chain_proto(limit=1)],
        has_function_score=False,
        schema=_search_schema(),
        num_queries=1,
        requested_output_fields=requested,
    )
    requested.append("category")

    assert plan.required_fields == ("popularity",)
    assert plan.requested_output_fields == ("title",)
    assert [type(op) for op in plan.chain.operators] == [
        MapOp,
        MapOp,
        LimitOp,
    ]
    with pytest.raises(FrozenInstanceError):
        plan.required_fields = ()


def test_execute_projects_requested_fields_and_hides_internal_values():
    plan = _prepare_plan(requested_output_fields=("category", "title"))
    results = [[{
        "id": 1,
        "distance": 0.2,
        "entity": {
            "title": "one",
            "popularity": 3.0,
            "category": "news",
        },
    }]]

    reranked = execute_search_function_chain(
        plan,
        results,
        metric_type="IP",
        schema=_search_schema(),
        primary_field_name="id",
    )

    assert reranked == [[{
        "id": 1,
        "distance": 3.2,
        "entity": {"category": "news", "title": "one"},
    }]]
    assert "popularity" not in reranked[0][0]["entity"]
    assert "tmp_score" not in reranked[0][0]["entity"]


def test_execute_none_projection_returns_all_non_primary_schema_fields():
    plan = _prepare_plan(requested_output_fields=None)
    results = [[{
        "id": 1,
        "distance": 0.2,
        "entity": {
            "id": 999,
            "vector": [0.1, 0.2],
            "title": "one",
            "popularity": 3.0,
            "category": "news",
        },
    }]]

    reranked = execute_search_function_chain(
        plan,
        results,
        metric_type="COSINE",
        schema=_search_schema(),
        primary_field_name="id",
    )

    assert reranked[0][0]["id"] == 1
    assert reranked[0][0]["entity"] == {
        "vector": [0.1, 0.2],
        "title": "one",
        "popularity": 3.0,
        "category": "news",
    }


def test_execute_dynamic_temp_collision_isolated_per_query_chunk():
    collision = schema_pb2.FunctionChainOp(
        op="map",
        expr=schema_pb2.FunctionChainExpr(
            name="num_combine",
            args=[_column("$score"), _column("popularity")],
            params={
                "mode": schema_pb2.FunctionParamValue(string_value="sum")
            },
        ),
        outputs=["dynamic_tag"],
    )
    plan = prepare_search_function_chain(
        function_chains=[_search_chain_proto(collision)],
        has_function_score=False,
        schema=_search_schema(),
        num_queries=2,
        requested_output_fields=["dynamic_tag"],
    )

    reranked = execute_search_function_chain(
        plan,
        [
            [{
                "id": 1,
                "distance": 0.2,
                "entity": {"popularity": 1.0, "dynamic_tag": "first"},
            }],
            [{
                "id": 1,
                "distance": 0.3,
                "entity": {"popularity": 2.0, "dynamic_tag": "second"},
            }],
        ],
        metric_type="IP",
        schema=_search_schema(),
        primary_field_name="id",
    )

    assert reranked[0][0]["entity"] == {"dynamic_tag": "first"}
    assert reranked[1][0]["entity"] == {"dynamic_tag": "second"}


def test_primary_field_name_is_runtime_id_alias_and_not_hidden_output():
    primary_score = schema_pb2.FunctionChainOp(
        op="map",
        expr=schema_pb2.FunctionChainExpr(
            name="num_combine",
            args=[_column("$score"), _column("id")],
            params={
                "mode": schema_pb2.FunctionParamValue(string_value="sum")
            },
        ),
        outputs=["$score"],
    )
    plan = prepare_search_function_chain(
        function_chains=[_search_chain_proto(primary_score)],
        has_function_score=False,
        schema=_search_schema(),
        num_queries=1,
        requested_output_fields=["title"],
    )

    assert plan.required_fields == ()
    assert merge_internal_output_fields(
        ["title"], plan.required_fields
    ) == ["title"]

    reranked = execute_search_function_chain(
        plan,
        [[{
            "id": 2,
            "distance": 0.5,
            "entity": {"title": "two"},
        }]],
        metric_type="IP",
        schema=_search_schema(),
        primary_field_name="id",
    )

    assert reranked == [[{
        "id": 2,
        "distance": 2.5,
        "entity": {"title": "two"},
    }]]


def test_execute_explicit_primary_field_is_excluded_from_entity():
    plan = _prepare_plan(requested_output_fields=("id", "title"))

    reranked = execute_search_function_chain(
        plan,
        [[{
            "id": 1,
            "distance": 0.2,
            "entity": {"id": 999, "title": "one", "popularity": 1.0},
        }]],
        metric_type="L2",
        schema=_search_schema(),
        primary_field_name="id",
    )

    assert reranked[0][0]["entity"] == {"title": "one"}


def test_execute_does_not_mutate_input_results_or_entities():
    plan = _prepare_plan()
    results = [[{
        "id": 1,
        "distance": 0.2,
        "entity": {"title": "one", "popularity": 3.0},
    }]]
    original = deepcopy(results)

    execute_search_function_chain(
        plan,
        results,
        metric_type="IP",
        schema=_search_schema(),
        primary_field_name="id",
    )

    assert results == original


@pytest.mark.parametrize(
    ("metric_type", "distance", "expected"),
    [
        ("BM25", 2.5, -2.5),
        ("COSINE", 0.8, 0.8),
        ("IP", 0.7, 0.7),
        ("L2", 1.2, 1.2),
        ("unknown", 4.0, 4.0),
    ],
)
def test_hit_score_for_chain_matches_search_metric_behavior(
    metric_type,
    distance,
    expected,
):
    assert hit_score_for_chain(
        {"distance": distance}, metric_type
    ) == expected


def test_execute_uses_bm25_score_conversion_for_final_distance():
    plan = prepare_search_function_chain(
        function_chains=[_search_chain_proto(_limit_proto(1))],
        has_function_score=False,
        schema=_search_schema(),
        num_queries=1,
        requested_output_fields=["title"],
    )

    reranked = execute_search_function_chain(
        plan,
        [[{"id": 1, "distance": 2.5, "entity": {"title": "one"}}]],
        metric_type="bm25",
        schema=_search_schema(),
        primary_field_name="id",
    )

    assert reranked[0][0]["distance"] == -2.5


@pytest.mark.parametrize("source", ["metadata", "entity"])
def test_execute_preserves_group_by_value_without_leaking_hidden_field(source):
    plan = _prepare_plan()
    hit = {
        "id": 1,
        "distance": 0.2,
        "entity": {"title": "one", "popularity": 3.0},
    }
    if source == "metadata":
        hit["_group_by_value"] = "news"
    else:
        hit["entity"]["category"] = "news"

    reranked = execute_search_function_chain(
        plan,
        [[hit]],
        metric_type="IP",
        schema=_search_schema(),
        primary_field_name="id",
        group_by_field="category",
    )

    assert reranked[0][0]["_group_by_value"] == "news"
    assert reranked[0][0]["entity"] == {"title": "one"}


def test_execute_handles_multiple_queries_empty_chunks_and_limit():
    plan = _prepare_plan(limit=1)
    results = [
        [
            {
                "id": 1,
                "distance": 0.2,
                "entity": {"title": "one", "popularity": 1.0},
            },
            {
                "id": 2,
                "distance": 0.3,
                "entity": {"title": "two", "popularity": 2.0},
            },
        ],
        [],
        [{
            "id": 3,
            "distance": 0.4,
            "entity": {"title": "three", "popularity": 3.0},
        }],
    ]

    reranked = execute_search_function_chain(
        plan,
        results,
        metric_type="IP",
        schema=_search_schema(),
        primary_field_name="id",
    )

    assert [[hit["id"] for hit in chunk] for chunk in reranked] == [
        [1],
        [],
        [3],
    ]
    assert execute_search_function_chain(
        plan,
        [],
        metric_type="IP",
        schema=_search_schema(),
        primary_field_name="id",
    ) == []


class _MalformedChain:
    def __init__(self, row):
        self._row = row

    def execute(self, dataframe):
        return DataFrame([[dict(self._row)]])


@pytest.mark.parametrize(
    ("row", "missing"),
    [
        ({"$score": 1.0}, "$id"),
        ({"$id": 1}, "$score"),
    ],
)
def test_execute_fails_clearly_for_malformed_runtime_output(row, missing):
    plan = SearchFunctionChainPlan(
        chain=_MalformedChain(row),
        required_fields=(),
        requested_output_fields=("title",),
    )

    with pytest.raises(ValueError) as error:
        execute_search_function_chain(
            plan,
            [[{"id": 1, "distance": 0.2, "entity": {"title": "one"}}]],
            metric_type="IP",
            schema=_search_schema(),
            primary_field_name="id",
        )

    assert f"missing required {missing}" in str(error.value)


def test_compile_and_execute_ordered_public_chain():
    map_op = _map(
        ExprRepr(
            "num_combine",
            (ColumnArg("$score"), ColumnArg("popularity"), LiteralArg(1.0)),
            {"mode": "sum"},
        )
    )
    sort_op = OpRepr(
        op="sort",
        expr=None,
        inputs=("$score", "$id"),
        outputs=(),
        params={
            "column": "$score",
            "desc": True,
            "tie_break_col": "$id",
        },
        read_names=("$score", "$id"),
        write_names=(),
    )
    limit_op = OpRepr(
        op="limit",
        expr=None,
        inputs=(),
        outputs=(),
        params={"limit": 2},
        read_names=(),
        write_names=(),
    )

    compiled = compile_function_chain(_validated(map_op, sort_op, limit_op))
    result = compiled.execute(
        DataFrame(
            [[
                {"$id": 1, "$score": 0.2, "popularity": 1.0},
                {"$id": 2, "$score": 0.1, "popularity": 5.0},
                {"$id": 3, "$score": 0.3, "popularity": 2.0},
            ]]
        )
    )

    assert [type(op) for op in compiled.operators] == [MapOp, SortOp, LimitOp]
    assert compiled.stage == STAGE_L2_RERANK
    assert [row["$id"] for row in result.chunk(0)] == [2, 3]
    assert [row["$score"] for row in result.chunk(0)] == [6.1, 3.3]
    bindings = compiled.operators[0].input_bindings
    assert bindings == [
        ColumnBinding("$score"),
        ColumnBinding("popularity"),
        LiteralBinding(1.0),
    ]


def test_compile_without_sort_preserves_order_and_adds_no_implicit_tail():
    compiled = compile_function_chain(
        _validated(
            _map(
                ExprRepr(
                    "round_decimal",
                    (ColumnArg("$score"),),
                    {"decimal": 1},
                )
            )
        )
    )
    result = compiled.execute(
        DataFrame(
            [[
                {"$id": 2, "$score": 0.24},
                {"$id": 1, "$score": 0.26},
                {"$id": 3, "$score": 0.25},
            ]]
        )
    )

    assert [type(op) for op in compiled.operators] == [MapOp]
    assert [row["$id"] for row in result.chunk(0)] == [2, 1, 3]
    assert [row["$score"] for row in result.chunk(0)] == [0.2, 0.3, 0.2]


def test_compile_sort_without_tie_break_is_stable():
    sort_op = OpRepr(
        op="sort",
        expr=None,
        inputs=("$score",),
        outputs=(),
        params={"column": "$score", "desc": True},
        read_names=("$score",),
        write_names=(),
    )
    result = compile_function_chain(_validated(sort_op)).execute(
        DataFrame(
            [[
                {"$id": 3, "$score": 1.0},
                {"$id": 1, "$score": 1.0},
                {"$id": 2, "$score": 0.5},
            ]]
        )
    )

    assert [row["$id"] for row in result.chunk(0)] == [3, 1, 2]


def test_compile_sort_with_optional_tie_break():
    sort_op = OpRepr(
        op="sort",
        expr=None,
        inputs=("$score", "$id"),
        outputs=(),
        params={"column": "$score", "tie_break_col": "$id"},
        read_names=("$score", "$id"),
        write_names=(),
    )
    result = compile_function_chain(_validated(sort_op)).execute(
        DataFrame(
            [[
                {"$id": 3, "$score": 1.0},
                {"$id": 1, "$score": 1.0},
                {"$id": 2, "$score": 0.5},
            ]]
        )
    )

    assert [row["$id"] for row in result.chunk(0)] == [1, 3, 2]


@pytest.mark.parametrize("function", ["gauss", "exp", "linear"])
def test_compile_decay_and_round_expressions(function):
    decay_op = _map(
        ExprRepr(
            "decay",
            (ColumnArg("popularity"),),
            {"function": function, "origin": 0.0, "scale": 10.0},
        ),
        output="decayed",
    )
    round_op = _map(
        ExprRepr(
            "round_decimal",
            (ColumnArg("decayed"),),
            {"decimal": 2},
        )
    )
    ops = (decay_op, round_op)
    chain = ChainRepr(
        name="public",
        stage="FunctionChainStageL2Rerank",
        ops=ops,
        info=build_chain_info(ops),
    )
    validated = validate_function_chain(chain, _search_schema(), 1)
    result = compile_function_chain(validated).execute(
        DataFrame(
            [[
                {"popularity": 0.0},
                {"popularity": 10.0},
                {"popularity": None},
            ]]
        )
    )

    assert [row["$score"] for row in result.chunk(0)] == [1.0, 0.5, 0.0]


def test_compile_num_combine_accepts_frozen_weight_sequence():
    compiled = compile_function_chain(
        _validated(
            _map(
                ExprRepr(
                    "num_combine",
                    (ColumnArg("left"), ColumnArg("right")),
                    {"mode": "weighted", "weights": [0.25, 0.75]},
                )
            )
        )
    )
    result = compiled.execute(DataFrame([[{"left": 2.0, "right": 4.0}]]))

    assert result.chunk(0)[0]["$score"] == 3.5


@dataclass
class _RerankResult:
    index: int
    relevance_score: float


class _MockProvider:
    def __init__(self) -> None:
        self.calls = []

    def rerank(self, query, documents, top_n=None):
        self.calls.append((query, documents, top_n))
        return [
            _RerankResult(index=index, relevance_score=index + 0.25)
            for index in range(len(documents))
        ]


def test_compile_rerank_model_copies_params_and_uses_per_chunk_queries(
    monkeypatch,
):
    provider = _MockProvider()
    factory_params = []

    def _factory(params):
        factory_params.append(params)
        return provider

    monkeypatch.setattr(
        "milvus_lite.rerank.factory.create_rerank_provider",
        _factory,
    )
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("text"),),
        {
            "provider": "cohere",
            "model_name": "test-model",
            "queries": ["first query", "second query"],
        },
    )
    compiled = compile_function_chain(_validated(_map(expr)))

    result = compiled.execute(
        DataFrame(
            [
                [{"text": "a"}, {"text": "b"}],
                [{"text": "c"}],
            ]
        )
    )

    assert factory_params == [
        {"provider": "cohere", "model_name": "test-model"}
    ]
    assert provider.calls == [
        ("first query", ["a", "b"], 2),
        ("second query", ["c"], 1),
    ]
    assert [row["$score"] for row in result.chunk(0)] == [0.25, 1.25]
    assert [row["$score"] for row in result.chunk(1)] == [0.25]
    assert compiled.operators[0].expr.query_texts == [
        "first query",
        "second query",
    ]


def test_prepare_rejects_public_rerank_credentials():
    queries = schema_pb2.FunctionParamValue(
        array_value=schema_pb2.FunctionParamArray(
            values=[schema_pb2.FunctionParamValue(string_value="query")]
        )
    )
    chain = _search_chain_proto(
        schema_pb2.FunctionChainOp(
            op="map",
            expr=schema_pb2.FunctionChainExpr(
                name="rerank_model",
                args=[_column("title")],
                params={
                    "provider": schema_pb2.FunctionParamValue(
                        string_value="cohere"
                    ),
                    "queries": queries,
                    "api_key": schema_pb2.FunctionParamValue(
                        string_value="secret"
                    ),
                },
            ),
            outputs=["$score"],
        )
    )

    with pytest.raises(SchemaValidationError, match="credentials"):
        prepare_search_function_chain(
            function_chains=[chain],
            has_function_score=False,
            schema=_search_schema(),
            num_queries=1,
            requested_output_fields=None,
        )


def test_prepare_preserves_rerank_provider_configuration_errors(monkeypatch):
    queries = schema_pb2.FunctionParamValue(
        array_value=schema_pb2.FunctionParamArray(
            values=[schema_pb2.FunctionParamValue(string_value="query")]
        )
    )
    chain = _search_chain_proto(
        schema_pb2.FunctionChainOp(
            op="map",
            expr=schema_pb2.FunctionChainExpr(
                name="rerank_model",
                args=[_column("title")],
                params={
                    "provider": schema_pb2.FunctionParamValue(
                        string_value="cohere"
                    ),
                    "queries": queries,
                },
            ),
            outputs=["$score"],
        )
    )

    def _factory(_params):
        raise ValueError("Cohere API key is required")

    monkeypatch.setattr(
        "milvus_lite.rerank.factory.create_rerank_provider",
        _factory,
    )

    with pytest.raises(ValueError, match="Cohere API key is required"):
        prepare_search_function_chain(
            function_chains=[chain],
            has_function_score=False,
            schema=_search_schema(),
            num_queries=1,
            requested_output_fields=["title"],
        )


def test_compile_rejects_unknown_operator_clearly():
    op = OpRepr(
        op="select",
        expr=None,
        inputs=(),
        outputs=(),
        params={},
        read_names=(),
        write_names=(),
    )

    with pytest.raises(ValueError, match="unsupported.*operator.*select"):
        compile_function_chain(_validated(op))


def test_compile_rejects_unknown_expression_clearly():
    expr = ExprRepr("mystery", (ColumnArg("$score"),), {})

    with pytest.raises(ValueError, match="unsupported.*expression.*mystery"):
        compile_function_chain(_validated(_map(expr)))


def test_compile_detects_impossible_map_without_expression():
    op = OpRepr(
        op="map",
        expr=None,
        inputs=(),
        outputs=("$score",),
        params={},
        read_names=("$score",),
        write_names=("$score",),
    )

    with pytest.raises(ValueError, match="map operator.*expression"):
        compile_function_chain(_validated(op))


def test_decay_accepts_timestamptz_values():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expr = DecayExpr(
        function="exp",
        origin=origin.timestamp(),
        scale=3600,
        decay=0.5,
    )

    result = expr.execute(
        _ctx(),
        [[origin, datetime(2026, 1, 1, 1, tzinfo=timezone.utc), None]],
    )

    assert result[0] == pytest.approx([1.0, 0.5, 0.0])


def test_decay_rejects_naive_datetime_values():
    expr = DecayExpr("exp", origin=0.0, scale=3600, decay=0.5)

    with pytest.raises(ValueError, match="timezone-aware"):
        expr.execute(_ctx(), [[datetime(2026, 1, 1)]])


def test_num_combine_public_metadata():
    assert NumCombineExpr.name == "num_combine"
    assert NumCombineExpr.supported_stages == frozenset({STAGE_L2_RERANK})


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("multiply", [6.0, 20.0]),
        ("sum", [5.0, 9.0]),
        ("max", [3.0, 5.0]),
        ("min", [2.0, 4.0]),
        ("avg", [2.5, 4.5]),
        ("weighted", [2.75, 4.75]),
    ],
)
def test_num_combine_modes(mode, expected):
    weights = (0.25, 0.75) if mode == "weighted" else None
    result = NumCombineExpr(mode, weights).execute(
        _ctx(), [[2.0, 4.0], [3.0, 5.0]]
    )
    assert result == [expected]


def test_num_combine_defaults_to_sum():
    assert NumCombineExpr().execute(_ctx(), [[2.0], [3.0]]) == [[5.0]]


def test_num_combine_detaches_and_converts_mutable_weights():
    weights = [1, 0.5]
    expr = NumCombineExpr("weighted", weights)

    weights[0] = 100

    assert expr.execute(_ctx(), [[2.0], [4.0]]) == [[4.0]]
    assert expr._weights == [1.0, 0.5]


def test_num_combine_none_becomes_zero():
    result = NumCombineExpr("sum").execute(
        _ctx(), [[None, 1.0], [2.0, None]]
    )
    assert result == [[0.0, 0.0]]


def test_num_combine_empty_columns_return_empty_output():
    assert NumCombineExpr().execute(_ctx(), [[], []]) == [[]]


def test_num_combine_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown num_combine mode"):
        NumCombineExpr("median")


@pytest.mark.parametrize("weights", [None, [], ()])
def test_weighted_num_combine_requires_nonempty_weights(weights):
    with pytest.raises(ValueError, match="requires weights"):
        NumCombineExpr("weighted", weights)


@pytest.mark.parametrize(
    "weights",
    [
        "12",
        [1.0, "bad"],
        [1.0, True],
        [1.0, float("nan")],
        [1.0, float("inf")],
        [1.0, float("-inf")],
    ],
)
def test_weighted_num_combine_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights must be finite numeric values"):
        NumCombineExpr("weighted", weights)


def test_nonweighted_num_combine_rejects_weights():
    with pytest.raises(ValueError, match="weights require weighted mode"):
        NumCombineExpr("sum", [1.0, 1.0])


@pytest.mark.parametrize("inputs", [[], [[1.0]]])
def test_num_combine_requires_at_least_two_input_columns(inputs):
    with pytest.raises(ValueError, match="at least two input columns"):
        NumCombineExpr().execute(_ctx(), inputs)


def test_num_combine_rejects_unequal_column_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        NumCombineExpr().execute(_ctx(), [[1.0], [2.0, 3.0]])


def test_weighted_num_combine_rejects_input_count_mismatch():
    expr = NumCombineExpr("weighted", [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="weights must match input column count"):
        expr.execute(_ctx(), [[1.0], [2.0]])


@pytest.mark.parametrize(
    "value",
    [True, "bad", object(), float("nan"), float("inf"), float("-inf")],
)
def test_num_combine_rejects_invalid_row_values(value):
    with pytest.raises(ValueError, match="finite numeric non-bool"):
        NumCombineExpr().execute(_ctx(), [[1.0], [value]])


def test_num_combine_accepts_finite_numeric_row_values():
    result = NumCombineExpr("avg").execute(_ctx(), [[1, 2.5], [3, 4.5]])

    assert result == [[2.0, 3.5]]
    assert all(math.isfinite(value) for value in result[0])


def test_num_combine_average_avoids_intermediate_overflow():
    result = NumCombineExpr("avg").execute(
        _ctx(), [[1e308], [1e308]]
    )

    assert result == [[1e308]]


@pytest.mark.parametrize(
    ("mode", "weights"),
    [
        ("sum", None),
        ("multiply", None),
        ("weighted", [1.0, 1.0]),
    ],
)
def test_num_combine_rejects_non_finite_results(mode, weights):
    with pytest.raises(ValueError, match="produced a non-finite result"):
        NumCombineExpr(mode, weights).execute(
            _ctx(), [[1e308], [1e308]]
        )
