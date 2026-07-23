"""Tests for schema-aware public Function Chain validation."""

from dataclasses import FrozenInstanceError

import pytest

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.function.repr import (
    ChainRepr,
    ColumnArg,
    ExprRepr,
    LiteralArg,
    OpRepr,
    build_chain_info,
)
from milvus_lite.function.validator import (
    ValidatedChain,
    ValueKind,
    validate_function_chain,
)
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def _schema(primary_type=DataType.INT64):
    return CollectionSchema(
        fields=[
            FieldSchema("id", primary_type, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
            FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("published_at", DataType.INT64),
            FieldSchema("event_time", DataType.TIMESTAMPTZ),
            FieldSchema("popularity", DataType.FLOAT),
            FieldSchema("doc", DataType.VARCHAR, max_length=512),
            FieldSchema("enabled", DataType.BOOL),
            FieldSchema("metadata", DataType.JSON),
            FieldSchema("tags", DataType.ARRAY, element_type=DataType.VARCHAR),
            FieldSchema("location", DataType.GEOMETRY),
        ]
    )


def _chain(*ops, stage="FunctionChainStageL2Rerank"):
    op_tuple = tuple(ops)
    return ChainRepr(
        name="test",
        stage=stage,
        ops=op_tuple,
        info=build_chain_info(op_tuple),
    )


def _map(expr, output="$score", inputs=(), outputs=None):
    actual_outputs = (output,) if outputs is None else outputs
    return OpRepr(
        op="map",
        expr=expr,
        inputs=inputs,
        outputs=actual_outputs,
        params={},
        read_names=tuple(
            arg.name for arg in expr.args if isinstance(arg, ColumnArg)
        ) if expr is not None else (),
        write_names=actual_outputs,
    )


def _round(arg=ColumnArg("$score"), decimal=2, output="$score"):
    return _map(
        ExprRepr("round_decimal", (arg,), {"decimal": decimal}),
        output=output,
    )


def _sort(column="$score", *, desc=True, tie_break_col=None, inputs=None):
    params = {"column": column, "desc": desc}
    if tie_break_col is not None:
        params["tie_break_col"] = tie_break_col
    expected_inputs = (
        (column,)
        if tie_break_col is None
        else (column, tie_break_col)
    )
    actual_inputs = expected_inputs if inputs is None else inputs
    return OpRepr(
        op="sort",
        expr=None,
        inputs=actual_inputs,
        outputs=(),
        params=params,
        read_names=actual_inputs,
        write_names=(),
    )


def _limit(limit=10, offset=0, *, expr=None, inputs=(), outputs=()):
    return OpRepr(
        op="limit",
        expr=expr,
        inputs=inputs,
        outputs=outputs,
        params={"limit": limit, "offset": offset},
        read_names=inputs,
        write_names=outputs,
    )


def test_public_validator_types_are_frozen_and_stable():
    assert [kind.value for kind in ValueKind] == [
        "numeric",
        "text",
        "bool",
        "timestamp",
    ]
    validated = validate_function_chain(_chain(_round()), _schema(), 1)

    assert isinstance(validated, ValidatedChain)
    assert validated.repr.ops[0].op == "map"
    with pytest.raises(FrozenInstanceError):
        validated.required_schema_fields = ()


def test_validator_plans_hidden_fields_once_in_first_seen_order():
    first = _map(
        ExprRepr(
            "decay",
            (ColumnArg("published_at"),),
            {
                "function": "exp",
                "origin": 100,
                "scale": 10,
                "offset": 0,
                "decay": 0.5,
            },
        ),
        output="freshness",
    )
    second = _map(
        ExprRepr(
            "num_combine",
            (
                ColumnArg("popularity"),
                ColumnArg("freshness"),
                ColumnArg("published_at"),
                ColumnArg("$score"),
            ),
            {"mode": "sum"},
        )
    )

    validated = validate_function_chain(_chain(first, second), _schema(), 1)

    assert validated.required_schema_fields == ("published_at", "popularity")


def test_previous_temporary_output_is_not_fetched_from_schema():
    first = _round(output="rounded")
    second = _map(
        ExprRepr(
            "num_combine",
            (ColumnArg("rounded"), ColumnArg("popularity")),
            {"mode": "sum"},
        )
    )

    validated = validate_function_chain(_chain(first, second), _schema(), 1)

    assert validated.required_schema_fields == ("popularity",)


def test_system_inputs_are_not_fetched_from_schema():
    validated = validate_function_chain(
        _chain(_sort("$score", tie_break_col="$id")),
        _schema(),
        1,
    )

    assert validated.required_schema_fields == ()


@pytest.mark.parametrize("name", ["$unknown", "$temporary"])
def test_validator_rejects_unknown_system_inputs(name):
    with pytest.raises(SchemaValidationError, match="system input"):
        validate_function_chain(_chain(_sort(name)), _schema(), 1)


@pytest.mark.parametrize("output", ["$id", "$temporary"])
def test_validator_rejects_readonly_or_unknown_system_outputs(output):
    with pytest.raises(SchemaValidationError, match="system output"):
        validate_function_chain(_chain(_round(output=output)), _schema(), 1)


def test_validator_allows_writing_score():
    validated = validate_function_chain(_chain(_round()), _schema(), 1)

    assert validated.required_schema_fields == ()


def test_validator_rejects_schema_field_outputs():
    with pytest.raises(SchemaValidationError, match="conflicts with a collection field"):
        validate_function_chain(
            _chain(_round(output="popularity")),
            _schema(),
            1,
        )


def test_validator_rejects_unknown_collection_input():
    with pytest.raises(SchemaValidationError, match="collection field"):
        validate_function_chain(_chain(_sort("missing")), _schema(), 1)


@pytest.mark.parametrize(
    "field_name",
    ["vector", "sparse", "metadata", "tags", "location"],
)
def test_validator_rejects_unsupported_collection_field_types(field_name):
    with pytest.raises(SchemaValidationError, match="unsupported type"):
        validate_function_chain(_chain(_sort(field_name)), _schema(), 1)


def test_validator_rejects_unknown_collection_field_type():
    class UnknownType:
        value = "future"

    schema = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("future", UnknownType()),
        ]
    )

    with pytest.raises(SchemaValidationError, match="unsupported type future"):
        validate_function_chain(_chain(_sort("future")), schema, 1)


@pytest.mark.parametrize(
    "dtype",
    [
        DataType.BOOL,
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
        DataType.FLOAT,
        DataType.DOUBLE,
        DataType.VARCHAR,
        DataType.TIMESTAMPTZ,
    ],
)
def test_validator_accepts_supported_collection_field_types(dtype):
    schema = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", dtype),
        ]
    )

    validated = validate_function_chain(_chain(_sort("value")), schema, 1)

    assert validated.required_schema_fields == ("value",)


def test_validator_reports_missing_primary_key_as_schema_error():
    schema = CollectionSchema(fields=[FieldSchema("value", DataType.FLOAT)])

    with pytest.raises(SchemaValidationError, match="primary key"):
        validate_function_chain(_chain(_round()), schema, 1)


@pytest.mark.parametrize(
    ("primary_name", "primary_type", "op"),
    [
        ("$id", DataType.INT64, _sort("$id")),
        ("$score", DataType.VARCHAR, _round()),
    ],
)
def test_validator_rejects_primary_names_conflicting_with_system_columns(
    primary_name,
    primary_type,
    op,
):
    schema = CollectionSchema(
        fields=[
            FieldSchema(primary_name, primary_type, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
        ]
    )

    with pytest.raises(SchemaValidationError) as error:
        validate_function_chain(_chain(op), schema, 1)

    assert str(error.value) == (
        f"function chain primary key field {primary_name!r} conflicts with "
        f"reserved system column {primary_name!r}"
    )


def test_string_primary_key_cannot_be_used_as_numeric_input():
    expr = ExprRepr(
        "num_combine",
        (ColumnArg("$score"), ColumnArg("$id")),
        {"mode": "sum"},
    )

    with pytest.raises(SchemaValidationError, match="numeric arguments"):
        validate_function_chain(_chain(_map(expr)), _schema(DataType.VARCHAR), 1)


def test_numeric_primary_field_name_is_id_alias_not_required_field():
    expr = ExprRepr(
        "num_combine",
        (ColumnArg("$score"), ColumnArg("id")),
        {"mode": "sum"},
    )

    validated = validate_function_chain(_chain(_map(expr)), _schema(), 1)

    assert validated.required_schema_fields == ()


def test_varchar_primary_field_name_resolves_as_text_id_alias():
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("id"),),
        {"provider": "cohere", "queries": ["one"]},
    )

    validated = validate_function_chain(
        _chain(_map(expr)),
        _schema(DataType.VARCHAR),
        1,
    )

    assert validated.required_schema_fields == ()


@pytest.mark.parametrize("stage", ["", "FunctionChainStageUnspecified", "rerank"])
def test_validator_rejects_wrong_stage(stage):
    with pytest.raises(SchemaValidationError, match="stage"):
        validate_function_chain(_chain(_round(), stage=stage), _schema(), 1)


def test_validator_rejects_empty_chain():
    with pytest.raises(SchemaValidationError, match="at least one operator"):
        validate_function_chain(_chain(), _schema(), 1)


def test_validator_rejects_unknown_operator():
    op = OpRepr("boost", None, (), (), {}, (), ())

    with pytest.raises(SchemaValidationError, match="unsupported.*operator"):
        validate_function_chain(_chain(op), _schema(), 1)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (_map(None), "requires an expression"),
        (_map(ExprRepr("round_decimal", (ColumnArg("$score"),), {"decimal": 2}), inputs=("$score",)), "must not declare op.inputs"),
        (_map(ExprRepr("round_decimal", (ColumnArg("$score"),), {"decimal": 2}), outputs=()), "exactly one output"),
        (_map(ExprRepr("round_decimal", (ColumnArg("$score"),), {"decimal": 2}), outputs=("one", "two")), "exactly one output"),
    ],
)
def test_map_operator_contract(op, message):
    with pytest.raises(SchemaValidationError, match=message):
        validate_function_chain(_chain(op), _schema(), 1)


def test_sort_defaults_desc_to_true_and_allows_no_tie_break():
    op = _sort("popularity")
    op = OpRepr(
        op=op.op,
        expr=op.expr,
        inputs=op.inputs,
        outputs=op.outputs,
        params={"column": "popularity"},
        read_names=op.read_names,
        write_names=op.write_names,
    )

    validated = validate_function_chain(_chain(op), _schema(), 1)

    assert validated.required_schema_fields == ("popularity",)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (OpRepr("sort", ExprRepr("round_decimal", (), {}), ("$score",), (), {"column": "$score"}, ("$score",), ()), "expression or outputs"),
        (OpRepr("sort", None, ("$score",), ("tmp",), {"column": "$score"}, ("$score",), ("tmp",)), "expression or outputs"),
        (_sort(""), "column must be a non-empty string"),
        (_sort("$score", desc=1), "desc must be a boolean"),
        (_sort("$score", tie_break_col=""), "tie_break_col must be a non-empty string"),
        (_sort("$score", inputs=()), "inputs must match"),
        (_sort("$score", tie_break_col="$id", inputs=("$score",)), "inputs must match"),
        (_sort("$score", tie_break_col="$id", inputs=("$id", "$score")), "inputs must match"),
    ],
)
def test_sort_operator_contract(op, message):
    with pytest.raises(SchemaValidationError, match=message):
        validate_function_chain(_chain(op), _schema(), 1)


def test_limit_accepts_default_offset():
    op = OpRepr("limit", None, (), (), {"limit": 10}, (), ())

    assert validate_function_chain(_chain(op), _schema(), 1).required_schema_fields == ()


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (_limit(expr=ExprRepr("round_decimal", (), {})), "expression, inputs, or outputs"),
        (_limit(inputs=("$score",)), "expression, inputs, or outputs"),
        (_limit(outputs=("tmp",)), "expression, inputs, or outputs"),
        (_limit(limit=True), "positive integer"),
        (_limit(limit=0), "positive integer"),
        (_limit(limit=-1), "positive integer"),
        (_limit(limit=1.5), "positive integer"),
        (_limit(offset=True), "non-negative integer"),
        (_limit(offset=-1), "non-negative integer"),
        (_limit(offset=1.5), "non-negative integer"),
    ],
)
def test_limit_operator_contract(op, message):
    with pytest.raises(SchemaValidationError, match=message):
        validate_function_chain(_chain(op), _schema(), 1)


@pytest.mark.parametrize("mode", ["multiply", "sum", "max", "min", "avg"])
def test_num_combine_accepts_supported_nonweighted_modes(mode):
    expr = ExprRepr(
        "num_combine",
        (ColumnArg("$score"), LiteralArg(2)),
        {"mode": mode},
    )

    validate_function_chain(_chain(_map(expr)), _schema(), 1)


def test_num_combine_accepts_tuple_weights_from_frozen_repr():
    expr = ExprRepr(
        "num_combine",
        (ColumnArg("$score"), LiteralArg(2)),
        {"mode": "weighted", "weights": [0.75, 0.25]},
    )

    assert expr.params["weights"] == (0.75, 0.25)
    validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("mode", [{"name": "sum"}, ["sum"]])
def test_num_combine_rejects_non_string_mode(mode):
    expr = ExprRepr(
        "num_combine",
        (ColumnArg("$score"), LiteralArg(2)),
        {"mode": mode},
    )

    with pytest.raises(SchemaValidationError, match="mode must be a string"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_num_combine_rejects_non_finite_weights(weight):
    expr = ExprRepr(
        "num_combine",
        (ColumnArg("$score"), LiteralArg(2)),
        {"mode": "weighted", "weights": [1, weight]},
    )

    with pytest.raises(SchemaValidationError, match="weights must be finite"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        (ExprRepr("num_combine", (ColumnArg("$score"),), {"mode": "sum"}), "at least two numeric arguments"),
        (ExprRepr("num_combine", (ColumnArg("$score"), ColumnArg("doc")), {"mode": "sum"}), "at least two numeric arguments"),
        (ExprRepr("num_combine", (ColumnArg("$score"), LiteralArg(True)), {"mode": "sum"}), "at least two numeric arguments"),
        (ExprRepr("num_combine", (ColumnArg("$score"), LiteralArg(1)), {"mode": "median"}), "unsupported num_combine mode"),
        (ExprRepr("num_combine", (ColumnArg("$score"), LiteralArg(1)), {"mode": "weighted"}), "requires 2 weights"),
        (ExprRepr("num_combine", (ColumnArg("$score"), LiteralArg(1)), {"mode": "weighted", "weights": [1]}), "requires 2 weights"),
        (ExprRepr("num_combine", (ColumnArg("$score"), LiteralArg(1)), {"mode": "weighted", "weights": "12"}), "requires 2 weights"),
        (ExprRepr("num_combine", (ColumnArg("$score"), LiteralArg(1)), {"mode": "weighted", "weights": [1, True]}), "weights must be numeric"),
        (ExprRepr("num_combine", (ColumnArg("$score"), LiteralArg(1)), {"mode": "weighted", "weights": [1, "bad"]}), "weights must be numeric"),
        (ExprRepr("num_combine", (ColumnArg("$score"), LiteralArg(1)), {"mode": "sum", "weights": [1, 1]}), "weights require mode='weighted'"),
    ],
)
def test_num_combine_rejects_invalid_arguments_and_params(expr, message):
    with pytest.raises(SchemaValidationError, match=message):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("function", ["gauss", "exp", "linear"])
@pytest.mark.parametrize("field", ["published_at", "event_time"])
def test_decay_accepts_supported_functions_and_input_kinds(function, field):
    expr = ExprRepr(
        "decay",
        (ColumnArg(field),),
        {"function": function, "origin": 0, "scale": 1},
    )

    validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("function", [{"name": "exp"}, ["exp"]])
def test_decay_rejects_non_string_function(function):
    expr = ExprRepr(
        "decay",
        (ColumnArg("published_at"),),
        {"function": function, "origin": 0, "scale": 1},
    )

    with pytest.raises(SchemaValidationError, match="function must be a string"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("parameter", ["origin", "scale", "offset", "decay"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_decay_rejects_non_finite_parameters(parameter, value):
    params = {
        "function": "exp",
        "origin": 0,
        "scale": 1,
        "offset": 0,
        "decay": 0.5,
    }
    params[parameter] = value
    expr = ExprRepr("decay", (ColumnArg("published_at"),), params)

    with pytest.raises(SchemaValidationError, match="parameters must be finite"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        (ExprRepr("decay", (), {"function": "exp", "origin": 0, "scale": 1}), "exactly one numeric or timestamp argument"),
        (ExprRepr("decay", (ColumnArg("published_at"), ColumnArg("popularity")), {"function": "exp", "origin": 0, "scale": 1}), "exactly one numeric or timestamp argument"),
        (ExprRepr("decay", (ColumnArg("doc"),), {"function": "exp", "origin": 0, "scale": 1}), "exactly one numeric or timestamp argument"),
        (ExprRepr("decay", (ColumnArg("enabled"),), {"function": "exp", "origin": 0, "scale": 1}), "exactly one numeric or timestamp argument"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "bad", "origin": 0, "scale": 1}), "unsupported decay function"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "scale": 1}), "parameters must be numeric"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "origin": True, "scale": 1}), "parameters must be numeric"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "origin": 0, "scale": "1"}), "parameters must be numeric"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "origin": 0, "scale": 0}), "scale > 0"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "origin": 0, "scale": float("nan")}), "parameters must be finite"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "origin": 0, "scale": 1, "offset": -1}), "offset >= 0"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "origin": 0, "scale": 1, "offset": float("nan")}), "parameters must be finite"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "origin": 0, "scale": 1, "decay": 0}), "0 < decay < 1"),
        (ExprRepr("decay", (ColumnArg("published_at"),), {"function": "exp", "origin": 0, "scale": 1, "decay": 1}), "0 < decay < 1"),
    ],
)
def test_decay_rejects_invalid_arguments_and_params(expr, message):
    with pytest.raises(SchemaValidationError, match=message):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("decimal", [0, 6])
def test_round_decimal_accepts_boundary_values(decimal):
    validate_function_chain(_chain(_round(decimal=decimal)), _schema(), 1)


@pytest.mark.parametrize("decimal", [None, True, -1, 7, 1.5])
def test_round_decimal_rejects_invalid_decimal(decimal):
    with pytest.raises(SchemaValidationError, match=r"integer in \[0, 6\]"):
        validate_function_chain(_chain(_round(decimal=decimal)), _schema(), 1)


@pytest.mark.parametrize("arg", [ColumnArg("doc"), ColumnArg("enabled")])
def test_round_decimal_rejects_non_numeric_argument(arg):
    with pytest.raises(SchemaValidationError, match="exactly one numeric argument"):
        validate_function_chain(_chain(_round(arg=arg)), _schema(), 1)


def test_rerank_model_accepts_tuple_queries_from_frozen_repr():
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("doc"),),
        {"provider": "cohere", "queries": ["one", "two"]},
    )

    assert expr.params["queries"] == ("one", "two")
    validated = validate_function_chain(_chain(_map(expr)), _schema(), 2)

    assert validated.required_schema_fields == ("doc",)


@pytest.mark.parametrize(
    ("expr", "num_queries", "message"),
    [
        (ExprRepr("rerank_model", (), {"provider": "cohere", "queries": ["one"]}), 1, "exactly one text argument"),
        (ExprRepr("rerank_model", (ColumnArg("doc"), ColumnArg("doc")), {"provider": "cohere", "queries": ["one"]}), 1, "exactly one text argument"),
        (ExprRepr("rerank_model", (ColumnArg("popularity"),), {"provider": "cohere", "queries": ["one"]}), 1, "exactly one text argument"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"queries": ["one"]}), 1, "provider must be a non-empty string"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"provider": "", "queries": ["one"]}), 1, "provider must be a non-empty string"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"provider": 1, "queries": ["one"]}), 1, "provider must be a non-empty string"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"provider": "cohere"}), 1, "queries must be a non-empty sequence"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"provider": "cohere", "queries": []}), 1, "queries must be a non-empty sequence"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"provider": "cohere", "queries": "one"}), 1, "queries must be a non-empty sequence"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"provider": "cohere", "queries": [""]}), 1, "queries must be a non-empty sequence"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"provider": "cohere", "queries": [1]}), 1, "queries must be a non-empty sequence"),
        (ExprRepr("rerank_model", (ColumnArg("doc"),), {"provider": "cohere", "queries": ["one"]}), 2, "query count"),
    ],
)
def test_rerank_model_rejects_invalid_arguments_and_params(
    expr,
    num_queries,
    message,
):
    with pytest.raises(SchemaValidationError, match=message):
        validate_function_chain(_chain(_map(expr)), _schema(), num_queries)


@pytest.mark.parametrize("credential", ["api_key", "API_KEY", "Token", "SeCrEt"])
def test_rerank_model_rejects_credential_params_case_insensitively(credential):
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("doc"),),
        {"provider": "cohere", "queries": ["one"], credential: "value"},
    )

    with pytest.raises(SchemaValidationError, match="credentials"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


def test_rerank_model_rejects_unknown_provider():
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("doc"),),
        {
            "provider": "unknown-provider",
            "queries": ["one"],
        },
    )

    with pytest.raises(SchemaValidationError, match="Unknown rerank provider"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize(
    ("params", "message"),
    [({"model_name": 123}, "model_name")],
)
def test_rerank_model_rejects_invalid_provider_params(params, message):
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("doc"),),
        {
            "provider": "cohere",
            "queries": ["one"],
            **params,
        },
    )

    with pytest.raises(SchemaValidationError, match=message):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


def test_rerank_model_accepts_supported_model_name_param():
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("doc"),),
        {
            "provider": "cohere",
            "queries": ["one"],
            "model_name": "rerank-v3.5",
        },
    )

    validated = validate_function_chain(_chain(_map(expr)), _schema(), 1)

    assert validated.required_schema_fields == ("doc",)


@pytest.mark.parametrize("key", ["model", "typo", 123])
def test_rerank_model_rejects_unsupported_provider_params(key):
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("doc"),),
        {
            "provider": "cohere",
            "queries": ["one"],
            key: "value",
        },
    )

    with pytest.raises(SchemaValidationError) as error:
        validate_function_chain(_chain(_map(expr)), _schema(), 1)

    assert "unsupported rerank_model parameter" in str(error.value)
    assert repr(key) in str(error.value)


def test_rerank_model_reports_unsupported_params_stably():
    errors = []
    for extra_params in (
        {"zeta": "value", "alpha": "value"},
        {"alpha": "value", "zeta": "value"},
    ):
        expr = ExprRepr(
            "rerank_model",
            (ColumnArg("doc"),),
            {
                "provider": "cohere",
                "queries": ["one"],
                **extra_params,
            },
        )

        with pytest.raises(SchemaValidationError) as error:
            validate_function_chain(_chain(_map(expr)), _schema(), 1)
        errors.append(str(error.value))

    assert errors == [
        "function chain op[0] expression rerank_model: "
        "unsupported rerank_model "
        "parameter 'alpha'"
    ] * 2


@pytest.mark.parametrize("key", ["base_url", "BASE_URL", "Base_Url"])
def test_rerank_model_rejects_request_endpoint_params(key):
    expr = ExprRepr(
        "rerank_model",
        (ColumnArg("doc"),),
        {
            "provider": "cohere",
            "queries": ["one"],
            key: "https://attacker.example",
        },
    )

    with pytest.raises(SchemaValidationError, match="endpoint"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


def test_validator_rejects_unknown_expression():
    expr = ExprRepr("custom", (ColumnArg("$score"),), {})

    with pytest.raises(SchemaValidationError, match="unsupported.*expression"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("literal", [None, b"bytes", (), {}, [1, 2]])
def test_validator_rejects_unsupported_literal_types(literal):
    expr = ExprRepr("round_decimal", (LiteralArg(literal),), {"decimal": 2})

    with pytest.raises(SchemaValidationError, match="unsupported.*literal type"):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("literal", [float("nan"), float("inf"), float("-inf")])
def test_round_decimal_rejects_non_finite_numeric_literal(literal):
    expr = ExprRepr("round_decimal", (LiteralArg(literal),), {"decimal": 2})

    with pytest.raises(
        SchemaValidationError,
        match=(
            r"function chain op\[0\] expression round_decimal: "
            r"numeric literal must be finite"
        ),
    ):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


@pytest.mark.parametrize("literal", [float("nan"), float("inf"), float("-inf")])
def test_num_combine_rejects_non_finite_numeric_literal(literal):
    expr = ExprRepr(
        "num_combine",
        (ColumnArg("$score"), LiteralArg(literal)),
        {"mode": "sum"},
    )

    with pytest.raises(
        SchemaValidationError,
        match=(
            r"function chain op\[0\] expression num_combine: "
            r"numeric literal must be finite"
        ),
    ):
        validate_function_chain(_chain(_map(expr)), _schema(), 1)


def test_multi_op_expression_error_includes_operator_and_expression_context():
    invalid = _map(
        ExprRepr(
            "num_combine",
            (ColumnArg("$score"), LiteralArg(1)),
            {"mode": "median"},
        )
    )

    with pytest.raises(
        SchemaValidationError,
        match=(
            r"function chain op\[1\] expression num_combine: "
            r"unsupported num_combine mode: median"
        ),
    ):
        validate_function_chain(_chain(_limit(), invalid), _schema(), 1)


def test_multi_op_operator_error_includes_operator_context():
    with pytest.raises(
        SchemaValidationError,
        match=r"function chain op\[1\]: sort desc must be a boolean",
    ):
        validate_function_chain(
            _chain(_round(), _sort(desc="descending")),
            _schema(),
            1,
        )


def test_multi_op_map_contract_error_uses_operator_context():
    invalid = _map(None)

    with pytest.raises(
        SchemaValidationError,
        match=r"function chain op\[1\]: map operator requires an expression",
    ):
        validate_function_chain(_chain(_limit(), invalid), _schema(), 1)
