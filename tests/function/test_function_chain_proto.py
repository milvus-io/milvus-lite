"""Tests for public Function Chain protobuf decoding."""

from dataclasses import FrozenInstanceError

import pytest
from pymilvus.grpc_gen import schema_pb2

from milvus_lite.adapter.grpc.translators.function_chain import (
    decode_function_param_value,
    function_chain_to_repr,
)
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.function.repr import (
    ColumnArg,
    ExprRepr,
    LiteralArg,
    OpRepr,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (schema_pb2.FunctionParamValue(bool_value=True), True),
        (schema_pb2.FunctionParamValue(int64_value=7), 7),
        (schema_pb2.FunctionParamValue(double_value=1.5), 1.5),
        (schema_pb2.FunctionParamValue(string_value="sum"), "sum"),
        (schema_pb2.FunctionParamValue(bytes_value=b"raw"), b"raw"),
    ],
)
def test_decode_function_param_scalar_values(value, expected):
    assert decode_function_param_value(value) == expected


def test_decode_nested_function_param_value():
    value = schema_pb2.FunctionParamValue(
        object_value=schema_pb2.FunctionParamObject(
            fields={
                "enabled": schema_pb2.FunctionParamValue(bool_value=True),
                "weights": schema_pb2.FunctionParamValue(
                    array_value=schema_pb2.FunctionParamArray(
                        values=[
                            schema_pb2.FunctionParamValue(double_value=0.7),
                            schema_pb2.FunctionParamValue(
                                object_value=schema_pb2.FunctionParamObject(
                                    fields={
                                        "rank": schema_pb2.FunctionParamValue(
                                            int64_value=2
                                        )
                                    }
                                )
                            ),
                        ]
                    )
                ),
            }
        )
    )

    assert decode_function_param_value(value) == {
        "enabled": True,
        "weights": [0.7, {"rank": 2}],
    }


def test_decode_function_param_value_uses_detached_list_and_dict_values():
    value = schema_pb2.FunctionParamValue(
        object_value=schema_pb2.FunctionParamObject(
            fields={
                "settings": schema_pb2.FunctionParamValue(
                    object_value=schema_pb2.FunctionParamObject(
                        fields={
                            "enabled": schema_pb2.FunctionParamValue(
                                bool_value=True
                            )
                        }
                    )
                ),
                "weights": schema_pb2.FunctionParamValue(
                    array_value=schema_pb2.FunctionParamArray(
                        values=[
                            schema_pb2.FunctionParamValue(double_value=0.7)
                        ]
                    )
                ),
            }
        )
    )

    decoded = decode_function_param_value(value)

    assert type(decoded) is dict
    assert type(decoded["settings"]) is dict
    assert type(decoded["weights"]) is list

    decoded["new"] = True
    decoded["settings"]["enabled"] = False
    decoded["weights"].append(0.3)

    object_value = value.object_value
    assert "new" not in object_value.fields
    assert object_value.fields["settings"].object_value.fields[
        "enabled"
    ].bool_value
    assert len(object_value.fields["weights"].array_value.values) == 1

    object_value.fields["weights"].array_value.values.append(
        schema_pb2.FunctionParamValue(double_value=0.9)
    )

    assert decoded["weights"] == [0.7, 0.3]


def test_representation_values_are_deeply_immutable_and_detached():
    expr_source = {"nested": {"weights": [0.7]}}
    op_source = {"options": [{"enabled": True}]}
    literal_source = {"values": [1]}

    expr = ExprRepr("combine", (), expr_source)
    op = OpRepr("map", expr, (), (), op_source, (), ())
    literal = LiteralArg(literal_source)

    expr_source["nested"]["weights"].append(0.3)
    op_source["options"][0]["enabled"] = False
    literal_source["values"].append(2)

    assert expr.params == {"nested": {"weights": (0.7,)}}
    assert op.params == {"options": ({"enabled": True},)}
    assert literal.value == {"values": (1,)}
    with pytest.raises(FrozenInstanceError):
        expr.params = {}
    with pytest.raises(FrozenInstanceError):
        op.params = {}
    with pytest.raises(FrozenInstanceError):
        literal.value = None
    with pytest.raises(TypeError):
        expr.params["new"] = True
    with pytest.raises(TypeError):
        expr.params["nested"]["weights"][0] = 0.3
    with pytest.raises(TypeError):
        op.params["options"][0]["enabled"] = False
    with pytest.raises(TypeError):
        literal.value["values"][0] = 2


def test_function_chain_to_repr_decodes_args_params_and_dependencies():
    chain = schema_pb2.FunctionChain(
        name="freshness",
        stage=schema_pb2.FunctionChainStageL2Rerank,
        ops=[
            schema_pb2.FunctionChainOp(
                op="map",
                expr=schema_pb2.FunctionChainExpr(
                    name="num_combine",
                    args=[
                        schema_pb2.FunctionChainExprArg(
                            column=schema_pb2.FunctionChainColumnArg(name="$score")
                        ),
                        schema_pb2.FunctionChainExprArg(
                            column=schema_pb2.FunctionChainColumnArg(
                                name="popularity"
                            )
                        ),
                        schema_pb2.FunctionChainExprArg(
                            literal=schema_pb2.FunctionParamValue(
                                double_value=1.0
                            )
                        ),
                    ],
                    params={
                        "mode": schema_pb2.FunctionParamValue(string_value="sum")
                    },
                ),
                inputs=["ignored_when_expr_is_present"],
                outputs=["tmp_score"],
                params={
                    "metadata": schema_pb2.FunctionParamValue(
                        object_value=schema_pb2.FunctionParamObject(
                            fields={
                                "enabled": schema_pb2.FunctionParamValue(
                                    bool_value=True
                                )
                            }
                        )
                    )
                },
            ),
            schema_pb2.FunctionChainOp(
                op="map",
                expr=schema_pb2.FunctionChainExpr(
                    name="round_decimal",
                    args=[
                        schema_pb2.FunctionChainExprArg(
                            column=schema_pb2.FunctionChainColumnArg(
                                name="tmp_score"
                            )
                        ),
                        schema_pb2.FunctionChainExprArg(
                            column=schema_pb2.FunctionChainColumnArg(name="$score")
                        ),
                    ],
                ),
                outputs=["tmp_score", "rounded"],
            ),
            schema_pb2.FunctionChainOp(
                op="sort",
                inputs=["rounded", "$id", "popularity"],
                params={
                    "column": schema_pb2.FunctionParamValue(
                        string_value="rounded"
                    ),
                    "desc": schema_pb2.FunctionParamValue(bool_value=True),
                },
            ),
        ],
    )

    result = function_chain_to_repr(chain)

    assert result.name == "freshness"
    assert result.stage == "FunctionChainStageL2Rerank"
    assert result.ops[0].inputs == ("ignored_when_expr_is_present",)
    assert result.ops[0].outputs == ("tmp_score",)
    assert result.ops[0].params == {"metadata": {"enabled": True}}
    assert result.ops[0].expr.params == {"mode": "sum"}
    assert result.ops[0].expr.args == (
        ColumnArg("$score"),
        ColumnArg("popularity"),
        LiteralArg(1.0),
    )
    assert result.ops[0].read_names == ("$score", "popularity")
    assert result.ops[0].write_names == ("tmp_score",)
    assert result.info.required_inputs == ("$score", "popularity", "$id")
    assert result.info.written_names == ("tmp_score", "rounded")


def test_decode_rejects_unset_parameter_value():
    with pytest.raises(SchemaValidationError, match="value is not set"):
        decode_function_param_value(schema_pb2.FunctionParamValue())


def test_decode_rejects_empty_object_key():
    value = schema_pb2.FunctionParamValue(
        object_value=schema_pb2.FunctionParamObject(
            fields={"": schema_pb2.FunctionParamValue(bool_value=True)}
        )
    )

    with pytest.raises(SchemaValidationError, match="object key must not be empty"):
        decode_function_param_value(value)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (schema_pb2.FunctionChainOp(), r"op\[0\] name must not be empty"),
        (
            schema_pb2.FunctionChainOp(
                op="map",
                expr=schema_pb2.FunctionChainExpr(
                    args=[
                        schema_pb2.FunctionChainExprArg(
                            column=schema_pb2.FunctionChainColumnArg(name="value")
                        )
                    ]
                ),
            ),
            r"op\[0\] expression name must not be empty",
        ),
        (
            schema_pb2.FunctionChainOp(
                op="map",
                expr=schema_pb2.FunctionChainExpr(
                    name="round_decimal",
                    args=[
                        schema_pb2.FunctionChainExprArg(
                            column=schema_pb2.FunctionChainColumnArg()
                        )
                    ],
                ),
            ),
            r"op\[0\]\.expr\.args\[0\] column name must not be empty",
        ),
        (
            schema_pb2.FunctionChainOp(op="sort", inputs=[""]),
            r"op\[0\] input\[0\] name must not be empty",
        ),
        (
            schema_pb2.FunctionChainOp(op="map", outputs=[""]),
            r"op\[0\] output\[0\] name must not be empty",
        ),
    ],
)
def test_function_chain_to_repr_rejects_empty_names(op, message):
    chain = schema_pb2.FunctionChain(
        stage=schema_pb2.FunctionChainStageL2Rerank,
        ops=[op],
    )

    with pytest.raises(SchemaValidationError, match=message):
        function_chain_to_repr(chain)


def test_function_chain_to_repr_rejects_unset_expression_argument():
    chain = schema_pb2.FunctionChain(
        stage=schema_pb2.FunctionChainStageL2Rerank,
        ops=[
            schema_pb2.FunctionChainOp(
                op="map",
                expr=schema_pb2.FunctionChainExpr(
                    name="round_decimal",
                    args=[schema_pb2.FunctionChainExprArg()],
                ),
            )
        ],
    )

    with pytest.raises(
        SchemaValidationError,
        match=r"op\[0\]\.expr\.args\[0\] argument is not set",
    ):
        function_chain_to_repr(chain)


def test_function_chain_to_repr_adds_context_to_unset_param_value():
    chain = schema_pb2.FunctionChain(
        stage=schema_pb2.FunctionChainStageL2Rerank,
        ops=[
            schema_pb2.FunctionChainOp(
                op="limit",
                params={"limit": schema_pb2.FunctionParamValue()},
            )
        ],
    )

    with pytest.raises(
        SchemaValidationError,
        match=r"op\[0\]\.params\['limit'\].*value is not set",
    ):
        function_chain_to_repr(chain)


def test_function_chain_to_repr_adds_recursive_param_error_context():
    chain = schema_pb2.FunctionChain(
        stage=schema_pb2.FunctionChainStageL2Rerank,
        ops=[
            schema_pb2.FunctionChainOp(
                op="limit",
                params={
                    "config": schema_pb2.FunctionParamValue(
                        object_value=schema_pb2.FunctionParamObject(
                            fields={
                                "items": schema_pb2.FunctionParamValue(
                                    array_value=schema_pb2.FunctionParamArray(
                                        values=[
                                            schema_pb2.FunctionParamValue()
                                        ]
                                    )
                                )
                            }
                        )
                    )
                },
            )
        ],
    )

    with pytest.raises(
        SchemaValidationError,
        match=(
            r"op\[0\]\.params\['config'\]\['items'\]\[0\]: "
            r"value is not set"
        ),
    ):
        function_chain_to_repr(chain)
