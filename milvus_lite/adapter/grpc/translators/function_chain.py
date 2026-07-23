"""Decode public Function Chain protobuf messages."""

from __future__ import annotations

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.function.repr import (
    ChainRepr,
    ColumnArg,
    ExprRepr,
    LiteralArg,
    OpRepr,
    build_chain_info,
)


def decode_function_param_value(value):
    return _decode_function_param_value(value, "function parameter")


def _decode_function_param_value(value, path: str):
    kind = value.WhichOneof("value")
    if kind is None:
        raise SchemaValidationError(f"{path}: value is not set")
    if kind in {
        "bool_value",
        "int64_value",
        "double_value",
        "string_value",
        "bytes_value",
    }:
        return getattr(value, kind)
    if kind == "array_value":
        return [
            _decode_function_param_value(item, f"{path}[{index}]")
            for index, item in enumerate(value.array_value.values)
        ]
    if kind == "object_value":
        decoded = {}
        for key, item in value.object_value.fields.items():
            if not key:
                raise SchemaValidationError(
                    f"{path}: object key must not be empty"
                )
            decoded[key] = _decode_function_param_value(
                item, f"{path}[{key!r}]"
            )
        return decoded
    raise SchemaValidationError(
        f"{path}: unsupported value kind: {kind}"
    )


def _decode_value_with_context(value, path: str):
    return _decode_function_param_value(value, path)


def _decode_expr_arg(arg, path: str):
    kind = arg.WhichOneof("arg")
    if kind == "column":
        if not arg.column.name:
            raise SchemaValidationError(
                f"{path} column name must not be empty"
            )
        return ColumnArg(arg.column.name)
    if kind == "literal":
        return LiteralArg(
            _decode_value_with_context(arg.literal, f"{path}.literal")
        )
    raise SchemaValidationError(f"{path} argument is not set")


def _decode_params(params, path: str):
    return {
        key: _decode_value_with_context(value, f"{path}[{key!r}]")
        for key, value in params.items()
    }


def _decode_expr(expr, path: str) -> ExprRepr:
    if not expr.name:
        raise SchemaValidationError(
            f"{path} expression name must not be empty"
        )
    return ExprRepr(
        name=expr.name,
        args=tuple(
            _decode_expr_arg(arg, f"{path}.expr.args[{index}]")
            for index, arg in enumerate(expr.args)
        ),
        params=_decode_params(expr.params, f"{path}.expr.params"),
    )


def _stage_name(chain_pb) -> str:
    field = chain_pb.DESCRIPTOR.fields_by_name["stage"]
    value = field.enum_type.values_by_number.get(chain_pb.stage)
    if value is None:
        raise SchemaValidationError(
            f"unknown function chain stage value: {chain_pb.stage}"
        )
    return value.name


def function_chain_to_repr(chain_pb) -> ChainRepr:
    ops = []
    for index, op_pb in enumerate(chain_pb.ops):
        path = f"function chain op[{index}]"
        if not op_pb.op:
            raise SchemaValidationError(f"{path} name must not be empty")

        expr = _decode_expr(op_pb.expr, path) if op_pb.HasField("expr") else None
        inputs = tuple(op_pb.inputs)
        outputs = tuple(op_pb.outputs)

        for input_index, name in enumerate(inputs):
            if not name:
                raise SchemaValidationError(
                    f"{path} input[{input_index}] name must not be empty"
                )
        for output_index, name in enumerate(outputs):
            if not name:
                raise SchemaValidationError(
                    f"{path} output[{output_index}] name must not be empty"
                )

        read_names = (
            tuple(
                arg.name
                for arg in expr.args
                if isinstance(arg, ColumnArg)
            )
            if expr is not None
            else inputs
        )
        ops.append(
            OpRepr(
                op=op_pb.op,
                expr=expr,
                inputs=inputs,
                outputs=outputs,
                params=_decode_params(op_pb.params, f"{path}.params"),
                read_names=read_names,
                write_names=outputs,
            )
        )

    op_tuple = tuple(ops)
    return ChainRepr(
        name=chain_pb.name,
        stage=_stage_name(chain_pb),
        ops=op_tuple,
        info=build_chain_info(op_tuple),
    )
