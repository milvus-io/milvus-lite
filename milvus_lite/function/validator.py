"""Schema-aware validation and dependency planning for public chains."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.function.repr import (
    ChainRepr,
    ColumnArg,
    ExprRepr,
    LiteralArg,
    OpRepr,
)
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD
from milvus_lite.schema.types import CollectionSchema, DataType


class ValueKind(Enum):
    NUMERIC = "numeric"
    TEXT = "text"
    BOOL = "bool"
    TIMESTAMP = "timestamp"


@dataclass(frozen=True)
class ValidatedChain:
    repr: ChainRepr
    required_schema_fields: tuple[str, ...]


_FIELD_KINDS = {
    DataType.BOOL: ValueKind.BOOL,
    DataType.INT8: ValueKind.NUMERIC,
    DataType.INT16: ValueKind.NUMERIC,
    DataType.INT32: ValueKind.NUMERIC,
    DataType.INT64: ValueKind.NUMERIC,
    DataType.FLOAT: ValueKind.NUMERIC,
    DataType.DOUBLE: ValueKind.NUMERIC,
    DataType.VARCHAR: ValueKind.TEXT,
    DataType.TIMESTAMPTZ: ValueKind.TIMESTAMP,
}

_NON_STRING_SEQUENCE_TYPES = (str, bytes, bytearray)


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _is_numeric(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _is_finite_numeric(value: object) -> bool:
    if not _is_numeric(value):
        return False
    if isinstance(value, int):
        return True
    return math.isfinite(value)


def _is_non_string_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, _NON_STRING_SEQUENCE_TYPES
    )


def _literal_kind(value: object) -> ValueKind:
    if isinstance(value, bool):
        return ValueKind.BOOL
    if _is_numeric(value):
        if not _is_finite_numeric(value):
            _fail("numeric literal must be finite")
        return ValueKind.NUMERIC
    if isinstance(value, str):
        return ValueKind.TEXT
    _fail(f"unsupported function chain literal type: {type(value).__name__}")


def _validate_map(op: OpRepr) -> ExprRepr:
    if op.expr is None:
        _fail("map operator requires an expression")
    if op.inputs:
        _fail("map operator must not declare op.inputs")
    if len(op.outputs) != 1:
        _fail("map operator requires exactly one output")
    return op.expr


def _validate_sort(op: OpRepr) -> None:
    if op.expr is not None or op.outputs:
        _fail("sort operator must not contain expression or outputs")
    column = op.params.get("column")
    desc = op.params.get("desc", True)
    tie_break = op.params.get("tie_break_col")
    if not isinstance(column, str) or not column:
        _fail("sort column must be a non-empty string")
    if not isinstance(desc, bool):
        _fail("sort desc must be a boolean")
    if tie_break is not None and (
        not isinstance(tie_break, str) or not tie_break
    ):
        _fail("sort tie_break_col must be a non-empty string")
    expected = (column,) if tie_break is None else (column, tie_break)
    if op.inputs != expected:
        _fail("sort inputs must match column and tie_break_col")


def _validate_limit(op: OpRepr) -> None:
    if op.expr is not None or op.inputs or op.outputs:
        _fail("limit operator must not contain expression, inputs, or outputs")
    limit = op.params.get("limit")
    offset = op.params.get("offset", 0)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        _fail("function chain limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        _fail("function chain offset must be a non-negative integer")


def _validate_num_combine(
    expr: ExprRepr,
    kinds: list[ValueKind],
) -> ValueKind:
    if len(kinds) < 2 or any(
        kind is not ValueKind.NUMERIC for kind in kinds
    ):
        _fail("num_combine requires at least two numeric arguments")
    mode = expr.params.get("mode", "sum")
    valid_modes = {"multiply", "sum", "max", "min", "avg", "weighted"}
    if not isinstance(mode, str):
        _fail("num_combine mode must be a string")
    if mode not in valid_modes:
        _fail(f"unsupported num_combine mode: {mode}")
    weights = expr.params.get("weights")
    if mode == "weighted":
        if not _is_non_string_sequence(weights) or len(weights) != len(kinds):
            _fail(f"weighted num_combine requires {len(kinds)} weights")
        if any(not _is_numeric(weight) for weight in weights):
            _fail("num_combine weights must be numeric")
        if any(not _is_finite_numeric(weight) for weight in weights):
            _fail("num_combine weights must be finite")
    elif weights is not None:
        _fail("num_combine weights require mode='weighted'")
    return ValueKind.NUMERIC


def _validate_decay(expr: ExprRepr, kinds: list[ValueKind]) -> ValueKind:
    if len(kinds) != 1 or kinds[0] not in {
        ValueKind.NUMERIC,
        ValueKind.TIMESTAMP,
    }:
        _fail("decay requires exactly one numeric or timestamp argument")
    function = expr.params.get("function")
    if not isinstance(function, str):
        _fail("decay function must be a string")
    if function not in {"gauss", "exp", "linear"}:
        _fail(f"unsupported decay function: {function}")
    origin = expr.params.get("origin")
    scale = expr.params.get("scale")
    offset = expr.params.get("offset", 0)
    decay = expr.params.get("decay", 0.5)
    values = (origin, scale, offset, decay)
    if any(not _is_numeric(value) for value in values):
        _fail("decay parameters must be numeric")
    if any(not _is_finite_numeric(value) for value in values):
        _fail("decay parameters must be finite")
    if not scale > 0 or not offset >= 0 or not 0 < decay < 1:
        _fail("decay requires scale > 0, offset >= 0, and 0 < decay < 1")
    return ValueKind.NUMERIC


def _validate_round_decimal(
    expr: ExprRepr,
    kinds: list[ValueKind],
) -> ValueKind:
    if kinds != [ValueKind.NUMERIC]:
        _fail("round_decimal requires exactly one numeric argument")
    decimal = expr.params.get("decimal")
    if (
        isinstance(decimal, bool)
        or not isinstance(decimal, int)
        or not 0 <= decimal <= 6
    ):
        _fail("round_decimal decimal must be an integer in [0, 6]")
    return ValueKind.NUMERIC


def _validate_rerank_model(
    expr: ExprRepr,
    kinds: list[ValueKind],
    num_queries: int,
) -> ValueKind:
    if kinds != [ValueKind.TEXT]:
        _fail("rerank_model requires exactly one text argument")
    lowered_keys = {key.lower() for key in expr.params if isinstance(key, str)}
    if "base_url" in lowered_keys:
        _fail("rerank_model endpoint must be configured on the server")
    if lowered_keys.intersection({"api_key", "token", "secret"}):
        _fail("rerank_model credentials must be configured on the server")
    queries = expr.params.get("queries")
    if (
        not _is_non_string_sequence(queries)
        or not queries
        or not all(isinstance(query, str) and query for query in queries)
    ):
        _fail("rerank_model queries must be a non-empty sequence of strings")
    if len(queries) != num_queries:
        _fail(
            f"rerank_model query count {len(queries)} does not match "
            f"search nq {num_queries}"
        )
    provider = expr.params.get("provider")
    if not isinstance(provider, str) or not provider:
        _fail("rerank_model provider must be a non-empty string")
    from milvus_lite.rerank.factory import (
        rerank_provider_param_names,
        validate_rerank_provider_params,
    )

    try:
        provider_name = validate_rerank_provider_params(dict(expr.params))
    except ValueError as error:
        raise SchemaValidationError(str(error)) from error
    supported_params = rerank_provider_param_names(provider_name) | {"queries"}
    unsupported_params = sorted(
        (key for key in expr.params if key not in supported_params),
        key=repr,
    )
    if unsupported_params:
        _fail(
            f"unsupported rerank_model parameter {unsupported_params[0]!r}"
        )
    return ValueKind.NUMERIC


def _validate_expr(
    expr: ExprRepr,
    kinds: list[ValueKind],
    num_queries: int,
) -> ValueKind:
    if expr.name == "num_combine":
        return _validate_num_combine(expr, kinds)
    if expr.name == "decay":
        return _validate_decay(expr, kinds)
    if expr.name == "round_decimal":
        return _validate_round_decimal(expr, kinds)
    if expr.name == "rerank_model":
        return _validate_rerank_model(expr, kinds, num_queries)
    _fail(f"unsupported function chain expression: {expr.name}")


def _dtype_name(dtype: object) -> str:
    value = getattr(dtype, "value", None)
    return value if isinstance(value, str) else type(dtype).__name__


def _fail_with_context(context: str, error: SchemaValidationError) -> None:
    raise SchemaValidationError(f"{context}: {error}") from error


def validate_function_chain(
    chain: ChainRepr,
    schema: CollectionSchema,
    num_queries: int,
) -> ValidatedChain:
    if chain.stage != "FunctionChainStageL2Rerank":
        _fail(
            f"function chain stage {chain.stage} is not supported in search request"
        )
    if not chain.ops:
        _fail("function chain must contain at least one operator")

    fields = {field.name: field for field in schema.fields}
    pk_field = next((field for field in schema.fields if field.is_primary), None)
    if pk_field is None:
        _fail("function chain schema has no primary key field")
    if pk_field.name in {ID_FIELD, SCORE_FIELD}:
        _fail(
            f"function chain primary key field {pk_field.name!r} conflicts "
            f"with reserved system column {pk_field.name!r}"
        )
    pk_kind = _FIELD_KINDS.get(pk_field.dtype)
    symbols = {
        SCORE_FIELD: ValueKind.NUMERIC,
        ID_FIELD: pk_kind,
        pk_field.name: pk_kind,
    }
    required: list[str] = []
    required_seen: set[str] = set()

    def resolve(name: str) -> ValueKind:
        kind = symbols.get(name)
        if kind is not None:
            return kind
        if name.startswith("$"):
            _fail(
                f"system input {name!r} is not supported by "
                "L2 rerank function chain"
            )
        field = fields.get(name)
        if field is None:
            _fail(
                f"function chain input {name!r} is neither a previous output "
                "nor a collection field"
            )
        kind = _FIELD_KINDS.get(field.dtype)
        if kind is None:
            _fail(
                f"function chain input field {name!r} has unsupported type "
                f"{_dtype_name(field.dtype)}"
            )
        symbols[name] = kind
        if name not in required_seen:
            required_seen.add(name)
            required.append(name)
        return kind

    for index, op in enumerate(chain.ops):
        op_context = f"function chain op[{index}]"
        if op.op == "map":
            try:
                expr = _validate_map(op)
            except SchemaValidationError as error:
                _fail_with_context(op_context, error)
            try:
                arg_kinds = [
                    resolve(arg.name)
                    if isinstance(arg, ColumnArg)
                    else _literal_kind(arg.value)
                    for arg in expr.args
                    if isinstance(arg, (ColumnArg, LiteralArg))
                ]
                if len(arg_kinds) != len(expr.args):
                    _fail("unsupported function chain expression argument type")
                output_kind = _validate_expr(expr, arg_kinds, num_queries)
            except SchemaValidationError as error:
                _fail_with_context(
                    f"{op_context} expression {expr.name}",
                    error,
                )
            try:
                output = op.outputs[0]
                if output == ID_FIELD or (
                    output.startswith("$") and output != SCORE_FIELD
                ):
                    _fail(
                        f"system output {output!r} is not writable by "
                        "L2 rerank function chain"
                    )
                if output in fields:
                    _fail(
                        f"function chain output {output!r} conflicts with "
                        "a collection field"
                    )
                symbols[output] = output_kind
            except SchemaValidationError as error:
                _fail_with_context(op_context, error)
        elif op.op == "sort":
            try:
                _validate_sort(op)
                for name in op.inputs:
                    resolve(name)
            except SchemaValidationError as error:
                _fail_with_context(op_context, error)
        elif op.op == "limit":
            try:
                _validate_limit(op)
            except SchemaValidationError as error:
                _fail_with_context(op_context, error)
        else:
            _fail(f"{op_context}: unsupported function chain operator: {op.op}")

    return ValidatedChain(chain, tuple(required))
