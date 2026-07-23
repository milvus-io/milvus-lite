# Function Chain API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ordinary Search support for one public PyMilvus `L2_RERANK` Function Chain using the existing Milvus Lite function runtime.

**Architecture:** Decode `SearchRequest.function_chains` into a protobuf-independent representation, validate and plan required schema fields, then compile the plan into the existing `FuncChain`/`DataFrame` runtime. Keep all protobuf handling in the gRPC adapter and preserve separate execution paths for public Function Chains, legacy FunctionScore rerank, and plain Search.

**Tech Stack:** Python 3.10+, dataclasses, PyMilvus generated protobufs, gRPC, pytest, existing Milvus Lite `FuncChain` runtime.

**Design Reference:** `docs/superpowers/specs/2026-07-15-function-chain-api-design.md`

---

## File Map

**Create:**

- `milvus_lite/function/repr.py` — protocol-neutral chain, operator, expression, and argument representations.
- `milvus_lite/adapter/grpc/translators/function_chain.py` — protobuf value and Function Chain decoding.
- `milvus_lite/function/validator.py` — stage, name, schema, operator, expression, and query-count validation.
- `milvus_lite/function/compiler.py` — validated representation to existing `FuncChain` compilation.
- `milvus_lite/function/expr/num_combine.py` — public numeric combination expression including weighted mode.
- `milvus_lite/adapter/grpc/function_chain.py` — ordinary Search preparation, DataFrame conversion, and result projection.
- `tests/function/test_function_chain_proto.py` — real protobuf translation tests.
- `tests/function/test_function_chain_validator.py` — dependency and semantic validation tests.
- `tests/function/test_public_function_chain.py` — compiler and runtime tests.
- `tests/adapter/test_function_chain.py` — PyMilvus-to-gRPC integration tests.

**Modify:**

- `milvus_lite/function/ops/map_op.py` — support column and literal bindings while retaining string input compatibility.
- `milvus_lite/function/ops/sort_op.py` — make tie-breaking optional.
- `milvus_lite/function/chain.py` — keep legacy fluent sorting deterministic by explicitly requesting `$id` tie-breaking.
- `milvus_lite/function/expr/decay_expr.py` — normalize timezone-aware timestamp inputs to Unix seconds.
- `milvus_lite/function/expr/score_combine.py` — delegate legacy modes to `NumCombineExpr`.
- `milvus_lite/function/expr/__init__.py` — export `NumCombineExpr`.
- `milvus_lite/adapter/grpc/translators/search.py` — expose compatibility-safe `function_chains` request data.
- `milvus_lite/adapter/grpc/servicer.py` — add public-chain Search branch and Hybrid Search rejection.
- `docs/modules.md` — document the new Function Chain modules and tests.

---

### Task 1: Add Protocol-Neutral Representation and Protobuf Decoder

**Files:**

- Create: `milvus_lite/function/repr.py`
- Create: `milvus_lite/adapter/grpc/translators/function_chain.py`
- Create: `tests/function/test_function_chain_proto.py`

- [ ] **Step 1: Write failing representation and decoder tests**

Add tests using the Function-Chain-capable sibling PyMilvus checkout:

```python
from pymilvus.grpc_gen import schema_pb2

from milvus_lite.adapter.grpc.translators.function_chain import (
    decode_function_param_value,
    function_chain_to_repr,
)
from milvus_lite.function.repr import ColumnArg, LiteralArg


def test_decode_nested_function_param_value():
    value = schema_pb2.FunctionParamValue(
        object_value=schema_pb2.FunctionParamObject(
            fields={
                "enabled": schema_pb2.FunctionParamValue(bool_value=True),
                "weights": schema_pb2.FunctionParamValue(
                    array_value=schema_pb2.FunctionParamArray(
                        values=[
                            schema_pb2.FunctionParamValue(double_value=0.7),
                            schema_pb2.FunctionParamValue(int64_value=2),
                        ]
                    )
                ),
            }
        )
    )

    assert decode_function_param_value(value) == {
        "enabled": True,
        "weights": [0.7, 2],
    }


def test_function_chain_to_repr_tracks_reads_and_writes():
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
                            column=schema_pb2.FunctionChainColumnArg(name="popularity")
                        ),
                        schema_pb2.FunctionChainExprArg(
                            literal=schema_pb2.FunctionParamValue(double_value=1.0)
                        ),
                    ],
                    params={"mode": schema_pb2.FunctionParamValue(string_value="sum")},
                ),
                outputs=["tmp_score"],
            ),
            schema_pb2.FunctionChainOp(
                op="sort",
                inputs=["tmp_score", "$id"],
                params={
                    "column": schema_pb2.FunctionParamValue(string_value="tmp_score"),
                    "desc": schema_pb2.FunctionParamValue(bool_value=True),
                    "tie_break_col": schema_pb2.FunctionParamValue(string_value="$id"),
                },
            ),
        ],
    )

    result = function_chain_to_repr(chain)

    assert result.name == "freshness"
    assert result.stage == "FunctionChainStageL2Rerank"
    assert isinstance(result.ops[0].expr.args[0], ColumnArg)
    assert isinstance(result.ops[0].expr.args[2], LiteralArg)
    assert result.ops[0].read_names == ("$score", "popularity")
    assert result.ops[0].write_names == ("tmp_score",)
    assert result.info.required_inputs == ("$score", "popularity", "$id")


def test_decode_rejects_unset_parameter_value():
    value = schema_pb2.FunctionParamValue()

    with pytest.raises(SchemaValidationError, match="value is not set"):
        decode_function_param_value(value)
```

Also add parameterized failures for empty operator names, empty column names, empty expression names, and unset expression argument oneofs.

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
PYTHONPATH=../pymilvus pytest tests/function/test_function_chain_proto.py -v
```

Expected: collection fails with `ModuleNotFoundError` for `milvus_lite.function.repr` or the new translator module.

- [ ] **Step 3: Implement the protocol-neutral representation**

Create `milvus_lite/function/repr.py` with these concrete types and dependency analysis:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias


@dataclass(frozen=True)
class ColumnArg:
    name: str


@dataclass(frozen=True)
class LiteralArg:
    value: object


ExprArg: TypeAlias = ColumnArg | LiteralArg


@dataclass(frozen=True)
class ExprRepr:
    name: str
    args: tuple[ExprArg, ...]
    params: Mapping[str, object]


@dataclass(frozen=True)
class OpRepr:
    op: str
    expr: ExprRepr | None
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: Mapping[str, object]
    read_names: tuple[str, ...]
    write_names: tuple[str, ...]


@dataclass(frozen=True)
class ChainInfo:
    required_inputs: tuple[str, ...]
    written_names: tuple[str, ...]


@dataclass(frozen=True)
class ChainRepr:
    name: str
    stage: str
    ops: tuple[OpRepr, ...]
    info: ChainInfo


def build_chain_info(ops: tuple[OpRepr, ...]) -> ChainInfo:
    available: set[str] = set()
    required: list[str] = []
    required_seen: set[str] = set()
    written: list[str] = []
    written_seen: set[str] = set()

    for op in ops:
        for name in op.read_names:
            if name not in available and name not in required_seen:
                required_seen.add(name)
                required.append(name)
        for name in op.write_names:
            available.add(name)
            if name not in written_seen:
                written_seen.add(name)
                written.append(name)

    return ChainInfo(tuple(required), tuple(written))
```

Do not seed `available` with `$id` or `$score`; they must remain visible in `required_inputs` so the schema-aware validator can classify them as runtime inputs.

- [ ] **Step 4: Implement protobuf decoding**

Create `milvus_lite/adapter/grpc/translators/function_chain.py` with:

```python
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
    kind = value.WhichOneof("value")
    if kind is None:
        raise SchemaValidationError("function parameter value is not set")
    if kind in {"bool_value", "int64_value", "double_value", "string_value", "bytes_value"}:
        return getattr(value, kind)
    if kind == "array_value":
        return [decode_function_param_value(item) for item in value.array_value.values]
    if kind == "object_value":
        decoded = {}
        for key, item in value.object_value.fields.items():
            if not key:
                raise SchemaValidationError("function parameter object key must not be empty")
            decoded[key] = decode_function_param_value(item)
        return decoded
    raise SchemaValidationError(f"unsupported function parameter value kind: {kind}")


def _decode_expr_arg(arg, path: str):
    kind = arg.WhichOneof("arg")
    if kind == "column":
        if not arg.column.name:
            raise SchemaValidationError(f"{path} column name must not be empty")
        return ColumnArg(arg.column.name)
    if kind == "literal":
        return LiteralArg(decode_function_param_value(arg.literal))
    raise SchemaValidationError(f"{path} argument is not set")


def _decode_params(params) -> dict[str, object]:
    return {key: decode_function_param_value(value) for key, value in params.items()}


def _decode_expr(expr, path: str) -> ExprRepr:
    if not expr.name:
        raise SchemaValidationError(f"{path} expression name must not be empty")
    return ExprRepr(
        name=expr.name,
        args=tuple(
            _decode_expr_arg(arg, f"{path}.args[{index}]")
            for index, arg in enumerate(expr.args)
        ),
        params=_decode_params(expr.params),
    )


def _stage_name(chain_pb) -> str:
    field = chain_pb.DESCRIPTOR.fields_by_name["stage"]
    value = field.enum_type.values_by_number.get(chain_pb.stage)
    if value is None:
        raise SchemaValidationError(f"unknown function chain stage value: {chain_pb.stage}")
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
        if any(not name for name in inputs):
            raise SchemaValidationError(f"{path} input name must not be empty")
        if any(not name for name in outputs):
            raise SchemaValidationError(f"{path} output name must not be empty")
        read_names = (
            tuple(arg.name for arg in expr.args if isinstance(arg, ColumnArg))
            if expr is not None
            else inputs
        )
        ops.append(
            OpRepr(
                op=op_pb.op,
                expr=expr,
                inputs=inputs,
                outputs=outputs,
                params=_decode_params(op_pb.params),
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
```

- [ ] **Step 5: Run decoder tests**

Run:

```bash
PYTHONPATH=../pymilvus pytest tests/function/test_function_chain_proto.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the representation and decoder**

```bash
git add milvus_lite/function/repr.py milvus_lite/adapter/grpc/translators/function_chain.py tests/function/test_function_chain_proto.py
git commit -m "feat: decode public function chain protobuf"
```

---

### Task 2: Add Schema-Aware Validation and Dependency Planning

**Files:**

- Create: `milvus_lite/function/validator.py`
- Create: `tests/function/test_function_chain_validator.py`

- [ ] **Step 1: Write failing validator tests**

Use direct `ChainRepr` construction so these tests do not depend on protobuf:

```python
from milvus_lite.function.repr import (
    ChainRepr,
    ColumnArg,
    ExprRepr,
    OpRepr,
    build_chain_info,
)
from milvus_lite.function.validator import validate_function_chain
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def _schema():
    return CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
            FieldSchema("published_at", DataType.INT64),
            FieldSchema("popularity", DataType.FLOAT),
            FieldSchema("doc", DataType.VARCHAR, max_length=512),
            FieldSchema("metadata", DataType.JSON),
        ]
    )


def _chain(*ops):
    op_tuple = tuple(ops)
    return ChainRepr(
        name="test",
        stage="FunctionChainStageL2Rerank",
        ops=op_tuple,
        info=build_chain_info(op_tuple),
    )


def test_validator_plans_hidden_fields_and_temporary_outputs():
    first = OpRepr(
        op="map",
        expr=ExprRepr(
            "decay",
            (ColumnArg("published_at"),),
            {"function": "exp", "origin": 100, "scale": 10, "offset": 0, "decay": 0.5},
        ),
        inputs=(),
        outputs=("freshness",),
        params={},
        read_names=("published_at",),
        write_names=("freshness",),
    )
    second = OpRepr(
        op="map",
        expr=ExprRepr(
            "num_combine",
            (ColumnArg("$score"), ColumnArg("freshness"), ColumnArg("popularity")),
            {"mode": "sum"},
        ),
        inputs=(),
        outputs=("$score",),
        params={},
        read_names=("$score", "freshness", "popularity"),
        write_names=("$score",),
    )

    validated = validate_function_chain(_chain(first, second), _schema(), num_queries=1)

    assert validated.required_schema_fields == ("published_at", "popularity")


@pytest.mark.parametrize("output", ["$id", "$temporary"])
def test_validator_rejects_readonly_or_unknown_system_outputs(output):
    op = OpRepr(
        op="map",
        expr=ExprRepr("round_decimal", (ColumnArg("$score"),), {"decimal": 2}),
        inputs=(),
        outputs=(output,),
        params={},
        read_names=("$score",),
        write_names=(output,),
    )

    with pytest.raises(SchemaValidationError, match="system output"):
        validate_function_chain(_chain(op), _schema(), num_queries=1)


def test_validator_rejects_rerank_query_count_mismatch():
    op = OpRepr(
        op="map",
        expr=ExprRepr(
            "rerank_model",
            (ColumnArg("doc"),),
            {"provider": "cohere", "queries": ["one"]},
        ),
        inputs=(),
        outputs=("$score",),
        params={},
        read_names=("doc",),
        write_names=("$score",),
    )

    with pytest.raises(SchemaValidationError, match="query count"):
        validate_function_chain(_chain(op), _schema(), num_queries=2)
```

Add tests for unsupported stage, empty chain, unknown operator, bad sort wire shape, invalid limit, unknown field, vector/JSON inputs, schema-field writes, invalid expression arity, weighted weights, credential parameters, and string primary keys used as numeric inputs.

- [ ] **Step 2: Run validator tests and verify they fail**

Run:

```bash
pytest tests/function/test_function_chain_validator.py -v
```

Expected: collection fails because `milvus_lite.function.validator` does not exist.

- [ ] **Step 3: Implement validator public types and stage/name rules**

Create `milvus_lite/function/validator.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.function.repr import ChainRepr, ColumnArg, ExprRepr, LiteralArg, OpRepr
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


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _literal_kind(value: object) -> ValueKind:
    if isinstance(value, bool):
        return ValueKind.BOOL
    if isinstance(value, (int, float)):
        return ValueKind.NUMERIC
    if isinstance(value, str):
        return ValueKind.TEXT
    _fail(f"unsupported function chain literal type: {type(value).__name__}")
```

Validate exactly one supported stage string, at least one op, `$id` read-only, `$score` read/write, and reject every other `$`-prefixed name.

- [ ] **Step 4: Implement operator and expression validation**

Add explicit helpers with these contracts:

```python
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
    expected = (column,) if tie_break is None else (column, tie_break)
    if not isinstance(column, str) or not column:
        _fail("sort column must be a non-empty string")
    if not isinstance(desc, bool):
        _fail("sort desc must be a boolean")
    if tie_break is not None and (not isinstance(tie_break, str) or not tie_break):
        _fail("sort tie_break_col must be a non-empty string")
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
```

Expression validation must enforce:

```python
def _validate_num_combine(expr: ExprRepr, kinds: list[ValueKind]) -> ValueKind:
    if len(kinds) < 2 or any(kind is not ValueKind.NUMERIC for kind in kinds):
        _fail("num_combine requires at least two numeric arguments")
    mode = expr.params.get("mode", "sum")
    valid_modes = {"multiply", "sum", "max", "min", "avg", "weighted"}
    if mode not in valid_modes:
        _fail(f"unsupported num_combine mode: {mode}")
    weights = expr.params.get("weights")
    if mode == "weighted":
        if not isinstance(weights, list) or len(weights) != len(kinds):
            _fail(f"weighted num_combine requires {len(kinds)} weights")
        if any(isinstance(weight, bool) or not isinstance(weight, (int, float)) for weight in weights):
            _fail("num_combine weights must be numeric")
    elif weights is not None:
        _fail("num_combine weights require mode='weighted'")
    return ValueKind.NUMERIC
```

Use these concrete helpers and dispatcher:

```python
def _validate_decay(expr: ExprRepr, kinds: list[ValueKind]) -> ValueKind:
    if len(kinds) != 1 or kinds[0] not in {ValueKind.NUMERIC, ValueKind.TIMESTAMP}:
        _fail("decay requires exactly one numeric or timestamp argument")
    function = expr.params.get("function")
    if function not in {"gauss", "exp", "linear"}:
        _fail(f"unsupported decay function: {function}")
    origin = expr.params.get("origin")
    scale = expr.params.get("scale")
    offset = expr.params.get("offset", 0)
    decay = expr.params.get("decay", 0.5)
    values = {"origin": origin, "scale": scale, "offset": offset, "decay": decay}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values.values()):
        _fail("decay parameters must be numeric")
    if scale <= 0 or offset < 0 or not 0 < decay < 1:
        _fail("decay requires scale > 0, offset >= 0, and 0 < decay < 1")
    return ValueKind.NUMERIC


def _validate_round_decimal(expr: ExprRepr, kinds: list[ValueKind]) -> ValueKind:
    if kinds != [ValueKind.NUMERIC]:
        _fail("round_decimal requires exactly one numeric argument")
    decimal = expr.params.get("decimal")
    if isinstance(decimal, bool) or not isinstance(decimal, int) or not 0 <= decimal <= 6:
        _fail("round_decimal decimal must be an integer in [0, 6]")
    return ValueKind.NUMERIC


def _validate_rerank_model(
    expr: ExprRepr,
    kinds: list[ValueKind],
    num_queries: int,
) -> ValueKind:
    if kinds != [ValueKind.TEXT]:
        _fail("rerank_model requires exactly one text argument")
    lowered_keys = {key.lower() for key in expr.params}
    forbidden = lowered_keys.intersection({"api_key", "token", "secret"})
    if forbidden:
        _fail("rerank_model credentials must be configured on the server")
    queries = expr.params.get("queries")
    if not isinstance(queries, list) or not queries or not all(
        isinstance(query, str) and query for query in queries
    ):
        _fail("rerank_model queries must be a non-empty list of strings")
    if len(queries) != num_queries:
        _fail(
            f"rerank_model query count {len(queries)} does not match search nq {num_queries}"
        )
    provider = expr.params.get("provider")
    if not isinstance(provider, str) or not provider:
        _fail("rerank_model provider must be a non-empty string")
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
```

- [ ] **Step 5: Implement sequential symbol and required-field planning**

Implement `validate_function_chain()` as a single forward scan:

```python
def validate_function_chain(
    chain: ChainRepr,
    schema: CollectionSchema,
    num_queries: int,
) -> ValidatedChain:
    if chain.stage != "FunctionChainStageL2Rerank":
        _fail(f"function chain stage {chain.stage} is not supported in search request")
    if not chain.ops:
        _fail("function chain must contain at least one operator")

    fields = {field.name: field for field in schema.fields}
    pk_field = next(field for field in schema.fields if field.is_primary)
    symbols = {
        SCORE_FIELD: ValueKind.NUMERIC,
        ID_FIELD: _FIELD_KINDS.get(pk_field.dtype),
    }
    required: list[str] = []
    required_seen: set[str] = set()

    def resolve(name: str) -> ValueKind:
        if name in symbols and symbols[name] is not None:
            return symbols[name]
        if name.startswith("$"):
            _fail(f"system input {name!r} is not supported by L2 rerank function chain")
        field = fields.get(name)
        if field is None:
            _fail(f"function chain input {name!r} is neither a previous output nor a collection field")
        kind = _FIELD_KINDS.get(field.dtype)
        if kind is None:
            _fail(f"function chain input field {name!r} has unsupported type {field.dtype.value}")
        symbols[name] = kind
        if name not in required_seen:
            required_seen.add(name)
            required.append(name)
        return kind

    for index, op in enumerate(chain.ops):
        if op.op == "map":
            expr = _validate_map(op)
            arg_kinds = [
                resolve(arg.name) if isinstance(arg, ColumnArg) else _literal_kind(arg.value)
                for arg in expr.args
            ]
            output_kind = _validate_expr(expr, arg_kinds, num_queries)
            output = op.outputs[0]
            if output == ID_FIELD or (output.startswith("$") and output != SCORE_FIELD):
                _fail(f"system output {output!r} is not writable by L2 rerank function chain")
            if output in fields:
                _fail(f"function chain output {output!r} conflicts with a collection field")
            symbols[output] = output_kind
        elif op.op == "sort":
            _validate_sort(op)
            for name in op.inputs:
                resolve(name)
        elif op.op == "limit":
            _validate_limit(op)
        else:
            _fail(f"unsupported function chain operator: {op.op}")

    return ValidatedChain(chain, tuple(required))
```

Use an explicit primary-key-not-found validation instead of relying on `next()` to raise `StopIteration`.

- [ ] **Step 6: Run validator tests**

Run:

```bash
pytest tests/function/test_function_chain_validator.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit validation and planning**

```bash
git add milvus_lite/function/validator.py tests/function/test_function_chain_validator.py
git commit -m "feat: validate and plan function chains"
```

---

### Task 3: Add Literal-Aware Map Inputs

**Files:**

- Modify: `milvus_lite/function/ops/map_op.py:1`
- Modify: `tests/function/test_map_op.py:1`

- [ ] **Step 1: Add failing literal binding tests**

Append:

```python
from milvus_lite.function.ops.map_op import ColumnBinding, LiteralBinding


def test_map_op_expands_literal_per_chunk():
    op = MapOp(
        _ConcatExpr(),
        [ColumnBinding("a"), LiteralBinding("!")],
        ["result"],
    )
    df = DataFrame([[{"a": "one"}, {"a": "two"}], [{"a": "three"}]])

    op.execute(FuncContext(STAGE_INGESTION), df)

    assert df.column("result", 0) == ["one!", "two!"]
    assert df.column("result", 1) == ["three!"]


def test_map_op_keeps_string_input_compatibility():
    op = MapOp(_DoubleExpr(), ["x"], ["y"])
    assert op.input_cols == ["x"]
```

- [ ] **Step 2: Run focused tests and verify the new test fails**

Run:

```bash
pytest tests/function/test_map_op.py -v
```

Expected: import failure for `ColumnBinding` and `LiteralBinding`.

- [ ] **Step 3: Implement input bindings without breaking callers**

Update `map_op.py`:

```python
from dataclasses import dataclass
from typing import List, TypeAlias


@dataclass(frozen=True)
class ColumnBinding:
    name: str


@dataclass(frozen=True)
class LiteralBinding:
    value: object


InputBinding: TypeAlias = ColumnBinding | LiteralBinding
InputSpec: TypeAlias = str | InputBinding
```

Normalize constructor inputs and resolve them per chunk:

```python
def __init__(self, expr, input_cols: List[InputSpec], output_cols: List[str]) -> None:
    self._expr = expr
    self._input_bindings = [
        ColumnBinding(value) if isinstance(value, str) else value
        for value in input_cols
    ]
    self._output_cols = output_cols


@property
def input_cols(self) -> List[str]:
    return [
        binding.name
        for binding in self._input_bindings
        if isinstance(binding, ColumnBinding)
    ]


def _resolve_input(self, binding: InputBinding, df: DataFrame, chunk_idx: int) -> list:
    if isinstance(binding, ColumnBinding):
        return df.column(binding.name, chunk_idx)
    return [binding.value] * len(df.chunk(chunk_idx))
```

Replace the existing input read with:

```python
inputs = [
    self._resolve_input(binding, df, chunk_idx)
    for binding in self._input_bindings
]
```

- [ ] **Step 4: Run MapOp and chain regression tests**

Run:

```bash
pytest tests/function/test_map_op.py tests/function/test_chain.py tests/function/test_builder_ingestion.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit literal-aware MapOp**

```bash
git add milvus_lite/function/ops/map_op.py tests/function/test_map_op.py
git commit -m "feat: support literal function chain inputs"
```

---

### Task 4: Implement Public Numeric Combination

**Files:**

- Create: `milvus_lite/function/expr/num_combine.py`
- Modify: `milvus_lite/function/expr/score_combine.py:1`
- Modify: `milvus_lite/function/expr/__init__.py:1`
- Create: `tests/function/test_public_function_chain.py`
- Modify: `tests/function/test_score_combine.py:1`

- [ ] **Step 1: Write failing `NumCombineExpr` tests**

Create `tests/function/test_public_function_chain.py` with:

```python
from milvus_lite.function.expr.num_combine import NumCombineExpr
from milvus_lite.function.types import FuncContext, STAGE_L2_RERANK


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("multiply", [6.0, 20.0]),
        ("sum", [5.0, 9.0]),
        ("max", [3.0, 5.0]),
        ("min", [2.0, 4.0]),
        ("avg", [2.5, 4.5]),
    ],
)
def test_num_combine_modes(mode, expected):
    result = NumCombineExpr(mode).execute(_ctx(), [[2.0, 4.0], [3.0, 5.0]])
    assert result == [expected]


def test_num_combine_weighted():
    result = NumCombineExpr("weighted", [0.25, 0.75]).execute(
        _ctx(), [[2.0], [6.0]]
    )
    assert result == [[5.0]]


def test_num_combine_none_becomes_zero():
    result = NumCombineExpr("sum").execute(_ctx(), [[None], [2.0]])
    assert result == [[0.0]]
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
pytest tests/function/test_public_function_chain.py -v
```

Expected: import failure for `NumCombineExpr`.

- [ ] **Step 3: Implement `NumCombineExpr`**

Create:

```python
from __future__ import annotations

from typing import FrozenSet, List, Optional

from milvus_lite.function.types import STAGE_L2_RERANK, FuncContext, FunctionExpr


class NumCombineExpr(FunctionExpr):
    name = "num_combine"
    supported_stages: FrozenSet[str] = frozenset({STAGE_L2_RERANK})
    _VALID_MODES = frozenset({"multiply", "sum", "max", "min", "avg", "weighted"})

    def __init__(self, mode: str = "sum", weights: Optional[List[float]] = None) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"unknown num_combine mode: {mode!r}")
        if mode == "weighted":
            if not weights:
                raise ValueError("weighted num_combine requires weights")
            self._weights = [float(weight) for weight in weights]
        elif weights is not None:
            raise ValueError("num_combine weights require weighted mode")
        else:
            self._weights = None
        self._mode = mode

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        if not inputs:
            return [[]]
        if self._weights is not None and len(self._weights) != len(inputs):
            raise ValueError("num_combine weights must match input count")
        output = []
        for row_index in range(len(inputs[0])):
            values = [column[row_index] for column in inputs]
            if any(value is None for value in values):
                output.append(0.0)
            elif self._mode == "multiply":
                result = 1.0
                for value in values:
                    result *= value
                output.append(result)
            elif self._mode == "sum":
                output.append(sum(values))
            elif self._mode == "max":
                output.append(max(values))
            elif self._mode == "min":
                output.append(min(values))
            elif self._mode == "avg":
                output.append(sum(values) / len(values))
            else:
                output.append(sum(value * weight for value, weight in zip(values, self._weights)))
        return [output]
```

- [ ] **Step 4: Delegate legacy score combination and export the new class**

Change `ScoreCombineExpr` to subclass or wrap `NumCombineExpr` while retaining:

```python
class ScoreCombineExpr(NumCombineExpr):
    name = "score_combine"

    def __init__(self, mode: str = "multiply") -> None:
        if mode == "weighted":
            raise ValueError("ScoreCombineExpr does not support weighted mode")
        super().__init__(mode=mode)
```

Export `NumCombineExpr` from `milvus_lite/function/expr/__init__.py`.

- [ ] **Step 5: Run public and legacy expression tests**

Run:

```bash
pytest tests/function/test_public_function_chain.py tests/function/test_score_combine.py tests/function/test_builder_rerank.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit numeric combination support**

```bash
git add milvus_lite/function/expr/num_combine.py milvus_lite/function/expr/score_combine.py milvus_lite/function/expr/__init__.py tests/function/test_public_function_chain.py tests/function/test_score_combine.py
git commit -m "feat: add public numeric combine expression"
```

---

### Task 5: Make Public Sort Tie-Breaking Optional

**Files:**

- Modify: `milvus_lite/function/ops/sort_op.py:15`
- Modify: `milvus_lite/function/chain.py:55`
- Modify: `tests/function/test_sort_op.py:1`
- Modify: `tests/function/test_chain.py:1`

- [ ] **Step 1: Add failing stable-sort and explicit tie-break tests**

Update tests to distinguish public and legacy behavior:

```python
def test_sort_without_tie_break_preserves_equal_input_order():
    df = DataFrame([[_hit(3, 0.5), _hit(1, 0.5), _hit(2, 0.5)]])

    SortOp(SCORE_FIELD, desc=True, tie_break_col=None).execute(_ctx(), df)

    assert [hit[ID_FIELD] for hit in df.chunk(0)] == [3, 1, 2]


def test_sort_with_explicit_id_tie_break():
    df = DataFrame([[_hit(3, 0.5), _hit(1, 0.5), _hit(2, 0.5)]])

    SortOp(SCORE_FIELD, desc=True, tie_break_col=ID_FIELD).execute(_ctx(), df)

    assert [hit[ID_FIELD] for hit in df.chunk(0)] == [1, 2, 3]
```

Add a chain-level regression proving `FuncChain.sort()` retains the old `$id` behavior.

- [ ] **Step 2: Run sort tests and verify the stable-sort test fails**

Run:

```bash
pytest tests/function/test_sort_op.py tests/function/test_chain.py -v
```

Expected: the no-tie-break case sorts by ID because current `SortOp` defaults to `$id`.

- [ ] **Step 3: Generalize `SortOp` and preserve legacy fluent behavior**

Change the constructor to:

```python
def __init__(
    self,
    column: str,
    desc: bool = True,
    tie_break_col: str | None = None,
) -> None:
    self._column = column
    self._desc = desc
    self._tie_break_col = tie_break_col
```

In the comparator, when primary values are equal and `tie_break_col is None`, return `0`; otherwise compare the tie-break values ascending. Keep `None` primary values last.

Change legacy `FuncChain.sort()` to:

```python
return self.add(SortOp(column, desc, tie_break_col=ID_FIELD))
```

Import `ID_FIELD` in `chain.py`.

- [ ] **Step 4: Run sort, chain, and rerank regression tests**

Run:

```bash
pytest tests/function/test_sort_op.py tests/function/test_chain.py tests/function/test_builder_rerank.py tests/rerank -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit optional tie-breaking**

```bash
git add milvus_lite/function/ops/sort_op.py milvus_lite/function/chain.py tests/function/test_sort_op.py tests/function/test_chain.py
git commit -m "feat: support optional function chain tie breaking"
```

---

### Task 6: Compile Validated Chains Into the Existing Runtime

**Files:**

- Create: `milvus_lite/function/compiler.py`
- Modify: `milvus_lite/function/expr/decay_expr.py:45`
- Modify: `tests/function/test_public_function_chain.py:1`

- [ ] **Step 1: Add failing compiler tests**

Add helpers that build a validated chain and verify operator order, literal expansion, score rewriting, sort, and limit:

```python
from milvus_lite.function.compiler import compile_function_chain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.repr import ColumnArg, ExprRepr, LiteralArg, OpRepr
from milvus_lite.function.validator import ValidatedChain


def test_compile_and_execute_ordered_public_chain():
    ops = (
        OpRepr(
            op="map",
            expr=ExprRepr(
                "num_combine",
                (ColumnArg("$score"), ColumnArg("popularity"), LiteralArg(1.0)),
                {"mode": "sum"},
            ),
            inputs=(),
            outputs=("$score",),
            params={},
            read_names=("$score", "popularity"),
            write_names=("$score",),
        ),
        OpRepr(
            op="sort",
            expr=None,
            inputs=("$score", "$id"),
            outputs=(),
            params={"column": "$score", "desc": True, "tie_break_col": "$id"},
            read_names=("$score", "$id"),
            write_names=(),
        ),
        OpRepr(
            op="limit",
            expr=None,
            inputs=(),
            outputs=(),
            params={"limit": 2, "offset": 0},
            read_names=(),
            write_names=(),
        ),
    )
    chain_repr = ChainRepr(
        name="public",
        stage="FunctionChainStageL2Rerank",
        ops=ops,
        info=build_chain_info(ops),
    )
    compiled = compile_function_chain(ValidatedChain(chain_repr, ("popularity",)))
    frame = DataFrame([[{
        "$id": 1,
        "$score": 0.2,
        "popularity": 1.0,
    }, {
        "$id": 2,
        "$score": 0.1,
        "popularity": 5.0,
    }, {
        "$id": 3,
        "$score": 0.3,
        "popularity": 2.0,
    }]])

    result = compiled.execute(frame)

    assert [row["$id"] for row in result.chunk(0)] == [2, 3]
    assert [row["$score"] for row in result.chunk(0)] == [6.1, 3.3]
```

Add tests for decay, round-decimal, a mocked rerank provider, and sort without a tie-break column.

Add a timestamp conversion test:

```python
from datetime import datetime, timezone


def test_decay_accepts_timestamptz_values():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expr = DecayExpr(
        function="exp",
        origin=origin.timestamp(),
        scale=3600,
        decay=0.5,
    )

    result = expr.execute(_ctx(), [[origin, datetime(2026, 1, 1, 1, tzinfo=timezone.utc)]])

    assert result[0][0] == pytest.approx(1.0)
    assert result[0][1] == pytest.approx(0.5)
```

- [ ] **Step 2: Run compiler tests and verify failure**

Run:

```bash
pytest tests/function/test_public_function_chain.py -v
```

Expected: import failure for `milvus_lite.function.compiler`.

- [ ] **Step 3: Implement explicit expression builders**

Create `compiler.py` with these helpers:

```python
from milvus_lite.function.chain import FuncChain
from milvus_lite.function.expr.decay_expr import DecayExpr
from milvus_lite.function.expr.num_combine import NumCombineExpr
from milvus_lite.function.expr.rerank_model import RerankModelExpr
from milvus_lite.function.expr.round_decimal import RoundDecimalExpr
from milvus_lite.function.ops.limit_op import LimitOp
from milvus_lite.function.ops.map_op import ColumnBinding, LiteralBinding, MapOp
from milvus_lite.function.ops.sort_op import SortOp
from milvus_lite.function.repr import ColumnArg, ExprRepr, LiteralArg
from milvus_lite.function.types import STAGE_L2_RERANK
from milvus_lite.function.validator import ValidatedChain


def _bindings(expr: ExprRepr):
    return [
        ColumnBinding(arg.name) if isinstance(arg, ColumnArg) else LiteralBinding(arg.value)
        for arg in expr.args
    ]


def _build_decay(expr: ExprRepr):
    return DecayExpr(
        function=expr.params["function"],
        origin=expr.params["origin"],
        scale=expr.params["scale"],
        offset=expr.params.get("offset", 0.0),
        decay=expr.params.get("decay", 0.5),
    )


def _build_num_combine(expr: ExprRepr):
    return NumCombineExpr(
        mode=expr.params.get("mode", "sum"),
        weights=expr.params.get("weights"),
    )


def _build_round_decimal(expr: ExprRepr):
    return RoundDecimalExpr(decimal=expr.params["decimal"])


def _build_rerank_model(expr: ExprRepr):
    from milvus_lite.rerank.factory import create_rerank_provider

    params = dict(expr.params)
    queries = list(params.pop("queries"))
    provider = create_rerank_provider(params)
    return RerankModelExpr(provider, query_texts=queries)


_EXPR_BUILDERS = {
    "decay": _build_decay,
    "num_combine": _build_num_combine,
    "round_decimal": _build_round_decimal,
    "rerank_model": _build_rerank_model,
}
```

Update `DecayExpr.execute()` so timestamps use Unix seconds:

```python
from datetime import datetime


def _numeric_decay_value(value) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("TIMESTAMPTZ decay input must be timezone-aware")
        return value.timestamp()
    return float(value)
```

Call `_numeric_decay_value(val)` before `compute_factor()`.

- [ ] **Step 4: Implement ordered operator compilation**

Add:

```python
def compile_function_chain(validated: ValidatedChain) -> FuncChain:
    result = FuncChain(validated.repr.name, STAGE_L2_RERANK)
    for op in validated.repr.ops:
        if op.op == "map":
            expr = _EXPR_BUILDERS[op.expr.name](op.expr)
            result.add(MapOp(expr, _bindings(op.expr), list(op.outputs)))
        elif op.op == "sort":
            result.add(
                SortOp(
                    column=op.params["column"],
                    desc=op.params.get("desc", True),
                    tie_break_col=op.params.get("tie_break_col"),
                )
            )
        elif op.op == "limit":
            result.add(LimitOp(op.params["limit"], op.params.get("offset", 0)))
        else:
            raise ValueError(f"unsupported validated function chain operator: {op.op}")
    return result
```

The compiler may assume validation has already succeeded; unknown validated entries indicate an internal error rather than a client validation error.

- [ ] **Step 5: Run compiler and runtime tests**

Run:

```bash
pytest tests/function/test_public_function_chain.py tests/function/test_map_op.py tests/function/test_sort_op.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the compiler**

```bash
git add milvus_lite/function/compiler.py milvus_lite/function/expr/decay_expr.py tests/function/test_public_function_chain.py
git commit -m "feat: compile public function chains"
```

---

### Task 7: Add Search-Specific Planning and Result Projection

**Files:**

- Create: `milvus_lite/adapter/grpc/function_chain.py`
- Modify: `tests/function/test_public_function_chain.py:1`

- [ ] **Step 1: Add failing Search helper tests**

Test API conflicts, hidden-field merging, DataFrame execution, and temporary-field removal:

```python
from milvus_lite.adapter.grpc.function_chain import (
    execute_search_function_chain,
    merge_internal_output_fields,
    prepare_search_function_chain,
)


def test_merge_internal_output_fields_preserves_user_order():
    assert merge_internal_output_fields(["title"], ("popularity", "title")) == [
        "title",
        "popularity",
    ]
    assert merge_internal_output_fields(None, ("popularity",)) is None


def test_prepare_rejects_function_score_conflict(function_chain_proto, schema):
    with pytest.raises(SchemaValidationError, match="cannot be used together"):
        prepare_search_function_chain(
            function_chains=[function_chain_proto],
            has_function_score=True,
            schema=schema,
            num_queries=1,
            requested_output_fields=["title"],
        )


def test_execute_projects_only_requested_collection_fields(public_chain_plan, schema):
    results = [[{
        "id": 1,
        "distance": 0.2,
        "entity": {"title": "one", "popularity": 3.0},
    }]]

    reranked = execute_search_function_chain(
        public_chain_plan,
        results,
        metric_type="IP",
        schema=schema,
        primary_field_name="id",
    )

    assert reranked[0][0]["entity"] == {"title": "one"}
    assert "tmp_score" not in reranked[0][0]["entity"]
```

- [ ] **Step 2: Run helper tests and verify failure**

Run:

```bash
PYTHONPATH=../pymilvus pytest tests/function/test_public_function_chain.py -v
```

Expected: import failure for `milvus_lite.adapter.grpc.function_chain`.

- [ ] **Step 3: Implement request preparation**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from milvus_lite.adapter.grpc.translators.function_chain import function_chain_to_repr
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.function.chain import FuncChain
from milvus_lite.function.compiler import compile_function_chain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD
from milvus_lite.function.validator import validate_function_chain


@dataclass(frozen=True)
class SearchFunctionChainPlan:
    chain: FuncChain
    required_fields: tuple[str, ...]
    requested_output_fields: tuple[str, ...] | None


def merge_internal_output_fields(
    requested_output_fields: Optional[list[str]],
    required_fields: tuple[str, ...],
) -> Optional[list[str]]:
    if requested_output_fields is None:
        return None
    return list(dict.fromkeys([*requested_output_fields, *required_fields]))


def prepare_search_function_chain(
    *,
    function_chains,
    has_function_score: bool,
    schema,
    num_queries: int,
    requested_output_fields: Optional[list[str]],
) -> SearchFunctionChainPlan | None:
    chains = list(function_chains or ())
    if not chains:
        return None
    if has_function_score:
        raise SchemaValidationError(
            "function_score and function_chains cannot be used together"
        )
    if len(chains) != 1:
        raise SchemaValidationError("ordinary search supports exactly one function chain")
    representation = function_chain_to_repr(chains[0])
    validated = validate_function_chain(representation, schema, num_queries)
    return SearchFunctionChainPlan(
        chain=compile_function_chain(validated),
        required_fields=validated.required_schema_fields,
        requested_output_fields=(
            tuple(requested_output_fields) if requested_output_fields is not None else None
        ),
    )
```

- [ ] **Step 4: Implement safe DataFrame conversion and projection**

Add:

```python
def execute_search_function_chain(
    plan: SearchFunctionChainPlan,
    results: list[list[dict]],
    *,
    metric_type: str,
    schema,
    primary_field_name: str,
    group_by_field: str | None = None,
) -> list[list[dict]]:
    chunks = []
    for query_hits in results:
        chunk = []
        for hit in query_hits:
            row = {
                ID_FIELD: hit["id"],
                SCORE_FIELD: hit_score_for_chain(hit, metric_type),
            }
            row.update(hit.get("entity", {}))
            chunk.append(row)
        chunks.append(chunk)

    output = plan.chain.execute(DataFrame(chunks))
    schema_names = {field.name for field in schema.fields if field.name != primary_field_name}
    allowed = (
        schema_names
        if plan.requested_output_fields is None
        else schema_names.intersection(plan.requested_output_fields)
    )

    reranked = []
    for chunk_index in range(output.num_chunks):
        hits = []
        for row in output.chunk(chunk_index):
            hit = {
                "id": row[ID_FIELD],
                "distance": row[SCORE_FIELD],
                "entity": {name: row[name] for name in allowed if name in row},
            }
            if group_by_field is not None and group_by_field in row:
                hit["_group_by_value"] = row[group_by_field]
            hits.append(hit)
        reranked.append(hits)
    return reranked
```

Define `hit_score_for_chain()` directly in this helper module and remove the existing private definition from `servicer.py` during Task 8:

```python
def hit_score_for_chain(hit: dict, metric_type: str) -> float:
    distance = hit["distance"]
    return -distance if metric_type.upper() == "BM25" else distance
```

Use `hit_score_for_chain()` in `execute_search_function_chain()`.

- [ ] **Step 5: Run Search helper tests**

Run:

```bash
PYTHONPATH=../pymilvus pytest tests/function/test_public_function_chain.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Search helper support**

```bash
git add milvus_lite/adapter/grpc/function_chain.py tests/function/test_public_function_chain.py
git commit -m "feat: prepare and execute search function chains"
```

---

### Task 8: Integrate Public Chains Into Search and Reject Hybrid Chains

**Files:**

- Modify: `milvus_lite/adapter/grpc/translators/search.py:44`
- Modify: `milvus_lite/adapter/grpc/servicer.py:501`
- Modify: `milvus_lite/adapter/grpc/servicer.py:1107`
- Create: `tests/adapter/test_function_chain.py`

- [ ] **Step 1: Write failing gRPC integration tests**

Create a collection with deterministic vectors and scalar fields, then exercise the real PyMilvus DSL:

```python
import pytest

from pymilvus import DataType, FunctionChain, FunctionChainStage
from pymilvus.function_chain import col, fn


def _create_collection(client, name):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
    schema.add_field("popularity", DataType.FLOAT)
    schema.add_field("title", DataType.VARCHAR, max_length=128)
    client.create_collection(name, schema=schema)
    client.insert(
        name,
        [
            {"id": 1, "vector": [0.0, 0.0], "popularity": 1.0, "title": "one"},
            {"id": 2, "vector": [0.1, 0.0], "popularity": 10.0, "title": "two"},
            {"id": 3, "vector": [0.2, 0.0], "popularity": 5.0, "title": "three"},
        ],
    )
    client.load_collection(name)


def test_search_function_chain_reranks_by_hidden_field(milvus_client):
    name = "function_chain_hidden"
    _create_collection(milvus_client, name)
    chain = (
        FunctionChain(FunctionChainStage.L2_RERANK)
        .map(
            "$score",
            fn.num_combine(col("$score"), col("popularity"), mode="sum"),
        )
        .sort(col("$score"), desc=True, tie_break_col=col("$id"))
    )

    result = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vector",
        search_params={"metric_type": "IP"},
        limit=3,
        output_fields=["title"],
        function_chains=chain,
    )

    assert [hit["id"] for hit in result[0]] == [2, 3, 1]
    assert all("popularity" not in hit["entity"] for hit in result[0])


def test_search_function_chain_limit_changes_topks(milvus_client):
    name = "function_chain_limit"
    _create_collection(milvus_client, name)
    chain = FunctionChain(FunctionChainStage.L2_RERANK).limit(2)

    result = milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vector",
        limit=3,
        function_chains=chain,
    )

    assert len(result[0]) == 2
```

Add integration tests for temporary-field non-leakage, final score serialization, no-sort order preservation, FunctionScore conflict through raw gRPC construction, multiple queries, and Hybrid Search rejection.

Add a candidate-budget regression test proving public chains do not use legacy `top_k * 10` expansion:

```python
def test_public_chain_keeps_requested_candidate_budget(milvus_client, monkeypatch):
    from milvus_lite.engine.collection import Collection

    name = "function_chain_budget"
    _create_collection(milvus_client, name)
    observed_top_k = []
    original_search = Collection.search

    def recording_search(self, *args, **kwargs):
        observed_top_k.append(kwargs["top_k"])
        return original_search(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "search", recording_search)
    chain = FunctionChain(FunctionChainStage.L2_RERANK).limit(2)

    milvus_client.search(
        collection_name=name,
        data=[[0.0, 0.0]],
        anns_field="vector",
        limit=3,
        function_chains=chain,
    )

    assert observed_top_k[-1] == 3
```

- [ ] **Step 2: Run integration tests and verify they fail**

Run:

```bash
PYTHONPATH=../pymilvus pytest tests/adapter/test_function_chain.py -v
```

Expected: Search ignores or rejects `function_chains`, so rerank assertions fail.

- [ ] **Step 3: Expose Function Chains from request parsing**

In `parse_search_request()`, add:

```python
function_chains = list(getattr(request, "function_chains", ()))
```

and return:

```python
"function_chains": function_chains,
"has_function_score": request.HasField("function_score"),
```

Guard `HasField` only if the request descriptor contains `function_score`; current supported request versions already do.

- [ ] **Step 4: Prepare the public plan before engine Search**

At the start of `MilvusServicer.Search()` after request parsing:

```python
from milvus_lite.adapter.grpc.function_chain import (
    execute_search_function_chain,
    merge_internal_output_fields,
    prepare_search_function_chain,
)

public_chain_plan = prepare_search_function_chain(
    function_chains=parsed["function_chains"],
    has_function_score=parsed["has_function_score"],
    schema=col.schema,
    num_queries=len(parsed["query_vectors"]),
    requested_output_fields=requested_output_fields,
)
```

Compute internal output fields as:

```python
if public_chain_plan is not None:
    search_output_fields = merge_internal_output_fields(
        requested_output_fields,
        public_chain_plan.required_fields,
    )
else:
    internal_output_fields = []
    if l2_func is not None:
        if group_by_field is not None:
            internal_output_fields.append(group_by_field)
        internal_output_fields.extend(
            list(getattr(l2_func, "input_field_names", []))
        )
    search_output_fields = (
        None
        if requested_output_fields is None
        else list(dict.fromkeys(requested_output_fields + internal_output_fields))
    )
```

Do not apply the legacy `top_k * 10`, offset reset, or group-by reset when `public_chain_plan` is present.

- [ ] **Step 5: Execute the public chain before the legacy branch**

After `Collection.search()` returns:

```python
if public_chain_plan is not None:
    results = execute_search_function_chain(
        public_chain_plan,
        results,
        metric_type=parsed["metric_type"],
        schema=col.schema,
        primary_field_name=col._pk_name,
        group_by_field=group_by_field,
    )
elif l2_func is not None:
    # Keep the current lines 582-630 legacy FunctionScore DataFrame block here,
    # changing only the leading `if` to this `elif`.
```

Remove the old `_hit_score_for_chain()` definition from `servicer.py` and import `hit_score_for_chain` only where the unchanged legacy block still needs it.

Keep the existing legacy rerank statements in place under the new `elif l2_func is not None:` branch. Do not extract or otherwise refactor that block in this task.

- [ ] **Step 6: Reject Hybrid Search Function Chains**

At the start of `HybridSearch()` after entering the `try` block:

```python
if list(getattr(request, "function_chains", ())):
    raise SchemaValidationError(
        "function_chains is not supported for hybrid search yet"
    )
```

Import `SchemaValidationError` through the existing servicer exception imports.

- [ ] **Step 7: Run integration and focused Search regression tests**

Run:

```bash
PYTHONPATH=../pymilvus pytest \
  tests/adapter/test_function_chain.py \
  tests/adapter/test_grpc_search.py \
  tests/adapter/test_boost_ranker.py \
  tests/adapter/test_hybrid_search.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit Search integration**

```bash
git add milvus_lite/adapter/grpc/translators/search.py milvus_lite/adapter/grpc/servicer.py tests/adapter/test_function_chain.py
git commit -m "feat: support function chains in search"
```

---

### Task 9: Complete Expression Coverage, Compatibility Tests, and Documentation

**Files:**

- Modify: `tests/function/test_public_function_chain.py:1`
- Modify: `tests/adapter/test_function_chain.py:1`
- Modify: `docs/modules.md:30`

- [ ] **Step 1: Add remaining expression and compatibility tests**

Add focused tests for:

```python
@pytest.mark.parametrize("function", ["gauss", "exp", "linear"])
def test_public_decay_expression(function):
    expr = DecayExpr(function=function, origin=100, scale=10, offset=0, decay=0.5)
    result = expr.execute(_ctx(), [[100, 110, None]])[0]
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == 0.0


def test_public_round_decimal_expression():
    assert RoundDecimalExpr(2).execute(_ctx(), [[1.234, None]]) == [[1.23, None]]


def test_prepare_with_old_request_shape_has_no_chain(schema):
    plan = prepare_search_function_chain(
        function_chains=getattr(SimpleNamespace(), "function_chains", ()),
        has_function_score=False,
        schema=schema,
        num_queries=1,
        requested_output_fields=None,
    )
    assert plan is None
```

Add a mocked Cohere provider test proving one query is selected per DataFrame chunk and a validation test proving `api_key` is rejected in public Chain params.

- [ ] **Step 2: Run the complete Function Chain test set**

Run:

```bash
PYTHONPATH=../pymilvus pytest \
  tests/function/test_function_chain_proto.py \
  tests/function/test_function_chain_validator.py \
  tests/function/test_public_function_chain.py \
  tests/adapter/test_function_chain.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Update the authoritative module documentation**

Update the repository tree in `docs/modules.md` to include:

```text
function/
├── repr.py                 # Protocol-neutral public Function Chain representation
├── validator.py            # L2 stage, schema, dependency, and expression validation
├── compiler.py             # Validated public plan -> existing FuncChain runtime
└── expr/num_combine.py     # Public numeric combination expression

adapter/grpc/
├── function_chain.py       # Search-specific planning, execution, and projection
└── translators/function_chain.py  # Function Chain protobuf decoding
```

Add the four new test files to the test layout and state that ordinary Search supports one public L2 chain while Hybrid Search remains unsupported.

- [ ] **Step 4: Run legacy rerank and full adapter regressions**

Run:

```bash
PYTHONPATH=../pymilvus pytest \
  tests/function \
  tests/rerank \
  tests/adapter/test_grpc_search.py \
  tests/adapter/test_boost_ranker.py \
  tests/adapter/test_hybrid_search.py \
  tests/adapter/test_function_chain.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run the full project suite**

Run:

```bash
PYTHONPATH=../pymilvus pytest
```

Expected: all tests pass. If an unrelated pre-existing failure appears, record the exact failing test and do not modify unrelated code.

- [ ] **Step 6: Check formatting and diff integrity**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 7: Commit documentation and final coverage**

```bash
git add tests/function/test_public_function_chain.py tests/adapter/test_function_chain.py docs/modules.md
git commit -m "test: complete function chain coverage"
```

---

## Final Verification Checklist

- [ ] `SearchRequest.function_chains` is accessed safely when the field is absent.
- [ ] Ordinary Search accepts exactly one `L2_RERANK` chain.
- [ ] Hybrid Search rejects public Function Chains.
- [ ] `function_score` and `function_chains` are mutually exclusive.
- [ ] Hidden schema inputs are fetched but not returned.
- [ ] Temporary outputs are not returned.
- [ ] `$id` is read-only and `$score` is writable.
- [ ] `map`, `sort`, and `limit` execute in request order.
- [ ] Literal arguments work without breaking legacy string column inputs.
- [ ] `num_combine` supports all six modes.
- [ ] `decay`, `round_decimal`, and `rerank_model` follow the validated contracts.
- [ ] A chain without sort preserves ANN order.
- [ ] A chain limit changes actual per-query result counts.
- [ ] Public chains do not use legacy candidate expansion or implicit tails.
- [ ] Legacy FunctionScore, hybrid rerank, and ordinary Search regressions pass.
- [ ] The full test suite passes with the Function-Chain-capable PyMilvus checkout.
