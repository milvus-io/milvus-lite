"""Tests for MapOp."""

from dataclasses import FrozenInstanceError

import pytest

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.map_op import ColumnBinding, LiteralBinding, MapOp
from milvus_lite.function.types import (
    STAGE_INGESTION,
    FuncContext,
    FunctionExpr,
)


class _DoubleExpr(FunctionExpr):
    name = "double"
    supported_stages = frozenset({STAGE_INGESTION})

    def execute(self, ctx, inputs):
        return [[v * 2 for v in inputs[0]]]


class _ConcatExpr(FunctionExpr):
    """Concatenates two string columns into one."""
    name = "concat"
    supported_stages = frozenset({STAGE_INGESTION})

    def execute(self, ctx, inputs):
        return [[a + b for a, b in zip(inputs[0], inputs[1])]]


def test_map_op_single_column():
    op = MapOp(_DoubleExpr(), ["x"], ["y"])
    assert op.input_cols == ["x"]
    assert op.input_bindings == [ColumnBinding("x")]

    df = DataFrame.from_records([{"x": 3}, {"x": 5}])
    ctx = FuncContext(STAGE_INGESTION)
    op.execute(ctx, df)
    assert df.column("y", 0) == [6, 10]
    # original column unchanged
    assert df.column("x", 0) == [3, 5]


def test_map_op_normalizes_strings_to_frozen_column_bindings():
    op = MapOp(_DoubleExpr(), ["x"], ["y"])

    bindings = op.input_bindings
    bindings.append(LiteralBinding(1))

    assert op.input_bindings == [ColumnBinding("x")]
    with pytest.raises(FrozenInstanceError):
        bindings[0].name = "other"


def test_map_op_expands_literal_inputs_per_chunk():
    seen_inputs = []

    class _RecordInputs(FunctionExpr):
        name = "record_inputs"
        supported_stages = frozenset({STAGE_INGESTION})

        def execute(self, ctx, inputs):
            seen_inputs.append(inputs)
            return [inputs[0]]

    op = MapOp(
        _RecordInputs(),
        [ColumnBinding("x"), LiteralBinding("suffix")],
        ["y"],
    )
    df = DataFrame([
        [{"x": 1}, {"x": 2}],
        [{"x": 3}],
        [],
    ])

    op.execute(FuncContext(STAGE_INGESTION), df)

    assert seen_inputs == [
        [[1, 2], ["suffix", "suffix"]],
        [[3], ["suffix"]],
        [[], []],
    ]
    assert op.input_cols == ["x"]
    assert df.column("y", 0) == [1, 2]
    assert df.column("y", 1) == [3]
    assert df.column("y", 2) == []


@pytest.mark.parametrize("unsupported", [None, 1, object()])
def test_map_op_rejects_unsupported_input_specs(unsupported):
    with pytest.raises(TypeError, match="input spec at index 1"):
        MapOp(_ConcatExpr(), ["x", unsupported], ["y"])


def test_map_op_overwrite_input():
    op = MapOp(_DoubleExpr(), ["x"], ["x"])
    df = DataFrame.from_records([{"x": 4}])
    ctx = FuncContext(STAGE_INGESTION)
    op.execute(ctx, df)
    assert df.column("x", 0) == [8]


def test_map_op_multi_input_cols():
    op = MapOp(_ConcatExpr(), ["a", "b"], ["c"])
    df = DataFrame.from_records([{"a": "hello", "b": " world"}])
    ctx = FuncContext(STAGE_INGESTION)
    op.execute(ctx, df)
    assert df.column("c", 0) == ["hello world"]


def test_map_op_multi_chunk():
    op = MapOp(_DoubleExpr(), ["v"], ["v2"])
    df = DataFrame([
        [{"v": 1}, {"v": 2}],
        [{"v": 10}],
    ])
    ctx = FuncContext(STAGE_INGESTION)
    op.execute(ctx, df)
    assert df.column("v2", 0) == [2, 4]
    assert df.column("v2", 1) == [20]


def test_map_op_sets_chunk_idx_on_context():
    """Verify MapOp sets ctx.chunk_idx for each chunk."""
    seen_idxs = []

    class _RecordIdx(FunctionExpr):
        name = "record_idx"
        supported_stages = frozenset({STAGE_INGESTION})

        def execute(self, ctx, inputs):
            seen_idxs.append(ctx.chunk_idx)
            return [inputs[0]]

    op = MapOp(_RecordIdx(), ["x"], ["x"])
    df = DataFrame([[{"x": 1}], [{"x": 2}], [{"x": 3}]])
    ctx = FuncContext(STAGE_INGESTION)
    op.execute(ctx, df)
    assert seen_idxs == [0, 1, 2]


def test_map_op_rejects_output_column_count_mismatch():
    class _NoOutputExpr(FunctionExpr):
        name = "no_output"
        supported_stages = frozenset({STAGE_INGESTION})

        def execute(self, ctx, inputs):
            return []

    op = MapOp(_NoOutputExpr(), ["x"], ["y"])
    df = DataFrame.from_records([{"x": 1}])
    ctx = FuncContext(STAGE_INGESTION)

    with pytest.raises(ValueError, match="expected 1 output columns"):
        op.execute(ctx, df)
