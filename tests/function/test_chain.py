"""Tests for function.chain — FuncChain."""

import pytest

from milvus_lite.function.chain import FuncChain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import (
    STAGE_INGESTION,
    STAGE_L2_RERANK,
    FuncContext,
    FunctionExpr,
)


# ── Test helpers ─────────────────────────────────────────────


class _AddColumnOp(Operator):
    """Trivial operator: adds a column with a fixed value to every record."""

    name = "AddColumn"

    def __init__(self, col_name: str, value):
        self._col_name = col_name
        self._value = value

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            for r in chunk:
                r[self._col_name] = self._value
        return df


class _DoubleExpr(FunctionExpr):
    """Doubles each value in a single input column."""

    name = "double"
    supported_stages = frozenset({STAGE_INGESTION})

    def execute(self, ctx, inputs):
        return [[v * 2 for v in inputs[0]]]


class _RerankOnlyExpr(FunctionExpr):
    """Expr that only supports STAGE_L2_RERANK."""

    name = "rerank_only"
    supported_stages = frozenset({STAGE_L2_RERANK})

    def execute(self, ctx, inputs):
        return inputs


# ── Empty chain ──────────────────────────────────────────────


def test_empty_chain_passthrough():
    chain = FuncChain("test", STAGE_INGESTION)
    records = [{"a": 1}]
    df = DataFrame.from_records(records)
    result = chain.execute(df)
    assert result.to_records() == [{"a": 1}]


# ── Single operator via add() ────────────────────────────────


def test_single_operator():
    chain = FuncChain("test", STAGE_INGESTION)
    chain.add(_AddColumnOp("extra", 42))
    df = DataFrame.from_records([{"a": 1}, {"a": 2}])
    result = chain.execute(df)
    assert result.to_records() == [{"a": 1, "extra": 42}, {"a": 2, "extra": 42}]


# ── Multi operator sequence ──────────────────────────────────


def test_multi_operator_sequence():
    chain = FuncChain("test", STAGE_INGESTION)
    chain.add(_AddColumnOp("x", 10))
    chain.add(_AddColumnOp("y", 20))
    df = DataFrame.from_records([{"a": 1}])
    result = chain.execute(df)
    rec = result.to_records()[0]
    assert rec == {"a": 1, "x": 10, "y": 20}


# ── __repr__ ─────────────────────────────────────────────────


def test_chain_repr():
    chain = FuncChain("my_chain", STAGE_INGESTION)
    chain.add(_AddColumnOp("x", 1))
    chain.add(_AddColumnOp("y", 2))
    r = repr(chain)
    assert "my_chain" in r
    assert "ingestion" in r
    assert "AddColumn" in r


# ── Error: multiple inputs without MergeOp ───────────────────


def test_execute_rejects_multiple_inputs_without_merge():
    chain = FuncChain("test", STAGE_L2_RERANK)
    df1 = DataFrame.from_records([{"a": 1}])
    df2 = DataFrame.from_records([{"a": 2}])
    with pytest.raises(ValueError, match="expects 1 input"):
        chain.execute(df1, df2)


# ── Stage validation in map() ────────────────────────────────


def test_map_rejects_wrong_stage():
    chain = FuncChain("test", STAGE_INGESTION)
    with pytest.raises(ValueError, match="does not support"):
        chain.map(_RerankOnlyExpr(), ["x"], ["y"])


# ── Properties ───────────────────────────────────────────────


def test_chain_stage_property():
    chain = FuncChain("test", STAGE_L2_RERANK)
    assert chain.stage == STAGE_L2_RERANK


def test_chain_operators_property():
    chain = FuncChain("test", STAGE_INGESTION)
    op = _AddColumnOp("x", 1)
    chain.add(op)
    assert chain.operators == [op]
    # operators returns a copy
    chain.operators.append(op)
    assert len(chain.operators) == 1


# ── MergeOp position validation via add() ────────────────────


def test_add_merge_op_at_non_first_position_rejected():
    from milvus_lite.function.ops.merge_op import MergeOp

    chain = FuncChain("test", STAGE_L2_RERANK)
    chain.add(_AddColumnOp("x", 1))
    with pytest.raises(ValueError, match="MergeOp must be the first"):
        chain.add(MergeOp("rrf"))


# ── GroupByOp scorer validation ──────────────────────────────


def test_group_by_rejects_unknown_scorer():
    from milvus_lite.function.ops.group_by_op import GroupByOp

    with pytest.raises(ValueError, match="Unknown group scorer"):
        GroupByOp("field", group_size=1, limit=10, scorer="invalid")


# ── Merge → Sort → Limit full pipeline ──────────────────────


def test_merge_sort_limit_pipeline():
    """MergeOp as first op followed by Sort and Limit."""
    from milvus_lite.function.types import ID_FIELD, SCORE_FIELD

    chain = FuncChain("test", STAGE_L2_RERANK)
    chain.merge("rrf", rrf_k=60)
    chain.sort(SCORE_FIELD, desc=True)
    chain.limit(2)

    path0 = DataFrame([
        [
            {ID_FIELD: 1, SCORE_FIELD: 0.9},
            {ID_FIELD: 2, SCORE_FIELD: 0.8},
            {ID_FIELD: 3, SCORE_FIELD: 0.7},
        ]
    ])
    path1 = DataFrame([
        [
            {ID_FIELD: 2, SCORE_FIELD: 0.6},
            {ID_FIELD: 4, SCORE_FIELD: 0.5},
        ]
    ])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    # limit=2
    assert len(chunk) == 2
    # sorted desc by $score
    assert chunk[0][SCORE_FIELD] >= chunk[1][SCORE_FIELD]
    # pk=2 appears in both routes → highest RRF
    assert chunk[0][ID_FIELD] == 2


def test_merge_map_sort_limit_pipeline():
    """MergeOp → MapOp → SortOp → LimitOp end-to-end."""
    from milvus_lite.function.types import ID_FIELD, SCORE_FIELD

    class _NegateScoreExpr(FunctionExpr):
        name = "negate"
        supported_stages = frozenset({STAGE_L2_RERANK})

        def execute(self, ctx, inputs):
            return [[-v for v in inputs[0]]]

    chain = FuncChain("test", STAGE_L2_RERANK)
    chain.merge("max")
    chain.map(_NegateScoreExpr(), [SCORE_FIELD], [SCORE_FIELD])
    chain.sort(SCORE_FIELD, desc=True)
    chain.limit(1)

    path0 = DataFrame([
        [{ID_FIELD: 1, SCORE_FIELD: 0.9}, {ID_FIELD: 2, SCORE_FIELD: 0.1}]
    ])
    path1 = DataFrame([
        [{ID_FIELD: 1, SCORE_FIELD: 0.5}]
    ])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    assert len(chunk) == 1
    # After max merge: pk1=0.9, pk2=0.1. After negate: pk1=-0.9, pk2=-0.1.
    # Sort desc → pk2 first (-0.1 > -0.9). Limit 1 → pk2.
    assert chunk[0][ID_FIELD] == 2
