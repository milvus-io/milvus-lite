"""Tests for LimitOp."""

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.limit_op import LimitOp
from milvus_lite.function.types import ID_FIELD, STAGE_L2_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def test_limit_basic():
    df = DataFrame([[{ID_FIELD: i} for i in range(10)]])
    result = LimitOp(3).execute(_ctx(), df)
    assert len(result.chunk(0)) == 3


def test_limit_with_offset():
    df = DataFrame([[{ID_FIELD: i} for i in range(10)]])
    result = LimitOp(3, offset=2).execute(_ctx(), df)
    ids = [h[ID_FIELD] for h in result.chunk(0)]
    assert ids == [2, 3, 4]


def test_limit_exceeds_chunk():
    df = DataFrame([[{ID_FIELD: 1}, {ID_FIELD: 2}]])
    result = LimitOp(100).execute(_ctx(), df)
    assert len(result.chunk(0)) == 2


def test_limit_offset_exceeds_chunk():
    df = DataFrame([[{ID_FIELD: 1}]])
    result = LimitOp(10, offset=100).execute(_ctx(), df)
    assert len(result.chunk(0)) == 0


def test_limit_multi_chunk():
    df = DataFrame([
        [{ID_FIELD: i} for i in range(5)],
        [{ID_FIELD: i} for i in range(3)],
    ])
    result = LimitOp(2).execute(_ctx(), df)
    assert len(result.chunk(0)) == 2
    assert len(result.chunk(1)) == 2
