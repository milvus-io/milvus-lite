"""Tests for SelectOp."""

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.select_op import SelectOp
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD, STAGE_L2_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def test_select_keeps_columns():
    df = DataFrame([[{ID_FIELD: 1, SCORE_FIELD: 0.5, "extra": "x"}]])
    result = SelectOp([ID_FIELD, SCORE_FIELD]).execute(_ctx(), df)
    rec = result.chunk(0)[0]
    assert set(rec.keys()) == {ID_FIELD, SCORE_FIELD}


def test_select_multi_chunk():
    df = DataFrame([
        [{ID_FIELD: 1, "a": 10, "b": 20}],
        [{ID_FIELD: 2, "a": 30, "b": 40}],
    ])
    result = SelectOp([ID_FIELD, "a"]).execute(_ctx(), df)
    assert set(result.chunk(0)[0].keys()) == {ID_FIELD, "a"}
    assert set(result.chunk(1)[0].keys()) == {ID_FIELD, "a"}


def test_select_missing_column_ignored():
    """If a record doesn't have a selected column, it just won't appear."""
    df = DataFrame([[{ID_FIELD: 1, "a": 10}]])
    result = SelectOp([ID_FIELD, "nonexistent"]).execute(_ctx(), df)
    rec = result.chunk(0)[0]
    assert ID_FIELD in rec
    assert "nonexistent" not in rec
