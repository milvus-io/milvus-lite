"""Tests for SortOp."""

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.sort_op import SortOp
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD, STAGE_L2_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def _hit(pk, score):
    return {ID_FIELD: pk, SCORE_FIELD: score}


def test_sort_desc_by_score():
    df = DataFrame([[_hit(1, 0.3), _hit(2, 0.9), _hit(3, 0.6)]])
    SortOp(SCORE_FIELD, desc=True).execute(_ctx(), df)
    scores = [h[SCORE_FIELD] for h in df.chunk(0)]
    assert scores == [0.9, 0.6, 0.3]


def test_sort_asc():
    df = DataFrame([[_hit(1, 0.3), _hit(2, 0.9), _hit(3, 0.6)]])
    SortOp(SCORE_FIELD, desc=False).execute(_ctx(), df)
    scores = [h[SCORE_FIELD] for h in df.chunk(0)]
    assert scores == [0.3, 0.6, 0.9]


def test_sort_none_values_last_desc():
    df = DataFrame([[_hit(1, None), _hit(2, 0.5), _hit(3, 0.9)]])
    SortOp(SCORE_FIELD, desc=True).execute(_ctx(), df)
    pks = [h[ID_FIELD] for h in df.chunk(0)]
    assert pks[-1] == 1  # None goes last


def test_sort_none_values_last_asc():
    df = DataFrame([[_hit(1, None), _hit(2, 0.5), _hit(3, 0.1)]])
    SortOp(SCORE_FIELD, desc=False).execute(_ctx(), df)
    pks = [h[ID_FIELD] for h in df.chunk(0)]
    assert pks[-1] == 1  # None still goes last


def test_sort_tie_break_by_id():
    df = DataFrame([[_hit(3, 0.5), _hit(1, 0.5), _hit(2, 0.5)]])
    SortOp(SCORE_FIELD, desc=True).execute(_ctx(), df)
    pks = [h[ID_FIELD] for h in df.chunk(0)]
    # same score → sorted by $id ascending
    assert pks == [1, 2, 3]


def test_sort_multi_chunk():
    df = DataFrame([
        [_hit(2, 0.3), _hit(1, 0.9)],
        [_hit(4, 0.1), _hit(3, 0.8)],
    ])
    SortOp(SCORE_FIELD, desc=True).execute(_ctx(), df)
    assert [h[ID_FIELD] for h in df.chunk(0)] == [1, 2]
    assert [h[ID_FIELD] for h in df.chunk(1)] == [3, 4]
