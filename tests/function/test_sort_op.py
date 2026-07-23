"""Tests for SortOp."""

import math
from itertools import permutations

import pytest

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.sort_op import SortOp
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD, STAGE_L2_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def _hit(pk, score):
    return {ID_FIELD: pk, SCORE_FIELD: score}


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


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


@pytest.mark.parametrize(
    ("desc", "ordered_ids"),
    [(False, [3, 1]), (True, [1, 3])],
)
def test_sort_primary_nan_permutations_are_stable_and_consistent(
    desc, ordered_ids
):
    records = [
        _hit(1, 2.0),
        _hit(2, float("nan")),
        _hit(3, 1.0),
        _hit(4, None),
    ]

    for permutation in permutations(records):
        df = DataFrame([[dict(record) for record in permutation]])
        SortOp(SCORE_FIELD, desc=desc).execute(_ctx(), df)
        missing_ids = [
            record[ID_FIELD]
            for record in permutation
            if _is_missing(record[SCORE_FIELD])
        ]
        assert [record[ID_FIELD] for record in df.chunk(0)] == [
            *ordered_ids,
            *missing_ids,
        ]


@pytest.mark.parametrize(
    ("desc", "expected_ids"),
    [(False, [3, 1, 2, 4]), (True, [1, 3, 2, 4])],
)
def test_sort_primary_nan_permutations_use_explicit_tie_break(
    desc, expected_ids
):
    records = [
        _hit(1, 2.0),
        _hit(2, float("nan")),
        _hit(3, 1.0),
        _hit(4, None),
    ]

    for permutation in permutations(records):
        df = DataFrame([[dict(record) for record in permutation]])
        SortOp(
            SCORE_FIELD,
            desc=desc,
            tie_break_col=ID_FIELD,
        ).execute(_ctx(), df)
        assert [record[ID_FIELD] for record in df.chunk(0)] == expected_ids


@pytest.mark.parametrize("desc", [False, True])
def test_sort_equal_values_without_tie_break_preserve_input_order(desc):
    df = DataFrame([[_hit(3, 0.5), _hit(1, 0.5), _hit(2, 0.5)]])
    SortOp(SCORE_FIELD, desc=desc).execute(_ctx(), df)
    pks = [h[ID_FIELD] for h in df.chunk(0)]
    assert pks == [3, 1, 2]


@pytest.mark.parametrize("desc", [False, True])
def test_sort_explicit_id_tie_break_is_always_ascending(desc):
    df = DataFrame([[_hit(3, 0.5), _hit(1, 0.5), _hit(2, 0.5)]])
    SortOp(SCORE_FIELD, desc=desc, tie_break_col=ID_FIELD).execute(_ctx(), df)
    pks = [h[ID_FIELD] for h in df.chunk(0)]
    assert pks == [1, 2, 3]


def test_sort_custom_text_tie_break():
    df = DataFrame([
        [
            {ID_FIELD: 1, SCORE_FIELD: 0.5, "title": "charlie"},
            {ID_FIELD: 2, SCORE_FIELD: 0.5, "title": "alpha"},
            {ID_FIELD: 3, SCORE_FIELD: 0.5, "title": "bravo"},
        ]
    ])
    SortOp(SCORE_FIELD, desc=True, tie_break_col="title").execute(_ctx(), df)
    assert [h["title"] for h in df.chunk(0)] == ["alpha", "bravo", "charlie"]


def test_sort_tie_break_none_and_missing_values_last_stably():
    df = DataFrame([
        [
            {ID_FIELD: 1, SCORE_FIELD: 0.5},
            {ID_FIELD: 2, SCORE_FIELD: 0.5, "title": "bravo"},
            {ID_FIELD: 3, SCORE_FIELD: 0.5, "title": None},
            {ID_FIELD: 4, SCORE_FIELD: 0.5, "title": "alpha"},
        ]
    ])
    SortOp(SCORE_FIELD, tie_break_col="title").execute(_ctx(), df)
    assert [h[ID_FIELD] for h in df.chunk(0)] == [4, 2, 1, 3]


@pytest.mark.parametrize("desc", [False, True])
def test_sort_tie_break_nan_permutations_are_stable_and_consistent(desc):
    records = [
        {ID_FIELD: 1, SCORE_FIELD: 0.5, "rank": 2.0},
        {ID_FIELD: 2, SCORE_FIELD: 0.5, "rank": float("nan")},
        {ID_FIELD: 3, SCORE_FIELD: 0.5, "rank": 1.0},
        {ID_FIELD: 4, SCORE_FIELD: 0.5, "rank": None},
    ]

    for permutation in permutations(records):
        df = DataFrame([[dict(record) for record in permutation]])
        SortOp(
            SCORE_FIELD,
            desc=desc,
            tie_break_col="rank",
        ).execute(_ctx(), df)
        missing_ids = [
            record[ID_FIELD]
            for record in permutation
            if _is_missing(record["rank"])
        ]
        assert [record[ID_FIELD] for record in df.chunk(0)] == [
            3,
            1,
            *missing_ids,
        ]


@pytest.mark.parametrize("desc", [False, True])
def test_sort_both_primary_none_without_tie_break_preserves_input_order(desc):
    df = DataFrame([[_hit(3, None), _hit(1, None), _hit(2, None)]])
    SortOp(SCORE_FIELD, desc=desc).execute(_ctx(), df)
    assert [h[ID_FIELD] for h in df.chunk(0)] == [3, 1, 2]


@pytest.mark.parametrize("desc", [False, True])
def test_sort_both_primary_none_uses_explicit_tie_break(desc):
    df = DataFrame([[_hit(3, None), _hit(1, None), _hit(2, None)]])
    SortOp(SCORE_FIELD, desc=desc, tie_break_col=ID_FIELD).execute(_ctx(), df)
    assert [h[ID_FIELD] for h in df.chunk(0)] == [1, 2, 3]


def test_sort_multi_chunk_with_explicit_tie_break():
    df = DataFrame([
        [_hit(2, 0.9), _hit(1, 0.9)],
        [_hit(4, 0.8), _hit(3, 0.8)],
    ])
    SortOp(SCORE_FIELD, desc=True, tie_break_col=ID_FIELD).execute(_ctx(), df)
    assert [h[ID_FIELD] for h in df.chunk(0)] == [1, 2]
    assert [h[ID_FIELD] for h in df.chunk(1)] == [3, 4]
