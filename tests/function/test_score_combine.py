"""Tests for ScoreCombineExpr."""

import pytest

from milvus_lite.function.expr.score_combine import ScoreCombineExpr
from milvus_lite.function.types import STAGE_L2_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def test_multiply():
    expr = ScoreCombineExpr("multiply")
    result = expr.execute(_ctx(), [[0.9, 0.8], [0.5, 0.4]])
    assert abs(result[0][0] - 0.45) < 1e-9
    assert abs(result[0][1] - 0.32) < 1e-9


def test_sum():
    expr = ScoreCombineExpr("sum")
    result = expr.execute(_ctx(), [[0.9], [0.5]])
    assert abs(result[0][0] - 1.4) < 1e-9


def test_max():
    expr = ScoreCombineExpr("max")
    result = expr.execute(_ctx(), [[0.3], [0.7]])
    assert result[0][0] == 0.7


def test_min():
    expr = ScoreCombineExpr("min")
    result = expr.execute(_ctx(), [[0.3], [0.7]])
    assert result[0][0] == 0.3


def test_avg():
    expr = ScoreCombineExpr("avg")
    result = expr.execute(_ctx(), [[0.2], [0.8]])
    assert abs(result[0][0] - 0.5) < 1e-9


def test_none_handling():
    expr = ScoreCombineExpr("multiply")
    result = expr.execute(_ctx(), [[None, 0.5], [0.3, 0.4]])
    assert result[0][0] == 0.0
    assert abs(result[0][1] - 0.2) < 1e-9


def test_default_remains_multiply():
    assert ScoreCombineExpr().execute(_ctx(), [[2.0], [3.0]]) == [[6.0]]


def test_public_name_remains_score_combine():
    assert ScoreCombineExpr.name == "score_combine"


def test_weighted_mode_is_not_supported():
    with pytest.raises(ValueError, match="does not support weighted mode"):
        ScoreCombineExpr("weighted")


@pytest.mark.parametrize("mode", ["multiply", "sum", "max", "min", "avg"])
def test_single_column_value_remains_identity(mode):
    assert ScoreCombineExpr(mode).execute(_ctx(), [[2.5]]) == [[2.5]]


@pytest.mark.parametrize("mode", ["multiply", "sum", "max", "min", "avg"])
def test_single_column_none_remains_zero(mode):
    assert ScoreCombineExpr(mode).execute(_ctx(), [[None]]) == [[0.0]]
