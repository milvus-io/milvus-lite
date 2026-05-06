"""Tests for RoundDecimalExpr."""

from milvus_lite.function.expr.round_decimal import RoundDecimalExpr
from milvus_lite.function.types import STAGE_INGESTION, STAGE_L2_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def test_round_2():
    expr = RoundDecimalExpr(2)
    result = expr.execute(_ctx(), [[0.12345, 0.6789]])
    assert result[0][0] == 0.12
    assert result[0][1] == 0.68


def test_round_0():
    expr = RoundDecimalExpr(0)
    result = expr.execute(_ctx(), [[3.7, 4.2]])
    assert result[0][0] == 4.0
    assert result[0][1] == 4.0


def test_round_none():
    expr = RoundDecimalExpr(2)
    result = expr.execute(_ctx(), [[None, 0.5]])
    assert result[0][0] is None
    assert result[0][1] == 0.5


def test_round_rerank_stage_only():
    expr = RoundDecimalExpr(2)
    assert expr.is_runnable(STAGE_L2_RERANK)
    assert not expr.is_runnable(STAGE_INGESTION)
