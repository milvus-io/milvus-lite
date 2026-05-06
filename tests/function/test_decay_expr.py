"""Tests for DecayExpr."""

import math

from milvus_lite.function.expr.decay_expr import DecayExpr
from milvus_lite.function.types import STAGE_INGESTION, STAGE_L2_RERANK, FuncContext
from milvus_lite.rerank.decay import DecayReranker


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def test_decay_gauss_at_origin():
    expr = DecayExpr("gauss", origin=100.0, scale=10.0, decay=0.5)
    result = expr.execute(_ctx(), [[100.0]])
    assert result[0][0] == 1.0


def test_decay_gauss_at_scale():
    expr = DecayExpr("gauss", origin=0.0, scale=10.0, decay=0.5)
    result = expr.execute(_ctx(), [[10.0]])
    assert abs(result[0][0] - 0.5) < 1e-6


def test_decay_exp_at_origin():
    expr = DecayExpr("exp", origin=0.0, scale=10.0, decay=0.5)
    result = expr.execute(_ctx(), [[0.0]])
    assert result[0][0] == 1.0


def test_decay_exp_at_scale():
    expr = DecayExpr("exp", origin=0.0, scale=10.0, decay=0.5)
    result = expr.execute(_ctx(), [[10.0]])
    assert abs(result[0][0] - 0.5) < 1e-6


def test_decay_linear_at_origin():
    expr = DecayExpr("linear", origin=0.0, scale=10.0, decay=0.5)
    result = expr.execute(_ctx(), [[0.0]])
    assert result[0][0] == 1.0


def test_decay_linear_at_scale():
    expr = DecayExpr("linear", origin=0.0, scale=10.0, decay=0.5)
    result = expr.execute(_ctx(), [[10.0]])
    assert abs(result[0][0] - 0.5) < 1e-6


def test_decay_linear_beyond_cutoff():
    expr = DecayExpr("linear", origin=0.0, scale=10.0, decay=0.5)
    result = expr.execute(_ctx(), [[100.0]])
    assert result[0][0] == 0.0


def test_decay_offset_safe_zone():
    expr = DecayExpr("gauss", origin=0.0, scale=10.0, offset=5.0, decay=0.5)
    # Within offset → factor = 1.0
    result = expr.execute(_ctx(), [[3.0]])
    assert result[0][0] == 1.0
    result = expr.execute(_ctx(), [[-5.0]])
    assert result[0][0] == 1.0


def test_decay_none_value():
    expr = DecayExpr("gauss", origin=0.0, scale=10.0, decay=0.5)
    result = expr.execute(_ctx(), [[None]])
    assert result[0][0] == 0.0


def test_decay_matches_existing_reranker():
    """Cross-check: DecayExpr output matches DecayReranker.compute_factor."""
    for func_name in ("gauss", "exp", "linear"):
        expr = DecayExpr(func_name, origin=50.0, scale=20.0, offset=3.0, decay=0.3)
        reranker = DecayReranker(func_name, origin=50.0, scale=20.0, offset=3.0, decay=0.3)
        test_vals = [0.0, 10.0, 47.0, 50.0, 53.0, 70.0, 100.0, 200.0]
        expr_out = expr.execute(_ctx(), [test_vals])[0]
        for val, expr_factor in zip(test_vals, expr_out):
            expected = reranker.compute_factor(val)
            assert abs(expr_factor - expected) < 1e-10, (
                f"{func_name} val={val}: expr={expr_factor} != reranker={expected}"
            )


def test_decay_stage():
    expr = DecayExpr("gauss", origin=0, scale=1, decay=0.5)
    assert expr.is_runnable(STAGE_L2_RERANK)
    assert not expr.is_runnable(STAGE_INGESTION)
