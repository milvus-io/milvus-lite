"""Tests for function.types — FunctionExpr, FuncContext, constants."""

import pytest

from milvus_lite.function.types import (
    STAGE_INGESTION,
    STAGE_L2_RERANK,
    ID_FIELD,
    SCORE_FIELD,
    FuncContext,
    FunctionExpr,
)


# ── Concrete impl for testing ────────────────────────────────


class _DummyExpr(FunctionExpr):
    name = "dummy"
    supported_stages = frozenset({STAGE_INGESTION})

    def execute(self, ctx, inputs):
        return inputs  # identity


# ── FunctionExpr ─────────────────────────────────────────────


def test_function_expr_abc_enforcement():
    """Cannot instantiate FunctionExpr without implementing all abstracts."""
    with pytest.raises(TypeError):
        FunctionExpr()  # type: ignore[abstract]


def test_function_expr_is_runnable():
    expr = _DummyExpr()
    assert expr.is_runnable(STAGE_INGESTION) is True
    assert expr.is_runnable(STAGE_L2_RERANK) is False


def test_function_expr_name_and_stages():
    expr = _DummyExpr()
    assert expr.name == "dummy"
    assert expr.supported_stages == frozenset({STAGE_INGESTION})


# ── FuncContext ──────────────────────────────────────────────


def test_func_context_stage():
    ctx = FuncContext(STAGE_INGESTION)
    assert ctx.stage == STAGE_INGESTION


def test_func_context_chunk_idx_default():
    ctx = FuncContext(STAGE_L2_RERANK)
    assert ctx.chunk_idx == 0


def test_func_context_chunk_idx_setter():
    ctx = FuncContext(STAGE_L2_RERANK)
    ctx.chunk_idx = 5
    assert ctx.chunk_idx == 5


# ── Constants ────────────────────────────────────────────────


def test_constants():
    assert STAGE_INGESTION == "ingestion"
    assert STAGE_L2_RERANK == "rerank"
    assert ID_FIELD == "$id"
    assert SCORE_FIELD == "$score"
