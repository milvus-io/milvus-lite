"""Tests for RerankModelExpr."""

from dataclasses import dataclass
from typing import List, Optional

import pytest

from milvus_lite.function.expr.rerank_model import RerankModelExpr
from milvus_lite.function.types import STAGE_L2_RERANK, FuncContext


@dataclass
class _MockResult:
    index: int
    relevance_score: float


class _MockProvider:
    """Mock RerankProvider."""

    def __init__(self):
        self.calls: list = []

    def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[_MockResult]:
        self.calls.append((query, documents, top_n))
        # Return results in reverse order with decreasing scores
        return [
            _MockResult(index=i, relevance_score=1.0 / (i + 1))
            for i in range(len(documents))
        ]


def _ctx(chunk_idx=0):
    ctx = FuncContext(STAGE_L2_RERANK)
    ctx.chunk_idx = chunk_idx
    return ctx


def test_rerank_model_basic():
    provider = _MockProvider()
    expr = RerankModelExpr(provider, query_texts=["what is AI?"])
    result = expr.execute(_ctx(0), [["doc A", "doc B", "doc C"]])
    scores = result[0]
    assert len(scores) == 3
    assert scores[0] == 1.0  # index 0 -> 1/(0+1)
    assert scores[1] == 0.5  # index 1 -> 1/(1+1)


def test_rerank_model_uses_chunk_idx():
    provider = _MockProvider()
    expr = RerankModelExpr(provider, query_texts=["q0", "q1", "q2"])
    expr.execute(_ctx(1), [["doc"]])
    # Provider should have received query "q1"
    assert provider.calls[0][0] == "q1"


def test_rerank_model_stage():
    provider = _MockProvider()
    expr = RerankModelExpr(provider, query_texts=[])
    assert expr.is_runnable(STAGE_L2_RERANK)
    assert not expr.is_runnable("ingestion")


def test_rerank_model_raises_without_query_texts():
    provider = _MockProvider()
    expr = RerankModelExpr(provider)  # query_texts defaults to []
    with pytest.raises(RuntimeError, match="query_texts not set"):
        expr.execute(_ctx(0), [["doc"]])


def test_rerank_model_chunk_idx_out_of_range():
    provider = _MockProvider()
    expr = RerankModelExpr(provider, query_texts=["q0"])
    with pytest.raises(IndexError):
        expr.execute(_ctx(5), [["doc"]])


def test_rerank_model_partial_provider_results():
    """Provider returns fewer results than doc count → unfilled slots stay 0.0."""

    class _PartialProvider:
        def rerank(self, query, documents, top_n=None):
            # Only return result for index 0, skip index 1
            return [_MockResult(index=0, relevance_score=0.99)]

    expr = RerankModelExpr(_PartialProvider(), query_texts=["q"])
    result = expr.execute(_ctx(0), [["doc A", "doc B"]])
    scores = result[0]
    assert scores[0] == 0.99
    assert scores[1] == 0.0  # unfilled
