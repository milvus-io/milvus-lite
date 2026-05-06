"""Tests for BM25Expr."""

from milvus_lite.analyzer.standard import StandardAnalyzer
from milvus_lite.analyzer.sparse import compute_tf
from milvus_lite.function.expr.bm25_expr import BM25Expr
from milvus_lite.function.types import STAGE_INGESTION, STAGE_L2_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_INGESTION)


def test_bm25_basic():
    analyzer = StandardAnalyzer()
    expr = BM25Expr(analyzer)
    texts = ["hello world", "foo bar baz"]
    result = expr.execute(_ctx(), [texts])
    assert len(result) == 1
    sparse_vecs = result[0]
    assert len(sparse_vecs) == 2
    # verify matches direct computation
    for text, sv in zip(texts, sparse_vecs):
        expected = compute_tf(analyzer.analyze(text))
        assert sv == expected


def test_bm25_null_text():
    expr = BM25Expr(StandardAnalyzer())
    result = expr.execute(_ctx(), [[None, "hello"]])
    assert result[0][0] == {}
    assert result[0][1] != {}


def test_bm25_non_string():
    expr = BM25Expr(StandardAnalyzer())
    result = expr.execute(_ctx(), [[123, True, []]])
    assert result[0] == [{}, {}, {}]


def test_bm25_empty_string():
    expr = BM25Expr(StandardAnalyzer())
    result = expr.execute(_ctx(), [[""]])
    # empty string -> analyzer returns [] -> compute_tf returns {}
    assert result[0] == [{}]


def test_bm25_stage():
    expr = BM25Expr(StandardAnalyzer())
    assert expr.is_runnable(STAGE_INGESTION) is True
    assert expr.is_runnable(STAGE_L2_RERANK) is False


def test_bm25_name():
    assert BM25Expr(StandardAnalyzer()).name == "bm25"
