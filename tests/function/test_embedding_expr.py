"""Tests for EmbeddingExpr."""

from milvus_lite.function.expr.embedding_expr import EmbeddingExpr
from milvus_lite.function.types import STAGE_INGESTION, STAGE_L2_RERANK, FuncContext


class _MockProvider:
    """Mock EmbeddingProvider for testing."""

    dimension = 4
    _call_count = 0

    def embed_documents(self, texts):
        self._call_count += 1
        return [[float(i)] * self.dimension for i in range(len(texts))]

    def embed_query(self, text):
        return [0.0] * self.dimension


def _ctx():
    return FuncContext(STAGE_INGESTION)


def test_embedding_basic():
    provider = _MockProvider()
    expr = EmbeddingExpr(provider)
    result = expr.execute(_ctx(), [["hello", "world"]])
    assert len(result) == 1
    assert len(result[0]) == 2
    assert len(result[0][0]) == 4  # dimension=4
    assert len(result[0][1]) == 4


def test_embedding_null_handling():
    provider = _MockProvider()
    expr = EmbeddingExpr(provider)
    result = expr.execute(_ctx(), [[None, "", "valid"]])
    vecs = result[0]
    # None -> zero vector
    assert vecs[0] == [0.0, 0.0, 0.0, 0.0]
    # empty string -> zero vector
    assert vecs[1] == [0.0, 0.0, 0.0, 0.0]
    # "valid" -> actual embedding
    assert vecs[2] == [0.0] * 4  # mock returns [0.0]*4 for first text in batch


def test_embedding_non_string():
    provider = _MockProvider()
    expr = EmbeddingExpr(provider)
    result = expr.execute(_ctx(), [[123, True]])
    vecs = result[0]
    assert vecs[0] == [0.0] * 4  # zero vector
    assert vecs[1] == [0.0] * 4  # zero vector


def test_embedding_batch_efficiency():
    provider = _MockProvider()
    expr = EmbeddingExpr(provider)
    expr.execute(_ctx(), [["a", "b", "c"]])
    # embed_documents should be called exactly once for the batch
    assert provider._call_count == 1


def test_embedding_all_null_no_api_call():
    provider = _MockProvider()
    expr = EmbeddingExpr(provider)
    expr.execute(_ctx(), [[None, None]])
    assert provider._call_count == 0


def test_embedding_stage():
    expr = EmbeddingExpr(_MockProvider())
    assert expr.is_runnable(STAGE_INGESTION) is True
    assert expr.is_runnable(STAGE_L2_RERANK) is False


def test_embedding_name():
    assert EmbeddingExpr(_MockProvider()).name == "text_embedding"
