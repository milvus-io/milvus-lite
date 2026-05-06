"""Rerank provider, schema, and decay math tests."""

import tempfile
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from milvus_lite.embedding.protocol import EmbeddingProvider
from milvus_lite.engine.collection import Collection
from milvus_lite.rerank.decay import DecayReranker
from milvus_lite.rerank.factory import create_rerank_provider
from milvus_lite.rerank.protocol import RerankProvider, RerankResult
from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)


class MockRerankProvider(RerankProvider):
    """Deterministic mock: scores by substring overlap with the query."""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        query_words = set(query.lower().split())
        scored = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words | doc_words), 1)
            scored.append(RerankResult(index=i, relevance_score=score))
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        return scored


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock: hashes text to a fixed-dim vector."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_text(text)

    def _hash_text(self, text: str) -> List[float]:
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def _mock_embedding_factory(params):
    return MockEmbeddingProvider(dim=params.get("dimensions", 8))


def _make_schema_no_rerank(dim=8):
    """Schema with TEXT_EMBEDDING only, no schema-level rerank."""
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim,
                    is_function_output=True),
    ], functions=[
        Function(
            name="text_emb",
            function_type=FunctionType.TEXT_EMBEDDING,
            input_field_names=["text"],
            output_field_names=["vec"],
            params={"provider": "openai", "model_name": "mock", "dimensions": dim},
        ),
    ])


class TestFactory:
    def test_missing_provider(self):
        with pytest.raises(ValueError, match="requires 'provider'"):
            create_rerank_provider({})

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown rerank provider"):
            create_rerank_provider({"provider": "foobar"})

    def test_cohere_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            old = os.environ.pop("COHERE_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="API key is required"):
                    create_rerank_provider({"provider": "cohere"})
            finally:
                if old is not None:
                    os.environ["COHERE_API_KEY"] = old


class TestSchemaValidation:
    def test_rerank_function_rejected_in_collection_schema(self):
        from milvus_lite.schema.validation import validate_schema

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="bad_rerank",
                function_type=FunctionType.RERANK,
                input_field_names=["text"],
                output_field_names=[],
                params={"provider": "cohere"},
            ),
        ])
        with pytest.raises(Exception, match="not supported in collection schema"):
            validate_schema(schema)

    def test_collection_rejects_rerank_function(self):
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="bad_rerank",
                function_type=FunctionType.RERANK,
                input_field_names=["text"],
                output_field_names=[],
                params={"provider": "cohere"},
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(Exception, match="not supported in collection schema"):
                Collection(name="test", data_dir=d, schema=schema)


class TestMockRerankProvider:
    def test_rerank_ordering(self):
        provider = MockRerankProvider()
        results = provider.rerank(
            query="machine learning",
            documents=[
                "web development frameworks",
                "machine learning algorithms",
                "deep learning neural networks",
            ],
        )
        assert results[0].index == 1

    def test_rerank_empty_documents(self):
        provider = MockRerankProvider()
        results = provider.rerank("test", [])
        assert results == []

    def test_rerank_top_n(self):
        provider = MockRerankProvider()
        results = provider.rerank(
            query="test",
            documents=["test one", "test two", "other"],
            top_n=2,
        )
        assert len(results) == 2


class TestSearchWithoutSchemaRerank:
    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_embedding_factory)
    def test_no_rerank_function_normal_search(self, mock_emb):
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d,
                             schema=_make_schema_no_rerank())
            col.insert([
                {"id": 1, "text": "machine learning"},
                {"id": 2, "text": "deep learning"},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=["machine learning"],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
            )
            assert len(results) == 1
            assert len(results[0]) == 2
            assert results[0][0]["id"] == 1


class TestDecayReranker:
    def test_gauss_at_origin(self):
        dr = DecayReranker("gauss", origin=0, scale=100, decay=0.5)
        assert dr.compute_factor(0.0) == 1.0

    def test_gauss_at_scale(self):
        dr = DecayReranker("gauss", origin=0, scale=100, decay=0.5)
        assert abs(dr.compute_factor(100.0) - 0.5) < 1e-9

    def test_gauss_symmetric(self):
        dr = DecayReranker("gauss", origin=50, scale=100, decay=0.5)
        assert abs(dr.compute_factor(80) - dr.compute_factor(20)) < 1e-9

    def test_exp_at_origin(self):
        dr = DecayReranker("exp", origin=0, scale=100, decay=0.5)
        assert dr.compute_factor(0.0) == 1.0

    def test_exp_at_scale(self):
        dr = DecayReranker("exp", origin=0, scale=100, decay=0.5)
        assert abs(dr.compute_factor(100.0) - 0.5) < 1e-9

    def test_exp_monotone_decrease(self):
        dr = DecayReranker("exp", origin=0, scale=100, decay=0.5)
        factors = [dr.compute_factor(d) for d in [0, 25, 50, 75, 100, 200]]
        for i in range(len(factors) - 1):
            assert factors[i] > factors[i + 1]

    def test_linear_at_origin(self):
        dr = DecayReranker("linear", origin=0, scale=100, decay=0.5)
        assert dr.compute_factor(0.0) == 1.0

    def test_linear_at_scale(self):
        dr = DecayReranker("linear", origin=0, scale=100, decay=0.5)
        assert abs(dr.compute_factor(100.0) - 0.5) < 1e-9

    def test_linear_clamps_to_zero(self):
        dr = DecayReranker("linear", origin=0, scale=100, decay=0.5)
        assert dr.compute_factor(300.0) == 0.0

    def test_offset_creates_safe_zone(self):
        dr = DecayReranker("gauss", origin=0, scale=100, offset=20, decay=0.5)
        assert dr.compute_factor(10.0) == 1.0
        assert dr.compute_factor(20.0) == 1.0
        assert dr.compute_factor(21.0) < 1.0

    def test_offset_shifts_scale(self):
        dr = DecayReranker("exp", origin=0, scale=100, offset=50, decay=0.5)
        assert abs(dr.compute_factor(150.0) - 0.5) < 1e-9

    def test_invalid_function(self):
        with pytest.raises(ValueError, match="invalid function"):
            DecayReranker("cubic", origin=0, scale=100)

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="scale must be > 0"):
            DecayReranker("gauss", origin=0, scale=-1)

    def test_invalid_offset(self):
        with pytest.raises(ValueError, match="offset must be >= 0"):
            DecayReranker("gauss", origin=0, scale=100, offset=-5)

    def test_invalid_decay(self):
        with pytest.raises(ValueError, match="decay must be 0 < decay < 1"):
            DecayReranker("gauss", origin=0, scale=100, decay=1.0)
        with pytest.raises(ValueError, match="decay must be 0 < decay < 1"):
            DecayReranker("gauss", origin=0, scale=100, decay=0.0)
