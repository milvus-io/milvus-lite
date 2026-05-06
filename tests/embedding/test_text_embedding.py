"""TEXT_EMBEDDING function tests.

Uses a mock EmbeddingProvider to avoid real API calls. Tests the full
insert → auto-embed → search → auto-embed-query pipeline.
"""

import tempfile
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from milvus_lite.embedding.protocol import EmbeddingProvider
from milvus_lite.embedding.factory import create_embedding_provider
from milvus_lite.schema.types import (
    CollectionSchema, DataType, FieldSchema, Function, FunctionType,
)
from milvus_lite.engine.collection import Collection


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

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
        """Produce a deterministic vector from text."""
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        # L2-normalize for cosine
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def _mock_factory(params):
    """Factory that returns MockEmbeddingProvider instead of real OpenAI."""
    return MockEmbeddingProvider(dim=params.get("dimensions", 8))


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------

def _make_schema(dim=8):
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


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_missing_provider(self):
        with pytest.raises(ValueError, match="requires 'provider'"):
            create_embedding_provider({})

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider({"provider": "foobar"})

    def test_openai_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove any existing OPENAI_API_KEY
            import os
            old = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="API key is required"):
                    create_embedding_provider({"provider": "openai"})
            finally:
                if old is not None:
                    os.environ["OPENAI_API_KEY"] = old


# ---------------------------------------------------------------------------
# Engine-level tests with mock provider
# ---------------------------------------------------------------------------

class TestTextEmbeddingInsert:
    """Test auto-embedding during insert."""

    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_factory)
    def test_insert_auto_generates_vector(self, mock_create):
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=_make_schema())
            pks = col.insert([
                {"id": 1, "text": "hello world"},
                {"id": 2, "text": "goodbye world"},
            ])
            assert pks == [1, 2]

            # Verify vectors were generated
            col.load()
            rows = col.query("id >= 1", output_fields=["vec"], limit=10)
            assert len(rows) == 2
            for r in rows:
                assert len(r["vec"]) == 8
                assert isinstance(r["vec"][0], float)

    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_factory)
    def test_different_texts_get_different_vectors(self, mock_create):
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=_make_schema())
            col.insert([
                {"id": 1, "text": "cats are fluffy"},
                {"id": 2, "text": "dogs are loyal"},
            ])
            col.load()
            rows = col.query("id >= 1", output_fields=["vec"], limit=10)
            v1 = rows[0]["vec"]
            v2 = rows[1]["vec"]
            assert v1 != v2

    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_factory)
    def test_null_text_gets_zero_vector(self, mock_create):
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8,
                        is_function_output=True),
        ], functions=[
            Function(
                name="text_emb",
                function_type=FunctionType.TEXT_EMBEDDING,
                input_field_names=["text"],
                output_field_names=["vec"],
                params={"provider": "openai", "dimensions": 8},
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            col.insert([{"id": 1, "text": None}])
            col.load()
            rows = col.query("id == 1", output_fields=["vec"], limit=1)
            assert rows[0]["vec"] == [0.0] * 8


class TestTextEmbeddingSearch:
    """Test auto-embedding of text queries during search."""

    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_factory)
    def test_search_with_text_query(self, mock_create):
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=_make_schema())
            col.insert([
                {"id": 1, "text": "machine learning"},
                {"id": 2, "text": "deep learning"},
                {"id": 3, "text": "web development"},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            # Search with text → should auto-embed
            results = col.search(
                query_vectors=["machine learning"],
                top_k=3,
                metric_type="COSINE",
                anns_field="vec",
            )
            assert len(results) == 1
            assert len(results[0]) == 3
            # Same text should match itself at rank 1
            assert results[0][0]["id"] == 1

    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_factory)
    def test_search_with_vector_query(self, mock_create):
        """Passing a pre-computed vector should also work."""
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=_make_schema())
            col.insert([
                {"id": 1, "text": "hello"},
                {"id": 2, "text": "world"},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            # Search with float vector directly
            mock_vec = MockEmbeddingProvider(dim=8).embed_query("hello")
            results = col.search(
                query_vectors=[mock_vec],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
            )
            assert len(results[0]) == 2

    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_factory)
    def test_search_text_query_without_embedding_function_fails(self, mock_create):
        """Text query on a field without TEXT_EMBEDDING should fail."""
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            col.insert([{"id": 1, "vec": [0.1] * 8}])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            from milvus_lite.exceptions import SchemaValidationError
            with pytest.raises(SchemaValidationError, match="TEXT_EMBEDDING"):
                col.search(
                    query_vectors=["some text"],
                    top_k=1,
                    metric_type="COSINE",
                    anns_field="vec",
                )
