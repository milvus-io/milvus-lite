"""Embedding provider subsystem.

Supports TEXT_EMBEDDING Function type: auto-generates dense vectors
from text fields during insert, and auto-embeds text queries during
search.

Public exports:
    EmbeddingProvider  — abstract protocol
    OpenAIProvider     — OpenAI embeddings (requires httpx)
    create_embedding_provider — factory dispatch
"""

from milvus_lite.embedding.protocol import EmbeddingProvider
from milvus_lite.embedding.factory import create_embedding_provider

__all__ = [
    "EmbeddingProvider",
    "create_embedding_provider",
]
