"""Embedding provider factory."""

from __future__ import annotations

from typing import Any, Dict, Optional

from milvus_lite.embedding.protocol import EmbeddingProvider


def create_embedding_provider(params: Optional[Dict[str, Any]] = None) -> EmbeddingProvider:
    """Create an EmbeddingProvider from Function params.

    Args:
        params: dict with at least ``provider`` key. Provider-specific
            keys are forwarded to the provider constructor.

    Supported providers:
        - ``openai``: OpenAI Embeddings API

    Raises:
        ValueError: unknown or missing provider
    """
    params = params or {}
    provider = params.get("provider", "").lower()

    if provider == "openai":
        from milvus_lite.embedding.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model_name=params.get("model_name", "text-embedding-3-small"),
            api_key=params.get("api_key"),
            dimensions=params.get("dimensions"),
            base_url=params.get("base_url"),
        )

    if not provider:
        raise ValueError(
            "TEXT_EMBEDDING function requires 'provider' in params. "
            "Supported: 'openai'"
        )
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. Supported: 'openai'"
    )
