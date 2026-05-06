"""Rerank provider factory."""

from __future__ import annotations

from typing import Any, Dict, Optional

from milvus_lite.rerank.protocol import RerankProvider


def create_rerank_provider(params: Optional[Dict[str, Any]] = None) -> RerankProvider:
    """Create a RerankProvider from Function params.

    Args:
        params: dict with at least ``provider`` key. Provider-specific
            keys are forwarded to the provider constructor.

    Supported providers:
        - ``cohere``: Cohere Rerank API

    Raises:
        ValueError: unknown or missing provider
    """
    params = params or {}
    provider = params.get("provider", "").lower()

    if provider == "cohere":
        from milvus_lite.rerank.cohere_provider import CohereProvider
        return CohereProvider(
            model_name=params.get("model_name", "rerank-v3.5"),
            api_key=params.get("api_key"),
            base_url=params.get("base_url"),
        )

    if not provider:
        raise ValueError(
            "RERANK function requires 'provider' in params. "
            "Supported: 'cohere'"
        )
    raise ValueError(
        f"Unknown rerank provider: {provider!r}. Supported: 'cohere'"
    )
