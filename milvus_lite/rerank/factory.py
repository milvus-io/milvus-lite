"""Rerank provider factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from milvus_lite.rerank.protocol import RerankProvider


@dataclass(frozen=True)
class _ProviderSpec:
    param_names: frozenset[str]
    validate: Callable[[Dict[str, Any]], None]
    create: Callable[[Dict[str, Any]], RerankProvider]


def _validate_optional_string(params: Dict[str, Any], name: str) -> None:
    value = params.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Cohere rerank parameter {name!r} must be a string")


def _validate_cohere_params(params: Dict[str, Any]) -> None:
    model_name = params.get("model_name", "rerank-v3.5")
    if not isinstance(model_name, str):
        raise ValueError(
            "Cohere rerank parameter 'model_name' must be a string"
        )
    _validate_optional_string(params, "api_key")
    _validate_optional_string(params, "base_url")


def _create_cohere_provider(params: Dict[str, Any]) -> RerankProvider:
    from milvus_lite.rerank.cohere_provider import CohereProvider

    return CohereProvider(
        model_name=params.get("model_name", "rerank-v3.5"),
        api_key=params.get("api_key"),
        base_url=params.get("base_url"),
    )


_PROVIDER_SPECS = {
    "cohere": _ProviderSpec(
        param_names=frozenset(
            {"provider", "model_name", "api_key", "base_url"}
        ),
        validate=_validate_cohere_params,
        create=_create_cohere_provider,
    ),
}


def _supported_provider_names() -> str:
    return ", ".join(repr(provider) for provider in _PROVIDER_SPECS)


def _provider_spec(provider: str) -> tuple[str, _ProviderSpec]:
    normalized = provider.lower()
    spec = _PROVIDER_SPECS.get(normalized)
    if spec is None:
        raise ValueError(
            f"Unknown rerank provider: {normalized!r}. "
            f"Supported: {_supported_provider_names()}"
        )
    return normalized, spec


def rerank_provider_param_names(provider: str) -> frozenset[str]:
    _, spec = _provider_spec(provider)
    return spec.param_names


def validate_rerank_provider_params(
    params: Optional[Dict[str, Any]] = None,
) -> str:
    params = params or {}
    provider = params.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ValueError(
            "RERANK function requires 'provider' in params. "
            f"Supported: {_supported_provider_names()}"
        )
    normalized, spec = _provider_spec(provider)
    spec.validate(params)
    return normalized


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
    provider = validate_rerank_provider_params(params)

    _, spec = _provider_spec(provider)
    return spec.create(params)
