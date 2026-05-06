"""OpenAI embedding provider.

Calls the OpenAI Embeddings API (POST /v1/embeddings) using urllib
from the standard library — no external HTTP dependency required.

Supports:
    - text-embedding-3-small (1536 dims, or custom via dimensions param)
    - text-embedding-3-large (3072 dims, or custom via dimensions param)
    - text-embedding-ada-002 (1536 dims, no dimensions param)

API key resolution order:
    1. Explicit ``api_key`` param
    2. ``OPENAI_API_KEY`` environment variable

Batching: the OpenAI API accepts up to 2048 inputs per request. This
provider chunks larger batches automatically.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

from milvus_lite.embedding.protocol import EmbeddingProvider

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_MAX_BATCH_SIZE = 2048


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embeddings via the REST API."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._model = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Pass api_key param or set "
                "OPENAI_API_KEY environment variable."
            )
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._user_dimensions = dimensions  # None if user did not specify

        # Validate: ada-002 does not support custom dimensions
        if dimensions is not None and self._model == "text-embedding-ada-002":
            if dimensions != 1536:
                raise ValueError(
                    f"text-embedding-ada-002 does not support custom dimensions "
                    f"(always 1536). Got dimensions={dimensions}."
                )

        # Infer default dimension from model if not specified
        self._dimensions = dimensions if dimensions is not None else _default_dimension(self._model)

    @property
    def dimension(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            chunk = texts[i:i + _MAX_BATCH_SIZE]
            result = self._call_api(chunk)
            # API returns embeddings sorted by index
            sorted_data = sorted(result["data"], key=lambda x: x["index"])
            all_embeddings.extend([d["embedding"] for d in sorted_data])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self._call_api([text])
        return result["data"][0]["embedding"]

    def _call_api(self, texts: List[str]) -> Dict[str, Any]:
        """Call the OpenAI embeddings endpoint."""
        body: Dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._user_dimensions is not None and self._model != "text-embedding-ada-002":
            body["dimensions"] = self._user_dimensions

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI API error {e.code}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"OpenAI API connection error: {e.reason}"
            ) from e


def _default_dimension(model: str) -> int:
    """Return the default output dimension for known OpenAI models."""
    defaults = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    return defaults.get(model, 1536)
