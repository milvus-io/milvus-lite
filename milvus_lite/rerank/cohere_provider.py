"""Cohere rerank provider.

Calls the Cohere Rerank API (POST /v2/rerank) using urllib
from the standard library — no external HTTP dependency required.

Supports:
    - rerank-v3.5 (default)
    - rerank-4.0

API key resolution order:
    1. Explicit ``api_key`` param
    2. ``COHERE_API_KEY`` environment variable
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

from milvus_lite.rerank.protocol import RerankProvider, RerankResult

_DEFAULT_MODEL = "rerank-v3.5"
_DEFAULT_BASE_URL = "https://api.cohere.com/v2"


class CohereProvider(RerankProvider):
    """Cohere reranking via the REST API."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._model = model_name
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Cohere API key is required. Pass api_key param or set "
                "COHERE_API_KEY environment variable."
            )
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []

        body: Dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            body["top_n"] = top_n

        resp = self._call_api(body)
        results = resp.get("results", [])
        return [
            RerankResult(index=r["index"], relevance_score=r["relevance_score"])
            for r in results
        ]

    def _call_api(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Cohere rerank endpoint."""
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/rerank",
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
                f"Cohere API error {e.code}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cohere API connection error: {e.reason}"
            ) from e
