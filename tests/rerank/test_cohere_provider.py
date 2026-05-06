"""Cohere rerank provider HTTP behavior with mocked urllib."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from milvus_lite.rerank import cohere_provider as provider_mod
from milvus_lite.rerank.cohere_provider import CohereProvider


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_cohere_provider_posts_expected_rerank_request(monkeypatch):
    captured = []

    def fake_urlopen(req, timeout):
        captured.append((req, timeout, json.loads(req.data.decode("utf-8"))))
        return _FakeResponse({
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.4},
            ]
        })

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = CohereProvider(
        model_name="rerank-4.0",
        api_key="co-test",
        base_url="https://example.test/v2/",
    )
    results = provider.rerank("query text", ["doc a", "doc b"], top_n=2)

    assert [(r.index, r.relevance_score) for r in results] == [(1, 0.9), (0, 0.4)]
    req, timeout, body = captured[0]
    assert timeout == 60
    assert req.full_url == "https://example.test/v2/rerank"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer co-test"
    assert body == {
        "model": "rerank-4.0",
        "query": "query text",
        "documents": ["doc a", "doc b"],
        "top_n": 2,
    }


def test_cohere_provider_omits_top_n_when_not_requested(monkeypatch):
    def fake_urlopen(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        assert "top_n" not in body
        return _FakeResponse({"results": []})

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = CohereProvider(api_key="co-test")
    assert provider.rerank("query", ["doc"]) == []


def test_cohere_provider_empty_documents_skip_api(monkeypatch):
    def fail_urlopen(req, timeout):
        raise AssertionError("urlopen should not be called for empty documents")

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fail_urlopen)

    provider = CohereProvider(api_key="co-test")
    assert provider.rerank("query", []) == []


def test_cohere_provider_wraps_http_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"bad key"}'),
        )

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = CohereProvider(api_key="bad-key")
    with pytest.raises(RuntimeError, match="Cohere API error 401"):
        provider.rerank("query", ["doc"])


def test_cohere_provider_wraps_url_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = CohereProvider(api_key="co-test")
    with pytest.raises(RuntimeError, match="Cohere API connection error"):
        provider.rerank("query", ["doc"])
