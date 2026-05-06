"""OpenAI embedding provider HTTP behavior.

These tests mock urllib at the boundary, so they verify request shape,
batching, response ordering, and error handling without making network calls.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from milvus_lite.embedding import openai_provider as provider_mod
from milvus_lite.embedding.openai_provider import OpenAIProvider


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_openai_provider_posts_expected_embedding_request(monkeypatch):
    captured = []

    def fake_urlopen(req, timeout):
        captured.append((req, timeout, json.loads(req.data.decode("utf-8"))))
        return _FakeResponse({
            "data": [
                {"index": 1, "embedding": [0.2, 0.3]},
                {"index": 0, "embedding": [0.0, 0.1]},
            ]
        })

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = OpenAIProvider(
        model_name="text-embedding-3-small",
        api_key="sk-test",
        dimensions=2,
        base_url="https://example.test/v1/",
    )
    embeddings = provider.embed_documents(["alpha", "beta"])

    assert embeddings == [[0.0, 0.1], [0.2, 0.3]]
    req, timeout, body = captured[0]
    assert timeout == 60
    assert req.full_url == "https://example.test/v1/embeddings"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer sk-test"
    assert body == {
        "input": ["alpha", "beta"],
        "model": "text-embedding-3-small",
        "dimensions": 2,
    }


def test_openai_provider_chunks_large_document_batches(monkeypatch):
    calls = []
    monkeypatch.setattr(provider_mod, "_MAX_BATCH_SIZE", 2)

    def fake_urlopen(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body["input"])
        return _FakeResponse({
            "data": [
                {"index": i, "embedding": [float(len(text))]}
                for i, text in enumerate(body["input"])
            ]
        })

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = OpenAIProvider(api_key="sk-test", dimensions=1)
    embeddings = provider.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert calls == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert embeddings == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_openai_provider_embed_query_uses_single_input(monkeypatch):
    def fake_urlopen(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        assert body["input"] == ["needle"]
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.4, 0.5]}]})

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = OpenAIProvider(api_key="sk-test", dimensions=2)
    assert provider.embed_query("needle") == [0.4, 0.5]


def test_openai_provider_rejects_bad_ada_dimensions():
    with pytest.raises(ValueError, match="does not support custom dimensions"):
        OpenAIProvider(
            model_name="text-embedding-ada-002",
            api_key="sk-test",
            dimensions=512,
        )


def test_openai_provider_wraps_http_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url,
            429,
            "rate limited",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"too many requests"}'),
        )

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = OpenAIProvider(api_key="sk-test")
    with pytest.raises(RuntimeError, match="OpenAI API error 429"):
        provider.embed_query("hello")


def test_openai_provider_wraps_url_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", fake_urlopen)

    provider = OpenAIProvider(api_key="sk-test")
    with pytest.raises(RuntimeError, match="OpenAI API connection error"):
        provider.embed_query("hello")
