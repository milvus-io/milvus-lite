"""Tests for request-level rerank chain builders."""

import pytest

from milvus_lite.function.builder import (
    build_hybrid_function_score_chain,
    build_hybrid_rerank_chain,
)
from milvus_lite.function.ops.map_op import MapOp
from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)


def _rerank_function(**params):
    return Function(
        name="rerank_fn",
        function_type=FunctionType.RERANK,
        input_field_names=["text"] if "provider" in params else ["ts"],
        output_field_names=[],
        params=params,
    )


def _op_names(chain):
    return [op.name for op in chain.operators]


def _schema():
    return CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("text", DataType.VARCHAR, max_length=128),
        FieldSchema("ts", DataType.INT64),
    ])


# ── build_hybrid_rerank_chain ────────────────────────────────


def test_build_hybrid_rrf():
    chain = build_hybrid_rerank_chain("rrf", {"k": 60}, {"limit": 10})
    names = _op_names(chain)
    assert names[0] == "Merge"
    assert "Sort" in names
    # Hybrid chains skip SelectOp (caller handles field filtering)
    assert "Select" not in names


def test_build_hybrid_weighted():
    chain = build_hybrid_rerank_chain(
        "weighted", {"weights": [0.7, 0.3]}, {"limit": 5}
    )
    names = _op_names(chain)
    assert names[0] == "Merge"


def test_build_hybrid_function_score_weighted_validates_weights():
    func = Function(
        name="weighted_fn",
        function_type=FunctionType.RERANK,
        input_field_names=[],
        output_field_names=[],
        params={"reranker": "weighted", "weights": [1.2]},
    )

    with pytest.raises(ValueError, match="range"):
        build_hybrid_function_score_chain(
            func,
            {"limit": 10},
            search_metrics=["IP"],
            collection_schema=_schema(),
        )


def test_build_hybrid_with_group_by():
    chain = build_hybrid_rerank_chain(
        "rrf", {}, {"limit": 10, "group_by_field": "cat", "group_size": 2}
    )
    names = _op_names(chain)
    assert "GroupBy" in names
    assert "Sort" not in names


def test_build_hybrid_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported hybrid rerank strategy"):
        build_hybrid_rerank_chain("unknown", {}, {"limit": 10})


def test_build_hybrid_model_requires_queries():
    func = _rerank_function(reranker="model", provider="mock")

    with pytest.raises(ValueError, match="queries"):
        build_hybrid_function_score_chain(func, {"limit": 10})


def test_build_hybrid_model_uses_queries_from_params(monkeypatch):
    class _Provider:
        def rerank(self, query, docs, top_n=None):
            return []

    monkeypatch.setattr(
        "milvus_lite.rerank.factory.create_rerank_provider",
        lambda params: _Provider(),
    )
    func = _rerank_function(
        reranker="model",
        provider="mock",
        queries=["query 1", "query 2"],
    )

    chain = build_hybrid_function_score_chain(func, {"limit": 10})
    model_maps = [
        op for op in chain.operators
        if isinstance(op, MapOp) and op.expr.name == "rerank_model"
    ]
    assert model_maps[0].expr.query_texts == ["query 1", "query 2"]


def test_build_hybrid_model_validates_varchar_input(monkeypatch):
    class _Provider:
        def rerank(self, query, docs, top_n=None):
            return []

    monkeypatch.setattr(
        "milvus_lite.rerank.factory.create_rerank_provider",
        lambda params: _Provider(),
    )
    func = Function(
        name="model_fn",
        function_type=FunctionType.RERANK,
        input_field_names=["ts"],
        output_field_names=[],
        params={"reranker": "model", "provider": "mock", "queries": ["q"]},
    )

    with pytest.raises(ValueError, match="VARCHAR"):
        build_hybrid_function_score_chain(
            func,
            {"limit": 10},
            search_metrics=["IP"],
            collection_schema=_schema(),
        )


def test_build_hybrid_function_score_decay_chain():
    func = _rerank_function(
        reranker="decay",
        function="gauss",
        origin=0,
        scale=100,
        decay=0.5,
    )
    chain = build_hybrid_function_score_chain(
        func,
        {"limit": 10},
        search_metrics=["IP"],
    )
    names = _op_names(chain)
    assert names[0] == "Merge"
    assert names.count("Map") == 2
    assert "Sort" in names


def test_build_hybrid_decay_validates_numeric_input():
    func = Function(
        name="decay_fn",
        function_type=FunctionType.RERANK,
        input_field_names=["text"],
        output_field_names=[],
        params={
            "reranker": "decay",
            "function": "gauss",
            "origin": 0,
            "scale": 100,
            "decay": 0.5,
        },
    )

    with pytest.raises(ValueError, match="numeric"):
        build_hybrid_function_score_chain(
            func,
            {"limit": 10},
            search_metrics=["IP"],
            collection_schema=_schema(),
        )
