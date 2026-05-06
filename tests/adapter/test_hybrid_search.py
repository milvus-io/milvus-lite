"""Phase 12 — HybridSearch integration tests.

Tests:
1. WeightedRanker: dense + BM25 fusion
2. RRFRanker: dense + BM25 fusion
3. Multiple dense vector fields fusion
4. Per-sub-request filter expressions
5. Output fields in hybrid results
6. Multiple queries (nq > 1)
7. Reranker unit tests (no gRPC)
"""

from types import SimpleNamespace

import pytest

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    WeightedRanker,
    RRFRanker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_hybrid_collection(client, name):
    """Collection with dense + BM25 sparse for hybrid search."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True, enable_match=True,
                     analyzer_params={"tokenizer": "standard"})
    schema.add_field("category", DataType.INT64)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("bm25_emb", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["bm25_emb"],
    ))
    client.create_collection(name, schema=schema)

    client.insert(name, [
        {"id": 1, "text": "python programming language", "category": 1,
         "dense": [1.0, 0.0, 0.0, 0.0]},
        {"id": 2, "text": "java programming language", "category": 1,
         "dense": [0.0, 1.0, 0.0, 0.0]},
        {"id": 3, "text": "machine learning algorithms", "category": 2,
         "dense": [0.0, 0.0, 1.0, 0.0]},
        {"id": 4, "text": "deep learning neural networks", "category": 2,
         "dense": [0.0, 0.0, 0.0, 1.0]},
        {"id": 5, "text": "python machine learning tutorial", "category": 2,
         "dense": [0.5, 0.0, 0.5, 0.0]},
    ])

    idx = client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    client.create_index(name, idx)
    client.load_collection(name)
    return name


def _create_hybrid_l2_projection_collection(client, name):
    """Collection for testing hidden fields needed by top-level L2 rerank."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("label", DataType.VARCHAR, max_length=64)
    schema.add_field("ts", DataType.INT64)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=2)

    idx = client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="IP", params={})
    client.create_collection(name, schema=schema, index_params=idx)
    client.insert(name, [
        {"id": 1, "text": "bad document", "label": "a", "ts": 0,
         "dense": [1.0, 0.0]},
        {"id": 2, "text": "good document", "label": "b", "ts": 100,
         "dense": [0.9, 0.0]},
        {"id": 3, "text": "ordinary document", "label": "c", "ts": 100,
         "dense": [0.8, 0.0]},
    ])
    client.load_collection(name)
    return name


# ---------------------------------------------------------------------------
# Reranker unit tests (no gRPC)
# ---------------------------------------------------------------------------

class TestRerankerUnit:
    def test_weighted_basic(self):
        from milvus_lite.adapter.grpc.reranker import rerank
        # Route 1: doc A best, Route 2: doc B best
        route1 = [[
            {"id": "A", "distance": -1.0, "entity": {}},
            {"id": "C", "distance": -0.5, "entity": {}},
        ]]
        route2 = [[
            {"id": "B", "distance": -1.0, "entity": {}},
            {"id": "A", "distance": -0.3, "entity": {}},
        ]]
        result = rerank("weighted", {"weights": [0.5, 0.5]},
                        [route1, route2], limit=3)
        assert len(result) == 1
        ids = [h["id"] for h in result[0]]
        assert "A" in ids  # appears in both routes
        assert "B" in ids

    def test_rrf_basic(self):
        from milvus_lite.adapter.grpc.reranker import rerank
        route1 = [[
            {"id": "A", "distance": -1.0, "entity": {}},
            {"id": "B", "distance": -0.5, "entity": {}},
        ]]
        route2 = [[
            {"id": "B", "distance": -1.0, "entity": {}},
            {"id": "C", "distance": -0.5, "entity": {}},
        ]]
        result = rerank("rrf", {"k": 60}, [route1, route2], limit=3)
        ids = [h["id"] for h in result[0]]
        # B appears rank 2 in route1 and rank 1 in route2 → highest RRF
        assert ids[0] == "B"

    def test_weighted_empty(self):
        from milvus_lite.adapter.grpc.reranker import rerank
        result = rerank("weighted", {"weights": [1.0]}, [[[]]],  limit=5)
        assert result == [[]]

    def test_rrf_single_route(self):
        from milvus_lite.adapter.grpc.reranker import rerank
        route = [[
            {"id": 1, "distance": -1.0, "entity": {}},
            {"id": 2, "distance": -0.5, "entity": {}},
        ]]
        result = rerank("rrf", {"k": 60}, [route], limit=5)
        assert result[0][0]["id"] == 1
        assert result[0][1]["id"] == 2

    def test_offset(self):
        from milvus_lite.adapter.grpc.reranker import rerank
        route = [[
            {"id": 1, "distance": -3.0, "entity": {}},
            {"id": 2, "distance": -2.0, "entity": {}},
            {"id": 3, "distance": -1.0, "entity": {}},
        ]]
        result = rerank("rrf", {"k": 60}, [route], limit=2, offset=1)
        assert result[0][0]["id"] == 2

    def test_chain_score_conversion_keeps_ip_direction(self):
        from milvus_lite.adapter.grpc.servicer import _hit_score_for_chain

        assert _hit_score_for_chain({"distance": 0.9}, "IP") == 0.9
        assert _hit_score_for_chain({"distance": 0.1}, "COSINE") == 0.9
        assert _hit_score_for_chain({"distance": 0.1}, "L2") == 0.1
        assert _hit_score_for_chain({"distance": -2.5}, "BM25") == 2.5


# ---------------------------------------------------------------------------
# gRPC integration tests
# ---------------------------------------------------------------------------

def test_hybrid_weighted_dense_bm25(milvus_client):
    """Hybrid search: dense + BM25 with WeightedRanker."""
    name = _create_hybrid_collection(milvus_client, "hybrid_w1")

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf

    # Dense query: closest to doc 1 [1,0,0,0]
    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        param={},
        limit=5,
    )
    # BM25 query: "machine learning" → docs 3, 4, 5
    bm25_query = compute_tf([term_to_id("machine"), term_to_id("learning")])
    bm25_req = AnnSearchRequest(
        data=[bm25_query],
        anns_field="bm25_emb",
        param={"metric_type": "BM25"},
        limit=5,
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[dense_req, bm25_req],
        ranker=WeightedRanker(0.5, 0.5),
        limit=5,
        output_fields=["text"],
    )

    assert len(results) == 1
    assert len(results[0]) > 0
    hit_ids = [h["id"] for h in results[0]]
    # Doc 5 has both dense similarity and BM25 match → should rank high
    assert 5 in hit_ids

    milvus_client.drop_collection(name)


def test_hybrid_top_level_function_score_weighted(milvus_client):
    """Hybrid FunctionScore weighted reranker runs at the L2 merge level."""
    name = _create_hybrid_collection(milvus_client, "hybrid_fn_weighted")

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf

    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        param={},
        limit=5,
    )
    bm25_query = compute_tf([term_to_id("machine"), term_to_id("learning")])
    bm25_req = AnnSearchRequest(
        data=[bm25_query],
        anns_field="bm25_emb",
        param={"metric_type": "BM25"},
        limit=5,
    )
    ranker = Function(
        name="weighted_l2",
        function_type=FunctionType.RERANK,
        input_field_names=[],
        output_field_names=[],
        params={"reranker": "weighted", "weights": [1.0, 0.0]},
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[dense_req, bm25_req],
        ranker=ranker,
        limit=3,
        output_fields=["text"],
    )

    assert [hit["id"] for hit in results[0]][0] == 1
    milvus_client.drop_collection(name)


def test_hybrid_top_level_decay_uses_hidden_input_field(milvus_client):
    """Top-level decay reranker can use a non-output input field."""
    name = _create_hybrid_l2_projection_collection(
        milvus_client, "hybrid_decay_hidden"
    )
    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0]],
        anns_field="dense",
        param={"metric_type": "IP"},
        limit=3,
    )
    ranker = Function(
        name="decay_l2",
        function_type=FunctionType.RERANK,
        input_field_names=["ts"],
        output_field_names=[],
        params={
            "reranker": "decay",
            "function": "linear",
            "origin": 100,
            "scale": 1,
            "decay": 0.5,
        },
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[dense_req],
        ranker=ranker,
        limit=3,
        output_fields=["label"],
    )

    expected_labels = {1: "a", 2: "b", 3: "c"}
    ids = [hit["id"] for hit in results[0]]
    distances = [hit["distance"] for hit in results[0]]

    assert ids[:2] == [2, 3]
    assert distances[0] == pytest.approx(0.9)
    assert distances[1] == pytest.approx(0.8)
    assert all("ts" not in hit["entity"] for hit in results[0])
    for hit in results[0]:
        assert hit["entity"]["label"] == expected_labels[hit["id"]]
    milvus_client.drop_collection(name)


def test_hybrid_top_level_model_uses_hidden_input_field(
    milvus_client,
    monkeypatch,
):
    """Top-level model reranker can use a non-output text input field."""
    class _Provider:
        def rerank(self, query, docs, top_n=None):
            return [
                SimpleNamespace(
                    index=i,
                    relevance_score=1.0 if query in doc else 0.0,
                )
                for i, doc in enumerate(docs)
            ]

    monkeypatch.setattr(
        "milvus_lite.rerank.factory.create_rerank_provider",
        lambda params: _Provider(),
    )

    name = _create_hybrid_l2_projection_collection(
        milvus_client, "hybrid_model_hidden"
    )
    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0]],
        anns_field="dense",
        param={"metric_type": "IP"},
        limit=3,
    )
    ranker = Function(
        name="model_l2",
        function_type=FunctionType.RERANK,
        input_field_names=["text"],
        output_field_names=[],
        params={
            "reranker": "model",
            "provider": "mock",
            "queries": ["good"],
        },
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[dense_req],
        ranker=ranker,
        limit=3,
        output_fields=["label"],
    )

    expected_labels = {1: "a", 2: "b", 3: "c"}
    ids = [hit["id"] for hit in results[0]]
    distances = [hit["distance"] for hit in results[0]]

    assert ids[0] == 2
    assert distances[0] == pytest.approx(1.0)
    assert all("text" not in hit["entity"] for hit in results[0])
    for hit in results[0]:
        assert hit["entity"]["label"] == expected_labels[hit["id"]]
    milvus_client.drop_collection(name)


def test_search_top_level_decay_uses_hidden_input_field(milvus_client):
    """Search L2 decay reranker can use a non-output input field."""
    name = _create_hybrid_l2_projection_collection(
        milvus_client, "search_decay_hidden"
    )
    ranker = Function(
        name="decay_l2",
        function_type=FunctionType.RERANK,
        input_field_names=["ts"],
        output_field_names=[],
        params={
            "reranker": "decay",
            "function": "linear",
            "origin": 100,
            "scale": 1,
            "decay": 0.5,
        },
    )

    results = milvus_client.search(
        name,
        data=[[1.0, 0.0]],
        anns_field="dense",
        search_params={"metric_type": "IP", "params": {}},
        limit=3,
        output_fields=["label"],
        ranker=ranker,
    )

    expected_labels = {1: "a", 2: "b", 3: "c"}
    ids = [hit["id"] for hit in results[0]]
    distances = [hit["distance"] for hit in results[0]]

    assert ids[:2] == [2, 3]
    assert distances[0] == pytest.approx(0.9)
    assert distances[1] == pytest.approx(0.8)
    assert all("ts" not in hit["entity"] for hit in results[0])
    for hit in results[0]:
        assert hit["entity"]["label"] == expected_labels[hit["id"]]
    milvus_client.drop_collection(name)


def test_search_top_level_model_uses_hidden_input_field(
    milvus_client,
    monkeypatch,
):
    """Search L2 model reranker can use a non-output text input field."""
    class _Provider:
        def rerank(self, query, docs, top_n=None):
            return [
                SimpleNamespace(
                    index=i,
                    relevance_score=1.0 if query in doc else 0.0,
                )
                for i, doc in enumerate(docs)
            ]

    monkeypatch.setattr(
        "milvus_lite.rerank.factory.create_rerank_provider",
        lambda params: _Provider(),
    )

    name = _create_hybrid_l2_projection_collection(
        milvus_client, "search_model_hidden"
    )
    ranker = Function(
        name="model_l2",
        function_type=FunctionType.RERANK,
        input_field_names=["text"],
        output_field_names=[],
        params={
            "reranker": "model",
            "provider": "mock",
            "queries": ["good"],
        },
    )

    results = milvus_client.search(
        name,
        data=[[1.0, 0.0]],
        anns_field="dense",
        search_params={"metric_type": "IP", "params": {}},
        limit=3,
        output_fields=["label"],
        ranker=ranker,
    )

    expected_labels = {1: "a", 2: "b", 3: "c"}
    ids = [hit["id"] for hit in results[0]]
    distances = [hit["distance"] for hit in results[0]]

    assert ids[0] == 2
    assert distances[0] == pytest.approx(1.0)
    assert all("text" not in hit["entity"] for hit in results[0])
    for hit in results[0]:
        assert hit["entity"]["label"] == expected_labels[hit["id"]]
    milvus_client.drop_collection(name)


def test_hybrid_rrf_dense_bm25(milvus_client):
    """Hybrid search: dense + BM25 with RRFRanker."""
    name = _create_hybrid_collection(milvus_client, "hybrid_rrf1")

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf

    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        param={},
        limit=5,
    )
    bm25_query = compute_tf([term_to_id("python")])
    bm25_req = AnnSearchRequest(
        data=[bm25_query],
        anns_field="bm25_emb",
        param={"metric_type": "BM25"},
        limit=5,
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[dense_req, bm25_req],
        ranker=RRFRanker(k=60),
        limit=5,
        output_fields=["text"],
    )

    assert len(results) == 1
    hit_ids = [h["id"] for h in results[0]]
    # Doc 1 is best for dense AND contains "python" → RRF should rank it top
    assert hit_ids[0] == 1

    milvus_client.drop_collection(name)


def test_hybrid_output_fields(milvus_client):
    """Hybrid search returns requested output fields."""
    name = _create_hybrid_collection(milvus_client, "hybrid_of")

    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        param={},
        limit=3,
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[dense_req],
        ranker=RRFRanker(),
        limit=3,
        output_fields=["text", "category"],
    )

    for hit in results[0]:
        assert "text" in hit["entity"]
        assert "category" in hit["entity"]

    milvus_client.drop_collection(name)


def test_hybrid_group_by_hidden_field_not_output(milvus_client):
    """Hybrid group_by uses an internal field without returning it."""
    name = _create_hybrid_collection(milvus_client, "hybrid_gb_hidden")
    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        param={},
        limit=5,
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[dense_req],
        ranker=RRFRanker(),
        limit=2,
        group_by_field="category",
        group_size=1,
        output_fields=["text"],
    )

    id_to_category = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
    ids = [hit["id"] for hit in results[0]]
    assert len(ids) == 2
    assert len({id_to_category[pk] for pk in ids}) == 2
    assert all("category" not in hit["entity"] for hit in results[0])

    milvus_client.drop_collection(name)


def test_hybrid_with_filter(milvus_client):
    """Sub-requests can have per-request filter expressions."""
    name = _create_hybrid_collection(milvus_client, "hybrid_filt")

    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        param={},
        limit=5,
        expr="category == 1",  # Only programming docs
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[dense_req],
        ranker=RRFRanker(),
        limit=5,
        output_fields=["category"],
    )

    for hit in results[0]:
        assert hit["entity"]["category"] == 1

    milvus_client.drop_collection(name)


def test_hybrid_single_route_same_as_search(milvus_client):
    """Hybrid with one route should produce same results as plain search."""
    name = _create_hybrid_collection(milvus_client, "hybrid_single")

    query = [[0.5, 0.0, 0.5, 0.0]]

    # Plain search
    plain = milvus_client.search(
        name, data=query, anns_field="dense", limit=3,
    )

    # Hybrid with one route
    req = AnnSearchRequest(data=query, anns_field="dense", param={}, limit=3)
    hybrid = milvus_client.hybrid_search(
        name, reqs=[req], ranker=RRFRanker(), limit=3,
    )

    # Same ids (order may differ slightly due to RRF vs raw distance)
    plain_ids = {h["id"] for h in plain[0]}
    hybrid_ids = {h["id"] for h in hybrid[0]}
    assert plain_ids == hybrid_ids

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# Issue #15 — BM25 sparse sub-request without explicit metric_type
# ---------------------------------------------------------------------------

def _create_hybrid_with_indexes(client, name):
    """Collection with dense + BM25 sparse, both indexed."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True,
                     analyzer_params={"tokenizer": "standard"})
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("bm25_emb", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["bm25_emb"],
    ))

    idx = client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    idx.add_index(field_name="bm25_emb", index_type="SPARSE_INVERTED_INDEX",
                  metric_type="BM25", params={})

    client.create_collection(name, schema=schema, index_params=idx)
    client.insert(name, [
        {"id": 1, "text": "python programming language",
         "dense": [1.0, 0.0, 0.0, 0.0]},
        {"id": 2, "text": "machine learning algorithms",
         "dense": [0.0, 0.0, 1.0, 0.0]},
    ])
    client.load_collection(name)
    return name


def test_hybrid_bm25_no_explicit_metric(milvus_client):
    """Issue #15: sparse sub-request with param={} should auto-resolve
    metric_type from the field's index spec (BM25)."""
    name = _create_hybrid_with_indexes(milvus_client, "hybrid_issue15")

    dense_req = AnnSearchRequest(
        data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        param={},  # no explicit metric_type
        limit=5,
    )
    sparse_req = AnnSearchRequest(
        data=["machine learning"],
        anns_field="bm25_emb",
        param={},  # no explicit metric_type — must auto-resolve to BM25
        limit=5,
    )

    results = milvus_client.hybrid_search(
        name,
        reqs=[sparse_req, dense_req],
        ranker=RRFRanker(k=60),
        limit=5,
        output_fields=["text"],
    )

    assert len(results) == 1
    assert len(results[0]) > 0
    milvus_client.drop_collection(name)
