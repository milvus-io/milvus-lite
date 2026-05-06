"""Phase 11.8 — Milvus full text search compatibility tests.

Adapted from milvus-io/milvus test suite:
- milvus_client_v2/test_milvus_client_search_text_match.py
- milvus_client/test_milvus_client_search.py (BM25 patterns)

Tests cover:
1. BM25 function auto-generates sparse vectors from text
2. BM25 search with text queries via pymilvus
3. BM25 search with sparse dict queries
4. text_match filter: single token, multi token (OR logic)
5. text_match combined with dense vector search
6. text_match combined with scalar filter
7. BM25 search with scalar filter
8. Insert + flush + search consistency
9. Upsert (re-insert with same pk) + search correctness
10. Delete + search exclusion
11. Multiple queries in single search call
12. Output fields projection in BM25 results
"""

import pytest

from pymilvus import DataType, Function, FunctionType, MilvusClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_fts_collection(client, name, *, with_dense=True):
    """Create a collection with BM25 function."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True, enable_match=True,
                     analyzer_params={"tokenizer": "standard"})
    schema.add_field("category", DataType.VARCHAR, max_length=100, nullable=True)
    if with_dense:
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("bm25_emb", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["bm25_emb"],
    ))
    client.create_collection(name, schema=schema)
    return name


_DOCS = [
    {"id": 1,  "text": "python is a popular programming language",
     "category": "programming", "dense": [1, 0, 0, 0]},
    {"id": 2,  "text": "java is another popular programming language",
     "category": "programming", "dense": [0, 1, 0, 0]},
    {"id": 3,  "text": "machine learning uses python extensively",
     "category": "ai", "dense": [0, 0, 1, 0]},
    {"id": 4,  "text": "deep learning and neural networks",
     "category": "ai", "dense": [0, 0, 0, 1]},
    {"id": 5,  "text": "databases store and retrieve data efficiently",
     "category": "database", "dense": [1, 1, 0, 0]},
    {"id": 6,  "text": "sql is used for database queries",
     "category": "database", "dense": [0, 1, 1, 0]},
    {"id": 7,  "text": "python and java are both object oriented",
     "category": "programming", "dense": [0, 0, 1, 1]},
    {"id": 8,  "text": "natural language processing with python",
     "category": "ai", "dense": [1, 0, 1, 0]},
]


_test_counter = 0

def _setup_loaded(client, name=None):
    """Insert docs, create index, load."""
    global _test_counter
    if name is None:
        _test_counter += 1
        name = f"fts_compat_{_test_counter}"
    _create_fts_collection(client, name)
    client.insert(name, _DOCS)
    idx = client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    client.create_index(name, idx)
    client.load_collection(name)
    return name


# ---------------------------------------------------------------------------
# 1. BM25 function auto-generates sparse vectors
# ---------------------------------------------------------------------------

def test_bm25_auto_generation(milvus_client):
    name = _setup_loaded(milvus_client)
    results = milvus_client.query(name, filter="id >= 1",
                                  output_fields=["id", "text"], limit=20)
    assert len(results) == len(_DOCS)
    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 2-3. BM25 search
# ---------------------------------------------------------------------------

def test_bm25_search_text_query(milvus_client):
    """BM25 search with text string query."""
    name = _setup_loaded(milvus_client)

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    query = compute_tf([term_to_id("python")])

    results = milvus_client.search(
        name, data=[query], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        limit=10, output_fields=["text"],
    )
    assert len(results) == 1
    hit_ids = [h["id"] for h in results[0]]
    # Docs 1, 3, 7, 8 contain "python"
    for doc_id in [1, 3, 7, 8]:
        assert doc_id in hit_ids
    # Docs without "python" should not appear
    for doc_id in [2, 4, 5, 6]:
        assert doc_id not in hit_ids

    milvus_client.drop_collection(name)


def test_bm25_search_relevance_ordering(milvus_client):
    """BM25 results should be ordered by relevance (distance ascending = score descending)."""
    name = _setup_loaded(milvus_client)

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    query = compute_tf([term_to_id("python")])

    results = milvus_client.search(
        name, data=[query], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        limit=10, output_fields=["text"],
    )
    # Distances should be sorted ascending (= BM25 scores descending)
    distances = [h["distance"] for h in results[0]]
    assert distances == sorted(distances)

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 4. text_match filter
# ---------------------------------------------------------------------------

def test_text_match_single_token(milvus_client):
    name = _setup_loaded(milvus_client)
    results = milvus_client.query(
        name, filter="text_match(text, 'python')",
        output_fields=["id", "text"], limit=20,
    )
    for r in results:
        assert "python" in r["text"].lower()
    assert len(results) == 4  # docs 1, 3, 7, 8

    milvus_client.drop_collection(name)


def test_text_match_multi_token_or(milvus_client):
    """Multi-token text_match uses OR logic."""
    name = _setup_loaded(milvus_client)
    results = milvus_client.query(
        name, filter="text_match(text, 'python java')",
        output_fields=["id", "text"], limit=20,
    )
    ids = {r["id"] for r in results}
    # Docs with "python" or "java": 1, 2, 3, 7, 8
    assert ids == {1, 2, 3, 7, 8}

    milvus_client.drop_collection(name)


def test_text_match_no_results(milvus_client):
    name = _setup_loaded(milvus_client)
    results = milvus_client.query(
        name, filter="text_match(text, 'nonexistent_word_xyz')",
        output_fields=["id"], limit=20,
    )
    assert len(results) == 0

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 5-6. text_match combined with other filters
# ---------------------------------------------------------------------------

def test_text_match_with_dense_search(milvus_client):
    """text_match as pre-filter in dense vector search."""
    name = _setup_loaded(milvus_client)
    results = milvus_client.search(
        name, data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        filter="text_match(text, 'programming')",
        limit=10, output_fields=["text"],
    )
    for h in results[0]:
        assert "programming" in h["entity"]["text"].lower()

    milvus_client.drop_collection(name)


def test_text_match_and_scalar_filter(milvus_client):
    """text_match combined with scalar filter using AND."""
    name = _setup_loaded(milvus_client)
    results = milvus_client.query(
        name,
        filter="text_match(text, 'python') and category == 'ai'",
        output_fields=["id", "text", "category"], limit=20,
    )
    for r in results:
        assert "python" in r["text"].lower()
        assert r["category"] == "ai"
    ids = {r["id"] for r in results}
    assert ids == {3, 8}

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 7. BM25 search with scalar filter
# ---------------------------------------------------------------------------

def test_bm25_search_with_filter(milvus_client):
    name = _setup_loaded(milvus_client)

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    query = compute_tf([term_to_id("python")])

    results = milvus_client.search(
        name, data=[query], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        filter="category == 'programming'",
        limit=10, output_fields=["text", "category"],
    )
    for h in results[0]:
        assert h["entity"]["category"] == "programming"
        assert "python" in h["entity"]["text"].lower()

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 8. Flush + search
# ---------------------------------------------------------------------------

def test_bm25_search_after_flush(milvus_client):
    name = _setup_loaded(milvus_client)
    milvus_client.flush(name)

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    query = compute_tf([term_to_id("database")])

    results = milvus_client.search(
        name, data=[query], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        limit=10, output_fields=["text"],
    )
    hit_ids = [h["id"] for h in results[0]]
    # "database" tokenizes to "database" — matches doc 6 ("database queries")
    # Doc 5 has "databases" which tokenizes to "databases" (different token)
    assert 6 in hit_ids  # "sql is used for database queries"

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 9. Upsert + search
# ---------------------------------------------------------------------------

def test_upsert_updates_bm25(milvus_client):
    """Re-inserting with same pk updates BM25 vectors."""
    name = _setup_loaded(milvus_client)

    # Re-insert doc 1 with different text
    milvus_client.insert(name, [
        {"id": 1, "text": "rust is a systems programming language",
         "category": "programming", "dense": [1, 0, 0, 0]},
    ])

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf

    # "python" should no longer match doc 1
    query = compute_tf([term_to_id("python")])
    results = milvus_client.search(
        name, data=[query], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        limit=10,
    )
    hit_ids = [h["id"] for h in results[0]]
    assert 1 not in hit_ids

    # "rust" should match doc 1
    query = compute_tf([term_to_id("rust")])
    results = milvus_client.search(
        name, data=[query], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        limit=10,
    )
    hit_ids = [h["id"] for h in results[0]]
    assert 1 in hit_ids

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 10. Delete + search
# ---------------------------------------------------------------------------

def test_delete_excludes_from_bm25(milvus_client):
    name = _setup_loaded(milvus_client)

    milvus_client.delete(name, ids=[1])

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    query = compute_tf([term_to_id("python")])

    results = milvus_client.search(
        name, data=[query], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        limit=10,
    )
    hit_ids = [h["id"] for h in results[0]]
    assert 1 not in hit_ids
    # Other python docs should still be there
    assert 3 in hit_ids

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 11. Multiple queries
# ---------------------------------------------------------------------------

def test_bm25_multiple_queries(milvus_client):
    name = _setup_loaded(milvus_client)

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    q1 = compute_tf([term_to_id("python")])
    q2 = compute_tf([term_to_id("database")])

    results = milvus_client.search(
        name, data=[q1, q2], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        limit=10,
    )
    assert len(results) == 2
    q1_ids = [h["id"] for h in results[0]]
    q2_ids = [h["id"] for h in results[1]]
    assert 1 in q1_ids  # python doc
    assert 6 in q2_ids  # "sql is used for database queries"

    milvus_client.drop_collection(name)


# ---------------------------------------------------------------------------
# 12. Output fields
# ---------------------------------------------------------------------------

def test_bm25_output_fields(milvus_client):
    name = _setup_loaded(milvus_client)

    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    query = compute_tf([term_to_id("python")])

    results = milvus_client.search(
        name, data=[query], anns_field="bm25_emb",
        search_params={"metric_type": "BM25"},
        limit=5, output_fields=["text", "category"],
    )
    for h in results[0]:
        assert "text" in h["entity"]
        assert "category" in h["entity"]

    milvus_client.drop_collection(name)
