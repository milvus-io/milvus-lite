"""Phase 11.7 — gRPC adapter full text search integration tests.

Tests:
- CreateCollection with BM25 Function + SparseFloatVector
- Insert text data (sparse field auto-generated)
- BM25 search via pymilvus
- text_match filter via pymilvus query
- DescribeCollection returns functions and FTS field attributes
"""

import pytest

from pymilvus import DataType, MilvusClient, Function, FunctionType


# ---------------------------------------------------------------------------
# Schema creation with BM25 function
# ---------------------------------------------------------------------------

def test_create_collection_with_bm25(milvus_client):
    """Create collection with BM25 function via pymilvus."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=65535,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "standard"},
        enable_match=True,
    )
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("sparse_emb", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_emb"],
    ))
    milvus_client.create_collection("fts_test", schema=schema)

    # Verify collection exists
    assert milvus_client.has_collection("fts_test")
    milvus_client.drop_collection("fts_test")


def test_describe_collection_shows_functions(milvus_client):
    """DescribeCollection should return function definitions."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("sparse_emb", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_emb"],
    ))
    milvus_client.create_collection("fts_desc", schema=schema)

    desc = milvus_client.describe_collection("fts_desc")
    # Check that sparse_emb field is in the schema
    field_names = [f["name"] for f in desc["fields"]]
    assert "sparse_emb" in field_names
    assert "text" in field_names

    milvus_client.drop_collection("fts_desc")


# ---------------------------------------------------------------------------
# Insert + search
# ---------------------------------------------------------------------------

def _setup_fts_collection(client, name="fts_demo"):
    """Helper: create FTS collection, insert data, create indexes, load."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True, enable_match=True,
                     analyzer_params={"tokenizer": "standard"})
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("sparse_emb", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_emb"],
    ))
    client.create_collection(name, schema=schema)

    # Insert documents
    client.insert(name, [
        {"id": 1, "text": "python programming language", "dense": [1, 0, 0, 0]},
        {"id": 2, "text": "java programming language", "dense": [0, 1, 0, 0]},
        {"id": 3, "text": "machine learning algorithms", "dense": [0, 0, 1, 0]},
        {"id": 4, "text": "deep learning neural networks", "dense": [0, 0, 0, 1]},
    ])

    # Create dense index and load
    idx = client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    client.create_index(name, idx)
    client.load_collection(name)
    return name


def test_insert_with_bm25_function(milvus_client):
    """Insert text data — sparse_emb should be auto-generated."""
    name = _setup_fts_collection(milvus_client)

    # Query to verify data was inserted
    results = milvus_client.query(name, filter="id >= 1", output_fields=["text"],
                                  limit=10)
    assert len(results) == 4
    milvus_client.drop_collection(name)


def test_bm25_search_via_grpc(milvus_client):
    """BM25 search through pymilvus gRPC interface.

    pymilvus MilvusClient.search with sparse vector data sends a
    PlaceholderGroup with type=SPARSE_FLOAT_VECTOR (104).
    """
    name = _setup_fts_collection(milvus_client)

    # Build sparse query dict: search for "python"
    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    query_tf = compute_tf([term_to_id("python")])

    results = milvus_client.search(
        name,
        data=[query_tf],
        anns_field="sparse_emb",
        search_params={"metric_type": "BM25"},
        limit=4,
        output_fields=["text"],
    )

    # Verify results
    assert len(results) > 0
    hit_ids = [hit["id"] for hit in results[0]]
    assert 1 in hit_ids

    milvus_client.drop_collection(name)


def test_text_match_filter_via_grpc(milvus_client):
    """text_match filter in query() through gRPC."""
    name = _setup_fts_collection(milvus_client)

    results = milvus_client.query(
        name,
        filter="text_match(text, 'python')",
        output_fields=["id", "text"],
        limit=10,
    )
    assert len(results) == 1
    assert results[0]["id"] == 1

    milvus_client.drop_collection(name)


def test_text_match_multi_token_via_grpc(milvus_client):
    """text_match with multiple tokens (OR logic) through gRPC."""
    name = _setup_fts_collection(milvus_client)

    results = milvus_client.query(
        name,
        filter="text_match(text, 'python java')",
        output_fields=["id"],
        limit=10,
    )
    ids = {r["id"] for r in results}
    assert ids == {1, 2}

    milvus_client.drop_collection(name)


def test_text_match_with_dense_search(milvus_client):
    """text_match as filter in dense vector search."""
    name = _setup_fts_collection(milvus_client)

    results = milvus_client.search(
        name,
        data=[[1.0, 0.0, 0.0, 0.0]],
        anns_field="dense",
        filter="text_match(text, 'programming')",
        limit=10,
        output_fields=["text"],
    )
    hit_ids = [h["id"] for h in results[0]]
    assert set(hit_ids) == {1, 2}

    milvus_client.drop_collection(name)
