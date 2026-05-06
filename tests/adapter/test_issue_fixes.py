"""Tests for GitHub issue fixes #1-#7."""

import tempfile

import numpy as np
import pytest

pymilvus = pytest.importorskip("pymilvus")
pytest.importorskip("grpc")

from pymilvus import MilvusClient, DataType


# ---------------------------------------------------------------------------
# #1: Dynamic field values returned in query output
# ---------------------------------------------------------------------------

def test_dynamic_field_query_output(milvus_client):
    """Issue #1: Dynamic field values must be returned in query output."""
    schema = MilvusClient.create_schema(enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    milvus_client.create_collection("dyn_test", schema=schema)
    milvus_client.insert("dyn_test", [
        {"id": 1, "vec": [1, 0, 0, 0], "category": "tech", "level": 5},
        {"id": 2, "vec": [0, 1, 0, 0], "category": "news", "level": 3},
    ])
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="FLAT",
                  metric_type="COSINE", params={})
    milvus_client.create_index("dyn_test", idx)
    milvus_client.load_collection("dyn_test")

    # Query with dynamic field in output_fields
    rows = milvus_client.query("dyn_test", filter="id >= 1",
                                output_fields=["category", "level"], limit=10)
    assert len(rows) == 2
    for r in rows:
        assert "category" in r
        assert "level" in r
    assert rows[0]["category"] in ("tech", "news")

    # Search with dynamic field in output_fields
    results = milvus_client.search("dyn_test", data=[[1, 0, 0, 0]], limit=2,
                                    output_fields=["category"])
    assert "category" in results[0][0]["entity"]

    milvus_client.drop_collection("dyn_test")


# ---------------------------------------------------------------------------
# #2: Multiple FLOAT_VECTOR fields
# ---------------------------------------------------------------------------

def test_multi_vector_fields(milvus_client):
    """Issue #2: Collections with multiple FLOAT_VECTOR fields must work."""
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("v1", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("v2", DataType.FLOAT_VECTOR, dim=4)
    milvus_client.create_collection("multi_vec", schema=schema)
    milvus_client.insert("multi_vec", [
        {"id": 1, "v1": [1, 0, 0, 0], "v2": [0, 0, 0, 1]},
        {"id": 2, "v1": [0, 1, 0, 0], "v2": [0, 0, 1, 0]},
    ])
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="v1", index_type="FLAT",
                  metric_type="COSINE", params={})
    idx.add_index(field_name="v2", index_type="FLAT",
                  metric_type="COSINE", params={})
    milvus_client.create_index("multi_vec", idx)
    milvus_client.load_collection("multi_vec")

    # Search on v1
    r1 = milvus_client.search("multi_vec", data=[[1, 0, 0, 0]],
                               anns_field="v1", limit=2)
    assert r1[0][0]["id"] == 1

    # Search on v2
    r2 = milvus_client.search("multi_vec", data=[[0, 0, 0, 1]],
                               anns_field="v2", limit=2)
    assert r2[0][0]["id"] == 1

    milvus_client.drop_collection("multi_vec")


# ---------------------------------------------------------------------------
# #3: IP distance sign
# ---------------------------------------------------------------------------

def test_ip_distance_sign():
    """Issue #3: IP self-search of unit vector should return positive distance."""
    from milvus_lite.schema.types import CollectionSchema, FieldSchema
    from milvus_lite.schema.types import DataType as LDT
    from milvus_lite.engine.collection import Collection

    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
    ])
    with tempfile.TemporaryDirectory() as d:
        col = Collection(name="ip_test", data_dir=d, schema=schema)
        col.insert([
            {"id": 1, "vec": [1, 0, 0, 0]},
            {"id": 2, "vec": [0, 1, 0, 0]},
        ])
        col.create_index("vec", {
            "index_type": "BRUTE_FORCE", "metric_type": "IP", "params": {},
        })
        col.load()
        results = col.search([[1, 0, 0, 0]], top_k=1, metric_type="IP")
        assert results[0][0]["id"] == 1
        # IP distance for self-match should be positive (dot product = 1.0)
        assert results[0][0]["distance"] > 0
        assert abs(results[0][0]["distance"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# #4: gRPC large batch insert
# ---------------------------------------------------------------------------

def test_grpc_large_batch_insert(milvus_client):
    """Issue #4: Insert ~10K vectors should not hit 4MB gRPC limit."""
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=128)
    milvus_client.create_collection("big_batch", schema=schema)

    data = [
        {"id": i, "vec": np.random.rand(128).tolist()}
        for i in range(5000)
    ]
    # Should not raise RESOURCE_EXHAUSTED
    result = milvus_client.insert("big_batch", data)
    assert result["insert_count"] == 5000

    milvus_client.drop_collection("big_batch")


# ---------------------------------------------------------------------------
# #6: describe_index with field_name
# ---------------------------------------------------------------------------

def test_describe_index_by_field_name(milvus_client):
    """Issue #6: describe_index should work with field_name parameter."""
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    milvus_client.create_collection("idx_desc", schema=schema)

    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="FLAT",
                  metric_type="COSINE", params={})
    milvus_client.create_index("idx_desc", idx)

    # Should not raise TypeError
    info = milvus_client.describe_index("idx_desc", index_name="vec")
    assert info is not None

    milvus_client.drop_collection("idx_desc")


# ---------------------------------------------------------------------------
# #7: AUTOINDEX uses HNSW when faiss available
# ---------------------------------------------------------------------------

def test_autoindex_uses_hnsw():
    """Issue #7: AUTOINDEX should build HNSW when faiss-cpu is available."""
    from milvus_lite.index.factory import is_faiss_available, build_index_from_spec
    from milvus_lite.index.spec import IndexSpec

    if not is_faiss_available():
        pytest.skip("faiss-cpu not installed")

    spec = IndexSpec(
        field_name="vec", index_type="AUTOINDEX",
        metric_type="COSINE", build_params={},
    )
    vectors = np.random.rand(100, 16).astype(np.float32)
    idx = build_index_from_spec(spec, vectors)
    assert idx.index_type == "HNSW"
