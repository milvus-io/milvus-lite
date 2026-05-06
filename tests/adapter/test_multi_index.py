"""Phase 18 — Multi-vector independent index tests.

Covers:
1. Engine: create separate indexes on dense + sparse fields
2. Engine: load builds both indexes
3. Engine: drop one index, keep the other
4. gRPC: pymilvus create_index on two vector fields
5. gRPC: DescribeIndex returns multiple descriptions
6. gRPC: search on each field works after loading
"""

import tempfile

import pytest

from pymilvus import DataType, Function, FunctionType, MilvusClient


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------

class TestMultiIndexEngine:
    def _make_collection(self, tmpdir):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
            Function as LFunc, FunctionType as LFT,
        )
        from milvus_lite.engine.collection import Collection

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
                FieldSchema(name="text", dtype=LDT.VARCHAR,
                            enable_analyzer=True),
                FieldSchema(name="dense", dtype=LDT.FLOAT_VECTOR, dim=4),
                FieldSchema(name="sparse", dtype=LDT.SPARSE_FLOAT_VECTOR,
                            is_function_output=True),
            ],
            functions=[
                LFunc(name="bm25", function_type=LFT.BM25,
                      input_field_names=["text"],
                      output_field_names=["sparse"]),
            ],
        )
        col = Collection(name="multi_idx", data_dir=tmpdir, schema=schema)
        col.insert([
            {"id": 1, "text": "python programming", "dense": [1, 0, 0, 0]},
            {"id": 2, "text": "java programming", "dense": [0, 1, 0, 0]},
            {"id": 3, "text": "machine learning", "dense": [0, 0, 1, 0]},
        ])
        return col

    def test_create_two_indexes(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.create_index("dense", {
                "index_type": "BRUTE_FORCE", "metric_type": "COSINE",
                "params": {},
            })
            col.create_index("sparse", {
                "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25",
                "params": {},
            })
            assert col.has_index("dense")
            assert col.has_index("sparse")
            assert col.has_index()  # any index

    def test_load_builds_both(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.create_index("dense", {
                "index_type": "BRUTE_FORCE", "metric_type": "COSINE",
                "params": {},
            })
            col.create_index("sparse", {
                "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25",
                "params": {},
            })
            col.load()
            assert col.load_state == "loaded"

            # Dense search works
            r1 = col.search([[1, 0, 0, 0]], top_k=2, anns_field="dense")
            assert len(r1[0]) == 2

            # BM25 search works
            r2 = col.search(["python"], top_k=2, metric_type="BM25",
                            anns_field="sparse")
            assert len(r2[0]) >= 1

    def test_drop_one_keeps_other(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.create_index("dense", {
                "index_type": "BRUTE_FORCE", "metric_type": "COSINE",
                "params": {},
            })
            col.create_index("sparse", {
                "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25",
                "params": {},
            })
            col.release()  # drop_index requires released state (Milvus semantics)
            col.drop_index("sparse")
            assert col.has_index("dense")
            assert not col.has_index("sparse")

    def test_get_index_info_per_field(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.create_index("dense", {
                "index_type": "BRUTE_FORCE", "metric_type": "COSINE",
                "params": {},
            })
            info = col.get_index_info("dense")
            assert info["index_type"] == "BRUTE_FORCE"
            assert col.get_index_info("sparse") is None


# ---------------------------------------------------------------------------
# gRPC integration tests
# ---------------------------------------------------------------------------

def _create_multi_idx_collection(client, name):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25", function_type=FunctionType.BM25,
        input_field_names=["text"], output_field_names=["sparse"],
    ))
    client.create_collection(name, schema=schema)
    client.insert(name, [
        {"id": 1, "text": "python programming", "dense": [1, 0, 0, 0]},
        {"id": 2, "text": "java programming", "dense": [0, 1, 0, 0]},
        {"id": 3, "text": "machine learning", "dense": [0, 0, 1, 0]},
    ])


def test_grpc_multi_index_create(milvus_client):
    _create_multi_idx_collection(milvus_client, "mi_create")
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                  metric_type="BM25", params={})
    milvus_client.create_index("mi_create", idx)
    milvus_client.load_collection("mi_create")

    # Both indexes should be queryable
    indexes = milvus_client.list_indexes("mi_create")
    assert len(indexes) >= 2

    milvus_client.drop_collection("mi_create")


def test_grpc_multi_index_search_both(milvus_client):
    _create_multi_idx_collection(milvus_client, "mi_search")
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    milvus_client.create_index("mi_search", idx)
    milvus_client.load_collection("mi_search")

    # Dense search
    r1 = milvus_client.search("mi_search", data=[[1, 0, 0, 0]], limit=3)
    assert len(r1[0]) == 3

    # BM25 search (no index needed — on-the-fly)
    from milvus_lite.analyzer.hash import term_to_id
    from milvus_lite.analyzer.sparse import compute_tf
    q = compute_tf([term_to_id("python")])
    r2 = milvus_client.search("mi_search", data=[q], anns_field="sparse",
                              search_params={"metric_type": "BM25"}, limit=3)
    assert len(r2[0]) >= 1

    milvus_client.drop_collection("mi_search")


def test_grpc_drop_one_index(milvus_client):
    _create_multi_idx_collection(milvus_client, "mi_drop")
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                  metric_type="BM25", params={})
    milvus_client.create_index("mi_drop", idx)

    # Drop requires released state (Milvus semantics).
    milvus_client.release_collection("mi_drop")
    # Drop only sparse index
    milvus_client.drop_index("mi_drop", index_name="sparse_idx")

    # Dense index still exists
    indexes = milvus_client.list_indexes("mi_drop")
    assert len(indexes) >= 1

    milvus_client.drop_collection("mi_drop")
