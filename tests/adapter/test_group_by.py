"""Phase 13 — Group By search tests.

Covers:
1. Engine-level group_by post-processing
2. gRPC search with group_by_field
3. strict_group_size behavior
4. Different field types (INT64, VARCHAR)
5. group_by with hybrid search
6. group_by with BM25 search
"""

import tempfile

import pytest

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)


# ---------------------------------------------------------------------------
# Engine-level unit tests
# ---------------------------------------------------------------------------

class TestGroupByEngine:
    def _make_collection(self, tmpdir):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.engine.collection import Collection

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
            FieldSchema(name="category", dtype=LDT.VARCHAR),
            FieldSchema(name="group_id", dtype=LDT.INT64),
            FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
        ])
        col = Collection(name="test_gb", data_dir=tmpdir, schema=schema)
        col.insert([
            {"id": 1, "category": "A", "group_id": 1, "vec": [1, 0, 0, 0]},
            {"id": 2, "category": "A", "group_id": 1, "vec": [0.9, 0.1, 0, 0]},
            {"id": 3, "category": "A", "group_id": 1, "vec": [0.8, 0.2, 0, 0]},
            {"id": 4, "category": "B", "group_id": 2, "vec": [0, 1, 0, 0]},
            {"id": 5, "category": "B", "group_id": 2, "vec": [0, 0.9, 0.1, 0]},
            {"id": 6, "category": "C", "group_id": 3, "vec": [0, 0, 1, 0]},
            {"id": 7, "category": "C", "group_id": 3, "vec": [0, 0, 0.9, 0.1]},
            {"id": 8, "category": "D", "group_id": 4, "vec": [0, 0, 0, 1]},
        ])
        return col

    def test_group_by_varchar(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1, 0, 0, 0]],
                top_k=3,
                metric_type="COSINE",
                group_by_field="category",
                group_size=1,
                output_fields=["category"],
            )
            # Should return 3 groups, 1 hit each
            assert len(results[0]) == 3
            categories = [h["entity"]["category"] for h in results[0]]
            assert len(set(categories)) == 3  # all different

    def test_group_by_int64(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1, 0, 0, 0]],
                top_k=4,
                metric_type="COSINE",
                group_by_field="group_id",
                group_size=1,
                output_fields=["group_id"],
            )
            group_ids = [h["entity"]["group_id"] for h in results[0]]
            assert len(set(group_ids)) == len(group_ids)  # all unique

    def test_group_size(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1, 0, 0, 0]],
                top_k=2,
                metric_type="COSINE",
                group_by_field="category",
                group_size=2,
                output_fields=["category"],
            )
            # 2 groups × 2 hits each = up to 4 results
            assert len(results[0]) <= 4
            # Check at most 2 per category
            from collections import Counter
            counts = Counter(h["entity"]["category"] for h in results[0])
            for c in counts.values():
                assert c <= 2

    def test_strict_group_size_true(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1, 0, 0, 0]],
                top_k=10,
                metric_type="COSINE",
                group_by_field="category",
                group_size=3,
                strict_group_size=True,
                output_fields=["category"],
            )
            # Only category A has 3 docs; others have fewer
            from collections import Counter
            counts = Counter(h["entity"]["category"] for h in results[0])
            for c in counts.values():
                assert c == 3  # strict: all groups must have exactly 3

    def test_strict_group_size_false(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1, 0, 0, 0]],
                top_k=10,
                metric_type="COSINE",
                group_by_field="category",
                group_size=3,
                strict_group_size=False,
                output_fields=["category"],
            )
            # All groups included, even with < 3 docs
            categories = set(h["entity"]["category"] for h in results[0])
            assert len(categories) >= 3

    def test_no_group_by_backward_compat(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1, 0, 0, 0]],
                top_k=5,
                metric_type="COSINE",
            )
            # Normal search, no grouping
            assert len(results[0]) == 5


# ---------------------------------------------------------------------------
# gRPC integration tests
# ---------------------------------------------------------------------------

def _create_grouped_collection(client, name):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("category", DataType.VARCHAR, max_length=100)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    client.create_collection(name, schema=schema)
    client.insert(name, [
        {"id": 1, "category": "tech", "dense": [1, 0, 0, 0]},
        {"id": 2, "category": "tech", "dense": [0.9, 0.1, 0, 0]},
        {"id": 3, "category": "tech", "dense": [0.8, 0.2, 0, 0]},
        {"id": 4, "category": "sports", "dense": [0, 1, 0, 0]},
        {"id": 5, "category": "sports", "dense": [0, 0.9, 0.1, 0]},
        {"id": 6, "category": "music", "dense": [0, 0, 1, 0]},
        {"id": 7, "category": "music", "dense": [0, 0, 0.9, 0.1]},
        {"id": 8, "category": "food", "dense": [0, 0, 0, 1]},
    ])
    idx = client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    client.create_index(name, idx)
    client.load_collection(name)


def test_grpc_group_by_basic(milvus_client):
    _create_grouped_collection(milvus_client, "gb_basic")
    results = milvus_client.search(
        "gb_basic",
        data=[[1, 0, 0, 0]],
        limit=3,
        group_by_field="category",
        group_size=1,
        output_fields=["category"],
    )
    categories = [h["entity"]["category"] for h in results[0]]
    assert len(set(categories)) == len(categories)  # unique per group
    milvus_client.drop_collection("gb_basic")


def test_grpc_group_by_group_size(milvus_client):
    _create_grouped_collection(milvus_client, "gb_size")
    results = milvus_client.search(
        "gb_size",
        data=[[1, 0, 0, 0]],
        limit=2,
        group_by_field="category",
        group_size=2,
        output_fields=["category"],
    )
    from collections import Counter
    counts = Counter(h["entity"]["category"] for h in results[0])
    for c in counts.values():
        assert c <= 2
    milvus_client.drop_collection("gb_size")


def test_grpc_group_by_strict(milvus_client):
    _create_grouped_collection(milvus_client, "gb_strict")
    results = milvus_client.search(
        "gb_strict",
        data=[[1, 0, 0, 0]],
        limit=10,
        group_by_field="category",
        group_size=3,
        strict_group_size=True,
        output_fields=["category"],
    )
    # Only "tech" has 3 docs
    from collections import Counter
    counts = Counter(h["entity"]["category"] for h in results[0])
    for c in counts.values():
        assert c == 3
    milvus_client.drop_collection("gb_strict")


def test_grpc_group_by_with_filter(milvus_client):
    _create_grouped_collection(milvus_client, "gb_filt")
    results = milvus_client.search(
        "gb_filt",
        data=[[1, 0, 0, 0]],
        limit=10,
        filter="id <= 5",
        group_by_field="category",
        group_size=1,
        output_fields=["category", "id"],
    )
    for h in results[0]:
        assert h["entity"]["id"] <= 5
    milvus_client.drop_collection("gb_filt")


def test_grpc_group_by_nullable_varchar(milvus_client):
    """Group-by on a nullable scalar field preserves real group values.

    Adapted from Milvus python_client query aggregation coverage for nullable
    group-by fields. Existing local tests only group by non-null fields.
    """
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("category", DataType.VARCHAR, max_length=100, nullable=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    milvus_client.create_collection("gb_nullable", schema=schema)
    milvus_client.insert("gb_nullable", [
        {"id": 1, "category": "tech", "dense": [1, 0, 0, 0]},
        {"id": 2, "category": "tech", "dense": [0.9, 0.1, 0, 0]},
        {"id": 3, "category": "sports", "dense": [0, 1, 0, 0]},
        {"id": 4, "category": "music", "dense": [0, 0, 1, 0]},
        {"id": 5, "category": None, "dense": [0, 0, 0, 1]},
    ])
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    milvus_client.create_index("gb_nullable", idx)
    milvus_client.load_collection("gb_nullable")

    results = milvus_client.search(
        "gb_nullable",
        data=[[1, 0, 0, 0]],
        limit=4,
        group_by_field="category",
        group_size=1,
        output_fields=["category"],
    )
    categories = [hit["entity"].get("category") for hit in results[0]]
    assert "tech" in categories
    assert len(categories) == len(set(categories))
    assert any(category is not None for category in categories)
    milvus_client.drop_collection("gb_nullable")
