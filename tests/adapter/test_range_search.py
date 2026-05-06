"""Phase 14 — Range Search tests.

Covers:
1. Engine-level range filter (radius/range_filter)
2. gRPC range search with COSINE metric
3. Only radius (no range_filter)
4. Only range_filter (no radius)
5. Range search with scalar filter
6. Range search returns empty when no matches
"""

import tempfile

import pytest

from pymilvus import DataType, MilvusClient


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------

class TestRangeSearchEngine:
    def _make_collection(self, tmpdir):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.engine.collection import Collection

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
        ])
        col = Collection(name="test_rs", data_dir=tmpdir, schema=schema)
        # Insert vectors at known positions
        col.insert([
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0]},   # dist ~0 from query
            {"id": 2, "vec": [0.7, 0.7, 0.0, 0.0]},   # dist ~0.29
            {"id": 3, "vec": [0.0, 1.0, 0.0, 0.0]},   # dist ~1.0
            {"id": 4, "vec": [0.0, 0.0, 1.0, 0.0]},   # dist ~1.0
            {"id": 5, "vec": [-1.0, 0.0, 0.0, 0.0]},  # dist ~2.0
        ])
        return col

    def test_range_both_bounds(self):
        """radius < distance <= range_filter."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=10,
                metric_type="COSINE",
                radius=0.1,
                range_filter=1.5,
            )
            for hit in results[0]:
                assert hit["distance"] > 0.1
                assert hit["distance"] <= 1.5

    def test_range_only_radius(self):
        """Only radius (outer bound for COSINE): distance <= radius."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=10,
                metric_type="COSINE",
                radius=0.5,
            )
            for hit in results[0]:
                assert hit["distance"] <= 0.5

    def test_range_only_range_filter(self):
        """Only range_filter (inner bound for COSINE): distance >= range_filter."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=10,
                metric_type="COSINE",
                range_filter=0.5,
            )
            for hit in results[0]:
                assert hit["distance"] >= 0.5

    def test_range_empty_result(self):
        """No results in range."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=10,
                metric_type="COSINE",
                radius=10.0,
                range_filter=20.0,
            )
            assert results[0] == []

    def test_range_respects_limit(self):
        """Limit still applies after range filtering."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=1,
                metric_type="COSINE",
                radius=-1.0,
                range_filter=10.0,
            )
            assert len(results[0]) <= 1

    def test_no_range_backward_compat(self):
        """Without range params, behaves normally."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=5,
                metric_type="COSINE",
            )
            assert len(results[0]) == 5


# ---------------------------------------------------------------------------
# gRPC integration tests
# ---------------------------------------------------------------------------

def _create_range_collection(client, name):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("label", DataType.INT64)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    client.create_collection(name, schema=schema)
    client.insert(name, [
        {"id": 1, "label": 1, "vec": [1.0, 0.0, 0.0, 0.0]},
        {"id": 2, "label": 1, "vec": [0.7, 0.7, 0.0, 0.0]},
        {"id": 3, "label": 2, "vec": [0.0, 1.0, 0.0, 0.0]},
        {"id": 4, "label": 2, "vec": [0.0, 0.0, 1.0, 0.0]},
        {"id": 5, "label": 3, "vec": [-1.0, 0.0, 0.0, 0.0]},
    ])
    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    client.create_index(name, idx)
    client.load_collection(name)


def test_grpc_range_search_cosine(milvus_client):
    _create_range_collection(milvus_client, "rs_cosine")
    results = milvus_client.search(
        "rs_cosine",
        data=[[1.0, 0.0, 0.0, 0.0]],
        search_params={
            "metric_type": "COSINE",
            "params": {"radius": 0.1, "range_filter": 1.5},
        },
        limit=10,
    )
    for hit in results[0]:
        assert hit["distance"] > 0.1
        assert hit["distance"] <= 1.5
    milvus_client.drop_collection("rs_cosine")


def test_grpc_range_only_range_filter(milvus_client):
    _create_range_collection(milvus_client, "rs_rf")
    results = milvus_client.search(
        "rs_rf",
        data=[[1.0, 0.0, 0.0, 0.0]],
        search_params={
            "metric_type": "COSINE",
            "params": {"range_filter": 0.5},
        },
        limit=10,
    )
    for hit in results[0]:
        assert hit["distance"] >= 0.5
    milvus_client.drop_collection("rs_rf")


def test_grpc_range_with_filter(milvus_client):
    _create_range_collection(milvus_client, "rs_filt")
    results = milvus_client.search(
        "rs_filt",
        data=[[1.0, 0.0, 0.0, 0.0]],
        search_params={
            "metric_type": "COSINE",
            "params": {"radius": -1.0, "range_filter": 2.0},
        },
        filter="label == 1",
        limit=10,
        output_fields=["label"],
    )
    for hit in results[0]:
        assert hit["entity"]["label"] == 1
    milvus_client.drop_collection("rs_filt")


def test_grpc_range_empty(milvus_client):
    _create_range_collection(milvus_client, "rs_empty")
    results = milvus_client.search(
        "rs_empty",
        data=[[1.0, 0.0, 0.0, 0.0]],
        search_params={
            "metric_type": "COSINE",
            "params": {"radius": 5.0, "range_filter": 10.0},
        },
        limit=10,
    )
    assert len(results[0]) == 0
    milvus_client.drop_collection("rs_empty")


def test_grpc_range_search_ip_bounds(milvus_client):
    """Range search with IP keeps scores in (radius, range_filter].

    Adapted from Milvus python_client range-search coverage; existing local
    tests only covered COSINE semantics.
    """
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    milvus_client.create_collection("rs_ip", schema=schema)
    milvus_client.insert("rs_ip", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0]},
        {"id": 2, "vec": [0.8, 0.0, 0.0, 0.0]},
        {"id": 3, "vec": [0.4, 0.0, 0.0, 0.0]},
        {"id": 4, "vec": [-1.0, 0.0, 0.0, 0.0]},
    ])
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="IP", params={})
    milvus_client.create_index("rs_ip", idx)
    milvus_client.load_collection("rs_ip")

    results = milvus_client.search(
        "rs_ip",
        data=[[1.0, 0.0, 0.0, 0.0]],
        search_params={
            "metric_type": "IP",
            "params": {"radius": 0.5, "range_filter": 0.9},
        },
        limit=10,
        output_fields=["id"],
    )
    assert [hit["entity"]["id"] for hit in results[0]] == [2]
    for hit in results[0]:
        assert 0.5 < hit["distance"] <= 0.9
    milvus_client.drop_collection("rs_ip")
