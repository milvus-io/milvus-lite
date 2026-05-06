"""Phase 19 — Array field type tests.

Covers:
1. Engine: insert/query with ARRAY fields
2. Filter: array_contains, array_contains_all, array_contains_any
3. Filter: array_length
4. Filter: array index access field[N]
5. gRPC: pymilvus create collection with ARRAY + insert + query
"""

import tempfile

import pytest
from pymilvus import DataType, MilvusClient


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------

class TestArrayEngine:
    def _make_collection(self, tmpdir):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.engine.collection import Collection

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
            FieldSchema(name="tags", dtype=LDT.ARRAY,
                        element_type=LDT.VARCHAR, max_capacity=10),
            FieldSchema(name="scores", dtype=LDT.ARRAY,
                        element_type=LDT.INT64, max_capacity=10),
        ])
        col = Collection(name="arr_test", data_dir=tmpdir, schema=schema)
        col.insert([
            {"id": 1, "vec": [1, 0, 0, 0],
             "tags": ["python", "ai"], "scores": [95, 88]},
            {"id": 2, "vec": [0, 1, 0, 0],
             "tags": ["java", "backend"], "scores": [80, 91, 75]},
            {"id": 3, "vec": [0, 0, 1, 0],
             "tags": ["python", "ml", "ai"], "scores": [70]},
        ])
        return col

    def test_insert_and_query_array(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            rows = col.query("id >= 1", output_fields=["tags", "scores"])
            assert len(rows) == 3
            assert rows[0]["tags"] == ["python", "ai"]
            assert rows[1]["scores"] == [80, 91, 75]

    def test_array_contains(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            rows = col.query('array_contains(tags, "python")')
            ids = {r["id"] for r in rows}
            assert ids == {1, 3}

    def test_array_contains_all(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            rows = col.query('array_contains_all(tags, ["python", "ai"])')
            ids = {r["id"] for r in rows}
            assert ids == {1, 3}

    def test_array_contains_any(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            rows = col.query('array_contains_any(tags, ["java", "ml"])')
            ids = {r["id"] for r in rows}
            assert ids == {2, 3}

    def test_array_length(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            rows = col.query("array_length(scores) >= 2")
            ids = {r["id"] for r in rows}
            assert ids == {1, 2}

    def test_array_index_access(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            rows = col.query("scores[0] >= 80")
            ids = {r["id"] for r in rows}
            assert ids == {1, 2}

    def test_array_combined_filter(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            rows = col.query('array_contains(tags, "python") and array_length(scores) >= 2')
            ids = {r["id"] for r in rows}
            assert ids == {1}


# ---------------------------------------------------------------------------
# gRPC integration tests
# ---------------------------------------------------------------------------

def test_grpc_array_create_insert_query(milvus_client):
    """Full lifecycle: create with ARRAY fields, insert, query."""
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("tags", DataType.ARRAY,
                     element_type=DataType.VARCHAR,
                     max_capacity=10, max_length=64)
    schema.add_field("nums", DataType.ARRAY,
                     element_type=DataType.INT64,
                     max_capacity=10)
    milvus_client.create_collection("arr_grpc", schema=schema)
    milvus_client.insert("arr_grpc", [
        {"id": 1, "vec": [1, 0, 0, 0], "tags": ["a", "b"], "nums": [10, 20]},
        {"id": 2, "vec": [0, 1, 0, 0], "tags": ["c"],      "nums": [30]},
    ])

    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    milvus_client.create_index("arr_grpc", idx)
    milvus_client.load_collection("arr_grpc")

    # Query all
    rows = milvus_client.query("arr_grpc", filter="id >= 1",
                               output_fields=["tags", "nums"], limit=10)
    assert len(rows) == 2
    assert rows[0]["tags"] == ["a", "b"]

    # array_contains
    rows = milvus_client.query("arr_grpc",
                               filter='array_contains(tags, "a")',
                               output_fields=["id"], limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == 1

    # array_length
    rows = milvus_client.query("arr_grpc",
                               filter="array_length(nums) == 2",
                               output_fields=["id"], limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == 1

    # array index access
    rows = milvus_client.query("arr_grpc",
                               filter="nums[0] >= 20",
                               output_fields=["id"], limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == 2

    milvus_client.drop_collection("arr_grpc")


def test_grpc_array_search_with_filter(milvus_client):
    """Search with array filter."""
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("tags", DataType.ARRAY,
                     element_type=DataType.VARCHAR,
                     max_capacity=10, max_length=64)
    milvus_client.create_collection("arr_search", schema=schema)
    milvus_client.insert("arr_search", [
        {"id": 1, "vec": [1, 0, 0, 0], "tags": ["python", "ai"]},
        {"id": 2, "vec": [0, 1, 0, 0], "tags": ["java"]},
        {"id": 3, "vec": [0, 0, 1, 0], "tags": ["python", "ml"]},
    ])

    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    milvus_client.create_index("arr_search", idx)
    milvus_client.load_collection("arr_search")

    results = milvus_client.search(
        "arr_search", data=[[1, 0, 0, 0]], limit=10,
        filter='array_contains(tags, "python")',
        output_fields=["tags"],
    )
    hit_ids = {h["id"] for h in results[0]}
    assert hit_ids == {1, 3}

    milvus_client.drop_collection("arr_search")
