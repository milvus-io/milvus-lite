"""JSON field comprehensive tests — insert, query, search, filter.

Covers: roundtrip, nested access, json_contains/all/any, nullable,
search output, mixed value types.
"""

import pytest
from pymilvus import DataType, MilvusClient

DIM = 8


@pytest.fixture
def col(milvus_client):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("info", DataType.JSON, nullable=True)

    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
    milvus_client.create_collection("json_test", schema=schema, index_params=idx)

    milvus_client.insert("json_test", [
        {"id": 0, "vec": [1.0] + [0.0] * (DIM - 1),
         "info": {"name": "alice", "age": 30, "score": 9.5,
                  "tags": ["python", "ml"], "nested": {"x": 1}}},
        {"id": 1, "vec": [0.0, 1.0] + [0.0] * (DIM - 2),
         "info": {"name": "bob", "age": 25, "score": 8.0,
                  "tags": ["java", "web"], "nested": {"x": 2}}},
        {"id": 2, "vec": [0.0, 0.0, 1.0] + [0.0] * (DIM - 3),
         "info": {"name": "charlie", "age": 35, "score": 7.5,
                  "tags": ["python", "data"], "nested": {"x": 3}}},
        {"id": 3, "vec": [0.0] * (DIM - 1) + [1.0],
         "info": {"name": "diana", "age": 28, "score": 9.0,
                  "tags": ["go", "ml"], "nested": {"x": 4}}},
        {"id": 4, "vec": [0.5] * DIM, "info": None},  # nullable JSON
    ])
    milvus_client.load_collection("json_test")
    yield milvus_client
    # cleanup handled by conftest


# ===========================================================================
# Insert + output roundtrip
# ===========================================================================

class TestJSONRoundtrip:
    def test_query_returns_full_json(self, col):
        res = col.query("json_test", filter="id == 0",
                        output_fields=["info"])
        info = res[0]["info"]
        assert info["name"] == "alice"
        assert info["age"] == 30
        assert info["score"] == 9.5
        assert info["tags"] == ["python", "ml"]
        assert info["nested"] == {"x": 1}

    def test_get_returns_json(self, col):
        res = col.get("json_test", ids=[1], output_fields=["info"])
        assert res[0]["info"]["name"] == "bob"

    def test_search_output_includes_json(self, col):
        res = col.search("json_test",
                         data=[[1.0] + [0.0] * (DIM - 1)], limit=1,
                         output_fields=["info"])
        entity = res[0][0]["entity"]
        assert "info" in entity
        assert entity["info"]["name"] == "alice"


# ===========================================================================
# Single-key access filters
# ===========================================================================

class TestJSONKeyAccess:
    def test_string_eq(self, col):
        res = col.query("json_test", filter='info["name"] == "alice"',
                        limit=10)
        assert len(res) == 1
        assert res[0]["id"] == 0

    def test_int_comparison(self, col):
        res = col.query("json_test", filter='info["age"] >= 30', limit=10)
        ids = {r["id"] for r in res}
        assert ids == {0, 2}  # alice(30), charlie(35)

    def test_float_comparison(self, col):
        res = col.query("json_test", filter='info["score"] > 8.5', limit=10)
        ids = {r["id"] for r in res}
        assert ids == {0, 3}  # alice(9.5), diana(9.0)

    def test_nested_key_access(self, col):
        res = col.query("json_test", filter='info["nested"]["x"] >= 3',
                        limit=10)
        ids = {r["id"] for r in res}
        assert ids == {2, 3}  # charlie(3), diana(4)

    def test_combined_json_and_scalar(self, col):
        res = col.query("json_test",
                        filter='info["age"] > 25 and id < 3', limit=10)
        ids = {r["id"] for r in res}
        assert ids == {0, 2}  # alice(30,id=0), charlie(35,id=2)


# ===========================================================================
# json_contains / json_contains_all / json_contains_any
# ===========================================================================

class TestJSONContains:
    def test_json_contains_string(self, col):
        res = col.query("json_test",
                        filter='json_contains(info["tags"], "python")',
                        limit=10)
        ids = {r["id"] for r in res}
        assert ids == {0, 2}  # alice, charlie

    def test_json_contains_string_no_match(self, col):
        res = col.query("json_test",
                        filter='json_contains(info["tags"], "rust")',
                        limit=10)
        assert len(res) == 0

    def test_json_contains_all(self, col):
        res = col.query("json_test",
                        filter='json_contains_all(info["tags"], ["python", "ml"])',
                        limit=10)
        ids = {r["id"] for r in res}
        assert ids == {0}  # only alice has both

    def test_json_contains_any(self, col):
        res = col.query("json_test",
                        filter='json_contains_any(info["tags"], ["ml", "web"])',
                        limit=10)
        ids = {r["id"] for r in res}
        assert ids == {0, 1, 3}  # alice(ml), bob(web), diana(ml)


# ===========================================================================
# Nullable JSON
# ===========================================================================

class TestJSONNullable:
    def test_null_json_returned(self, col):
        res = col.query("json_test", filter="id == 4",
                        output_fields=["info"])
        assert res[0]["info"] is None

    def test_is_null_filter(self, col):
        res = col.query("json_test", filter="info is null", limit=10)
        assert len(res) == 1
        assert res[0]["id"] == 4

    def test_is_not_null_filter(self, col):
        res = col.query("json_test", filter="info is not null", limit=10)
        assert len(res) == 4
        ids = {r["id"] for r in res}
        assert 4 not in ids


# ===========================================================================
# Search with JSON filter
# ===========================================================================

class TestJSONSearch:
    def test_search_with_json_filter(self, col):
        res = col.search("json_test",
                         data=[[1.0] + [0.0] * (DIM - 1)], limit=10,
                         filter='info["age"] >= 30',
                         output_fields=["info"])
        ids = {h["id"] for h in res[0]}
        assert ids == {0, 2}  # alice, charlie

    def test_search_with_json_contains_filter(self, col):
        res = col.search("json_test",
                         data=[[1.0] + [0.0] * (DIM - 1)], limit=10,
                         filter='json_contains(info["tags"], "ml")')
        ids = {h["id"] for h in res[0]}
        assert ids == {0, 3}  # alice, diana

    def test_search_with_nested_json_filter(self, col):
        res = col.search("json_test",
                         data=[[1.0] + [0.0] * (DIM - 1)], limit=10,
                         filter='info["nested"]["x"] <= 2')
        ids = {h["id"] for h in res[0]}
        assert ids == {0, 1}  # alice(1), bob(2)


# ===========================================================================
# LIKE on JSON string values
# ===========================================================================

class TestJSONLike:
    def test_json_string_like(self, col):
        res = col.query("json_test",
                        filter='info["name"] like "a%"', limit=10)
        assert len(res) == 1
        assert res[0]["id"] == 0  # alice


# ===========================================================================
# After flush (segment path)
# ===========================================================================

class TestJSONAfterFlush:
    def test_json_roundtrip_after_flush(self, col):
        col.flush("json_test")
        res = col.query("json_test", filter="id == 0",
                        output_fields=["info"])
        assert res[0]["info"]["name"] == "alice"
        assert res[0]["info"]["tags"] == ["python", "ml"]

    def test_json_filter_after_flush(self, col):
        col.flush("json_test")
        res = col.query("json_test", filter='info["age"] >= 30', limit=10)
        ids = {r["id"] for r in res}
        assert ids == {0, 2}
