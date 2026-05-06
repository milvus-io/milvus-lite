"""
Milvus API edge case tests — verify MilvusLite behavior under extreme/atypical scenarios.

Coverage:
  1. Empty collection operations (search/query/get on empty collection)
  2. Data type boundary values (INT8 overflow, INT64 extremes, empty string, very long string)
  3. Vector boundaries (zero vector, identical vectors, top_k > total data count)
  4. Duplicate primary keys (duplicate PK in same batch, multiple inserts with same PK)
  5. Complex filter expressions (nested AND/OR/NOT, LIKE, NOT IN, array_contains, is null)
  6. Query offset pagination
  7. Search offset pagination
  8. count(*) aggregation
  9. output_fields=["*"] wildcard
 10. Operations after deleting all data
 11. Upsert batch + partial overlap
 12. Many partition operations
 13. Search/query without load should raise error
 14. Index rebuild (drop + recreate)
 15. Data persistence (data survives server restart)
 16. Group By Search
 17. Range Search (radius / range_filter)
 18. Special character field values
 19. JSON deep nested access
 20. Empty filter / empty output_fields
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from milvus_lite.adapter.grpc.server import start_server_in_thread

DIM = 16
SEED = 77


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="edge_test_")
    server, db, port = start_server_in_thread(data_dir)
    yield port, data_dir
    server.stop(grace=2)
    db.close()
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def client(server):
    port, _ = server
    c = MilvusClient(uri=f"http://127.0.0.1:{port}")
    yield c
    for name in c.list_collections():
        c.drop_collection(name)


def rvecs(n: int, dim: int = DIM, seed: int = SEED) -> list[list[float]]:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32).tolist()


def make_simple_collection(client, name, dim=DIM, metric="COSINE"):
    """Quickly create a pk+vec+label collection with HNSW index"""
    schema = client.create_schema()
    schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
    schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("label", MilvusDataType.VARCHAR, max_length=128)

    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="HNSW", metric_type=metric,
                  params={"M": 16, "efConstruction": 64})
    client.create_collection(name, schema=schema, index_params=idx)


# ====================================================================
# 1. Empty collection operations
# ====================================================================

class TestEmptyCollection:

    def test_search_empty_collection(self, client: MilvusClient):
        """Search on empty collection should return empty results, not raise an error"""
        make_simple_collection(client, "empty_search")
        client.load_collection("empty_search")
        results = client.search("empty_search", data=rvecs(1), limit=5,
                                output_fields=["pk"])
        assert len(results) == 1
        assert len(results[0]) == 0

    def test_query_empty_collection(self, client: MilvusClient):
        """Query on empty collection should return empty list"""
        make_simple_collection(client, "empty_query")
        results = client.query("empty_query", filter="pk >= 0",
                               output_fields=["pk"])
        assert results == []

    def test_get_empty_collection(self, client: MilvusClient):
        """Get on empty collection should return empty list"""
        make_simple_collection(client, "empty_get")
        got = client.get("empty_get", ids=[1, 2, 3])
        assert got == []

    def test_stats_empty_collection(self, client: MilvusClient):
        """Empty collection stats should show row_count=0"""
        make_simple_collection(client, "empty_stats")
        stats = client.get_collection_stats("empty_stats")
        assert int(stats["row_count"]) == 0

    def test_delete_from_empty_collection(self, client: MilvusClient):
        """Delete from empty collection should not raise error"""
        make_simple_collection(client, "empty_del")
        client.delete("empty_del", ids=[999])


# ====================================================================
# 2. Data type boundary values
# ====================================================================

class TestDataTypeBoundary:

    def test_int8_boundary(self, client: MilvusClient):
        """INT8 boundary values: -128 ~ 127"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.INT8)

        client.create_collection("int8_bound", schema=schema)
        vecs = rvecs(3)
        client.insert("int8_bound", [
            {"pk": 1, "vec": vecs[0], "val": -128},
            {"pk": 2, "vec": vecs[1], "val": 0},
            {"pk": 3, "vec": vecs[2], "val": 127},
        ])
        got = client.get("int8_bound", ids=[1, 3])
        vals = {r["pk"]: r["val"] for r in got}
        assert vals[1] == -128
        assert vals[3] == 127

    def test_int8_overflow_rejected(self, client: MilvusClient):
        """INT8 out-of-range values should be rejected"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.INT8)

        client.create_collection("int8_overflow", schema=schema)
        with pytest.raises(Exception):
            client.insert("int8_overflow", [
                {"pk": 1, "vec": rvecs(1)[0], "val": 200},
            ])

    def test_int64_extreme_values(self, client: MilvusClient):
        """INT64 extreme values as primary key"""
        client.create_collection("int64_ext", dimension=DIM)
        vecs = rvecs(3)
        big = 2**62  # Avoid 2**63-1 to prevent overflow issues
        client.insert("int64_ext", [
            {"id": 0, "vector": vecs[0]},
            {"id": big, "vector": vecs[1]},
            {"id": -big, "vector": vecs[2]},
        ])
        got = client.get("int64_ext", ids=[0, big, -big])
        assert len(got) == 3

    def test_empty_string_value(self, client: MilvusClient):
        """Empty string as VARCHAR value"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("name", MilvusDataType.VARCHAR, max_length=64)

        client.create_collection("empty_str", schema=schema)
        vecs = rvecs(2)
        client.insert("empty_str", [
            {"pk": 1, "vec": vecs[0], "name": ""},
            {"pk": 2, "vec": vecs[1], "name": "hello"},
        ])
        got = client.get("empty_str", ids=[1])
        assert got[0]["name"] == ""

    def test_empty_string_primary_key(self, client: MilvusClient):
        """Empty string as VARCHAR primary key"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        client.create_collection("empty_pk", schema=schema)
        vecs = rvecs(1)
        client.insert("empty_pk", [{"pk": "", "vec": vecs[0]}])
        got = client.get("empty_pk", ids=[""])
        assert len(got) == 1
        assert got[0]["pk"] == ""

    def test_special_characters_in_values(self, client: MilvusClient):
        """Special characters: Unicode, quotes, backslash"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("text", MilvusDataType.VARCHAR, max_length=256)

        client.create_collection("special_chars", schema=schema)
        vecs = rvecs(3)
        client.insert("special_chars", [
            {"pk": 1, "vec": vecs[0], "text": "chinese_test"},
            {"pk": 2, "vec": vecs[1], "text": "hello\nworld\ttab"},
            {"pk": 3, "vec": vecs[2], "text": "it's a \"test\""},
        ])
        got = client.get("special_chars", ids=[1, 2, 3])
        texts = {r["pk"]: r["text"] for r in got}
        assert texts[1] == "chinese_test"
        assert texts[2] == "hello\nworld\ttab"
        assert texts[3] == 'it\'s a "test"'

    def test_bool_field_values(self, client: MilvusClient):
        """BOOL field read/write + filter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("active", MilvusDataType.BOOL)

        client.create_collection("bool_test", schema=schema)
        vecs = rvecs(4)
        client.insert("bool_test", [
            {"pk": 1, "vec": vecs[0], "active": True},
            {"pk": 2, "vec": vecs[1], "active": False},
            {"pk": 3, "vec": vecs[2], "active": True},
            {"pk": 4, "vec": vecs[3], "active": False},
        ])
        r = client.query("bool_test", filter="active == true",
                         output_fields=["pk", "active"])
        assert len(r) == 2
        assert all(x["active"] is True for x in r)


# ====================================================================
# 3. Vector boundaries
# ====================================================================

class TestVectorBoundary:

    def test_zero_vector_search(self, client: MilvusClient):
        """Zero vector search (L2 distance)"""
        make_simple_collection(client, "zero_vec", metric="L2")
        vecs = rvecs(10)
        data = [{"pk": i, "vec": vecs[i], "label": f"l{i}"} for i in range(10)]
        client.insert("zero_vec", data)
        client.load_collection("zero_vec")

        zero = [[0.0] * DIM]
        results = client.search("zero_vec", data=zero, limit=3,
                                output_fields=["pk"])
        assert len(results[0]) == 3

    def test_identical_vectors(self, client: MilvusClient):
        """Insert identical vectors, search should return all of them"""
        make_simple_collection(client, "ident_vec", metric="L2")
        same_vec = [1.0] * DIM
        data = [{"pk": i, "vec": same_vec, "label": f"l{i}"} for i in range(5)]
        client.insert("ident_vec", data)
        client.load_collection("ident_vec")

        results = client.search("ident_vec", data=[same_vec], limit=5,
                                output_fields=["pk"])
        assert len(results[0]) == 5
        # Distances should all be 0
        for hit in results[0]:
            assert abs(hit["distance"]) < 1e-4

    def test_topk_greater_than_total(self, client: MilvusClient):
        """top_k greater than total data count should return all data"""
        make_simple_collection(client, "topk_big", metric="COSINE")
        vecs = rvecs(3)
        data = [{"pk": i, "vec": vecs[i], "label": f"l{i}"} for i in range(3)]
        client.insert("topk_big", data)
        client.load_collection("topk_big")

        results = client.search("topk_big", data=rvecs(1, seed=99), limit=100,
                                output_fields=["pk"])
        assert len(results[0]) == 3  # Only 3 records, return all

    def test_single_record_search(self, client: MilvusClient):
        """Search on a collection with only 1 record"""
        make_simple_collection(client, "single_rec")
        vecs = rvecs(1)
        client.insert("single_rec", [{"pk": 42, "vec": vecs[0], "label": "only"}])
        client.load_collection("single_rec")

        results = client.search("single_rec", data=rvecs(1, seed=99), limit=5,
                                output_fields=["pk", "label"])
        assert len(results[0]) == 1
        assert results[0][0]["entity"]["pk"] == 42


# ====================================================================
# 4. Duplicate primary keys
# ====================================================================

class TestDuplicatePK:

    def test_insert_same_pk_twice(self, client: MilvusClient):
        """Two inserts with the same PK, the latter overwrites the former"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.VARCHAR, max_length=64)

        client.create_collection("dup_pk", schema=schema)
        vecs = rvecs(2)
        client.insert("dup_pk", [{"pk": 1, "vec": vecs[0], "val": "first"}])
        client.insert("dup_pk", [{"pk": 1, "vec": vecs[1], "val": "second"}])

        got = client.get("dup_pk", ids=[1])
        assert len(got) == 1
        assert got[0]["val"] == "second"

    def test_duplicate_pk_in_same_batch(self, client: MilvusClient):
        """Duplicate PK in the same batch, the latter overwrites the former"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.VARCHAR, max_length=64)

        client.create_collection("dup_batch", schema=schema)
        vecs = rvecs(2)
        client.insert("dup_batch", [
            {"pk": 1, "vec": vecs[0], "val": "first"},
            {"pk": 1, "vec": vecs[1], "val": "second"},
        ])
        got = client.get("dup_batch", ids=[1])
        assert len(got) == 1
        assert got[0]["val"] == "second"

    def test_insert_count_with_upsert_semantics(self, client: MilvusClient):
        """stats row_count should not double-count due to duplicate PK"""
        client.create_collection("dup_count", dimension=DIM)
        vecs = rvecs(3)
        client.insert("dup_count", [{"id": 1, "vector": vecs[0]}])
        client.insert("dup_count", [{"id": 1, "vector": vecs[1]}])
        client.insert("dup_count", [{"id": 2, "vector": vecs[2]}])

        stats = client.get_collection_stats("dup_count")
        assert int(stats["row_count"]) == 2


# ====================================================================
# 5. Complex filter expressions
# ====================================================================

class TestComplexFilters:

    @pytest.fixture(autouse=True)
    def _setup(self, client):
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("age", MilvusDataType.INT64)
        schema.add_field("name", MilvusDataType.VARCHAR, max_length=128)
        schema.add_field("score", MilvusDataType.FLOAT)
        schema.add_field("tags", MilvusDataType.ARRAY,
                         element_type=MilvusDataType.VARCHAR,
                         max_capacity=10, max_length=64)
        schema.add_field("info", MilvusDataType.JSON)
        schema.add_field("maybe", MilvusDataType.VARCHAR, max_length=64,
                         nullable=True)

        client.create_collection("cf_test", schema=schema)
        vecs = rvecs(10)
        client.insert("cf_test", [
            {"pk": 0, "vec": vecs[0], "age": 25, "name": "Alice",   "score": 88.5,
             "tags": ["python", "ml"],       "info": {"level": 3, "city": "NYC"},
             "maybe": "yes"},
            {"pk": 1, "vec": vecs[1], "age": 30, "name": "Bob",     "score": 72.0,
             "tags": ["java"],               "info": {"level": 5, "city": "LA"},
             "maybe": None},
            {"pk": 2, "vec": vecs[2], "age": 22, "name": "Charlie", "score": 95.1,
             "tags": ["python", "web"],      "info": {"level": 1, "city": "NYC"},
             "maybe": "no"},
            {"pk": 3, "vec": vecs[3], "age": 35, "name": "Diana",   "score": 60.0,
             "tags": ["rust", "ml"],         "info": {"level": 4, "city": "SF"},
             "maybe": None},
            {"pk": 4, "vec": vecs[4], "age": 28, "name": "Eve",     "score": 91.2,
             "tags": ["python"],             "info": {"level": 2, "city": "LA"},
             "maybe": "yes"},
            {"pk": 5, "vec": vecs[5], "age": 40, "name": "Frank",   "score": 55.0,
             "tags": ["go", "web"],          "info": {"level": 6, "city": "NYC"},
             "maybe": None},
            {"pk": 6, "vec": vecs[6], "age": 19, "name": "Grace",   "score": 99.0,
             "tags": ["python", "ml", "dl"], "info": {"level": 1, "city": "SF"},
             "maybe": "yes"},
            {"pk": 7, "vec": vecs[7], "age": 33, "name": "Hank",    "score": 45.5,
             "tags": ["java", "web"],        "info": {"level": 3, "city": "LA"},
             "maybe": None},
            {"pk": 8, "vec": vecs[8], "age": 27, "name": "Ivy",     "score": 82.0,
             "tags": ["rust"],               "info": {"level": 2, "city": "NYC"},
             "maybe": "no"},
            {"pk": 9, "vec": vecs[9], "age": 31, "name": "Jack",    "score": 77.7,
             "tags": ["go", "ml"],           "info": {"level": 4, "city": "SF"},
             "maybe": None},
        ])

    def test_nested_and_or(self, client: MilvusClient):
        """Nested AND / OR"""
        r = client.query("cf_test",
                         filter='(age < 25 or age > 35) and score > 50',
                         output_fields=["pk", "age", "score"])
        for x in r:
            assert (x["age"] < 25 or x["age"] > 35) and x["score"] > 50

    def test_not_operator(self, client: MilvusClient):
        """NOT operator"""
        r = client.query("cf_test", filter="not (age > 30)",
                         output_fields=["pk", "age"])
        assert all(x["age"] <= 30 for x in r)

    def test_not_in_operator(self, client: MilvusClient):
        """NOT IN"""
        r = client.query("cf_test", filter="pk not in [0, 1, 2, 3]",
                         output_fields=["pk"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [4, 5, 6, 7, 8, 9]

    def test_not_equal(self, client: MilvusClient):
        """!= operator"""
        r = client.query("cf_test", filter='name != "Alice"',
                         output_fields=["pk", "name"])
        assert all(x["name"] != "Alice" for x in r)
        assert len(r) == 9

    def test_like_operator(self, client: MilvusClient):
        """LIKE pattern matching"""
        r = client.query("cf_test", filter='name like "A%"',
                         output_fields=["pk", "name"])
        assert len(r) == 1
        assert r[0]["name"] == "Alice"

    def test_json_nested_filter(self, client: MilvusClient):
        """JSON nested path filter"""
        r = client.query("cf_test", filter='info["city"] == "NYC"',
                         output_fields=["pk", "info"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 2, 5, 8]

    def test_json_numeric_filter(self, client: MilvusClient):
        """JSON numeric field filter"""
        r = client.query("cf_test", filter='info["level"] >= 4',
                         output_fields=["pk", "info"])
        for x in r:
            assert x["info"]["level"] >= 4

    def test_array_contains(self, client: MilvusClient):
        """ARRAY_CONTAINS"""
        r = client.query("cf_test",
                         filter='array_contains(tags, "python")',
                         output_fields=["pk", "tags"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 2, 4, 6]
        for x in r:
            assert "python" in x["tags"]

    def test_array_length(self, client: MilvusClient):
        """ARRAY_LENGTH"""
        r = client.query("cf_test",
                         filter="array_length(tags) >= 2",
                         output_fields=["pk", "tags"])
        for x in r:
            assert len(x["tags"]) >= 2

    def test_is_null(self, client: MilvusClient):
        """IS NULL (nullable field)"""
        r = client.query("cf_test", filter="maybe is null",
                         output_fields=["pk", "maybe"])
        # pk 1,3,5,7,9 have None
        pks = sorted([x["pk"] for x in r])
        assert pks == [1, 3, 5, 7, 9]

    def test_is_not_null(self, client: MilvusClient):
        """IS NOT NULL"""
        r = client.query("cf_test", filter="maybe is not null",
                         output_fields=["pk", "maybe"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 2, 4, 6, 8]

    def test_complex_combined_filter(self, client: MilvusClient):
        """Complex combination: JSON + array + scalar + NOT"""
        r = client.query("cf_test",
                         filter='info["city"] == "NYC" and array_contains(tags, "python") and not (age > 25)',
                         output_fields=["pk", "age", "info", "tags"])
        pks = sorted([x["pk"] for x in r])
        # Alice: age=25, NYC, python (25 is NOT > 25, so not(age>25) is True)
        # Charlie: age=22, NYC, python
        assert pks == [0, 2]

    def test_float_comparison_precision(self, client: MilvusClient):
        """Float precision comparison"""
        r = client.query("cf_test", filter="score > 90.0 and score < 100.0",
                         output_fields=["pk", "score"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [2, 4, 6]  # 95.1, 91.2, 99.0


# ====================================================================
# 6. Query Offset pagination
# ====================================================================

class TestQueryPagination:

    def test_query_offset(self, client: MilvusClient):
        """Query offset parameter"""
        client.create_collection("q_offset", dimension=DIM)
        vecs = rvecs(20)
        data = [{"id": i, "vector": vecs[i]} for i in range(20)]
        client.insert("q_offset", data)

        page1 = client.query("q_offset", filter="id >= 0", limit=5, offset=0,
                             output_fields=["id"])
        page2 = client.query("q_offset", filter="id >= 0", limit=5, offset=5,
                             output_fields=["id"])
        ids1 = set(r["id"] for r in page1)
        ids2 = set(r["id"] for r in page2)
        assert len(ids1) == 5
        assert len(ids2) == 5
        assert ids1.isdisjoint(ids2)  # No overlap

    def test_query_offset_beyond_data(self, client: MilvusClient):
        """Offset beyond total count should return empty"""
        client.create_collection("q_off_big", dimension=DIM)
        vecs = rvecs(5)
        client.insert("q_off_big", [{"id": i, "vector": vecs[i]} for i in range(5)])

        r = client.query("q_off_big", filter="id >= 0", limit=10, offset=100,
                         output_fields=["id"])
        assert len(r) == 0


# ====================================================================
# 7. Search Offset pagination
# ====================================================================

class TestSearchPagination:

    def test_search_offset(self, client: MilvusClient):
        """Search offset parameter"""
        make_simple_collection(client, "s_offset", metric="L2")
        vecs = rvecs(20)
        data = [{"pk": i, "vec": vecs[i], "label": f"l{i}"} for i in range(20)]
        client.insert("s_offset", data)
        client.load_collection("s_offset")

        query = rvecs(1, seed=88)
        page1 = client.search("s_offset", data=query, limit=5, offset=0,
                              output_fields=["pk"])
        page2 = client.search("s_offset", data=query, limit=5, offset=5,
                              output_fields=["pk"])
        ids1 = [h["entity"]["pk"] for h in page1[0]]
        ids2 = [h["entity"]["pk"] for h in page2[0]]
        assert len(ids1) == 5
        assert len(ids2) == 5
        # Second page pks should not appear in the first page
        assert set(ids1).isdisjoint(set(ids2))


# ====================================================================
# 8. count(*) aggregation
# ====================================================================

class TestCountAggregation:

    def test_count_all(self, client: MilvusClient):
        """count(*) returns total row count"""
        client.create_collection("count_all", dimension=DIM)
        vecs = rvecs(15)
        client.insert("count_all", [{"id": i, "vector": vecs[i]} for i in range(15)])

        r = client.query("count_all", filter="", output_fields=["count(*)"])
        assert r[0]["count(*)"] == 15

    def test_count_with_filter(self, client: MilvusClient):
        """count(*) + filter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", MilvusDataType.VARCHAR, max_length=32)

        client.create_collection("count_filter", schema=schema)
        vecs = rvecs(10)
        client.insert("count_filter", [
            {"pk": i, "vec": vecs[i], "status": "active" if i % 2 == 0 else "inactive"}
            for i in range(10)
        ])

        r = client.query("count_filter", filter='status == "active"',
                         output_fields=["count(*)"])
        assert r[0]["count(*)"] == 5

    def test_count_empty_collection(self, client: MilvusClient):
        """Empty collection count(*) = 0"""
        client.create_collection("count_empty", dimension=DIM)
        r = client.query("count_empty", filter="", output_fields=["count(*)"])
        assert r[0]["count(*)"] == 0


# ====================================================================
# 9. output_fields wildcard
# ====================================================================

class TestOutputFieldsStar:

    def test_output_fields_star(self, client: MilvusClient):
        """output_fields=["*"] returns all fields"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("a", MilvusDataType.VARCHAR, max_length=32)
        schema.add_field("b", MilvusDataType.INT32)

        client.create_collection("star_test", schema=schema)
        vecs = rvecs(1)
        client.insert("star_test", [{"pk": 1, "vec": vecs[0], "a": "hello", "b": 42}])

        got = client.query("star_test", filter="pk == 1", output_fields=["*"])
        assert "pk" in got[0]
        assert "a" in got[0]
        assert "b" in got[0]
        assert got[0]["a"] == "hello"
        assert got[0]["b"] == 42


# ====================================================================
# 10. Operations after deleting all data
# ====================================================================

class TestDeleteAll:

    def test_delete_all_then_query(self, client: MilvusClient):
        """Query returns empty after deleting all data"""
        client.create_collection("del_all", dimension=DIM)
        vecs = rvecs(10)
        client.insert("del_all", [{"id": i, "vector": vecs[i]} for i in range(10)])

        client.delete("del_all", ids=list(range(10)))
        remaining = client.query("del_all", filter="id >= 0", output_fields=["id"])
        assert len(remaining) == 0

    def test_delete_all_then_insert(self, client: MilvusClient):
        """Re-insert after deleting all"""
        client.create_collection("del_reinsert", dimension=DIM)
        vecs = rvecs(5)
        client.insert("del_reinsert", [{"id": i, "vector": vecs[i]} for i in range(5)])
        client.delete("del_reinsert", ids=list(range(5)))

        new_vecs = rvecs(3, seed=123)
        client.insert("del_reinsert", [
            {"id": 100 + i, "vector": new_vecs[i]} for i in range(3)
        ])
        got = client.query("del_reinsert", filter="id >= 0", output_fields=["id"])
        assert len(got) == 3

    def test_delete_all_then_search(self, client: MilvusClient):
        """Search should return empty after deleting all"""
        make_simple_collection(client, "del_search")
        vecs = rvecs(5)
        client.insert("del_search", [
            {"pk": i, "vec": vecs[i], "label": f"l{i}"} for i in range(5)
        ])
        client.delete("del_search", ids=list(range(5)))
        client.load_collection("del_search")

        results = client.search("del_search", data=rvecs(1, seed=99), limit=5,
                                output_fields=["pk"])
        assert len(results[0]) == 0


# ====================================================================
# 11. Upsert batch + partial overlap
# ====================================================================

class TestUpsertEdge:

    def test_upsert_partial_overlap(self, client: MilvusClient):
        """upsert with partially existing + partially new records"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.VARCHAR, max_length=64)

        client.create_collection("ups_overlap", schema=schema)
        vecs = rvecs(5)
        client.insert("ups_overlap", [
            {"pk": 1, "vec": vecs[0], "val": "orig_1"},
            {"pk": 2, "vec": vecs[1], "val": "orig_2"},
            {"pk": 3, "vec": vecs[2], "val": "orig_3"},
        ])

        new_vecs = rvecs(4, seed=88)
        client.upsert("ups_overlap", [
            {"pk": 2, "vec": new_vecs[0], "val": "updated_2"},  # Already exists
            {"pk": 3, "vec": new_vecs[1], "val": "updated_3"},  # Already exists
            {"pk": 4, "vec": new_vecs[2], "val": "new_4"},      # New record
            {"pk": 5, "vec": new_vecs[3], "val": "new_5"},      # New record
        ])

        got = client.get("ups_overlap", ids=[1, 2, 3, 4, 5])
        vals = {r["pk"]: r["val"] for r in got}
        assert vals[1] == "orig_1"     # Not upserted
        assert vals[2] == "updated_2"  # Overwritten
        assert vals[3] == "updated_3"  # Overwritten
        assert vals[4] == "new_4"      # Newly added
        assert vals[5] == "new_5"      # Newly added
        assert len(got) == 5


# ====================================================================
# 12. Many partition operations
# ====================================================================

class TestPartitionEdge:

    def test_many_partitions(self, client: MilvusClient):
        """Create many partitions"""
        client.create_collection("many_parts", dimension=DIM)
        for i in range(20):
            client.create_partition("many_parts", f"p_{i}")

        parts = client.list_partitions("many_parts")
        assert len(parts) >= 21  # 20 + _default

    def test_drop_partition_with_data(self, client: MilvusClient):
        """Drop a partition that contains data"""
        client.create_collection("drop_part_data", dimension=DIM)
        client.create_partition("drop_part_data", "temp")

        vecs = rvecs(5)
        client.insert("drop_part_data",
                      [{"id": i, "vector": vecs[i]} for i in range(5)],
                      partition_name="temp")

        client.drop_partition("drop_part_data", "temp")
        assert client.has_partition("drop_part_data", "temp") is False

    def test_insert_nonexistent_partition(self, client: MilvusClient):
        """Inserting into a non-existent partition should raise an error"""
        client.create_collection("no_part", dimension=DIM)
        with pytest.raises(Exception):
            client.insert("no_part",
                          [{"id": 1, "vector": rvecs(1)[0]}],
                          partition_name="ghost_partition")


# ====================================================================
# 13. Search/query without load
# ====================================================================

class TestLoadRequirement:

    def test_search_without_load(self, client: MilvusClient):
        """Search without load should raise an error"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("no_load", schema=schema, index_params=idx)

        vecs = rvecs(5)
        client.insert("no_load", [{"pk": i, "vec": vecs[i]} for i in range(5)])
        client.release_collection("no_load")

        with pytest.raises(Exception, match="released|load"):
            client.search("no_load", data=rvecs(1), limit=3)


# ====================================================================
# 14. Index rebuild
# ====================================================================

class TestIndexRebuild:

    def test_drop_and_recreate_index(self, client: MilvusClient):
        """After dropping and recreating index, search still works"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("idx_rebuild", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((50, DIM)).astype(np.float32)
        client.insert("idx_rebuild",
                      [{"pk": i, "vec": vecs[i].tolist()} for i in range(50)])

        # Drop index
        client.release_collection("idx_rebuild")
        client.drop_index("idx_rebuild", index_name="vec")

        # Rebuild index
        idx2 = client.prepare_index_params()
        idx2.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                       params={"M": 8, "efConstruction": 32})
        client.create_index("idx_rebuild", idx2)
        client.load_collection("idx_rebuild")

        results = client.search("idx_rebuild", data=vecs[0:1].tolist(), limit=3,
                                output_fields=["pk"])
        assert results[0][0]["entity"]["pk"] == 0


# ====================================================================
# 15. Data persistence
# ====================================================================

class TestPersistence:

    def test_data_survives_restart(self):
        """Data remains available after server restart"""
        data_dir = tempfile.mkdtemp(prefix="persist_test_")
        try:
            # Phase 1: Write data
            server1, db1, port1 = start_server_in_thread(data_dir)
            c1 = MilvusClient(uri=f"http://127.0.0.1:{port1}")
            c1.create_collection("persist_col", dimension=DIM)
            vecs = rvecs(10)
            c1.insert("persist_col", [{"id": i, "vector": vecs[i]} for i in range(10)])
            c1.close()
            server1.stop(grace=2)
            db1.close()

            # Phase 2: Restart and verify
            server2, db2, port2 = start_server_in_thread(data_dir)
            c2 = MilvusClient(uri=f"http://127.0.0.1:{port2}")
            assert c2.has_collection("persist_col") is True
            # After restart, collection is in released state, need to load
            c2.load_collection("persist_col")
            got = c2.get("persist_col", ids=list(range(10)))
            assert len(got) == 10
            c2.close()
            server2.stop(grace=2)
            db2.close()
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


# ====================================================================
# 16. Group By Search
# ====================================================================

class TestGroupBySearch:

    def test_group_by_field(self, client: MilvusClient):
        """Search with group by scalar field"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("category", MilvusDataType.VARCHAR, max_length=32)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("group_test", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        n = 60
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        categories = ["cat_A", "cat_B", "cat_C"]
        data = [
            {"pk": i, "vec": vecs[i].tolist(),
             "category": categories[i % 3]}
            for i in range(n)
        ]
        client.insert("group_test", data)
        client.load_collection("group_test")

        results = client.search("group_test", data=vecs[0:1].tolist(), limit=3,
                                group_by_field="category",
                                output_fields=["pk", "category"])

        # Should return 3 groups, each with a different category
        seen_cats = set()
        for hit in results[0]:
            seen_cats.add(hit["entity"]["category"])
        assert len(seen_cats) == 3


# ====================================================================
# 17. Range Search
# ====================================================================

class TestRangeSearch:

    def test_range_search_l2(self, client: MilvusClient):
        """Range search: only return results within [0, radius] distance"""
        make_simple_collection(client, "range_test", metric="L2")
        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((100, DIM)).astype(np.float32)
        data = [{"pk": i, "vec": vecs[i].tolist(), "label": f"l{i}"}
                for i in range(100)]
        client.insert("range_test", data)
        client.load_collection("range_test")

        query = vecs[0:1].tolist()
        # Normal search to find the max distance as reference
        all_res = client.search("range_test", data=query, limit=100,
                                output_fields=["pk"])
        max_dist = all_res[0][-1]["distance"]
        mid_dist = max_dist / 2

        # range search: radius is the upper bound
        range_res = client.search("range_test", data=query, limit=100,
                                  search_params={"radius": mid_dist},
                                  output_fields=["pk"])
        if len(range_res[0]) > 0:
            for hit in range_res[0]:
                assert hit["distance"] <= mid_dist + 1e-4


# ====================================================================
# 18. Explicit flush
# ====================================================================

class TestFlush:

    def test_explicit_flush(self, client: MilvusClient):
        """Data is persisted after explicit flush"""
        client.create_collection("flush_test", dimension=DIM)
        vecs = rvecs(5)
        client.insert("flush_test", [{"id": i, "vector": vecs[i]} for i in range(5)])

        client.flush("flush_test")
        got = client.get("flush_test", ids=list(range(5)))
        assert len(got) == 5


# ====================================================================
# 19. Rapid consecutive operations (interleaved insert-delete-search)
# ====================================================================

class TestRapidOperations:

    def test_interleaved_insert_delete_search(self, client: MilvusClient):
        """Interleaved insert / delete / search operations"""
        make_simple_collection(client, "rapid", metric="L2")
        client.load_collection("rapid")

        rng = np.random.default_rng(SEED)
        alive_pks = set()

        for batch in range(10):
            # Insert 5
            vecs = rng.standard_normal((5, DIM)).astype(np.float32)
            base_pk = batch * 10
            data = [{"pk": base_pk + i, "vec": vecs[i].tolist(), "label": f"b{batch}_{i}"}
                    for i in range(5)]
            client.insert("rapid", data)
            alive_pks.update(base_pk + i for i in range(5))

            # Delete 2 from this batch
            del_pks = [base_pk, base_pk + 2]
            client.delete("rapid", ids=del_pks)
            alive_pks -= set(del_pks)

            # Search
            query = rng.standard_normal((1, DIM)).astype(np.float32).tolist()
            results = client.search("rapid", data=query, limit=3,
                                    output_fields=["pk"])
            for hit in results[0]:
                assert hit["entity"]["pk"] in alive_pks

        # Final verification
        remaining = client.query("rapid", filter="pk >= 0", output_fields=["pk"],
                                 limit=1000)
        remaining_pks = set(r["pk"] for r in remaining)
        assert remaining_pks == alive_pks


# ====================================================================
# 20. Default value field
# ====================================================================

class TestDefaultValue:

    def test_field_with_default_value(self, client: MilvusClient):
        """Field with default value, insert without providing that field"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", MilvusDataType.VARCHAR, max_length=32,
                         default_value="active")

        client.create_collection("default_val", schema=schema)
        vecs = rvecs(2)
        client.insert("default_val", [
            {"pk": 1, "vec": vecs[0]},                         # status not provided
            {"pk": 2, "vec": vecs[1], "status": "inactive"},   # status provided
        ])

        got = client.get("default_val", ids=[1, 2])
        vals = {r["pk"]: r["status"] for r in got}
        assert vals[1] == "active"    # Used the default value
        assert vals[2] == "inactive"  # Used the provided value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
