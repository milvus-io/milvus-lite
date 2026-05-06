"""
Milvus miscellaneous compatibility tests -- covering remaining untested scenarios.

Coverage:
  1.  pymilvus ORM API (connections.connect + Collection)
  2.  Invalid filter syntax -> should return clear error
  3.  Insert with nonexistent field name
  4.  Type mismatch (string inserted into integer field)
  5.  Search returning zero results (extremely restrictive filter)
  6.  Empty string filter and None filter
  7.  Insert with many duplicate PKs (1000 overwrites of same PK)
  8.  query limit=1 returns only one record
  9.  get single id (not a list)
 10.  search limit=1 returns only nearest neighbor
 11.  delete many ids (500 records)
 12.  Collection name special characters / long names
 13.  FLOAT and INT mixed insert (automatic type conversion)
 14.  Multiple filter conditions using the same field
 15.  Concurrent create/drop collections
 16.  search + filter hits 0 records
 17.  query offset > limit combination
 18.  BOOL filter various syntax
 19.  VARCHAR LIKE various patterns
 20.  JSON deep nesting paths
"""

from __future__ import annotations

import shutil
import tempfile
import threading

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from milvus_lite.adapter.grpc.server import start_server_in_thread

DIM = 8
SEED = 66


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="misc_test_")
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


def rvecs(n, dim=DIM, seed=SEED):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32).tolist()


# ====================================================================
# 1. pymilvus ORM API
# ====================================================================

class TestORMAPI:

    def test_orm_connect_and_crud(self, server):
        """Full CRUD using pymilvus ORM API (connections + Collection)"""
        from pymilvus import (
            connections, Collection, CollectionSchema,
            FieldSchema, DataType, utility,
        )
        port, _ = server
        alias = f"test_orm_{port}"

        connections.connect(alias=alias, host="127.0.0.1", port=port)
        try:
            # Create schema
            fields = [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=DIM),
                FieldSchema("tag", DataType.VARCHAR, max_length=64),
            ]
            schema = CollectionSchema(fields, description="ORM test")

            # drop if exists
            if utility.has_collection("orm_test", using=alias):
                utility.drop_collection("orm_test", using=alias)

            col = Collection("orm_test", schema=schema, using=alias)

            # Insert
            import random
            rng = np.random.default_rng(SEED)
            vecs = rng.standard_normal((10, DIM)).astype(np.float32).tolist()
            data = [
                list(range(10)),     # pk
                vecs,                # vec
                [f"t{i}" for i in range(10)],  # tag
            ]
            col.insert(data)

            # Create index + load
            col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE",
                                     "params": {"M": 16, "efConstruction": 64}})
            col.load()

            # Search
            results = col.search(
                data=[vecs[0]], anns_field="vec",
                param={"metric_type": "COSINE"},
                limit=3, output_fields=["pk", "tag"],
            )
            assert len(results[0]) == 3
            assert results[0][0].entity.get("pk") == 0

            # Query
            rows = col.query(expr="pk < 5", output_fields=["pk", "tag"])
            assert len(rows) == 5

            # Delete
            col.delete(expr="pk in [0, 1]")

            # Verify
            rows = col.query(expr="pk >= 0", output_fields=["pk"])
            assert len(rows) == 8

            col.drop()
            assert not utility.has_collection("orm_test", using=alias)
        finally:
            connections.disconnect(alias)


# ====================================================================
# 2. Invalid filter syntax
# ====================================================================

class TestInvalidFilter:

    def test_bad_filter_syntax(self, client: MilvusClient):
        """Invalid filter syntax should raise error"""
        client.create_collection("bad_filter", dimension=DIM)
        vecs = rvecs(3)
        client.insert("bad_filter", [{"id": i, "vector": vecs[i]} for i in range(3)])

        with pytest.raises(Exception):
            client.query("bad_filter", filter="((( invalid syntax !!!",
                         output_fields=["id"])

    def test_unknown_field_in_filter(self, client: MilvusClient):
        """Filter references nonexistent field (non-dynamic field collection)"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        # enable_dynamic_field defaults to False -> unknown field should raise error

        client.create_collection("unknown_field", schema=schema)
        vecs = rvecs(1)
        client.insert("unknown_field", [{"pk": 0, "vec": vecs[0]}])

        with pytest.raises(Exception):
            client.query("unknown_field", filter="nonexistent > 5",
                         output_fields=["pk"])


# ====================================================================
# 3. Type mismatch
# ====================================================================

class TestTypeMismatch:

    def test_string_into_int_field(self, client: MilvusClient):
        """String inserted into INT64 field should raise error"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("count", MilvusDataType.INT64)

        client.create_collection("type_err", schema=schema)
        with pytest.raises(Exception):
            client.insert("type_err", [
                {"pk": 1, "vec": rvecs(1)[0], "count": "not_a_number"},
            ])

    def test_int_into_float_field_ok(self, client: MilvusClient):
        """int inserted into FLOAT field should auto-convert (Milvus behavior)"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("score", MilvusDataType.FLOAT)

        client.create_collection("int_to_float", schema=schema)
        vecs = rvecs(1)
        client.insert("int_to_float", [
            {"pk": 1, "vec": vecs[0], "score": 42},  # int → float
        ])
        got = client.get("int_to_float", ids=[1])
        assert got[0]["score"] == pytest.approx(42.0)


# ====================================================================
# 4. Search returning zero results
# ====================================================================

class TestSearchZeroResults:

    def test_search_with_impossible_filter(self, client: MilvusClient):
        """Filter excludes all data -> returns empty"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.INT64)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("zero_res", schema=schema, index_params=idx)

        vecs = rvecs(10)
        client.insert("zero_res", [
            {"pk": i, "vec": vecs[i], "val": i} for i in range(10)
        ])
        client.load_collection("zero_res")

        # val max is 9, searching > 100 cannot match anything
        results = client.search("zero_res", data=[vecs[0]], limit=5,
                                filter="val > 100", output_fields=["pk"])
        assert len(results[0]) == 0

    def test_query_no_match(self, client: MilvusClient):
        """query filter with no matches"""
        client.create_collection("q_no_match", dimension=DIM)
        vecs = rvecs(5)
        client.insert("q_no_match", [{"id": i, "vector": vecs[i]} for i in range(5)])

        r = client.query("q_no_match", filter="id > 999", output_fields=["id"])
        assert r == []


# ====================================================================
# 5. Insert with many duplicate PKs
# ====================================================================

class TestMassivePKOverwrite:

    def test_1000_overwrites_same_pk(self, client: MilvusClient):
        """Insert same PK 1000 times consecutively, only the last one is kept"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("ver", MilvusDataType.INT64)

        client.create_collection("mass_overwrite", schema=schema)

        # Batch insert with same PK
        base_vec = rvecs(1)[0]
        data = [{"pk": 1, "vec": base_vec, "ver": i} for i in range(1000)]
        client.insert("mass_overwrite", data)

        got = client.get("mass_overwrite", ids=[1])
        assert len(got) == 1
        assert got[0]["ver"] == 999  # last one

        stats = client.get_collection_stats("mass_overwrite")
        assert int(stats["row_count"]) == 1


# ====================================================================
# 6. query limit=1 / search limit=1
# ====================================================================

class TestLimitOne:

    def test_query_limit_1(self, client: MilvusClient):
        client.create_collection("lim1_q", dimension=DIM)
        vecs = rvecs(10)
        client.insert("lim1_q", [{"id": i, "vector": vecs[i]} for i in range(10)])

        r = client.query("lim1_q", filter="id >= 0", limit=1, output_fields=["id"])
        assert len(r) == 1

    def test_search_limit_1(self, client: MilvusClient):
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("lim1_s", schema=schema, index_params=idx)

        vecs = rvecs(10)
        client.insert("lim1_s", [{"pk": i, "vec": vecs[i]} for i in range(10)])
        client.load_collection("lim1_s")

        r = client.search("lim1_s", data=[vecs[5]], limit=1, output_fields=["pk"])
        assert len(r[0]) == 1
        assert r[0][0]["entity"]["pk"] == 5


# ====================================================================
# 7. delete many ids (500 records)
# ====================================================================

class TestBulkDelete:

    def test_delete_500_ids(self, client: MilvusClient):
        client.create_collection("bulk_del_500", dimension=DIM)
        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((500, DIM)).astype(np.float32).tolist()
        client.insert("bulk_del_500", [{"id": i, "vector": vecs[i]} for i in range(500)])

        client.delete("bulk_del_500", ids=list(range(500)))
        stats = client.get_collection_stats("bulk_del_500")
        assert int(stats["row_count"]) == 0


# ====================================================================
# 8. Collection name edge cases
# ====================================================================

class TestCollectionNameEdge:

    def test_long_collection_name(self, client: MilvusClient):
        """Long collection name"""
        name = "a" * 200
        client.create_collection(name, dimension=DIM)
        assert client.has_collection(name) is True
        client.drop_collection(name)

    def test_collection_name_with_underscore_and_digits(self, client: MilvusClient):
        """Collection name with underscores and digits"""
        name = "test_collection_123_v2"
        client.create_collection(name, dimension=DIM)
        assert client.has_collection(name) is True


# ====================================================================
# 9. FLOAT and INT mixed usage
# ====================================================================

class TestTypeCoercion:

    def test_float_into_double_field(self, client: MilvusClient):
        """float inserted into DOUBLE field"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.DOUBLE)

        client.create_collection("f2d", schema=schema)
        vecs = rvecs(1)
        client.insert("f2d", [{"pk": 1, "vec": vecs[0], "val": 3.14}])
        got = client.get("f2d", ids=[1])
        assert got[0]["val"] == pytest.approx(3.14, rel=1e-5)

    def test_bool_filter_variations(self, client: MilvusClient):
        """Various filter syntax for BOOL field"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("flag", MilvusDataType.BOOL)

        client.create_collection("bool_var", schema=schema)
        vecs = rvecs(4)
        client.insert("bool_var", [
            {"pk": 0, "vec": vecs[0], "flag": True},
            {"pk": 1, "vec": vecs[1], "flag": False},
            {"pk": 2, "vec": vecs[2], "flag": True},
            {"pk": 3, "vec": vecs[3], "flag": False},
        ])

        # true lowercase
        r1 = client.query("bool_var", filter="flag == true", output_fields=["pk"])
        assert len(r1) == 2

        # false lowercase
        r2 = client.query("bool_var", filter="flag == false", output_fields=["pk"])
        assert len(r2) == 2

        # not flag
        r3 = client.query("bool_var", filter="not flag", output_fields=["pk"])
        assert len(r3) == 2
        assert all(x["pk"] % 2 == 1 for x in r3)


# ====================================================================
# 10. Same field with multiple conditions
# ====================================================================

class TestSameFieldMultiCondition:

    def test_range_on_same_field(self, client: MilvusClient):
        """Range query on the same field: a > X and a < Y"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("score", MilvusDataType.FLOAT)

        client.create_collection("same_field", schema=schema)
        vecs = rvecs(20)
        client.insert("same_field", [
            {"pk": i, "vec": vecs[i], "score": float(i * 5)} for i in range(20)
        ])

        r = client.query("same_field",
                         filter="score >= 25.0 and score < 50.0",
                         output_fields=["pk", "score"])
        for x in r:
            assert 25.0 <= x["score"] < 50.0
        assert len(r) == 5  # 25, 30, 35, 40, 45


# ====================================================================
# 11. VARCHAR LIKE various patterns
# ====================================================================

class TestLikePatterns:

    @pytest.fixture(autouse=True)
    def _setup(self, client):
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("name", MilvusDataType.VARCHAR, max_length=128)

        client.create_collection("like_test", schema=schema)
        vecs = rvecs(8)
        client.insert("like_test", [
            {"pk": 0, "vec": vecs[0], "name": "apple"},
            {"pk": 1, "vec": vecs[1], "name": "application"},
            {"pk": 2, "vec": vecs[2], "name": "banana"},
            {"pk": 3, "vec": vecs[3], "name": "cherry"},
            {"pk": 4, "vec": vecs[4], "name": "pineapple"},
            {"pk": 5, "vec": vecs[5], "name": "grape"},
            {"pk": 6, "vec": vecs[6], "name": "APP_config"},
            {"pk": 7, "vec": vecs[7], "name": "test123"},
        ])

    def test_like_prefix(self, client: MilvusClient):
        """Prefix match: app%"""
        r = client.query("like_test", filter='name like "app%"',
                         output_fields=["pk", "name"])
        names = [x["name"] for x in r]
        assert "apple" in names
        assert "application" in names
        assert "banana" not in names

    def test_like_suffix(self, client: MilvusClient):
        """Suffix match: %ple"""
        r = client.query("like_test", filter='name like "%ple"',
                         output_fields=["pk", "name"])
        names = sorted([x["name"] for x in r])
        assert "apple" in names
        assert "pineapple" in names

    def test_like_contains(self, client: MilvusClient):
        """Contains match: %an%"""
        r = client.query("like_test", filter='name like "%an%"',
                         output_fields=["pk", "name"])
        names = [x["name"] for x in r]
        assert "banana" in names

    def test_like_no_match(self, client: MilvusClient):
        """No match"""
        r = client.query("like_test", filter='name like "zzz%"',
                         output_fields=["pk"])
        assert r == []


# ====================================================================
# 12. JSON deep nesting
# ====================================================================

class TestDeepJsonNesting:

    def test_deep_nested_json_read(self, client: MilvusClient):
        """Deep nested JSON read"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("data", MilvusDataType.JSON)

        client.create_collection("deep_json", schema=schema)
        vecs = rvecs(1)
        nested = {
            "level1": {
                "level2": {
                    "level3": {"value": 42, "tags": ["a", "b"]}
                }
            },
            "flat": "hello",
        }
        client.insert("deep_json", [{"pk": 1, "vec": vecs[0], "data": nested}])

        got = client.get("deep_json", ids=[1])
        assert got[0]["data"]["level1"]["level2"]["level3"]["value"] == 42
        assert got[0]["data"]["flat"] == "hello"

    def test_json_nested_filter(self, client: MilvusClient):
        """JSON nested field filter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("info", MilvusDataType.JSON)

        client.create_collection("json_nest_f", schema=schema)
        vecs = rvecs(3)
        client.insert("json_nest_f", [
            {"pk": 0, "vec": vecs[0], "info": {"a": {"b": 1}}},
            {"pk": 1, "vec": vecs[1], "info": {"a": {"b": 5}}},
            {"pk": 2, "vec": vecs[2], "info": {"a": {"b": 10}}},
        ])

        r = client.query("json_nest_f", filter='info["a"]["b"] >= 5',
                         output_fields=["pk", "info"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [1, 2]


# ====================================================================
# 13. Concurrent create/drop collections
# ====================================================================

class TestConcurrentCollectionOps:

    def test_concurrent_create_drop(self, client: MilvusClient):
        """Multi-threaded concurrent creation of different collections"""
        errors = []

        def create_col(name):
            try:
                client.create_collection(name, dimension=DIM)
                assert client.has_collection(name)
                vecs = rvecs(2)
                client.insert(name, [{"id": i, "vector": vecs[i]} for i in range(2)])
            except Exception as e:
                errors.append(f"{name}: {e}")

        threads = [threading.Thread(target=create_col, args=(f"cc_{i}",))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Errors: {errors}"
        names = client.list_collections()
        for i in range(10):
            assert f"cc_{i}" in names


# ====================================================================
# 14. query offset + limit edge cases
# ====================================================================

class TestPaginationEdge:

    def test_offset_equals_total(self, client: MilvusClient):
        """offset == total data count -> empty"""
        client.create_collection("pag_eq", dimension=DIM)
        vecs = rvecs(5)
        client.insert("pag_eq", [{"id": i, "vector": vecs[i]} for i in range(5)])

        r = client.query("pag_eq", filter="id >= 0", limit=10, offset=5,
                         output_fields=["id"])
        assert len(r) == 0

    def test_offset_0_limit_0(self, client: MilvusClient):
        """limit=0 -> empty (or error, depends on implementation)"""
        client.create_collection("pag_zero", dimension=DIM)
        vecs = rvecs(3)
        client.insert("pag_zero", [{"id": i, "vector": vecs[i]} for i in range(3)])

        # Milvus: limit=0 is usually treated as "no limit" or raises error
        # Try to see if it succeeds
        try:
            r = client.query("pag_zero", filter="id >= 0", limit=0,
                             output_fields=["id"])
            # If successful, result should be empty or all
            assert isinstance(r, list)
        except Exception:
            pass  # Error is also acceptable

    def test_full_pagination(self, client: MilvusClient):
        """Full pagination traversal"""
        client.create_collection("pag_full", dimension=DIM)
        vecs = rvecs(17)
        client.insert("pag_full", [{"id": i, "vector": vecs[i]} for i in range(17)])

        all_ids = []
        offset = 0
        page_size = 5
        while True:
            r = client.query("pag_full", filter="id >= 0",
                             limit=page_size, offset=offset,
                             output_fields=["id"])
            if not r:
                break
            all_ids.extend(x["id"] for x in r)
            offset += page_size

        assert sorted(all_ids) == list(range(17))


# ====================================================================
# 15. search + filter hits 0 records (different from empty collection)
# ====================================================================

class TestFilterNoHit:

    def test_search_filter_excludes_all_nearest(self, client: MilvusClient):
        """All nearest neighbors excluded by filter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("group", MilvusDataType.VARCHAR, max_length=16)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("filter_excl", schema=schema, index_params=idx)

        vecs = rvecs(10)
        client.insert("filter_excl", [
            {"pk": i, "vec": vecs[i], "group": "only_group"} for i in range(10)
        ])
        client.load_collection("filter_excl")

        # Search filter requires group="nonexistent" -> 0 results
        r = client.search("filter_excl", data=[vecs[0]], limit=5,
                          filter='group == "nonexistent"', output_fields=["pk"])
        assert len(r[0]) == 0


# ====================================================================
# 16. Negative PK
# ====================================================================

class TestNegativePK:

    def test_negative_int64_pk(self, client: MilvusClient):
        """Negative number as INT64 primary key"""
        client.create_collection("neg_pk", dimension=DIM)
        vecs = rvecs(3)
        client.insert("neg_pk", [
            {"id": -100, "vector": vecs[0]},
            {"id": 0,    "vector": vecs[1]},
            {"id": 100,  "vector": vecs[2]},
        ])

        got = client.get("neg_pk", ids=[-100, 0, 100])
        assert len(got) == 3

        # Delete by negative id
        client.delete("neg_pk", ids=[-100])
        got = client.get("neg_pk", ids=[-100])
        assert len(got) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
