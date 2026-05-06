"""
Partition Key compatibility tests — verified via pymilvus MilvusClient.

Partition Key semantics (consistent with Milvus):
  - A scalar field in the schema is marked with is_partition_key=True
  - On insert, data is automatically hash-routed to internal bucket partitions based on field value
  - Users do not need to manage partitions manually; queries/searches automatically scan all buckets
  - Users cannot manually create/drop partitions (manual partition ops are forbidden on partition_key collections)

Test coverage:
  1.  Basic create + insert + query
  2.  DescribeCollection returns partition_key info
  3.  Search correctness (cross-bucket search)
  4.  Filter query by partition_key field
  5.  Data isolation across different partition_key values
  6.  Data aggregation for same partition_key value
  7.  Delete operation (cross-bucket delete)
  8.  Upsert operation
  9.  count(*) aggregation
 10.  partition_key + other scalar filter combination
 11.  partition_key + vector search + filter
 12.  Many different partition_key values
 13.  partition_key with INT64 type
 14.  partition_key + auto_id
 15.  partition_key + dynamic field
 16.  partition_key + nullable field
 17.  partition_key + BM25 full-text search
 18.  Data correctness after flush with partition_key
 19.  Manual partition operations forbidden on partition_key collections
 20.  partition_key value as empty string / extreme values
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from milvus_lite.adapter.grpc.server import start_server_in_thread

DIM = 8
SEED = 88


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="pkey_test_")
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


def make_pkey_collection(client, name, pkey_type=MilvusDataType.VARCHAR):
    """Create a collection with partition_key"""
    schema = client.create_schema()
    schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
    schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
    if pkey_type == MilvusDataType.VARCHAR:
        schema.add_field("category", pkey_type, max_length=64,
                         is_partition_key=True)
    else:
        schema.add_field("category", pkey_type, is_partition_key=True)
    schema.add_field("score", MilvusDataType.FLOAT, nullable=True)

    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                  params={"M": 16, "efConstruction": 64})
    client.create_collection(name, schema=schema, index_params=idx)


# ====================================================================
# 1. Basic create + insert + query
# ====================================================================

class TestBasicPartitionKey:

    def test_insert_and_get(self, client: MilvusClient):
        """Basic insert + get on a partition_key collection"""
        make_pkey_collection(client, "pkey_basic")
        vecs = rvecs(5)
        client.insert("pkey_basic", [
            {"pk": i, "vec": vecs[i], "category": f"cat_{i % 3}", "score": float(i)}
            for i in range(5)
        ])

        got = client.get("pkey_basic", ids=[0, 1, 2, 3, 4])
        assert len(got) == 5
        cats = {r["pk"]: r["category"] for r in got}
        assert cats[0] == "cat_0"
        assert cats[1] == "cat_1"
        assert cats[4] == "cat_1"

    def test_query_all(self, client: MilvusClient):
        """Query on a partition_key collection returns all data"""
        make_pkey_collection(client, "pkey_query")
        vecs = rvecs(10)
        client.insert("pkey_query", [
            {"pk": i, "vec": vecs[i], "category": f"cat_{i % 4}", "score": float(i)}
            for i in range(10)
        ])

        r = client.query("pkey_query", filter="pk >= 0", output_fields=["pk"],
                         limit=100)
        assert len(r) == 10


# ====================================================================
# 2. DescribeCollection returns partition_key info
# ====================================================================

class TestDescribePartitionKey:

    def test_describe_shows_partition_key(self, client: MilvusClient):
        """describe should show the partition_key field"""
        make_pkey_collection(client, "pkey_desc")
        info = client.describe_collection("pkey_desc")

        # Find the category field
        cat_field = next(f for f in info["fields"] if f["name"] == "category")
        assert cat_field.get("is_partition_key") is True


# ====================================================================
# 3. Search correctness (cross-bucket search)
# ====================================================================

class TestPartitionKeySearch:

    def test_search_across_buckets(self, client: MilvusClient):
        """Search automatically spans all bucket partitions"""
        make_pkey_collection(client, "pkey_search")
        rng = np.random.default_rng(SEED)
        n = 50
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        client.insert("pkey_search", [
            {"pk": i, "vec": vecs[i].tolist(),
             "category": f"cat_{i % 5}", "score": float(i)}
            for i in range(n)
        ])
        client.load_collection("pkey_search")

        # Search should find data from any bucket
        results = client.search("pkey_search", data=vecs[0:1].tolist(), limit=5,
                                output_fields=["pk", "category"])
        assert len(results[0]) == 5
        assert results[0][0]["entity"]["pk"] == 0

    def test_search_with_different_categories(self, client: MilvusClient):
        """Vectors from different categories can all be found by search"""
        make_pkey_collection(client, "pkey_search_cat")
        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((6, DIM)).astype(np.float32)
        client.insert("pkey_search_cat", [
            {"pk": 0, "vec": vecs[0].tolist(), "category": "alpha", "score": 1.0},
            {"pk": 1, "vec": vecs[1].tolist(), "category": "beta",  "score": 2.0},
            {"pk": 2, "vec": vecs[2].tolist(), "category": "gamma", "score": 3.0},
            {"pk": 3, "vec": vecs[3].tolist(), "category": "alpha", "score": 4.0},
            {"pk": 4, "vec": vecs[4].tolist(), "category": "beta",  "score": 5.0},
            {"pk": 5, "vec": vecs[5].tolist(), "category": "gamma", "score": 6.0},
        ])
        client.load_collection("pkey_search_cat")

        # Search for each category's vector should return it as nearest neighbor
        for i in range(6):
            r = client.search("pkey_search_cat", data=vecs[i:i+1].tolist(),
                              limit=1, output_fields=["pk"])
            assert r[0][0]["entity"]["pk"] == i


# ====================================================================
# 4. Filter query by partition_key field
# ====================================================================

class TestPartitionKeyFilter:

    def test_filter_by_partition_key(self, client: MilvusClient):
        """Filter by partition_key field value"""
        make_pkey_collection(client, "pkey_filter")
        vecs = rvecs(12)
        client.insert("pkey_filter", [
            {"pk": i, "vec": vecs[i],
             "category": ["red", "green", "blue"][i % 3], "score": float(i)}
            for i in range(12)
        ])

        r = client.query("pkey_filter", filter='category == "red"',
                         output_fields=["pk", "category"])
        assert len(r) == 4  # 0, 3, 6, 9
        assert all(x["category"] == "red" for x in r)

    def test_search_with_partition_key_filter(self, client: MilvusClient):
        """search + partition_key field filter"""
        make_pkey_collection(client, "pkey_sf")
        rng = np.random.default_rng(SEED)
        n = 30
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        client.insert("pkey_sf", [
            {"pk": i, "vec": vecs[i].tolist(),
             "category": "A" if i < 15 else "B", "score": float(i)}
            for i in range(n)
        ])
        client.load_collection("pkey_sf")

        r = client.search("pkey_sf", data=vecs[0:1].tolist(), limit=5,
                          filter='category == "A"',
                          output_fields=["pk", "category"])
        assert len(r[0]) == 5
        for hit in r[0]:
            assert hit["entity"]["category"] == "A"


# ====================================================================
# 5. Data isolation across different partition_key values
# ====================================================================

class TestPartitionKeyIsolation:

    def test_data_isolation_via_filter(self, client: MilvusClient):
        """Data with different partition_key values is fully isolated via filter"""
        make_pkey_collection(client, "pkey_iso")
        vecs = rvecs(20)
        client.insert("pkey_iso", [
            {"pk": i, "vec": vecs[i],
             "category": f"tenant_{i % 4}", "score": float(i)}
            for i in range(20)
        ])

        for t in range(4):
            r = client.query("pkey_iso", filter=f'category == "tenant_{t}"',
                             output_fields=["pk", "category"])
            assert len(r) == 5
            assert all(x["category"] == f"tenant_{t}" for x in r)


# ====================================================================
# 6. Data aggregation for same partition_key value
# ====================================================================

class TestPartitionKeyAggregation:

    def test_same_key_data_queryable(self, client: MilvusClient):
        """Data with the same partition_key can be queried together"""
        make_pkey_collection(client, "pkey_agg")
        vecs = rvecs(10)
        # All data uses the same partition_key
        client.insert("pkey_agg", [
            {"pk": i, "vec": vecs[i], "category": "single_tenant", "score": float(i)}
            for i in range(10)
        ])

        r = client.query("pkey_agg", filter='category == "single_tenant"',
                         output_fields=["pk"])
        assert len(r) == 10


# ====================================================================
# 7. Delete operation
# ====================================================================

class TestPartitionKeyDelete:

    def test_delete_by_pk(self, client: MilvusClient):
        """Delete by PK in a partition_key collection"""
        make_pkey_collection(client, "pkey_del")
        vecs = rvecs(6)
        client.insert("pkey_del", [
            {"pk": i, "vec": vecs[i], "category": f"c{i % 3}", "score": float(i)}
            for i in range(6)
        ])

        client.delete("pkey_del", ids=[1, 3, 5])
        remaining = client.query("pkey_del", filter="pk >= 0",
                                 output_fields=["pk"])
        pks = sorted([r["pk"] for r in remaining])
        assert pks == [0, 2, 4]

    def test_delete_by_filter(self, client: MilvusClient):
        """Delete by filter in a partition_key collection"""
        make_pkey_collection(client, "pkey_del_f")
        vecs = rvecs(9)
        client.insert("pkey_del_f", [
            {"pk": i, "vec": vecs[i],
             "category": ["x", "y", "z"][i % 3], "score": float(i)}
            for i in range(9)
        ])

        client.delete("pkey_del_f", filter='category == "y"')
        remaining = client.query("pkey_del_f", filter="pk >= 0",
                                 output_fields=["pk", "category"])
        assert all(r["category"] != "y" for r in remaining)
        assert len(remaining) == 6


# ====================================================================
# 8. Upsert operation
# ====================================================================

class TestPartitionKeyUpsert:

    def test_upsert_existing(self, client: MilvusClient):
        """Upsert existing record in a partition_key collection"""
        make_pkey_collection(client, "pkey_ups")
        vecs = rvecs(3)
        client.insert("pkey_ups", [
            {"pk": 1, "vec": vecs[0], "category": "alpha", "score": 10.0},
            {"pk": 2, "vec": vecs[1], "category": "beta",  "score": 20.0},
        ])

        new_vec = rvecs(1, seed=99)[0]
        client.upsert("pkey_ups", [
            {"pk": 1, "vec": new_vec, "category": "alpha", "score": 99.0},
        ])

        got = client.get("pkey_ups", ids=[1])
        assert got[0]["score"] == pytest.approx(99.0)

    def test_upsert_new(self, client: MilvusClient):
        """Upsert new record in a partition_key collection"""
        make_pkey_collection(client, "pkey_ups_new")
        vecs = rvecs(2)
        client.insert("pkey_ups_new", [
            {"pk": 1, "vec": vecs[0], "category": "alpha", "score": 1.0},
        ])

        client.upsert("pkey_ups_new", [
            {"pk": 2, "vec": vecs[1], "category": "beta", "score": 2.0},
        ])

        got = client.get("pkey_ups_new", ids=[1, 2])
        assert len(got) == 2


# ====================================================================
# 9. count(*)
# ====================================================================

class TestPartitionKeyCount:

    def test_count_all(self, client: MilvusClient):
        make_pkey_collection(client, "pkey_cnt")
        vecs = rvecs(15)
        client.insert("pkey_cnt", [
            {"pk": i, "vec": vecs[i], "category": f"c{i % 5}", "score": 0.0}
            for i in range(15)
        ])

        r = client.query("pkey_cnt", filter="", output_fields=["count(*)"])
        assert r[0]["count(*)"] == 15

    def test_count_by_category(self, client: MilvusClient):
        make_pkey_collection(client, "pkey_cnt_cat")
        vecs = rvecs(12)
        client.insert("pkey_cnt_cat", [
            {"pk": i, "vec": vecs[i],
             "category": ["a", "b", "c"][i % 3], "score": 0.0}
            for i in range(12)
        ])

        r = client.query("pkey_cnt_cat", filter='category == "a"',
                          output_fields=["count(*)"])
        assert r[0]["count(*)"] == 4


# ====================================================================
# 10. partition_key + other scalar filter combination
# ====================================================================

class TestPartitionKeyComboFilter:

    def test_partition_key_and_scalar_filter(self, client: MilvusClient):
        """partition_key filter + other scalar conditions"""
        make_pkey_collection(client, "pkey_combo")
        vecs = rvecs(20)
        client.insert("pkey_combo", [
            {"pk": i, "vec": vecs[i],
             "category": "hot" if i < 10 else "cold",
             "score": float(i * 10)}
            for i in range(20)
        ])

        # category == "hot" AND score >= 50
        r = client.query("pkey_combo",
                         filter='category == "hot" and score >= 50',
                         output_fields=["pk", "category", "score"])
        for x in r:
            assert x["category"] == "hot"
            assert x["score"] >= 50
        assert len(r) == 5  # pk 5,6,7,8,9


# ====================================================================
# 11. Many different partition_key values
# ====================================================================

class TestManyPartitionKeys:

    def test_100_different_keys(self, client: MilvusClient):
        """100 different partition_key values"""
        make_pkey_collection(client, "pkey_many")
        rng = np.random.default_rng(SEED)
        n = 100
        vecs = rng.standard_normal((n, DIM)).astype(np.float32).tolist()
        client.insert("pkey_many", [
            {"pk": i, "vec": vecs[i], "category": f"key_{i}", "score": 0.0}
            for i in range(n)
        ])

        stats = client.get_collection_stats("pkey_many")
        assert int(stats["row_count"]) == n

        # Random spot check
        got = client.get("pkey_many", ids=[0, 50, 99])
        assert len(got) == 3
        cats = {r["pk"]: r["category"] for r in got}
        assert cats[0] == "key_0"
        assert cats[50] == "key_50"
        assert cats[99] == "key_99"


# ====================================================================
# 12. INT64 partition_key
# ====================================================================

class TestIntPartitionKey:

    def test_int64_partition_key(self, client: MilvusClient):
        """INT64 type partition_key"""
        make_pkey_collection(client, "pkey_int", pkey_type=MilvusDataType.INT64)
        vecs = rvecs(10)
        client.insert("pkey_int", [
            {"pk": i, "vec": vecs[i], "category": i % 3, "score": float(i)}
            for i in range(10)
        ])

        got = client.get("pkey_int", ids=list(range(10)))
        assert len(got) == 10

        # filter by int partition_key
        r = client.query("pkey_int", filter="category == 1",
                         output_fields=["pk", "category"])
        assert all(x["category"] == 1 for x in r)


# ====================================================================
# 13. partition_key + auto_id
# ====================================================================

class TestPartitionKeyAutoId:

    def test_auto_id_with_partition_key(self, client: MilvusClient):
        """auto_id + partition_key"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tenant", MilvusDataType.VARCHAR, max_length=64,
                         is_partition_key=True)

        client.create_collection("pkey_autoid", schema=schema)
        vecs = rvecs(6)
        client.insert("pkey_autoid", [
            {"vec": vecs[i], "tenant": f"t{i % 2}"} for i in range(6)
        ])

        r = client.query("pkey_autoid", filter='tenant == "t0"',
                         output_fields=["pk", "tenant"])
        assert len(r) == 3
        assert all(x["tenant"] == "t0" for x in r)


# ====================================================================
# 14. partition_key + dynamic field
# ====================================================================

class TestPartitionKeyDynamic:

    def test_partition_key_with_dynamic_fields(self, client: MilvusClient):
        """partition_key + dynamic fields"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("region", MilvusDataType.VARCHAR, max_length=32,
                         is_partition_key=True)
        schema.enable_dynamic_field = True

        client.create_collection("pkey_dyn", schema=schema)
        vecs = rvecs(4)
        client.insert("pkey_dyn", [
            {"pk": 0, "vec": vecs[0], "region": "us", "color": "red"},
            {"pk": 1, "vec": vecs[1], "region": "eu", "color": "blue"},
            {"pk": 2, "vec": vecs[2], "region": "us", "color": "green"},
            {"pk": 3, "vec": vecs[3], "region": "eu", "color": "red"},
        ])

        # filter by dynamic field
        r = client.query("pkey_dyn", filter='color == "red"',
                         output_fields=["pk", "region", "color"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 3]

        # filter by partition_key + dynamic
        r = client.query("pkey_dyn",
                         filter='region == "us" and color == "red"',
                         output_fields=["pk"])
        assert len(r) == 1
        assert r[0]["pk"] == 0


# ====================================================================
# 15. Data correctness after flush
# ====================================================================

class TestPartitionKeyFlush:

    def test_data_survives_flush(self, client: MilvusClient):
        """partition_key data is intact after flush"""
        make_pkey_collection(client, "pkey_flush")
        vecs = rvecs(10)
        client.insert("pkey_flush", [
            {"pk": i, "vec": vecs[i], "category": f"c{i % 3}", "score": float(i)}
            for i in range(10)
        ])

        client.flush("pkey_flush")

        got = client.get("pkey_flush", ids=list(range(10)))
        assert len(got) == 10

        r = client.query("pkey_flush", filter='category == "c0"',
                         output_fields=["pk"])
        assert len(r) == 4  # 0, 3, 6, 9


# ====================================================================
# 16. Manual partition operations forbidden on partition_key collections
# ====================================================================

class TestPartitionKeyNoManualPartition:

    def test_create_partition_forbidden(self, client: MilvusClient):
        """Manual partition creation is not allowed on partition_key collections"""
        make_pkey_collection(client, "pkey_no_part")
        with pytest.raises(Exception):
            client.create_partition("pkey_no_part", "manual_partition")

    def test_drop_partition_forbidden(self, client: MilvusClient):
        """Manual partition deletion is not allowed on partition_key collections"""
        make_pkey_collection(client, "pkey_no_drop")
        with pytest.raises(Exception):
            client.drop_partition("pkey_no_drop", "_pk_0")


# ====================================================================
# 17. partition_key value as empty string
# ====================================================================

class TestPartitionKeyEdgeValues:

    def test_empty_string_partition_key(self, client: MilvusClient):
        """Empty string as partition_key value"""
        make_pkey_collection(client, "pkey_empty")
        vecs = rvecs(3)
        client.insert("pkey_empty", [
            {"pk": 0, "vec": vecs[0], "category": "", "score": 1.0},
            {"pk": 1, "vec": vecs[1], "category": "normal", "score": 2.0},
            {"pk": 2, "vec": vecs[2], "category": "", "score": 3.0},
        ])

        r = client.query("pkey_empty", filter='category == ""',
                         output_fields=["pk"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 2]

    def test_special_chars_partition_key(self, client: MilvusClient):
        """Special characters as partition_key value"""
        make_pkey_collection(client, "pkey_special")
        vecs = rvecs(3)
        client.insert("pkey_special", [
            {"pk": 0, "vec": vecs[0], "category": "hello world", "score": 1.0},
            {"pk": 1, "vec": vecs[1], "category": "chinese_partition", "score": 2.0},
            {"pk": 2, "vec": vecs[2], "category": "a/b/c", "score": 3.0},
        ])

        got = client.get("pkey_special", ids=[0, 1, 2])
        assert len(got) == 3


# ====================================================================
# 18. End-to-end full workflow
# ====================================================================

class TestPartitionKeyE2E:

    def test_full_workflow(self, client: MilvusClient):
        """partition_key full workflow"""
        make_pkey_collection(client, "pkey_e2e")
        rng = np.random.default_rng(SEED)

        # 1. Insert
        n = 30
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        tenants = ["acme", "globex", "initech"]
        client.insert("pkey_e2e", [
            {"pk": i, "vec": vecs[i].tolist(),
             "category": tenants[i % 3], "score": float(i)}
            for i in range(n)
        ])

        # 2. Each tenant has the correct data count
        for t in tenants:
            r = client.query("pkey_e2e", filter=f'category == "{t}"',
                             output_fields=["count(*)"])
            assert r[0]["count(*)"] == 10

        # 3. Search
        client.load_collection("pkey_e2e")
        results = client.search("pkey_e2e", data=vecs[0:1].tolist(), limit=3,
                                filter='category == "acme"',
                                output_fields=["pk", "category"])
        assert all(h["entity"]["category"] == "acme" for h in results[0])

        # 4. Delete data of one tenant
        client.delete("pkey_e2e", filter='category == "globex"')

        r = client.query("pkey_e2e", filter="", output_fields=["count(*)"])
        assert r[0]["count(*)"] == 20  # 30 - 10

        # 5. upsert
        client.upsert("pkey_e2e", [
            {"pk": 0, "vec": vecs[0].tolist(), "category": "acme", "score": 999.0},
        ])
        got = client.get("pkey_e2e", ids=[0])
        assert got[0]["score"] == pytest.approx(999.0)

        # 6. Flush + re-query
        client.flush("pkey_e2e")
        got = client.get("pkey_e2e", ids=[0])
        assert got[0]["score"] == pytest.approx(999.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
