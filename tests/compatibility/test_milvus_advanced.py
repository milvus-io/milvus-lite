"""
Milvus advanced scenario compatibility tests -- covering previously untested dimensions.

Coverage:
  1. MilvusClient("./local.db") drop-in mode
  2. HNSW search_params (ef)
  3. IVF_FLAT search_params (nprobe)
  4. High-dimensional vectors (768d, simulating real embeddings)
  5. Multi-collection parallel operations
  6. Schema with many fields (20+ fields)
  7. get with output_fields for selective field return
  8. search with output_fields excluding certain fields
  9. Multi-threaded concurrent insert + search
 10. insert empty list
 11. Large JSON / large VARCHAR values
 12. query with OR joining multiple IN expressions
 13. Same collection drop + recreate with different schema
 14. search returning vector fields
 15. Hybrid Search + WeightedRanker
 16. search consistency before and after flush
 17. Cross-partition search
 18. query without filter (return all)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from milvus_lite.adapter.grpc.server import start_server_in_thread

DIM = 16
SEED = 55


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="advanced_test_")
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
# 1. MilvusClient("./local.db") drop-in mode
# ====================================================================

class TestLocalDBMode:

    def test_local_db_uri(self):
        """MilvusClient('./xxx.db') automatically starts a local server"""
        db_path = tempfile.mktemp(suffix=".db", prefix="lite_dropin_")
        try:
            c = MilvusClient(db_path)
            c.create_collection("dropin_test", dimension=DIM)
            assert c.has_collection("dropin_test") is True

            vecs = rvecs(5)
            c.insert("dropin_test", [{"id": i, "vector": vecs[i]} for i in range(5)])
            got = c.get("dropin_test", ids=[0, 1, 2])
            assert len(got) == 3

            # Search also works
            results = c.search("dropin_test", data=[vecs[0]], limit=3,
                               output_fields=["id"])
            assert results[0][0]["entity"]["id"] == 0
            c.close()
        finally:
            shutil.rmtree(db_path, ignore_errors=True)

    def test_local_db_persistence(self):
        """local.db mode data persistence"""
        db_path = tempfile.mktemp(suffix=".db", prefix="lite_persist_")
        try:
            # Write phase
            c1 = MilvusClient(db_path)
            c1.create_collection("persist", dimension=DIM)
            vecs = rvecs(3)
            c1.insert("persist", [{"id": i, "vector": vecs[i]} for i in range(3)])
            c1.close()

            # Read phase
            c2 = MilvusClient(db_path)
            assert c2.has_collection("persist") is True
            c2.load_collection("persist")
            got = c2.get("persist", ids=[0, 1, 2])
            assert len(got) == 3
            c2.close()
        finally:
            shutil.rmtree(db_path, ignore_errors=True)


# ====================================================================
# 2. HNSW search_params (ef)
# ====================================================================

class TestHNSWSearchParams:

    def test_search_with_ef_param(self, client: MilvusClient):
        """HNSW search with specified ef parameter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("hnsw_ef", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((100, DIM)).astype(np.float32)
        client.insert("hnsw_ef", [{"pk": i, "vec": vecs[i].tolist()} for i in range(100)])
        client.load_collection("hnsw_ef")

        # ef=10 (small) vs ef=200 (large) -- both should succeed
        r_small = client.search("hnsw_ef", data=vecs[0:1].tolist(), limit=5,
                                search_params={"ef": 10}, output_fields=["pk"])
        assert len(r_small[0]) == 5
        assert r_small[0][0]["entity"]["pk"] == 0

        r_large = client.search("hnsw_ef", data=vecs[0:1].tolist(), limit=5,
                                search_params={"ef": 200}, output_fields=["pk"])
        assert len(r_large[0]) == 5
        assert r_large[0][0]["entity"]["pk"] == 0


# ====================================================================
# 3. IVF_FLAT search_params (nprobe)
# ====================================================================

class TestIVFSearchParams:

    def test_search_with_nprobe(self, client: MilvusClient):
        """IVF_FLAT search with specified nprobe"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        client.create_collection("ivf_nprobe", schema=schema)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((200, DIM)).astype(np.float32)
        client.insert("ivf_nprobe", [{"pk": i, "vec": vecs[i].tolist()} for i in range(200)])

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="IVF_FLAT", metric_type="L2",
                      params={"nlist": 8})
        client.create_index("ivf_nprobe", idx)
        client.load_collection("ivf_nprobe")

        r = client.search("ivf_nprobe", data=vecs[0:1].tolist(), limit=5,
                          search_params={"nprobe": 4}, output_fields=["pk"])
        assert len(r[0]) == 5
        assert r[0][0]["entity"]["pk"] == 0


# ====================================================================
# 4. High-dimensional vectors
# ====================================================================

class TestHighDimension:

    def test_768d_vectors(self, client: MilvusClient):
        """768-dimensional vectors (common sentence-transformer dimension)"""
        dim = 768
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=dim)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("high_dim", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        n = 100
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        client.insert("high_dim", [{"pk": i, "vec": vecs[i].tolist()} for i in range(n)])
        client.load_collection("high_dim")

        results = client.search("high_dim", data=vecs[0:1].tolist(), limit=5,
                                output_fields=["pk"])
        assert results[0][0]["entity"]["pk"] == 0


# ====================================================================
# 5. Multi-collection parallel operations
# ====================================================================

class TestMultiCollection:

    def test_operations_across_collections(self, client: MilvusClient):
        """Operations across multiple collections do not interfere with each other"""
        for name in ["mc_a", "mc_b", "mc_c"]:
            client.create_collection(name, dimension=DIM)

        vecs_a = rvecs(5, seed=1)
        vecs_b = rvecs(5, seed=2)
        vecs_c = rvecs(5, seed=3)

        client.insert("mc_a", [{"id": i, "vector": vecs_a[i]} for i in range(5)])
        client.insert("mc_b", [{"id": i + 100, "vector": vecs_b[i]} for i in range(5)])
        client.insert("mc_c", [{"id": i + 200, "vector": vecs_c[i]} for i in range(5)])

        # Data isolation across collections
        got_a = client.get("mc_a", ids=[0, 1])
        got_b = client.get("mc_b", ids=[100, 101])
        got_c = client.get("mc_c", ids=[200, 201])
        assert len(got_a) == 2
        assert len(got_b) == 2
        assert len(got_c) == 2

        # Dropping mc_b does not affect others
        client.drop_collection("mc_b")
        assert client.has_collection("mc_a") is True
        assert client.has_collection("mc_b") is False
        assert client.has_collection("mc_c") is True
        assert len(client.get("mc_a", ids=[0])) == 1


# ====================================================================
# 6. Schema with many fields
# ====================================================================

class TestManyFields:

    def test_20_field_schema(self, client: MilvusClient):
        """Collection with 20 scalar fields"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        for i in range(18):
            schema.add_field(f"f{i}", MilvusDataType.VARCHAR, max_length=64, nullable=True)

        client.create_collection("many_fields", schema=schema)
        vecs = rvecs(3)
        records = []
        for r in range(3):
            rec = {"pk": r, "vec": vecs[r]}
            for i in range(18):
                rec[f"f{i}"] = f"val_{r}_{i}"
            records.append(rec)

        client.insert("many_fields", records)
        got = client.get("many_fields", ids=[1])
        assert got[0]["f0"] == "val_1_0"
        assert got[0]["f17"] == "val_1_17"


# ====================================================================
# 7. get/query with selective output_fields return
# ====================================================================

class TestSelectiveOutputFields:

    def test_get_with_output_fields(self, client: MilvusClient):
        """get returns only the specified fields"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("a", MilvusDataType.VARCHAR, max_length=32)
        schema.add_field("b", MilvusDataType.INT32)
        schema.add_field("c", MilvusDataType.FLOAT)

        client.create_collection("sel_out", schema=schema)
        vecs = rvecs(1)
        client.insert("sel_out", [{"pk": 1, "vec": vecs[0], "a": "hello", "b": 42, "c": 3.14}])

        # Only retrieve a and c
        got = client.get("sel_out", ids=[1], output_fields=["a", "c"])
        assert "a" in got[0]
        assert "c" in got[0]
        assert "b" not in got[0]
        assert "vec" not in got[0]

    def test_search_without_vector_in_output(self, client: MilvusClient):
        """search output_fields does not include vector"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("label", MilvusDataType.VARCHAR, max_length=32)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("sel_search", schema=schema, index_params=idx)

        vecs = rvecs(10)
        client.insert("sel_search", [
            {"pk": i, "vec": vecs[i], "label": f"l{i}"} for i in range(10)
        ])
        client.load_collection("sel_search")

        results = client.search("sel_search", data=[vecs[0]], limit=3,
                                output_fields=["label"])
        hit = results[0][0]["entity"]
        assert "label" in hit
        assert "vec" not in hit  # vector not requested


# ====================================================================
# 8. Multi-threaded concurrent insert + search
# ====================================================================

class TestConcurrency:

    def test_concurrent_insert_and_search(self, client: MilvusClient):
        """Multi-threaded concurrent insert and search"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("concurrent", schema=schema, index_params=idx)

        # Insert initial batch of data first
        rng = np.random.default_rng(SEED)
        init_vecs = rng.standard_normal((50, DIM)).astype(np.float32)
        client.insert("concurrent", [
            {"pk": i, "vec": init_vecs[i].tolist()} for i in range(50)
        ])
        client.load_collection("concurrent")

        errors = []

        def inserter():
            try:
                for batch in range(5):
                    base = 1000 + batch * 10
                    v = rng.standard_normal((10, DIM)).astype(np.float32)
                    client.insert("concurrent", [
                        {"pk": base + j, "vec": v[j].tolist()} for j in range(10)
                    ])
            except Exception as e:
                errors.append(f"insert: {e}")

        def searcher():
            try:
                for _ in range(10):
                    q = rng.standard_normal((1, DIM)).astype(np.float32).tolist()
                    r = client.search("concurrent", data=q, limit=5,
                                      output_fields=["pk"])
                    assert len(r[0]) > 0
            except Exception as e:
                errors.append(f"search: {e}")

        threads = [
            threading.Thread(target=inserter),
            threading.Thread(target=searcher),
            threading.Thread(target=searcher),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Concurrent errors: {errors}"


# ====================================================================
# 9. insert empty list
# ====================================================================

class TestEmptyInsert:

    def test_insert_empty_list(self, client: MilvusClient):
        """insert empty list should not raise error, insert_count=0"""
        client.create_collection("empty_ins", dimension=DIM)
        res = client.insert("empty_ins", [])
        assert res["insert_count"] == 0


# ====================================================================
# 10. Large JSON / large VARCHAR values
# ====================================================================

class TestLargeValues:

    def test_large_varchar(self, client: MilvusClient):
        """Long VARCHAR value"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("text", MilvusDataType.VARCHAR, max_length=65535)

        client.create_collection("large_str", schema=schema)
        big_text = "x" * 10000
        vecs = rvecs(1)
        client.insert("large_str", [{"pk": 1, "vec": vecs[0], "text": big_text}])

        got = client.get("large_str", ids=[1])
        assert len(got[0]["text"]) == 10000

    def test_large_json(self, client: MilvusClient):
        """Large JSON object"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("data", MilvusDataType.JSON)

        client.create_collection("large_json", schema=schema)
        big_json = {f"key_{i}": f"value_{i}" * 10 for i in range(100)}
        vecs = rvecs(1)
        client.insert("large_json", [{"pk": 1, "vec": vecs[0], "data": big_json}])

        got = client.get("large_json", ids=[1])
        assert got[0]["data"]["key_0"] == "value_0" * 10
        assert len(got[0]["data"]) == 100


# ====================================================================
# 11. Complex OR + IN combinations
# ====================================================================

class TestComplexOrIn:

    def test_or_with_multiple_in(self, client: MilvusClient):
        """OR joining multiple IN expressions"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("cat", MilvusDataType.VARCHAR, max_length=32)
        schema.add_field("tier", MilvusDataType.INT32)

        client.create_collection("or_in", schema=schema)
        vecs = rvecs(10)
        client.insert("or_in", [
            {"pk": i, "vec": vecs[i], "cat": ["a", "b", "c"][i % 3], "tier": i % 4}
            for i in range(10)
        ])

        r = client.query("or_in",
                         filter='cat in ["a"] or tier in [3]',
                         output_fields=["pk", "cat", "tier"])
        for x in r:
            assert x["cat"] == "a" or x["tier"] == 3


# ====================================================================
# 12. drop + recreate with different schema
# ====================================================================

class TestSchemaRecreate:

    def test_drop_and_recreate_different_schema(self, client: MilvusClient):
        """Drop same-named collection then recreate with different schema"""
        # v1: INT64 pk
        client.create_collection("evolve", dimension=DIM)
        vecs = rvecs(3)
        client.insert("evolve", [{"id": i, "vector": vecs[i]} for i in range(3)])
        client.drop_collection("evolve")

        # v2: VARCHAR pk + extra fields
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("label", MilvusDataType.VARCHAR, max_length=128)

        client.create_collection("evolve", schema=schema)
        vecs2 = rvecs(2, seed=99)
        client.insert("evolve", [
            {"pk": "doc_1", "vec": vecs2[0], "label": "hello"},
            {"pk": "doc_2", "vec": vecs2[1], "label": "world"},
        ])

        got = client.get("evolve", ids=["doc_1"])
        assert got[0]["label"] == "hello"

        # Old data should not exist
        got_old = client.get("evolve", ids=["0"])
        assert len(got_old) == 0


# ====================================================================
# 13. Hybrid Search + WeightedRanker
# ====================================================================

class TestHybridSearchWeighted:

    def test_weighted_ranker(self, client: MilvusClient):
        """Hybrid Search using WeightedRanker"""
        from pymilvus import AnnSearchRequest, WeightedRanker

        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("hybrid_w", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((30, DIM)).astype(np.float32)
        client.insert("hybrid_w", [{"pk": i, "vec": vecs[i].tolist()} for i in range(30)])
        client.load_collection("hybrid_w")

        q = vecs[0:1].tolist()
        req = AnnSearchRequest(data=q, anns_field="vec", param={}, limit=10)
        results = client.hybrid_search(
            "hybrid_w",
            reqs=[req, req],
            ranker=WeightedRanker(0.5, 0.5),
            limit=5,
            output_fields=["pk"],
        )
        assert len(results[0]) == 5
        assert results[0][0]["entity"]["pk"] == 0


# ====================================================================
# 14. search consistency before and after flush
# ====================================================================

class TestFlushSearchConsistency:

    def test_search_before_and_after_flush(self, client: MilvusClient):
        """search results are consistent before and after flush"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("flush_search", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((50, DIM)).astype(np.float32)
        client.insert("flush_search", [{"pk": i, "vec": vecs[i].tolist()} for i in range(50)])
        client.load_collection("flush_search")

        q = vecs[0:1].tolist()

        # Before flush
        r_before = client.search("flush_search", data=q, limit=10,
                                 output_fields=["pk"])
        pks_before = [h["entity"]["pk"] for h in r_before[0]]

        # Flush
        client.flush("flush_search")

        # After flush
        r_after = client.search("flush_search", data=q, limit=10,
                                output_fields=["pk"])
        pks_after = [h["entity"]["pk"] for h in r_after[0]]

        assert pks_before == pks_after


# ====================================================================
# 15. Cross-partition search
# ====================================================================

class TestCrossPartitionSearch:

    def test_search_across_partitions(self, client: MilvusClient):
        """Search across partitions"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("cross_part", schema=schema, index_params=idx)

        client.create_partition("cross_part", "p1")
        client.create_partition("cross_part", "p2")

        rng = np.random.default_rng(SEED)
        v1 = rng.standard_normal((10, DIM)).astype(np.float32)
        v2 = rng.standard_normal((10, DIM)).astype(np.float32)

        client.insert("cross_part",
                      [{"pk": i, "vec": v1[i].tolist()} for i in range(10)],
                      partition_name="p1")
        client.insert("cross_part",
                      [{"pk": 100 + i, "vec": v2[i].tolist()} for i in range(10)],
                      partition_name="p2")
        client.load_collection("cross_part")

        # Search across both partitions
        q = v1[0:1].tolist()
        r_both = client.search("cross_part", data=q, limit=5,
                               partition_names=["p1", "p2"],
                               output_fields=["pk"])
        all_pks = [h["entity"]["pk"] for h in r_both[0]]
        assert len(all_pks) == 5
        # Nearest neighbor should be pk=0 in p1
        assert r_both[0][0]["entity"]["pk"] == 0

        # Search only p2
        r_p2 = client.search("cross_part", data=q, limit=5,
                             partition_names=["p2"],
                             output_fields=["pk"])
        for h in r_p2[0]:
            assert h["entity"]["pk"] >= 100


# ====================================================================
# 16. query without filter (return all)
# ====================================================================

class TestQueryAll:

    def test_query_no_filter(self, client: MilvusClient):
        """query filter="" returns all (requires limit)"""
        client.create_collection("q_all", dimension=DIM)
        vecs = rvecs(8)
        client.insert("q_all", [{"id": i, "vector": vecs[i]} for i in range(8)])

        r = client.query("q_all", filter="", limit=100, output_fields=["id"])
        assert len(r) == 8


# ====================================================================
# 17. Search returning vector fields
# ====================================================================

class TestSearchReturnVector:

    def test_search_with_vector_output(self, client: MilvusClient):
        """search output_fields includes vector field"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("vec_output", schema=schema, index_params=idx)

        vecs = rvecs(5)
        client.insert("vec_output", [{"pk": i, "vec": vecs[i]} for i in range(5)])
        client.load_collection("vec_output")

        results = client.search("vec_output", data=[vecs[0]], limit=1,
                                output_fields=["pk", "vec"])
        hit = results[0][0]["entity"]
        assert "vec" in hit
        assert len(hit["vec"]) == DIM
        assert hit["vec"] == pytest.approx(vecs[0], rel=1e-5)


# ====================================================================
# 18. Bulk delete + query consistency
# ====================================================================

class TestBulkDeleteConsistency:

    def test_delete_half_then_verify(self, client: MilvusClient):
        """Verify remaining data integrity after deleting half the data"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.INT64)

        client.create_collection("bulk_del", schema=schema)
        vecs = rvecs(100, seed=1)
        client.insert("bulk_del", [
            {"pk": i, "vec": vecs[i], "val": i * 10} for i in range(100)
        ])

        # Delete even pks
        client.delete("bulk_del", ids=list(range(0, 100, 2)))

        # Verify odd pks still exist with correct data
        for pk in [1, 11, 51, 99]:
            got = client.get("bulk_del", ids=[pk])
            assert len(got) == 1
            assert got[0]["val"] == pk * 10

        # Verify even pks have been deleted
        for pk in [0, 10, 50, 98]:
            got = client.get("bulk_del", ids=[pk])
            assert len(got) == 0

        stats = client.get_collection_stats("bulk_del")
        assert int(stats["row_count"]) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
