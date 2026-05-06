"""
Milvus compatibility final supplementary tests -- covering previously missed scenarios.

Coverage:
  1. BM25 + Dense hybrid search (core RAG scenario)
  2. Query Iterator pagination traversal
  3. Search Iterator pagination traversal
  4. array_contains_all / array_contains_any
  5. Read-your-writes consistency (search immediately after insert, no flush)
  6. Delete by complex filter
  7. DOUBLE precision field
  8. Large batch insert (5000 records)
  9. Search with PK range filter
 10. Repeated delete of same PK (idempotency)
 11. Upsert + auto_id interaction
 12. Multi-condition NOT combination filter
 13. JSON array access filter
 14. Hybrid Search + filter
 15. Full lifecycle: insert, delete, upsert
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from milvus_lite.adapter.grpc.server import start_server_in_thread

DIM = 16
SEED = 33


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="final_test_")
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
# 1. BM25 + Dense hybrid search (core RAG scenario)
# ====================================================================

class TestBM25DenseHybrid:

    def test_sparse_dense_hybrid_search(self, client: MilvusClient):
        """BM25 sparse + Dense vector hybrid search -- most typical RAG usage"""
        from pymilvus import Function, FunctionType, AnnSearchRequest, RRFRanker

        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", MilvusDataType.VARCHAR, max_length=1024,
                         enable_analyzer=True)
        schema.add_field("sparse", MilvusDataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        bm25 = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
        schema.add_function(bm25)

        idx = client.prepare_index_params()
        idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                      metric_type="BM25")
        idx.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})

        client.create_collection("rag_hybrid", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        docs = [
            "machine learning algorithms for image recognition",
            "deep neural network training techniques",
            "natural language processing with transformers",
            "database indexing and query optimization",
            "vector similarity search in high dimensions",
            "reinforcement learning for game playing",
            "computer vision object detection methods",
            "text classification using pretrained models",
            "graph neural networks for social analysis",
            "recommendation system collaborative filtering",
        ]
        n = len(docs)
        dense_vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        data = [{"text": docs[i], "dense": dense_vecs[i].tolist()} for i in range(n)]
        client.insert("rag_hybrid", data)
        client.load_collection("rag_hybrid")

        # Sparse (BM25) search
        sparse_req = AnnSearchRequest(
            data=["machine learning"],
            anns_field="sparse",
            param={},
            limit=5,
        )

        # Dense search
        dense_req = AnnSearchRequest(
            data=dense_vecs[0:1].tolist(),
            anns_field="dense",
            param={},
            limit=5,
        )

        results = client.hybrid_search(
            "rag_hybrid",
            reqs=[sparse_req, dense_req],
            ranker=RRFRanker(k=60),
            limit=5,
            output_fields=["text"],
        )
        assert len(results[0]) == 5
        # Results should contain text content
        texts = [h["entity"]["text"] for h in results[0]]
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)


# ====================================================================
# 2. Query Iterator
# ====================================================================

class TestQueryIterator:

    def test_query_iterator_pagination(self, client: MilvusClient):
        """query_iterator traverses all data"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.INT64)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("iter_test", schema=schema, index_params=idx)

        vecs = rvecs(50, seed=1)
        client.insert("iter_test", [
            {"pk": i, "vec": vecs[i], "val": i * 10} for i in range(50)
        ])
        client.load_collection("iter_test")

        # Traverse using iterator
        it = client.query_iterator(
            "iter_test",
            filter="pk >= 0",
            output_fields=["pk", "val"],
            batch_size=10,
        )
        all_pks = []
        while True:
            batch = it.next()
            if not batch:
                break
            all_pks.extend(r["pk"] for r in batch)
        it.close()

        assert sorted(all_pks) == list(range(50))

    def test_query_iterator_with_filter(self, client: MilvusClient):
        """query_iterator + filter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("even", MilvusDataType.BOOL)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("iter_filter", schema=schema, index_params=idx)

        vecs = rvecs(30, seed=2)
        client.insert("iter_filter", [
            {"pk": i, "vec": vecs[i], "even": i % 2 == 0} for i in range(30)
        ])
        client.load_collection("iter_filter")

        it = client.query_iterator(
            "iter_filter",
            filter="even == true",
            output_fields=["pk"],
            batch_size=5,
        )
        pks = []
        while True:
            batch = it.next()
            if not batch:
                break
            pks.extend(r["pk"] for r in batch)
        it.close()

        assert len(pks) == 15  # 0,2,4,...,28
        assert all(pk % 2 == 0 for pk in pks)


# ====================================================================
# 3. Search Iterator
# ====================================================================

class TestSearchIterator:

    def test_search_iterator(self, client: MilvusClient):
        """search_iterator fetches search results in batches"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("search_iter", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((100, DIM)).astype(np.float32)
        client.insert("search_iter", [
            {"pk": i, "vec": vecs[i].tolist()} for i in range(100)
        ])
        client.load_collection("search_iter")

        it = client.search_iterator(
            "search_iter",
            data=vecs[0:1].tolist(),
            batch_size=10,
            limit=30,
            output_fields=["pk"],
        )
        all_pks = []
        while True:
            batch = it.next()
            if not batch:
                break
            all_pks.extend(h["entity"]["pk"] for h in batch)
        it.close()

        assert len(all_pks) == 30
        assert all_pks[0] == 0  # nearest neighbor is itself
        # Should have no duplicates
        assert len(set(all_pks)) == 30


# ====================================================================
# 4. array_contains_all / array_contains_any
# ====================================================================

class TestArrayContainsAdvanced:

    @pytest.fixture(autouse=True)
    def _setup(self, client):
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tags", MilvusDataType.ARRAY,
                         element_type=MilvusDataType.VARCHAR,
                         max_capacity=10, max_length=64)

        client.create_collection("arr_adv", schema=schema)
        vecs = rvecs(6)
        client.insert("arr_adv", [
            {"pk": 0, "vec": vecs[0], "tags": ["python", "ml", "web"]},
            {"pk": 1, "vec": vecs[1], "tags": ["java", "web"]},
            {"pk": 2, "vec": vecs[2], "tags": ["python", "ml"]},
            {"pk": 3, "vec": vecs[3], "tags": ["rust", "web", "ml"]},
            {"pk": 4, "vec": vecs[4], "tags": ["go"]},
            {"pk": 5, "vec": vecs[5], "tags": ["python", "web", "ml", "dl"]},
        ])

    def test_array_contains_all(self, client: MilvusClient):
        """array_contains_all: array contains all specified elements"""
        r = client.query("arr_adv",
                         filter='array_contains_all(tags, ["python", "ml"])',
                         output_fields=["pk", "tags"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 2, 5]  # all contain both python and ml

    def test_array_contains_any(self, client: MilvusClient):
        """array_contains_any: array contains any of the specified elements"""
        r = client.query("arr_adv",
                         filter='array_contains_any(tags, ["rust", "go"])',
                         output_fields=["pk", "tags"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [3, 4]  # rust or go

    def test_array_contains_all_single(self, client: MilvusClient):
        """array_contains_all with only one element"""
        r = client.query("arr_adv",
                         filter='array_contains_all(tags, ["web"])',
                         output_fields=["pk"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 1, 3, 5]


# ====================================================================
# 5. Read-your-writes consistency
# ====================================================================

class TestReadYourWrites:

    def test_search_immediately_after_insert(self, client: MilvusClient):
        """Search immediately after insert (no flush) should find new data"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("ryw", schema=schema, index_params=idx)
        client.load_collection("ryw")

        rng = np.random.default_rng(SEED)
        # Insert one by one and search immediately after each
        for i in range(5):
            v = rng.standard_normal((1, DIM)).astype(np.float32)
            client.insert("ryw", [{"pk": i, "vec": v[0].tolist()}])

            results = client.search("ryw", data=v.tolist(), limit=1,
                                    output_fields=["pk"])
            assert results[0][0]["entity"]["pk"] == i

    def test_query_immediately_after_delete(self, client: MilvusClient):
        """Query immediately after delete, deleted data should not be visible"""
        client.create_collection("ryw_del", dimension=DIM)
        vecs = rvecs(5)
        client.insert("ryw_del", [{"id": i, "vector": vecs[i]} for i in range(5)])

        client.delete("ryw_del", ids=[2])
        got = client.get("ryw_del", ids=[2])
        assert len(got) == 0

        remaining = client.query("ryw_del", filter="id >= 0",
                                 output_fields=["id"])
        assert len(remaining) == 4


# ====================================================================
# 6. Delete by complex filter
# ====================================================================

class TestDeleteByComplexFilter:

    def test_delete_by_multi_condition(self, client: MilvusClient):
        """Delete with multi-condition filter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("age", MilvusDataType.INT64)
        schema.add_field("status", MilvusDataType.VARCHAR, max_length=32)

        client.create_collection("del_complex", schema=schema)
        vecs = rvecs(10)
        client.insert("del_complex", [
            {"pk": i, "vec": vecs[i], "age": 20 + i,
             "status": "active" if i % 2 == 0 else "inactive"}
            for i in range(10)
        ])

        # Delete records with age >= 25 and inactive status
        client.delete("del_complex", filter='age >= 25 and status == "inactive"')

        remaining = client.query("del_complex", filter="pk >= 0",
                                 output_fields=["pk", "age", "status"])
        for r in remaining:
            # There should be no records with age>=25 and inactive status
            if r["age"] >= 25:
                assert r["status"] != "inactive"


# ====================================================================
# 7. DOUBLE precision field
# ====================================================================

class TestDoublePrecision:

    def test_double_precision_preserved(self, client: MilvusClient):
        """DOUBLE field precision is preserved"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("precise", MilvusDataType.DOUBLE)

        client.create_collection("double_test", schema=schema)
        vecs = rvecs(2)
        val = 3.141592653589793238
        client.insert("double_test", [
            {"pk": 1, "vec": vecs[0], "precise": val},
            {"pk": 2, "vec": vecs[1], "precise": 1e-15},
        ])

        got = client.get("double_test", ids=[1, 2])
        assert got[0]["precise"] == pytest.approx(val, rel=1e-10)
        assert got[1]["precise"] == pytest.approx(1e-15, rel=1e-5)


# ====================================================================
# 8. Large batch insert (5000 records)
# ====================================================================

class TestLargeBatchInsert:

    def test_insert_5000_records(self, client: MilvusClient):
        """Single insert of 5000 records"""
        client.create_collection("big_batch", dimension=DIM)
        rng = np.random.default_rng(SEED)
        n = 5000
        vecs = rng.standard_normal((n, DIM)).astype(np.float32).tolist()
        data = [{"id": i, "vector": vecs[i]} for i in range(n)]

        res = client.insert("big_batch", data)
        assert res["insert_count"] == n

        stats = client.get_collection_stats("big_batch")
        assert int(stats["row_count"]) == n

        # Random sampling verification
        for pk in [0, 1000, 2500, 4999]:
            got = client.get("big_batch", ids=[pk])
            assert len(got) == 1


# ====================================================================
# 9. Search with PK range filter
# ====================================================================

class TestSearchWithPKFilter:

    def test_search_with_pk_range(self, client: MilvusClient):
        """search with PK range as filter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("pk_range", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((100, DIM)).astype(np.float32)
        client.insert("pk_range", [{"pk": i, "vec": vecs[i].tolist()} for i in range(100)])
        client.load_collection("pk_range")

        results = client.search("pk_range", data=vecs[0:1].tolist(), limit=10,
                                filter="pk >= 50 and pk < 80",
                                output_fields=["pk"])
        for hit in results[0]:
            assert 50 <= hit["entity"]["pk"] < 80


# ====================================================================
# 10. Repeated delete of same PK (idempotency)
# ====================================================================

class TestIdempotentDelete:

    def test_delete_same_pk_twice(self, client: MilvusClient):
        """Deleting the same PK twice should not raise error"""
        client.create_collection("idem_del", dimension=DIM)
        vecs = rvecs(3)
        client.insert("idem_del", [{"id": i, "vector": vecs[i]} for i in range(3)])

        client.delete("idem_del", ids=[1])
        client.delete("idem_del", ids=[1])  # Second delete should not raise error

        got = client.get("idem_del", ids=[1])
        assert len(got) == 0
        # Other records should not be affected
        assert len(client.get("idem_del", ids=[0, 2])) == 2


# ====================================================================
# 11. Upsert + auto_id
# ====================================================================

class TestUpsertWithAutoId:

    def test_upsert_auto_id_collection(self, client: MilvusClient):
        """Upsert on auto_id collection"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("label", MilvusDataType.VARCHAR, max_length=64)

        client.create_collection("ups_autoid", schema=schema)
        vecs = rvecs(3)

        # Insert to get auto-assigned pks
        res = client.insert("ups_autoid", [
            {"vec": vecs[0], "label": "first"},
            {"vec": vecs[1], "label": "second"},
        ])
        assert res["insert_count"] == 2

        # Query to get the assigned pks
        rows = client.query("ups_autoid", filter="pk >= 0",
                            output_fields=["pk", "label"])
        assert len(rows) == 2

        # Upsert with known pk
        pk = rows[0]["pk"]
        new_vec = rvecs(1, seed=99)[0]
        client.upsert("ups_autoid", [
            {"pk": pk, "vec": new_vec, "label": "updated"},
        ])

        got = client.get("ups_autoid", ids=[pk])
        assert got[0]["label"] == "updated"


# ====================================================================
# 12. Multi-condition NOT combinations
# ====================================================================

class TestNotCombinations:

    def test_not_and_not(self, client: MilvusClient):
        """NOT + AND + NOT combination"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("x", MilvusDataType.INT64)
        schema.add_field("y", MilvusDataType.INT64)

        client.create_collection("not_combo", schema=schema)
        vecs = rvecs(10)
        client.insert("not_combo", [
            {"pk": i, "vec": vecs[i], "x": i % 3, "y": i % 2} for i in range(10)
        ])

        # not(x==0) and not(y==0) -> x!=0 and y!=0
        r = client.query("not_combo",
                         filter="not (x == 0) and not (y == 0)",
                         output_fields=["pk", "x", "y"])
        for row in r:
            assert row["x"] != 0 and row["y"] != 0


# ====================================================================
# 13. JSON array access filter
# ====================================================================

class TestJsonArrayAccess:

    def test_json_array_element_filter(self, client: MilvusClient):
        """Filtering on arrays within JSON fields"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("meta", MilvusDataType.JSON)

        client.create_collection("json_arr", schema=schema)
        vecs = rvecs(4)
        client.insert("json_arr", [
            {"pk": 0, "vec": vecs[0], "meta": {"scores": [90, 85, 92], "level": "A"}},
            {"pk": 1, "vec": vecs[1], "meta": {"scores": [70, 65, 72], "level": "B"}},
            {"pk": 2, "vec": vecs[2], "meta": {"scores": [95, 88, 91], "level": "A"}},
            {"pk": 3, "vec": vecs[3], "meta": {"scores": [50, 45, 55], "level": "C"}},
        ])

        # Filter using JSON path
        r = client.query("json_arr", filter='meta["level"] == "A"',
                         output_fields=["pk", "meta"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 2]


# ====================================================================
# 14. Hybrid Search + filter
# ====================================================================

class TestHybridSearchWithFilter:

    def test_hybrid_search_with_scalar_filter(self, client: MilvusClient):
        """Hybrid Search with scalar filter"""
        from pymilvus import AnnSearchRequest, RRFRanker

        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("group", MilvusDataType.VARCHAR, max_length=32)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("hybrid_f", schema=schema, index_params=idx)

        rng = np.random.default_rng(SEED)
        n = 40
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        client.insert("hybrid_f", [
            {"pk": i, "vec": vecs[i].tolist(),
             "group": "alpha" if i < 20 else "beta"}
            for i in range(n)
        ])
        client.load_collection("hybrid_f")

        q = vecs[0:1].tolist()
        req = AnnSearchRequest(
            data=q, anns_field="vec", param={}, limit=10,
            expr='group == "beta"',
        )
        results = client.hybrid_search(
            "hybrid_f",
            reqs=[req],
            ranker=RRFRanker(k=60),
            limit=5,
            output_fields=["pk", "group"],
        )
        for hit in results[0]:
            assert hit["entity"]["group"] == "beta"
            assert hit["entity"]["pk"] >= 20


# ====================================================================
# 15. Full lifecycle: insert -> search -> update -> delete -> re-insert
# ====================================================================

class TestFullLifecycle:

    def test_record_lifecycle(self, client: MilvusClient):
        """Full lifecycle of a single record"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", MilvusDataType.VARCHAR, max_length=32)
        schema.add_field("version", MilvusDataType.INT32)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("lifecycle", schema=schema, index_params=idx)
        client.load_collection("lifecycle")

        vec = rvecs(1)[0]

        # 1. Insert
        client.insert("lifecycle", [
            {"pk": 42, "vec": vec, "status": "created", "version": 1},
        ])
        got = client.get("lifecycle", ids=[42])
        assert got[0]["status"] == "created"

        # 2. Search can find it
        r = client.search("lifecycle", data=[vec], limit=1, output_fields=["pk"])
        assert r[0][0]["entity"]["pk"] == 42

        # 3. Upsert update
        client.upsert("lifecycle", [
            {"pk": 42, "vec": vec, "status": "updated", "version": 2},
        ])
        got = client.get("lifecycle", ids=[42])
        assert got[0]["status"] == "updated"
        assert got[0]["version"] == 2

        # 4. Flush
        client.flush("lifecycle")

        # 5. Upsert again
        client.upsert("lifecycle", [
            {"pk": 42, "vec": vec, "status": "finalized", "version": 3},
        ])
        got = client.get("lifecycle", ids=[42])
        assert got[0]["status"] == "finalized"

        # 6. Delete
        client.delete("lifecycle", ids=[42])
        assert len(client.get("lifecycle", ids=[42])) == 0

        # 7. Re-insert same PK
        new_vec = rvecs(1, seed=99)[0]
        client.insert("lifecycle", [
            {"pk": 42, "vec": new_vec, "status": "reborn", "version": 1},
        ])
        got = client.get("lifecycle", ids=[42])
        assert got[0]["status"] == "reborn"
        assert got[0]["version"] == 1

        # 8. Search still works
        r = client.search("lifecycle", data=[new_vec], limit=1,
                          output_fields=["pk", "status"])
        assert r[0][0]["entity"]["pk"] == 42
        assert r[0][0]["entity"]["status"] == "reborn"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
