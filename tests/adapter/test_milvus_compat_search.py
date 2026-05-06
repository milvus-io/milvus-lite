"""Milvus compatibility test suite — search features.

Migrated from milvus/tests/python_client/milvus_client_v2/ test files:
- test_milvus_client_search_json.py (JSON filter in search)
- test_milvus_client_search_pagination.py (search pagination)
- test_milvus_client_search_text_match.py (text_match filter)
- test_milvus_client_partial_update.py (partial upsert via gRPC)
"""

import numpy as np
import pytest

from pymilvus import DataType, Function, FunctionType, MilvusClient


DIM = 16
rng = np.random.default_rng(seed=42)


# ===========================================================================
# JSON filter in search
# ===========================================================================

class TestSearchJSON:
    @pytest.fixture(autouse=True)
    def _setup(self, milvus_client):
        self.client = milvus_client
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("info", DataType.JSON)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("json_s", schema=schema, index_params=idx)

        rows = []
        for i in range(100):
            rows.append({
                "id": i,
                "vec": rng.random(DIM).tolist(),
                "info": {
                    "number": i,
                    "tag": f"cat_{i % 5}",
                    "nested": {"a": i * 10, "b": f"nested_{i}"},
                    "list": [i, i + 1, i + 2],
                },
            })
        milvus_client.insert("json_s", rows)
        milvus_client.load_collection("json_s")
        self.rows = rows

    def test_search_json_comparison(self):
        q = [self.rows[0]["vec"]]
        res = self.client.search("json_s", data=q, limit=100,
                                 filter='info["number"] >= 50')
        for h in res[0]:
            assert h["id"] >= 50

    def test_search_json_nested(self):
        q = [self.rows[0]["vec"]]
        res = self.client.search("json_s", data=q, limit=100,
                                 filter='info["nested"]["a"] >= 500')
        for h in res[0]:
            assert h["id"] >= 50

    def test_search_json_string_eq(self):
        q = [self.rows[0]["vec"]]
        res = self.client.search("json_s", data=q, limit=100,
                                 filter='info["tag"] == "cat_0"',
                                 output_fields=["info"])
        for h in res[0]:
            assert h["entity"]["info"]["tag"] == "cat_0"

    def test_query_json_nested_like(self):
        res = self.client.query("json_s",
                                filter='info["nested"]["b"] like "nested_1%"',
                                limit=100)
        for r in res:
            assert r["info"]["nested"]["b"].startswith("nested_1")


# ===========================================================================
# Search pagination
# ===========================================================================

class TestSearchPagination:
    @pytest.fixture(autouse=True)
    def _setup(self, milvus_client):
        self.client = milvus_client
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("pag_s", schema=schema, index_params=idx)

        vecs = rng.random((200, DIM)).astype(np.float32)
        rows = [{"id": i, "vec": vecs[i].tolist()} for i in range(200)]
        milvus_client.insert("pag_s", rows)
        milvus_client.load_collection("pag_s")
        self.query_vec = vecs[0].tolist()

    def test_paginated_search_matches_full(self):
        """Page 1 + Page 2 should equal top-20 full search."""
        full = self.client.search("pag_s", data=[self.query_vec], limit=20)
        p1 = self.client.search("pag_s", data=[self.query_vec], limit=10,
                                search_params={"offset": 0})
        p2 = self.client.search("pag_s", data=[self.query_vec], limit=10,
                                search_params={"offset": 10})
        full_ids = [h["id"] for h in full[0]]
        paged_ids = [h["id"] for h in p1[0]] + [h["id"] for h in p2[0]]
        assert full_ids == paged_ids

    def test_search_offset_beyond_total(self):
        res = self.client.search("pag_s", data=[self.query_vec], limit=10,
                                 search_params={"offset": 9999})
        assert len(res[0]) == 0

    def test_search_pagination_with_filter(self):
        full = self.client.search("pag_s", data=[self.query_vec], limit=20,
                                  filter="id >= 100")
        p1 = self.client.search("pag_s", data=[self.query_vec], limit=10,
                                filter="id >= 100",
                                search_params={"offset": 0})
        p2 = self.client.search("pag_s", data=[self.query_vec], limit=10,
                                filter="id >= 100",
                                search_params={"offset": 10})
        full_ids = [h["id"] for h in full[0]]
        paged_ids = [h["id"] for h in p1[0]] + [h["id"] for h in p2[0]]
        assert full_ids == paged_ids

    def test_search_pagination_in_partition(self):
        self.client.create_partition("pag_s", "p1")
        self.client.insert("pag_s", [
            {"id": 1000 + i, "vec": rng.random(DIM).tolist()}
            for i in range(50)
        ], partition_name="p1")
        res = self.client.search("pag_s", data=[self.query_vec], limit=10,
                                 partition_names=["p1"],
                                 search_params={"offset": 0})
        assert len(res[0]) == 10
        res2 = self.client.search("pag_s", data=[self.query_vec], limit=10,
                                  partition_names=["p1"],
                                  search_params={"offset": 45})
        assert len(res2[0]) == 5  # only 50 total in p1


# ===========================================================================
# Text match filter in search
# ===========================================================================

class TestSearchTextMatch:
    @pytest.fixture(autouse=True)
    def _setup(self, milvus_client):
        self.client = milvus_client
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("text", DataType.VARCHAR, max_length=1024,
                         enable_analyzer=True, enable_match=True,
                         analyzer_params={"tokenizer": "standard"})

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("tm_s", schema=schema, index_params=idx)
        milvus_client.insert("tm_s", [
            {"id": 1, "vec": [1.0] + [0.0] * (DIM - 1),
             "text": "python programming language"},
            {"id": 2, "vec": [0.0, 1.0] + [0.0] * (DIM - 2),
             "text": "java programming language"},
            {"id": 3, "vec": [0.0, 0.0, 1.0] + [0.0] * (DIM - 3),
             "text": "machine learning algorithms"},
            {"id": 4, "vec": [0.0] * (DIM - 1) + [1.0],
             "text": "deep learning neural networks"},
        ])
        milvus_client.load_collection("tm_s")

    def test_search_with_text_match_single_token(self):
        q = [[1.0] + [0.0] * (DIM - 1)]
        res = self.client.search("tm_s", data=q, limit=10,
                                 filter='text_match(text, "python")',
                                 output_fields=["text"])
        assert len(res[0]) == 1
        assert res[0][0]["id"] == 1

    def test_search_with_text_match_multi_token(self):
        q = [[1.0] + [0.0] * (DIM - 1)]
        res = self.client.search("tm_s", data=q, limit=10,
                                 filter='text_match(text, "programming")',
                                 output_fields=["text"])
        ids = {h["id"] for h in res[0]}
        assert ids == {1, 2}

    def test_query_with_text_match(self):
        res = self.client.query("tm_s",
                                filter='text_match(text, "learning")',
                                output_fields=["text"], limit=10)
        ids = {r["id"] for r in res}
        assert ids == {3, 4}

    def test_search_text_match_combined_with_scalar(self):
        q = [[1.0] + [0.0] * (DIM - 1)]
        res = self.client.search("tm_s", data=q, limit=10,
                                 filter='text_match(text, "learning") and id >= 4',
                                 output_fields=["text"])
        assert len(res[0]) == 1
        assert res[0][0]["id"] == 4


# ===========================================================================
# Partial update via gRPC upsert
# ===========================================================================

class TestUpsertViaGRPC:
    """Upsert (full replace) via gRPC — pymilvus sends all columns."""

    @pytest.fixture(autouse=True)
    def _setup(self, milvus_client):
        self.client = milvus_client
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("name", DataType.VARCHAR, max_length=128, nullable=True)
        schema.add_field("score", DataType.FLOAT, nullable=True)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("ups_grpc", schema=schema, index_params=idx)
        milvus_client.insert("ups_grpc", [
            {"id": 1, "vec": [1.0] + [0.0] * (DIM - 1),
             "name": "alice", "score": 90.0},
            {"id": 2, "vec": [0.0, 1.0] + [0.0] * (DIM - 2),
             "name": "bob", "score": 80.0},
        ])
        milvus_client.load_collection("ups_grpc")

    def test_upsert_overwrites(self):
        """Full upsert replaces entire record."""
        self.client.upsert("ups_grpc", [
            {"id": 1, "vec": [0.5] * DIM,
             "name": "alice_v2", "score": 99.0},
        ])
        res = self.client.query("ups_grpc", filter="id == 1",
                                output_fields=["name", "score"])
        assert res[0]["name"] == "alice_v2"
        assert res[0]["score"] == pytest.approx(99.0)

    def test_upsert_new_pk_as_insert(self):
        self.client.upsert("ups_grpc", [
            {"id": 3, "vec": [0.3] * DIM, "name": "charlie", "score": 70.0},
        ])
        res = self.client.query("ups_grpc", filter="id == 3",
                                output_fields=["name"])
        assert res[0]["name"] == "charlie"

    def test_upsert_then_search(self):
        self.client.upsert("ups_grpc", [
            {"id": 1, "vec": [1.0] + [0.0] * (DIM - 1),
             "name": "alice_v2", "score": 100.0},
        ])
        res = self.client.search("ups_grpc",
                                 data=[[1.0] + [0.0] * (DIM - 1)], limit=1,
                                 output_fields=["name", "score"])
        assert res[0][0]["id"] == 1
        assert res[0][0]["entity"]["score"] == pytest.approx(100.0)
        assert res[0][0]["entity"]["name"] == "alice_v2"

    def test_upsert_multiple_times(self):
        """Latest upsert wins."""
        for i in range(5):
            self.client.upsert("ups_grpc", [
                {"id": 1, "vec": [0.1] * DIM,
                 "name": f"v{i}", "score": float(i)},
            ])
        res = self.client.query("ups_grpc", filter="id == 1",
                                output_fields=["name", "score"])
        assert res[0]["name"] == "v4"
        assert res[0]["score"] == pytest.approx(4.0)
