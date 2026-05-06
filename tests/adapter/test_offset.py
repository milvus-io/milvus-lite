"""Phase 17 — Search/Query offset pagination tests.

Covers:
1. Engine search offset
2. Engine query offset
3. gRPC search offset via pymilvus
4. gRPC query offset via pymilvus
5. offset + limit combination
6. offset beyond result count → empty
"""

import tempfile

import pytest

from pymilvus import DataType, MilvusClient


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------

class TestOffsetEngine:
    def _make_collection(self, tmpdir):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.engine.collection import Collection

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
        ])
        col = Collection(name="t", data_dir=tmpdir, schema=schema)
        col.insert([
            {"id": i, "vec": [float(10 - i), 0, 0, 0]}
            for i in range(10)
        ])
        return col

    def test_search_offset_skips_top(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            # Without offset
            no_off = col.search([[10, 0, 0, 0]], top_k=10)
            # With offset=3, top_k=5: should skip best 3, return next 5
            off_results = col.search([[10, 0, 0, 0]], top_k=5, offset=3)
            assert len(off_results[0]) <= 5
            # The first offset result should NOT be any of the top-3
            top3_ids = {h["id"] for h in no_off[0][:3]}
            for h in off_results[0]:
                assert h["id"] not in top3_ids

    def test_search_offset_zero_is_default(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            r1 = col.search([[10, 0, 0, 0]], top_k=5, offset=0)
            r2 = col.search([[10, 0, 0, 0]], top_k=5)
            assert [h["id"] for h in r1[0]] == [h["id"] for h in r2[0]]

    def test_search_offset_beyond_count(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search([[10, 0, 0, 0]], top_k=5, offset=100)
            assert results[0] == []

    def test_query_offset(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            all_rows = col.query("id >= 0")
            offset_rows = col.query("id >= 0", limit=3, offset=2)
            assert len(offset_rows) == 3
            # Should skip first 2 rows
            all_ids = [r["id"] for r in all_rows]
            offset_ids = [r["id"] for r in offset_rows]
            assert offset_ids == all_ids[2:5]

    def test_query_offset_without_limit(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            all_rows = col.query("id >= 0")
            offset_rows = col.query("id >= 0", offset=5)
            assert len(offset_rows) == len(all_rows) - 5


# ---------------------------------------------------------------------------
# gRPC integration tests
# ---------------------------------------------------------------------------

def _setup(client, name, n=20):
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    client.create_collection(name, schema=schema)
    client.insert(name, [
        {"id": i, "vec": [float(20 - i), 0, 0, 0]} for i in range(n)
    ])
    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    client.create_index(name, idx)
    client.load_collection(name)


def test_grpc_search_offset(milvus_client):
    _setup(milvus_client, "off_search")
    all_res = milvus_client.search("off_search", data=[[20, 0, 0, 0]], limit=20)
    off_res = milvus_client.search("off_search", data=[[20, 0, 0, 0]],
                                   limit=5, offset=3)
    assert len(off_res[0]) <= 5
    # Offset results should not contain the top-3 best matches
    top3_ids = {h["id"] for h in all_res[0][:3]}
    for h in off_res[0]:
        assert h["id"] not in top3_ids
    milvus_client.drop_collection("off_search")


def test_grpc_query_offset(milvus_client):
    _setup(milvus_client, "off_query")
    all_rows = milvus_client.query("off_query", filter="id >= 0",
                                   output_fields=["id"], limit=20)
    off_rows = milvus_client.query("off_query", filter="id >= 0",
                                   output_fields=["id"],
                                   limit=5, offset=3)
    assert len(off_rows) == 5
    milvus_client.drop_collection("off_query")


def test_grpc_search_offset_empty(milvus_client):
    _setup(milvus_client, "off_empty", n=5)
    results = milvus_client.search("off_empty", data=[[20, 0, 0, 0]],
                                   limit=5, offset=100)
    assert len(results[0]) == 0
    milvus_client.drop_collection("off_empty")
