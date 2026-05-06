"""Run our compatibility tests against a real Milvus server.

Usage:
    MILVUS_URI=http://host:19530 pytest tests/adapter/test_against_milvus.py -v

Requires a running Milvus at MILVUS_URI env var.
Skipped automatically if MILVUS_URI is not set or the server is unreachable.
"""

import os
import numpy as np
import pytest

MILVUS_URI = os.environ.get("MILVUS_URI", "")

pymilvus = pytest.importorskip("pymilvus")
from pymilvus import DataType, MilvusClient, Function, FunctionType


# ---------------------------------------------------------------------------
# Fixture: connect to real Milvus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mclient():
    if not MILVUS_URI:
        pytest.skip("MILVUS_URI env var not set")
    try:
        client = MilvusClient(uri=MILVUS_URI)
        client.get_server_version()
    except Exception:
        pytest.skip(f"Milvus server not reachable at {MILVUS_URI}")
    yield client
    # Clean up all test collections
    for name in client.list_collections():
        if name.startswith("mtest_"):
            try:
                client.drop_collection(name)
            except Exception:
                pass
    client.close()


DIM = 16
rng = np.random.default_rng(seed=42)


def _schema(dim=DIM, enable_dynamic=False):
    schema = MilvusClient.create_schema(auto_id=False,
                                        enable_dynamic_field=enable_dynamic)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("title", DataType.VARCHAR, max_length=256, nullable=True)
    schema.add_field("score", DataType.FLOAT, nullable=True)
    return schema


def _create(client, name, dim=DIM, rows=None, **kw):
    schema = _schema(dim=dim, **kw)
    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(name, schema=schema, index_params=idx,
                             consistency_level="Strong")
    if rows:
        client.insert(name, rows)
    client.load_collection(name)


# ===========================================================================
# CRUD
# ===========================================================================

class TestCRUD:
    def test_insert_and_query(self, mclient):
        rows = [{"id": i, "vec": rng.random(DIM).tolist(), "title": f"d_{i}"}
                for i in range(50)]
        _create(mclient, "mtest_iq", rows=rows)
        res = mclient.query("mtest_iq", filter="id >= 0", limit=100)
        assert len(res) == 50

    def test_upsert_overwrites(self, mclient):
        _create(mclient, "mtest_ups", rows=[
            {"id": 1, "vec": [0.1] * DIM, "title": "old"},
        ])
        mclient.upsert("mtest_ups", [
            {"id": 1, "vec": [0.1] * DIM, "title": "new"},
        ])
        res = mclient.query("mtest_ups", filter="id == 1",
                            output_fields=["title"])
        assert res[0]["title"] == "new"

    def test_delete_by_ids(self, mclient):
        _create(mclient, "mtest_del", rows=[
            {"id": i, "vec": rng.random(DIM).tolist()} for i in range(10)
        ])
        mclient.delete("mtest_del", ids=[0, 1, 2])
        res = mclient.query("mtest_del", filter="id in [0,1,2]", limit=10)
        assert len(res) == 0

    def test_delete_by_filter(self, mclient):
        _create(mclient, "mtest_delflt", rows=[
            {"id": i, "vec": rng.random(DIM).tolist()} for i in range(20)
        ])
        mclient.delete("mtest_delflt", filter="id < 10")
        res = mclient.query("mtest_delflt", filter="id < 10", limit=100)
        assert len(res) == 0

    def test_get_by_ids(self, mclient):
        _create(mclient, "mtest_get", rows=[
            {"id": i, "vec": rng.random(DIM).tolist()} for i in range(10)
        ])
        res = mclient.get("mtest_get", ids=[0, 1, 2])
        assert len(res) == 3

    def test_search_basic(self, mclient):
        rows = [{"id": i, "vec": rng.random(DIM).tolist()} for i in range(50)]
        _create(mclient, "mtest_search", rows=rows)
        res = mclient.search("mtest_search", data=[rows[0]["vec"]], limit=5)
        assert len(res[0]) == 5

    def test_search_with_filter(self, mclient):
        rows = [{"id": i, "vec": rng.random(DIM).tolist()} for i in range(50)]
        _create(mclient, "mtest_sflt", rows=rows)
        res = mclient.search("mtest_sflt", data=[rows[0]["vec"]], limit=50,
                             filter="id >= 25")
        for h in res[0]:
            assert h["id"] >= 25


# ===========================================================================
# Query features
# ===========================================================================

class TestQuery:
    def test_query_pagination(self, mclient):
        rows = [{"id": i, "vec": rng.random(DIM).tolist()} for i in range(50)]
        _create(mclient, "mtest_qpage", rows=rows)
        p1 = mclient.query("mtest_qpage", filter="id >= 0", limit=10, offset=0)
        p2 = mclient.query("mtest_qpage", filter="id >= 0", limit=10, offset=10)
        ids1 = {r["id"] for r in p1}
        ids2 = {r["id"] for r in p2}
        assert len(ids1) == 10
        assert ids1.isdisjoint(ids2)

    def test_query_like(self, mclient):
        rows = [{"id": i, "vec": rng.random(DIM).tolist(), "title": f"doc_{i}"}
                for i in range(50)]
        _create(mclient, "mtest_qlike", rows=rows)
        res = mclient.query("mtest_qlike", filter='title like "doc_1%"',
                            limit=100, output_fields=["title"])
        for r in res:
            assert r["title"].startswith("doc_1")

    def test_query_not_in_empty(self, mclient):
        rows = [{"id": i, "vec": rng.random(DIM).tolist()} for i in range(20)]
        _create(mclient, "mtest_qnotine", rows=rows)
        res = mclient.query("mtest_qnotine", filter="id not in []", limit=100)
        assert len(res) == 20

    def test_query_is_null(self, mclient):
        _create(mclient, "mtest_qnull", rows=[
            {"id": 1, "vec": [0.1] * DIM, "title": "yes"},
            {"id": 2, "vec": [0.2] * DIM, "title": None},
        ])
        res = mclient.query("mtest_qnull", filter="title is null", limit=10)
        assert len(res) == 1
        assert res[0]["id"] == 2


# ===========================================================================
# Dynamic fields
# ===========================================================================

class TestDynamicFields:
    def test_dynamic_all_types(self, mclient):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        idx = mclient.prepare_index_params()
        idx.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
        mclient.create_collection("mtest_dyn", schema=schema, index_params=idx,
                                        consistency_level="Strong")
        mclient.insert("mtest_dyn", [{
            "id": 1, "vec": [0.1] * DIM,
            "str_f": "hello", "int_f": 42, "float_f": 3.14,
            "bool_f": True, "list_f": [1, 2, 3],
            "dict_f": {"nested": "val"},
        }])
        mclient.load_collection("mtest_dyn")
        res = mclient.query("mtest_dyn", filter="id == 1",
                            output_fields=["str_f", "int_f", "float_f",
                                           "bool_f", "list_f", "dict_f"])
        r = res[0]
        assert r["str_f"] == "hello"
        assert r["int_f"] == 42
        assert isinstance(r["int_f"], int)
        assert abs(r["float_f"] - 3.14) < 0.01
        assert r["bool_f"] is True
        assert r["list_f"] == [1, 2, 3]
        assert r["dict_f"] == {"nested": "val"}

    def test_dynamic_field_filter(self, mclient):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        idx = mclient.prepare_index_params()
        idx.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
        mclient.create_collection("mtest_dynflt", schema=schema, index_params=idx,
                                        consistency_level="Strong")
        mclient.insert("mtest_dynflt", [
            {"id": i, "vec": rng.random(DIM).tolist(), "color": f"c_{i % 3}"}
            for i in range(30)
        ])
        mclient.load_collection("mtest_dynflt")
        res = mclient.query("mtest_dynflt", filter='color == "c_0"', limit=100)
        assert len(res) == 10


# ===========================================================================
# JSON field
# ===========================================================================

class TestJSON:
    def test_json_nested_filter(self, mclient):
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("info", DataType.JSON)
        idx = mclient.prepare_index_params()
        idx.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
        mclient.create_collection("mtest_json", schema=schema, index_params=idx,
                                        consistency_level="Strong")
        mclient.insert("mtest_json", [
            {"id": i, "vec": rng.random(DIM).tolist(),
             "info": {"a": {"b": i * 10}}}
            for i in range(10)
        ])
        mclient.load_collection("mtest_json")
        res = mclient.query("mtest_json", filter='info["a"]["b"] >= 50',
                            limit=100)
        assert len(res) == 5


# ===========================================================================
# Partition key
# ===========================================================================

class TestPartitionKey:
    def test_partition_key_insert_query(self, mclient):
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tenant", DataType.VARCHAR, max_length=64,
                         is_partition_key=True)
        idx = mclient.prepare_index_params()
        idx.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
        mclient.create_collection("mtest_pk", schema=schema, index_params=idx,
                                        consistency_level="Strong")
        mclient.insert("mtest_pk", [
            {"id": i, "vec": rng.random(DIM).tolist(), "tenant": f"t_{i % 5}"}
            for i in range(50)
        ])
        mclient.load_collection("mtest_pk")
        res = mclient.query("mtest_pk", filter='tenant == "t_0"', limit=100)
        assert len(res) == 10

    def test_partition_key_describe(self, mclient):
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("grp", DataType.INT64, is_partition_key=True)
        idx = mclient.prepare_index_params()
        idx.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
        mclient.create_collection("mtest_pkdesc", schema=schema, index_params=idx,
                                        consistency_level="Strong")
        desc = mclient.describe_collection("mtest_pkdesc")
        pk_fields = [f for f in desc["fields"] if f.get("is_partition_key")]
        assert len(pk_fields) == 1
        assert pk_fields[0]["name"] == "grp"


# ===========================================================================
# Default value
# ===========================================================================

class TestDefaultValue:
    def test_insert_with_default(self, mclient):
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", DataType.VARCHAR, max_length=32,
                         default_value="active")
        idx = mclient.prepare_index_params()
        idx.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
        mclient.create_collection("mtest_defval", schema=schema, index_params=idx,
                                        consistency_level="Strong")
        mclient.insert("mtest_defval", [{"id": 1, "vec": [0.1] * DIM}])
        mclient.load_collection("mtest_defval")
        res = mclient.query("mtest_defval", filter="id == 1",
                            output_fields=["status"])
        assert res[0]["status"] == "active"
