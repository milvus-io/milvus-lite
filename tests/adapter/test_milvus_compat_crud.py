"""Milvus compatibility test suite — CRUD operations.

Migrated from milvus/tests/python_client/milvus_client/ test files:
- test_milvus_client_insert.py
- test_milvus_client_upsert.py
- test_milvus_client_delete.py
- test_milvus_client_query.py
- test_milvus_client_search.py

Covers core insert/upsert/delete/query/search/get round-trips that
any pymilvus-compatible backend must pass.
"""

import numpy as np
import pytest

from pymilvus import DataType, MilvusClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema(dim=128, auto_id=False, enable_dynamic=False):
    schema = MilvusClient.create_schema(auto_id=auto_id,
                                        enable_dynamic_field=enable_dynamic)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("title", DataType.VARCHAR, max_length=256, nullable=True)
    schema.add_field("score", DataType.FLOAT, nullable=True)
    schema.add_field("active", DataType.BOOL, nullable=True)
    return schema


def _schema_string_pk(dim=128):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=dim)
    return schema


DIM = 16
NB = 200  # keep small for test speed
rng = np.random.default_rng(seed=42)


def _gen_rows(nb=NB, dim=DIM, id_offset=0):
    vecs = rng.random((nb, dim)).astype(np.float32)
    return [
        {"id": id_offset + i, "vec": vecs[i].tolist(),
         "title": f"doc_{id_offset + i}", "score": float(i) / nb,
         "active": i % 2 == 0}
        for i in range(nb)
    ]


def _create_and_load(client, name, dim=DIM, rows=None, **schema_kw):
    schema = _schema(dim=dim, **schema_kw)
    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
    client.create_collection(name, schema=schema, index_params=idx)
    if rows is None:
        rows = _gen_rows(dim=dim)
    if rows:
        client.insert(name, rows)
    client.load_collection(name)
    return rows


# ===========================================================================
# Insert
# ===========================================================================

class TestInsert:
    def test_insert_basic(self, milvus_client):
        rows = _create_and_load(milvus_client, "ins_basic")
        res = milvus_client.query("ins_basic", filter="id >= 0", limit=NB + 10)
        assert len(res) == NB

    def test_insert_empty_list(self, milvus_client):
        _create_and_load(milvus_client, "ins_empty", rows=[])
        res = milvus_client.query("ins_empty", filter="id >= 0", limit=10)
        assert len(res) == 0

    def test_insert_missing_vector_raises(self, milvus_client):
        _create_and_load(milvus_client, "ins_no_vec", rows=[])
        with pytest.raises(Exception):
            milvus_client.insert("ins_no_vec", [{"id": 1, "title": "x"}])

    def test_insert_dim_mismatch_raises(self, milvus_client):
        _create_and_load(milvus_client, "ins_dim", rows=[])
        with pytest.raises(Exception):
            milvus_client.insert("ins_dim", [{"id": 1, "vec": [0.1] * 999}])

    def test_insert_extra_field_no_dynamic_raises(self, milvus_client):
        _create_and_load(milvus_client, "ins_extra", rows=[])
        with pytest.raises(Exception):
            milvus_client.insert("ins_extra", [
                {"id": 1, "vec": [0.1] * DIM, "unknown_field": "x"}
            ])

    def test_insert_with_dynamic_field(self, milvus_client):
        _create_and_load(milvus_client, "ins_dyn", rows=[], enable_dynamic=True)
        milvus_client.insert("ins_dyn", [
            {"id": 1, "vec": [0.1] * DIM, "color": "red"},
        ])
        res = milvus_client.query("ins_dyn", filter="id == 1",
                                  output_fields=["color"])
        assert res[0]["color"] == "red"

    def test_insert_string_pk(self, milvus_client):
        schema = _schema_string_pk(dim=DIM)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("ins_strpk", schema=schema, index_params=idx)
        milvus_client.insert("ins_strpk", [
            {"id": "hello", "vec": [0.1] * DIM},
            {"id": "world", "vec": [0.2] * DIM},
        ])
        milvus_client.load_collection("ins_strpk")
        res = milvus_client.query("ins_strpk", filter='id == "hello"')
        assert len(res) == 1


# ===========================================================================
# Upsert
# ===========================================================================

class TestUpsert:
    def test_upsert_as_insert(self, milvus_client):
        """Upsert to non-existent PKs acts as insert."""
        _create_and_load(milvus_client, "ups_new", rows=[])
        milvus_client.upsert("ups_new", [
            {"id": 1, "vec": [0.1] * DIM, "title": "a"},
            {"id": 2, "vec": [0.2] * DIM, "title": "b"},
        ])
        res = milvus_client.query("ups_new", filter="id >= 0", limit=10)
        assert len(res) == 2

    def test_upsert_overwrites_existing(self, milvus_client):
        """Upsert existing PKs overwrites the record."""
        _create_and_load(milvus_client, "ups_over", rows=[
            {"id": 1, "vec": [0.1] * DIM, "title": "old"},
        ])
        milvus_client.upsert("ups_over", [
            {"id": 1, "vec": [0.9] * DIM, "title": "new"},
        ])
        res = milvus_client.query("ups_over", filter="id == 1",
                                  output_fields=["title"])
        assert res[0]["title"] == "new"

    def test_upsert_multiple_times(self, milvus_client):
        """Upsert same PK multiple times, latest wins."""
        _create_and_load(milvus_client, "ups_multi", rows=[])
        for i in range(5):
            milvus_client.upsert("ups_multi", [
                {"id": 1, "vec": [float(i)] * DIM, "title": f"v{i}"},
            ])
        res = milvus_client.query("ups_multi", filter="id == 1",
                                  output_fields=["title"])
        assert res[0]["title"] == "v4"

    def test_upsert_with_dynamic_field(self, milvus_client):
        _create_and_load(milvus_client, "ups_dyn", rows=[], enable_dynamic=True)
        milvus_client.insert("ups_dyn", [
            {"id": 1, "vec": [0.1] * DIM, "color": "red"},
        ])
        milvus_client.upsert("ups_dyn", [
            {"id": 1, "vec": [0.1] * DIM, "color": "blue"},
        ])
        res = milvus_client.query("ups_dyn", filter="id == 1",
                                  output_fields=["color"])
        assert res[0]["color"] == "blue"

    def test_upsert_duplicate_pk_in_batch(self, milvus_client):
        """Duplicate PKs in same upsert batch — last one wins."""
        _create_and_load(milvus_client, "ups_dup", rows=[])
        milvus_client.upsert("ups_dup", [
            {"id": 1, "vec": [0.1] * DIM, "title": "first"},
            {"id": 1, "vec": [0.2] * DIM, "title": "second"},
        ])
        res = milvus_client.query("ups_dup", filter="id == 1",
                                  output_fields=["title"])
        assert res[0]["title"] == "second"


# ===========================================================================
# Delete
# ===========================================================================

class TestDelete:
    def test_delete_by_ids(self, milvus_client):
        rows = _create_and_load(milvus_client, "del_ids")
        milvus_client.delete("del_ids", ids=[0, 1, 2])
        res = milvus_client.query("del_ids", filter="id in [0, 1, 2]", limit=10)
        assert len(res) == 0

    def test_delete_by_filter(self, milvus_client):
        _create_and_load(milvus_client, "del_flt")
        milvus_client.delete("del_flt", filter="id < 10")
        res = milvus_client.query("del_flt", filter="id < 10", limit=100)
        assert len(res) == 0

    def test_delete_then_search_excludes(self, milvus_client):
        rows = _create_and_load(milvus_client, "del_search")
        target_vec = rows[0]["vec"]
        milvus_client.delete("del_search", ids=[0])
        res = milvus_client.search("del_search", data=[target_vec], limit=NB)
        hit_ids = {h["id"] for h in res[0]}
        assert 0 not in hit_ids

    def test_delete_all_then_query_empty(self, milvus_client):
        _create_and_load(milvus_client, "del_all")
        milvus_client.delete("del_all", filter="id >= 0")
        res = milvus_client.query("del_all", filter="id >= 0", limit=NB + 10)
        assert len(res) == 0


# ===========================================================================
# Query
# ===========================================================================

class TestQuery:
    def test_query_default(self, milvus_client):
        _create_and_load(milvus_client, "q_default")
        res = milvus_client.query("q_default", filter="id >= 0", limit=10)
        assert len(res) == 10

    def test_query_output_fields(self, milvus_client):
        _create_and_load(milvus_client, "q_out")
        res = milvus_client.query("q_out", filter="id == 0",
                                  output_fields=["title", "score"])
        assert "title" in res[0]
        assert "score" in res[0]

    def test_query_output_fields_wildcard(self, milvus_client):
        _create_and_load(milvus_client, "q_wild")
        res = milvus_client.query("q_wild", filter="id == 0",
                                  output_fields=["*"])
        assert "title" in res[0]
        assert "score" in res[0]
        assert "vec" in res[0]

    def test_query_empty_collection(self, milvus_client):
        _create_and_load(milvus_client, "q_empty", rows=[])
        res = milvus_client.query("q_empty", filter="id >= 0", limit=10)
        assert len(res) == 0

    def test_query_empty_in_list(self, milvus_client):
        """id in [] returns empty."""
        _create_and_load(milvus_client, "q_emptylist")
        res = milvus_client.query("q_emptylist", filter="id in []", limit=10)
        assert len(res) == 0

    def test_query_not_in(self, milvus_client):
        _create_and_load(milvus_client, "q_notin")
        res = milvus_client.query("q_notin", filter="id not in [0, 1]",
                                  limit=NB)
        ids = {r["id"] for r in res}
        assert 0 not in ids and 1 not in ids

    def test_query_not_in_empty_returns_all(self, milvus_client):
        """not in [] returns all records."""
        _create_and_load(milvus_client, "q_notinall")
        res = milvus_client.query("q_notinall", filter="id not in []",
                                  limit=NB + 10)
        assert len(res) == NB

    def test_query_with_limit(self, milvus_client):
        _create_and_load(milvus_client, "q_limit")
        res = milvus_client.query("q_limit", filter="id >= 0", limit=5)
        assert len(res) == 5

    def test_query_pagination(self, milvus_client):
        _create_and_load(milvus_client, "q_page")
        page1 = milvus_client.query("q_page", filter="id >= 0",
                                    limit=10, offset=0)
        page2 = milvus_client.query("q_page", filter="id >= 0",
                                    limit=10, offset=10)
        ids1 = {r["id"] for r in page1}
        ids2 = {r["id"] for r in page2}
        assert len(ids1) == 10
        assert len(ids2) == 10
        assert ids1.isdisjoint(ids2)

    def test_query_offset_beyond_total(self, milvus_client):
        _create_and_load(milvus_client, "q_offbig")
        res = milvus_client.query("q_offbig", filter="id >= 0",
                                  limit=10, offset=NB + 100)
        assert len(res) == 0

    def test_query_by_bool_field(self, milvus_client):
        _create_and_load(milvus_client, "q_bool")
        res = milvus_client.query("q_bool", filter="active == true",
                                  limit=NB)
        for r in res:
            assert r["active"] is True

    def test_query_by_float_field(self, milvus_client):
        _create_and_load(milvus_client, "q_float")
        res = milvus_client.query("q_float", filter="score > 0.5",
                                  limit=NB, output_fields=["score"])
        for r in res:
            assert r["score"] > 0.5

    def test_query_multi_logical(self, milvus_client):
        _create_and_load(milvus_client, "q_logic")
        res = milvus_client.query(
            "q_logic",
            filter="id >= 10 and id < 20 and active == true",
            limit=NB,
        )
        for r in res:
            assert 10 <= r["id"] < 20

    def test_query_like(self, milvus_client):
        _create_and_load(milvus_client, "q_like")
        res = milvus_client.query("q_like", filter='title like "doc_1%"',
                                  limit=NB, output_fields=["title"])
        for r in res:
            assert r["title"].startswith("doc_1")

    def test_query_dup_pk_returns_latest(self, milvus_client):
        """Insert same PK twice, query returns latest version."""
        _create_and_load(milvus_client, "q_dup", rows=[
            {"id": 1, "vec": [0.1] * DIM, "title": "old"},
        ])
        milvus_client.insert("q_dup", [
            {"id": 1, "vec": [0.1] * DIM, "title": "new"},
        ])
        res = milvus_client.query("q_dup", filter="id == 1",
                                  output_fields=["title"])
        assert res[0]["title"] == "new"

    def test_query_without_loading_raises(self, milvus_client):
        """Collection with index but not loaded should reject query."""
        schema = _schema(dim=DIM)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("q_noload", schema=schema,
                                        index_params=idx)
        milvus_client.release_collection("q_noload")
        with pytest.raises(Exception):
            milvus_client.query("q_noload", filter="id >= 0", limit=10)

    def test_query_not_exist_collection_raises(self, milvus_client):
        with pytest.raises(Exception):
            milvus_client.query("nonexistent_col_xyz", filter="id >= 0",
                                limit=10)


# ===========================================================================
# Get
# ===========================================================================

class TestGet:
    def test_get_basic(self, milvus_client):
        _create_and_load(milvus_client, "get_basic")
        res = milvus_client.get("get_basic", ids=[0, 1, 2])
        assert len(res) == 3

    def test_get_nonexistent_ids(self, milvus_client):
        _create_and_load(milvus_client, "get_none")
        res = milvus_client.get("get_none", ids=[99999])
        assert len(res) == 0

    def test_get_output_fields(self, milvus_client):
        _create_and_load(milvus_client, "get_out")
        res = milvus_client.get("get_out", ids=[0],
                                output_fields=["title"])
        assert "title" in res[0]


# ===========================================================================
# Search
# ===========================================================================

class TestSearch:
    def test_search_basic(self, milvus_client):
        rows = _create_and_load(milvus_client, "s_basic")
        q = [rows[0]["vec"]]
        res = milvus_client.search("s_basic", data=q, limit=5)
        assert len(res) == 1
        assert len(res[0]) == 5
        # Self should be closest
        assert res[0][0]["id"] == 0

    def test_search_with_filter(self, milvus_client):
        rows = _create_and_load(milvus_client, "s_flt")
        q = [rows[0]["vec"]]
        res = milvus_client.search("s_flt", data=q, limit=NB,
                                   filter="id >= 100")
        for h in res[0]:
            assert h["id"] >= 100

    def test_search_output_fields(self, milvus_client):
        rows = _create_and_load(milvus_client, "s_out")
        q = [rows[0]["vec"]]
        res = milvus_client.search("s_out", data=q, limit=3,
                                   output_fields=["title", "score"])
        for h in res[0]:
            assert "title" in h["entity"]
            assert "score" in h["entity"]

    def test_search_l2_metric(self, milvus_client):
        schema = _schema(dim=DIM)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="L2")
        milvus_client.create_collection("s_l2", schema=schema, index_params=idx)
        rows = _gen_rows()
        milvus_client.insert("s_l2", rows)
        milvus_client.load_collection("s_l2")
        res = milvus_client.search("s_l2", data=[rows[0]["vec"]], limit=1)
        assert res[0][0]["id"] == 0
        assert res[0][0]["distance"] >= 0  # L2 distance >= 0

    def test_search_offset(self, milvus_client):
        rows = _create_and_load(milvus_client, "s_offset")
        q = [rows[0]["vec"]]
        res_no_off = milvus_client.search("s_offset", data=q, limit=5)
        res_off = milvus_client.search("s_offset", data=q, limit=5,
                                       search_params={"offset": 5})
        ids_no = [h["id"] for h in res_no_off[0]]
        ids_off = [h["id"] for h in res_off[0]]
        # Offset results should not overlap with first page
        assert set(ids_no).isdisjoint(set(ids_off))

    def test_search_batch_queries(self, milvus_client):
        """Multiple query vectors (nq > 1)."""
        rows = _create_and_load(milvus_client, "s_batch")
        q = [rows[0]["vec"], rows[1]["vec"]]
        res = milvus_client.search("s_batch", data=q, limit=3)
        assert len(res) == 2
        assert len(res[0]) == 3
        assert len(res[1]) == 3

    def test_search_string_pk(self, milvus_client):
        schema = _schema_string_pk(dim=DIM)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("s_strpk", schema=schema, index_params=idx)
        milvus_client.insert("s_strpk", [
            {"id": "alpha", "vec": [1.0] + [0.0] * (DIM - 1)},
            {"id": "beta", "vec": [0.0, 1.0] + [0.0] * (DIM - 2)},
        ])
        milvus_client.load_collection("s_strpk")
        res = milvus_client.search("s_strpk",
                                   data=[[1.0] + [0.0] * (DIM - 1)], limit=1)
        assert res[0][0]["id"] == "alpha"

    def test_search_nonexistent_collection_raises(self, milvus_client):
        with pytest.raises(Exception):
            milvus_client.search("ghost_col", data=[[0.1] * DIM], limit=5)


# ===========================================================================
# Partition
# ===========================================================================

class TestPartition:
    def test_query_specific_partition(self, milvus_client):
        schema = _schema(dim=DIM)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("part_q", schema=schema, index_params=idx)
        milvus_client.create_partition("part_q", "p1")
        milvus_client.create_partition("part_q", "p2")
        milvus_client.insert("part_q", [
            {"id": 1, "vec": [0.1] * DIM, "title": "in_p1"},
        ], partition_name="p1")
        milvus_client.insert("part_q", [
            {"id": 2, "vec": [0.2] * DIM, "title": "in_p2"},
        ], partition_name="p2")
        milvus_client.load_collection("part_q")
        res = milvus_client.query("part_q", filter="id >= 0",
                                  partition_names=["p1"], limit=10)
        ids = {r["id"] for r in res}
        assert ids == {1}

    def test_query_multiple_partitions(self, milvus_client):
        schema = _schema(dim=DIM)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("part_multi", schema=schema, index_params=idx)
        milvus_client.create_partition("part_multi", "p1")
        milvus_client.create_partition("part_multi", "p2")
        milvus_client.insert("part_multi", [
            {"id": 1, "vec": [0.1] * DIM},
        ], partition_name="p1")
        milvus_client.insert("part_multi", [
            {"id": 2, "vec": [0.2] * DIM},
        ], partition_name="p2")
        milvus_client.load_collection("part_multi")
        res = milvus_client.query("part_multi", filter="id >= 0",
                                  partition_names=["p1", "p2"], limit=10)
        assert len(res) == 2

    def test_query_empty_partition(self, milvus_client):
        schema = _schema(dim=DIM)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("part_emp", schema=schema, index_params=idx)
        milvus_client.create_partition("part_emp", "empty")
        milvus_client.load_collection("part_emp")
        res = milvus_client.query("part_emp", filter="id >= 0",
                                  partition_names=["empty"], limit=10)
        assert len(res) == 0


# ===========================================================================
# Default value
# ===========================================================================

class TestDefaultValue:
    def test_insert_with_default_value(self, milvus_client):
        schema = MilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", DataType.VARCHAR, max_length=32,
                         default_value="active")
        schema.add_field("count", DataType.INT64, default_value=0)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("def_val", schema=schema, index_params=idx)
        # Omit status and count — should use defaults
        milvus_client.insert("def_val", [
            {"id": 1, "vec": [0.1] * DIM},
        ])
        milvus_client.load_collection("def_val")
        res = milvus_client.query("def_val", filter="id == 1",
                                  output_fields=["status", "count"])
        assert res[0]["status"] == "active"
        assert res[0]["count"] == 0

    def test_insert_override_default_value(self, milvus_client):
        schema = MilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", DataType.VARCHAR, max_length=32,
                         default_value="active")

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("def_over", schema=schema, index_params=idx)
        milvus_client.insert("def_over", [
            {"id": 1, "vec": [0.1] * DIM, "status": "inactive"},
        ])
        milvus_client.load_collection("def_over")
        res = milvus_client.query("def_over", filter="id == 1",
                                  output_fields=["status"])
        assert res[0]["status"] == "inactive"


# ===========================================================================
# Nullable
# ===========================================================================

class TestNullable:
    def test_insert_null_fields(self, milvus_client):
        _create_and_load(milvus_client, "null_ins", rows=[
            {"id": 1, "vec": [0.1] * DIM, "title": None, "score": None},
        ])
        res = milvus_client.query("null_ins", filter="id == 1",
                                  output_fields=["title", "score"])
        assert res[0]["title"] is None
        assert res[0]["score"] is None

    def test_query_is_null(self, milvus_client):
        _create_and_load(milvus_client, "null_q", rows=[
            {"id": 1, "vec": [0.1] * DIM, "title": "yes"},
            {"id": 2, "vec": [0.2] * DIM, "title": None},
        ])
        res = milvus_client.query("null_q", filter="title is null", limit=10)
        assert len(res) == 1
        assert res[0]["id"] == 2

    def test_query_is_not_null(self, milvus_client):
        _create_and_load(milvus_client, "null_qnn", rows=[
            {"id": 1, "vec": [0.1] * DIM, "title": "yes"},
            {"id": 2, "vec": [0.2] * DIM, "title": None},
        ])
        res = milvus_client.query("null_qnn", filter="title is not null",
                                  limit=10)
        assert len(res) == 1
        assert res[0]["id"] == 1
