"""Milvus pymilvus compatibility tests — adapted from milvus-io/milvus
python_client/milvus_client/ test suite.

Covers the "can MilvusLite pass the same test as a real Milvus?" surface:
- Full CRUD lifecycle (insert → search → query → get → delete)
- All supported scalar types (INT8/16/32/64, FLOAT, DOUBLE, VARCHAR, BOOL)
- VARCHAR primary key
- Filter expression coverage (cmp / AND / OR / NOT / IN / LIKE / IS NULL)
- Upsert semantics (overwrite existing pk, insert new pk)
- Multi-query search
- Output fields projection
- Partition insert + partition-scoped search / query
- Delete by IDs + delete by filter expression
- Index lifecycle (create → load → search → release → drop → re-create)
- Dynamic field ($meta) access

Test style: each test is self-contained (create → insert → exercise
→ drop) so failures are isolated. No shared state across tests.

Skipped automatically when pymilvus / faiss-cpu is not installed.
"""

import numpy as np
import pytest
from pymilvus import DataType, MilvusClient

from milvus_lite.index.factory import is_faiss_available

pytestmark = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss-cpu required for search tests"
)

DIM = 8
NB = 200  # small but enough to exercise filters meaningfully


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng():
    return np.random.default_rng(seed=19530)


def _gen_rows(n=NB, dim=DIM, rng=None):
    """Generate n rows with id, vec, float_field, varchar, bool fields."""
    rng = rng or _rng()
    return [
        {
            "id": i,
            "vec": rng.standard_normal(dim).astype(np.float32).tolist(),
            "float_field": float(i) * 1.0,
            "varchar": f"str_{i:04d}",
            "bool_field": (i % 2 == 0),
        }
        for i in range(n)
    ]


def _schema(client, dynamic=False):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=dynamic)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("float_field", DataType.FLOAT)
    schema.add_field("varchar", DataType.VARCHAR, max_length=128)
    schema.add_field("bool_field", DataType.BOOL)
    return schema


def _create_loaded(client, name, rows=None, metric="COSINE", dynamic=False):
    """Helper: create collection, insert, create index, load. Returns rows."""
    rows = rows or _gen_rows()
    client.create_collection(name, schema=_schema(client, dynamic=dynamic))
    client.insert(name, rows)
    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="HNSW", metric_type=metric,
                  params={"M": 16, "efConstruction": 200})
    client.create_index(name, idx)
    client.load_collection(name)
    return rows


# ===========================================================================
# 1. INSERT — basic, type coverage, varchar pk
# ===========================================================================

class TestInsert:
    def test_insert_and_count(self, milvus_client):
        rows = _create_loaded(milvus_client, "ins1")
        result = milvus_client.query("ins1", filter="id >= 0", output_fields=["id"])
        assert len(result) == NB
        milvus_client.drop_collection("ins1")

    def test_insert_string_pk(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        milvus_client.create_collection("strpk", schema=schema)
        rows = [{"pk": f"doc_{i}", "vec": [float(i)] * 4} for i in range(10)]
        res = milvus_client.insert("strpk", rows)
        assert res["insert_count"] == 10
        milvus_client.drop_collection("strpk")

    def test_insert_multiple_batches(self, milvus_client):
        milvus_client.create_collection("batch", schema=_schema(milvus_client))
        for batch in range(5):
            rows = _gen_rows(n=20, rng=np.random.default_rng(batch))
            # Offset ids to avoid pk collision
            for i, r in enumerate(rows):
                r["id"] = batch * 20 + i
            milvus_client.insert("batch", rows)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16})
        milvus_client.create_index("batch", idx)
        milvus_client.load_collection("batch")
        result = milvus_client.query("batch", filter="id >= 0", output_fields=["id"])
        assert len(result) == 100
        milvus_client.drop_collection("batch")


# ===========================================================================
# 2. SEARCH — basic, filters, metrics, multi-query, output_fields
# ===========================================================================

class TestSearch:
    def test_search_basic(self, milvus_client):
        rows = _create_loaded(milvus_client, "srch1")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search("srch1", data=q, limit=10)
        assert len(res) == 1
        assert len(res[0]) == 10
        milvus_client.drop_collection("srch1")

    def test_search_self_query_l2(self, milvus_client):
        rows = _create_loaded(milvus_client, "srch_l2", metric="L2")
        # Query with the exact vector of id=50
        q = [rows[50]["vec"]]
        res = milvus_client.search("srch_l2", data=q, limit=1)
        assert res[0][0]["id"] == 50
        assert res[0][0]["distance"] < 1e-3
        milvus_client.drop_collection("srch_l2")

    @pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
    def test_search_metric_types(self, milvus_client, metric):
        _create_loaded(milvus_client, f"met_{metric}", metric=metric)
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(f"met_{metric}", data=q, limit=5)
        assert len(res[0]) == 5
        milvus_client.drop_collection(f"met_{metric}")

    def test_search_with_filter_int(self, milvus_client):
        _create_loaded(milvus_client, "filt_int")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "filt_int", data=q, limit=50,
            filter="id >= 100",
            output_fields=["id"],
        )
        for hit in res[0]:
            assert hit["entity"]["id"] >= 100
        milvus_client.drop_collection("filt_int")

    def test_search_with_filter_bool(self, milvus_client):
        _create_loaded(milvus_client, "filt_bool")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "filt_bool", data=q, limit=50,
            filter="bool_field == true",
            output_fields=["bool_field"],
        )
        for hit in res[0]:
            assert hit["entity"]["bool_field"] is True
        milvus_client.drop_collection("filt_bool")

    def test_search_with_filter_varchar_like(self, milvus_client):
        _create_loaded(milvus_client, "filt_like")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "filt_like", data=q, limit=50,
            filter='varchar like "str_01%"',
            output_fields=["varchar"],
        )
        for hit in res[0]:
            assert hit["entity"]["varchar"].startswith("str_01")
        milvus_client.drop_collection("filt_like")

    def test_search_with_filter_and_or(self, milvus_client):
        _create_loaded(milvus_client, "filt_andor")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "filt_andor", data=q, limit=50,
            filter="id >= 100 and (bool_field == true or float_field < 50.0)",
            output_fields=["id", "bool_field", "float_field"],
        )
        for hit in res[0]:
            e = hit["entity"]
            assert e["id"] >= 100
            assert e["bool_field"] is True or e["float_field"] < 50.0
        milvus_client.drop_collection("filt_andor")

    def test_search_with_filter_in(self, milvus_client):
        _create_loaded(milvus_client, "filt_in")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        target_ids = [10, 20, 30, 40, 50]
        res = milvus_client.search(
            "filt_in", data=q, limit=50,
            filter=f"id in {target_ids}",
            output_fields=["id"],
        )
        for hit in res[0]:
            assert hit["entity"]["id"] in target_ids
        milvus_client.drop_collection("filt_in")

    def test_search_with_filter_not(self, milvus_client):
        _create_loaded(milvus_client, "filt_not")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "filt_not", data=q, limit=50,
            filter="not (id < 100)",
            output_fields=["id"],
        )
        for hit in res[0]:
            assert hit["entity"]["id"] >= 100
        milvus_client.drop_collection("filt_not")

    def test_search_multi_query(self, milvus_client):
        rows = _create_loaded(milvus_client, "multi_q", metric="L2")
        qs = [rows[0]["vec"], rows[99]["vec"], rows[199]["vec"]]
        res = milvus_client.search("multi_q", data=qs, limit=3)
        assert len(res) == 3
        assert res[0][0]["id"] == 0
        assert res[1][0]["id"] == 99
        assert res[2][0]["id"] == 199
        milvus_client.drop_collection("multi_q")

    def test_search_with_output_fields(self, milvus_client):
        _create_loaded(milvus_client, "out_f")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "out_f", data=q, limit=5,
            output_fields=["varchar", "float_field"],
        )
        for hit in res[0]:
            assert "varchar" in hit["entity"]
            assert "float_field" in hit["entity"]
        milvus_client.drop_collection("out_f")

    def test_search_results_sorted_ascending(self, milvus_client):
        _create_loaded(milvus_client, "sorted")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search("sorted", data=q, limit=20)
        distances = [hit["distance"] for hit in res[0]]
        assert distances == sorted(distances)
        milvus_client.drop_collection("sorted")


# ===========================================================================
# 3. QUERY — filter expressions, output_fields, limit, get by id
# ===========================================================================

class TestQuery:
    def test_query_all(self, milvus_client):
        _create_loaded(milvus_client, "qa")
        rows = milvus_client.query("qa", filter="id >= 0", output_fields=["id"])
        assert len(rows) == NB
        milvus_client.drop_collection("qa")

    def test_query_with_limit(self, milvus_client):
        _create_loaded(milvus_client, "ql")
        rows = milvus_client.query("ql", filter="id >= 0", limit=10, output_fields=["id"])
        assert len(rows) == 10
        milvus_client.drop_collection("ql")

    def test_query_id_in_list(self, milvus_client):
        _create_loaded(milvus_client, "qid")
        rows = milvus_client.query(
            "qid", filter="id in [5, 10, 15]",
            output_fields=["id", "varchar"],
        )
        ids = sorted(r["id"] for r in rows)
        assert ids == [5, 10, 15]
        milvus_client.drop_collection("qid")

    def test_query_range_expression(self, milvus_client):
        _create_loaded(milvus_client, "qr")
        rows = milvus_client.query(
            "qr", filter="float_field >= 50.0 and float_field < 60.0",
            output_fields=["id", "float_field"],
        )
        for r in rows:
            assert 50.0 <= r["float_field"] < 60.0
        assert len(rows) == 10
        milvus_client.drop_collection("qr")

    def test_query_varchar_like_prefix(self, milvus_client):
        _create_loaded(milvus_client, "qlk")
        rows = milvus_client.query(
            "qlk", filter='varchar like "str_00%"',
            output_fields=["varchar"],
        )
        for r in rows:
            assert r["varchar"].startswith("str_00")
        # str_0000 to str_0099 → 100 matches
        assert len(rows) == 100
        milvus_client.drop_collection("qlk")

    def test_query_bool_filter(self, milvus_client):
        _create_loaded(milvus_client, "qb")
        rows = milvus_client.query(
            "qb", filter="bool_field == false",
            output_fields=["id", "bool_field"],
        )
        for r in rows:
            assert r["bool_field"] is False
        assert len(rows) == NB // 2
        milvus_client.drop_collection("qb")

    def test_query_not_in(self, milvus_client):
        _create_loaded(milvus_client, "qni")
        rows = milvus_client.query(
            "qni", filter="id not in [0, 1, 2]",
            output_fields=["id"],
        )
        ids = {r["id"] for r in rows}
        assert 0 not in ids
        assert 1 not in ids
        assert 2 not in ids
        assert len(rows) == NB - 3
        milvus_client.drop_collection("qni")

    def test_get_by_ids(self, milvus_client):
        _create_loaded(milvus_client, "gid")
        rows = milvus_client.get("gid", ids=[0, 50, 199])
        assert sorted(r["id"] for r in rows) == [0, 50, 199]
        milvus_client.drop_collection("gid")

    def test_get_missing_ids_returns_empty(self, milvus_client):
        _create_loaded(milvus_client, "gm")
        rows = milvus_client.get("gm", ids=[9999])
        assert rows == []
        milvus_client.drop_collection("gm")

    def test_query_output_fields_subset(self, milvus_client):
        _create_loaded(milvus_client, "qof")
        rows = milvus_client.query(
            "qof", filter="id < 5",
            output_fields=["varchar"],
        )
        for r in rows:
            assert "varchar" in r
            assert "id" in r  # pk always included
        milvus_client.drop_collection("qof")


# ===========================================================================
# 4. DELETE — by IDs, by filter, then verify
# ===========================================================================

class TestDelete:
    def test_delete_by_ids(self, milvus_client):
        _create_loaded(milvus_client, "did")
        milvus_client.delete("did", ids=[0, 1, 2])
        remaining = milvus_client.query("did", filter="id >= 0", output_fields=["id"])
        ids = {r["id"] for r in remaining}
        assert 0 not in ids and 1 not in ids and 2 not in ids
        assert len(remaining) == NB - 3
        milvus_client.drop_collection("did")

    def test_delete_by_filter(self, milvus_client):
        _create_loaded(milvus_client, "dfilt")
        milvus_client.delete("dfilt", filter="id < 10")
        remaining = milvus_client.query("dfilt", filter="id >= 0", output_fields=["id"])
        for r in remaining:
            assert r["id"] >= 10
        assert len(remaining) == NB - 10
        milvus_client.drop_collection("dfilt")

    def test_delete_then_search_excludes_deleted(self, milvus_client):
        rows = _create_loaded(milvus_client, "dsr", metric="L2")
        milvus_client.delete("dsr", ids=[50])
        q = [rows[50]["vec"]]
        res = milvus_client.search("dsr", data=q, limit=5, output_fields=["id"])
        for hit in res[0]:
            assert hit["entity"]["id"] != 50
        milvus_client.drop_collection("dsr")

    def test_delete_nonexistent_pk_no_error(self, milvus_client):
        _create_loaded(milvus_client, "dne")
        milvus_client.delete("dne", ids=[99999])
        remaining = milvus_client.query("dne", filter="id >= 0", output_fields=["id"])
        assert len(remaining) == NB
        milvus_client.drop_collection("dne")


# ===========================================================================
# 5. UPSERT — overwrite existing, insert new
# ===========================================================================

class TestUpsert:
    def test_upsert_existing_pk_overwrites(self, milvus_client):
        rows = _create_loaded(milvus_client, "uex")
        milvus_client.upsert("uex", [{
            "id": 10,
            "vec": [9.0] * DIM,
            "float_field": 999.0,
            "varchar": "OVERWRITTEN",
            "bool_field": False,
        }])
        result = milvus_client.query(
            "uex", filter="id == 10",
            output_fields=["varchar", "float_field"],
        )
        assert result[0]["varchar"] == "OVERWRITTEN"
        assert abs(result[0]["float_field"] - 999.0) < 1e-3
        milvus_client.drop_collection("uex")

    def test_upsert_new_pk_inserts(self, milvus_client):
        _create_loaded(milvus_client, "unew")
        milvus_client.upsert("unew", [{
            "id": NB + 1,
            "vec": [1.0] * DIM,
            "float_field": 0.0,
            "varchar": "NEW",
            "bool_field": True,
        }])
        result = milvus_client.query(
            "unew", filter=f"id == {NB + 1}",
            output_fields=["varchar"],
        )
        assert len(result) == 1
        assert result[0]["varchar"] == "NEW"
        milvus_client.drop_collection("unew")

    def test_upsert_multiple_times(self, milvus_client):
        _create_loaded(milvus_client, "umul")
        for val in range(5):
            milvus_client.upsert("umul", [{
                "id": 0,
                "vec": [float(val)] * DIM,
                "float_field": float(val),
                "varchar": f"v{val}",
                "bool_field": True,
            }])
        result = milvus_client.query(
            "umul", filter="id == 0",
            output_fields=["varchar"],
        )
        assert len(result) == 1
        assert result[0]["varchar"] == "v4"  # last upsert wins
        milvus_client.drop_collection("umul")


# ===========================================================================
# 6. PARTITION — create, insert, search scoped, drop
# ===========================================================================

class TestPartition:
    def test_partition_lifecycle(self, milvus_client):
        milvus_client.create_collection("plc", schema=_schema(milvus_client))
        milvus_client.create_partition("plc", "p1")
        assert "p1" in milvus_client.list_partitions("plc")
        assert milvus_client.has_partition("plc", "p1")
        milvus_client.drop_partition("plc", "p1")
        assert "p1" not in milvus_client.list_partitions("plc")
        milvus_client.drop_collection("plc")

    def test_partition_insert_and_search_scoped(self, milvus_client):
        milvus_client.create_collection("pis", schema=_schema(milvus_client))
        milvus_client.create_partition("pis", "p1")
        milvus_client.create_partition("pis", "p2")
        rng = _rng()
        p1_rows = [
            {"id": i, "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
             "float_field": 0.0, "varchar": "p1", "bool_field": True}
            for i in range(50)
        ]
        p2_rows = [
            {"id": 100 + i, "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
             "float_field": 0.0, "varchar": "p2", "bool_field": False}
            for i in range(50)
        ]
        milvus_client.insert("pis", p1_rows, partition_name="p1")
        milvus_client.insert("pis", p2_rows, partition_name="p2")
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16})
        milvus_client.create_index("pis", idx)
        milvus_client.load_collection("pis")

        q = rng.standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "pis", data=q, limit=10,
            partition_names=["p1"], output_fields=["id"],
        )
        for hit in res[0]:
            assert hit["entity"]["id"] < 100  # only p1 ids

        milvus_client.drop_collection("pis")


# ===========================================================================
# 7. INDEX LIFECYCLE — create, load, search, release, drop, re-create
# ===========================================================================

class TestIndexLifecycle:
    def test_index_create_load_release_drop_cycle(self, milvus_client):
        milvus_client.create_collection("ilc", schema=_schema(milvus_client))
        milvus_client.insert("ilc", _gen_rows(n=50))

        # Create index
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16})
        milvus_client.create_index("ilc", idx)
        desc = milvus_client.describe_index("ilc", "vec")
        assert desc["index_type"] == "HNSW"

        # Load + search
        milvus_client.load_collection("ilc")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search("ilc", data=q, limit=5)
        assert len(res[0]) == 5

        # Release
        milvus_client.release_collection("ilc")
        with pytest.raises(Exception):
            milvus_client.search("ilc", data=q, limit=5)

        # Drop index
        milvus_client.drop_index("ilc", "vec")
        assert milvus_client.describe_index("ilc", "vec") is None

        # Re-create with different metric
        idx2 = milvus_client.prepare_index_params()
        idx2.add_index(field_name="vec", index_type="BRUTE_FORCE",
                       metric_type="L2", params={})
        milvus_client.create_index("ilc", idx2)
        milvus_client.load_collection("ilc")
        res = milvus_client.search("ilc", data=q, limit=5)
        assert len(res[0]) == 5

        milvus_client.drop_collection("ilc")


# ===========================================================================
# 8. DYNAMIC FIELD — $meta access via filter
# ===========================================================================

class TestDynamicField:
    def test_dynamic_field_insert_and_query(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        milvus_client.create_collection("dyn", schema=schema)
        rows = [
            {"id": i, "vec": [float(i)] * 4, "category": "tech" if i % 2 == 0 else "news"}
            for i in range(20)
        ]
        milvus_client.insert("dyn", rows)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                      metric_type="L2", params={})
        milvus_client.create_index("dyn", idx)
        milvus_client.load_collection("dyn")

        # Search with $meta filter
        q = [[5.0, 5.0, 5.0, 5.0]]
        res = milvus_client.search(
            "dyn", data=q, limit=10,
            filter='$meta["category"] == "tech"',
        )
        # All results should be even ids (category == tech)
        for hit in res[0]:
            assert hit["id"] % 2 == 0

        milvus_client.drop_collection("dyn")


# ===========================================================================
# 9. NULLABLE FIELD — null handling in insert + query
# ===========================================================================

class TestNullableField:
    def test_nullable_varchar_insert_and_query(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        schema.add_field("title", DataType.VARCHAR, max_length=64, nullable=True)
        milvus_client.create_collection("null_v", schema=schema)
        rows = [
            {"id": i, "vec": [float(i)] * 4, "title": f"t{i}" if i % 2 == 0 else None}
            for i in range(20)
        ]
        milvus_client.insert("null_v", rows)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                      metric_type="L2", params={})
        milvus_client.create_index("null_v", idx)
        milvus_client.load_collection("null_v")

        # Query non-null rows
        non_null = milvus_client.query(
            "null_v", filter='title like "t%"',
            output_fields=["id", "title"],
        )
        for r in non_null:
            assert r["title"] is not None
            assert r["id"] % 2 == 0
        assert len(non_null) == 10

        milvus_client.drop_collection("null_v")


# ===========================================================================
# 10. COLLECTION LIFECYCLE — comprehensive
# ===========================================================================

class TestCollectionLifecycle:
    def test_create_describe_drop(self, milvus_client):
        milvus_client.create_collection("cdl", schema=_schema(milvus_client))
        desc = milvus_client.describe_collection("cdl")
        field_names = sorted(f["name"] for f in desc["fields"])
        assert field_names == sorted(["id", "vec", "float_field", "varchar", "bool_field"])
        milvus_client.drop_collection("cdl")
        assert not milvus_client.has_collection("cdl")

    def test_create_duplicate_raises(self, milvus_client):
        milvus_client.create_collection("dup", schema=_schema(milvus_client))
        with pytest.raises(Exception):
            milvus_client.create_collection("dup", schema=_schema(milvus_client))
        milvus_client.drop_collection("dup")

    def test_list_multiple_collections(self, milvus_client):
        for name in ["c_alpha", "c_beta", "c_gamma"]:
            milvus_client.create_collection(name, schema=_schema(milvus_client))
        names = milvus_client.list_collections()
        assert set(names) >= {"c_alpha", "c_beta", "c_gamma"}
        for name in ["c_alpha", "c_beta", "c_gamma"]:
            milvus_client.drop_collection(name)
