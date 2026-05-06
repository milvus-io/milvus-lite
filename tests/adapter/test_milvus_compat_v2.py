"""Milvus compatibility tests v2 — advanced filter expressions and
nullable field handling.

Adapted from milvus-io/milvus python_client/milvus_client_v2/
test_milvus_client_e2e.py and test_milvus_client_search_v2.py.

Focuses on NEW patterns not covered by test_milvus_compat.py:
  - IS NULL / IS NOT NULL filters
  - Nullable field insert + query + search
  - Arithmetic in filter expressions (field * 2 > 1.0)
  - Complex compound expressions (3+ conditions with NULL checks)
  - NOT operator combinations
  - LIKE with suffix and inner match
  - Dynamic field ($meta) access in search filters
  - Mixed metric types across separate searches
  - Large batch insert + search
  - Delete-then-verify lifecycle

All tests go through pymilvus → gRPC → MilvusLite engine.
"""

import numpy as np
import pytest
from pymilvus import DataType, MilvusClient

from milvus_lite.index.factory import is_faiss_available

pytestmark = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss-cpu required for search tests"
)

DIM = 8
NB = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed=42):
    return np.random.default_rng(seed=seed)


def _nullable_schema(client):
    """Schema with multiple nullable scalar fields."""
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("int_val", DataType.INT64)
    schema.add_field("float_val", DataType.FLOAT, nullable=True)
    schema.add_field("title", DataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field("active", DataType.BOOL)
    schema.add_field("score", DataType.DOUBLE)
    return schema


def _gen_nullable_rows(n=NB):
    rng = _rng()
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
            "int_val": i,
            "float_val": float(i) / 10 if i % 3 != 0 else None,   # ~33% null
            "title": f"doc_{i:04d}" if i % 4 != 0 else None,       # ~25% null
            "active": (i % 2 == 0),
            "score": float(i) * 0.01,
        })
    return rows


def _create_loaded_nullable(client, name, rows=None, metric="COSINE"):
    rows = rows or _gen_nullable_rows()
    client.create_collection(name, schema=_nullable_schema(client))
    client.insert(name, rows)
    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="HNSW", metric_type=metric,
                  params={"M": 16, "efConstruction": 200})
    client.create_index(name, idx)
    client.load_collection(name)
    return rows


# ===========================================================================
# 1. IS NULL / IS NOT NULL filter expressions
# ===========================================================================

class TestNullFilters:
    def test_query_is_null(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "qn1")
        result = milvus_client.query(
            "qn1", filter="float_val is null",
            output_fields=["id", "float_val"],
        )
        for r in result:
            assert r["float_val"] is None
        # ~33% of rows have null float_val
        expected = sum(1 for r in rows if r["float_val"] is None)
        assert len(result) == expected
        milvus_client.drop_collection("qn1")

    def test_query_is_not_null(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "qn2")
        result = milvus_client.query(
            "qn2", filter="float_val is not null",
            output_fields=["id", "float_val"],
        )
        for r in result:
            assert r["float_val"] is not None
        expected = sum(1 for r in rows if r["float_val"] is not None)
        assert len(result) == expected
        milvus_client.drop_collection("qn2")

    def test_query_is_null_varchar(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "qn3")
        result = milvus_client.query(
            "qn3", filter="title is null",
            output_fields=["id", "title"],
        )
        for r in result:
            assert r["title"] is None
        expected = sum(1 for r in rows if r["title"] is None)
        assert len(result) == expected
        milvus_client.drop_collection("qn3")

    def test_search_with_is_null_filter(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "sn1")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "sn1", data=q, limit=50,
            filter="title is null",
            output_fields=["title"],
        )
        for hit in res[0]:
            assert hit["entity"]["title"] is None
        milvus_client.drop_collection("sn1")

    def test_search_with_is_not_null_filter(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "sn2")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "sn2", data=q, limit=50,
            filter="float_val is not null",
            output_fields=["float_val"],
        )
        for hit in res[0]:
            assert hit["entity"]["float_val"] is not None
        milvus_client.drop_collection("sn2")


# ===========================================================================
# 2. Complex compound expressions with NULL checks
# ===========================================================================

class TestCompoundNullExpressions:
    def test_null_and_comparison(self, milvus_client):
        """float_val is not null AND float_val > 5.0"""
        rows = _create_loaded_nullable(milvus_client, "cn1")
        result = milvus_client.query(
            "cn1",
            filter="float_val is not null and float_val > 5.0",
            output_fields=["id", "float_val"],
        )
        for r in result:
            assert r["float_val"] is not None
            assert r["float_val"] > 5.0
        expected = sum(1 for r in rows if r["float_val"] is not None and r["float_val"] > 5.0)
        assert len(result) == expected
        milvus_client.drop_collection("cn1")

    def test_null_and_like(self, milvus_client):
        """title is not null AND title like 'doc_00%'"""
        rows = _create_loaded_nullable(milvus_client, "cn2")
        result = milvus_client.query(
            "cn2",
            filter='title is not null and title like "doc_00%"',
            output_fields=["id", "title"],
        )
        for r in result:
            assert r["title"] is not None
            assert r["title"].startswith("doc_00")
        expected = sum(1 for r in rows
                       if r["title"] is not None and r["title"].startswith("doc_00"))
        assert len(result) == expected
        milvus_client.drop_collection("cn2")

    def test_multi_null_checks(self, milvus_client):
        """float_val is null AND title is not null"""
        rows = _create_loaded_nullable(milvus_client, "cn3")
        result = milvus_client.query(
            "cn3",
            filter="float_val is null and title is not null",
            output_fields=["id", "float_val", "title"],
        )
        for r in result:
            assert r["float_val"] is None
            assert r["title"] is not None
        expected = sum(1 for r in rows
                       if r["float_val"] is None and r["title"] is not None)
        assert len(result) == expected
        milvus_client.drop_collection("cn3")

    def test_null_with_bool_and_comparison(self, milvus_client):
        """active == true AND float_val is not null AND float_val > 3.0"""
        rows = _create_loaded_nullable(milvus_client, "cn4")
        result = milvus_client.query(
            "cn4",
            filter="active == true and float_val is not null and float_val > 3.0",
            output_fields=["id", "active", "float_val"],
        )
        for r in result:
            assert r["active"] is True
            assert r["float_val"] is not None
            assert r["float_val"] > 3.0
        expected = sum(1 for r in rows
                       if r["active"] is True
                       and r["float_val"] is not None
                       and r["float_val"] > 3.0)
        assert len(result) == expected
        milvus_client.drop_collection("cn4")

    def test_complex_or_with_null(self, milvus_client):
        """(float_val is null) OR (float_val > 8.0)"""
        rows = _create_loaded_nullable(milvus_client, "cn5")
        result = milvus_client.query(
            "cn5",
            filter="float_val is null or float_val > 8.0",
            output_fields=["id", "float_val"],
        )
        for r in result:
            assert r["float_val"] is None or r["float_val"] > 8.0
        expected = sum(1 for r in rows
                       if r["float_val"] is None or
                       (r["float_val"] is not None and r["float_val"] > 8.0))
        assert len(result) == expected
        milvus_client.drop_collection("cn5")


# ===========================================================================
# 3. Arithmetic in filter expressions
# ===========================================================================

class TestArithmeticFilters:
    def test_multiply_in_filter(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "ar1")
        result = milvus_client.query(
            "ar1",
            filter="score * 100 > 50",
            output_fields=["id", "score"],
        )
        for r in result:
            assert r["score"] * 100 > 50
        expected = sum(1 for r in rows if r["score"] * 100 > 50)
        assert len(result) == expected
        milvus_client.drop_collection("ar1")

    def test_add_in_filter(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "ar2")
        result = milvus_client.query(
            "ar2",
            filter="int_val + 10 > 50",
            output_fields=["id", "int_val"],
        )
        for r in result:
            assert r["int_val"] + 10 > 50
        expected = sum(1 for r in rows if r["int_val"] + 10 > 50)
        assert len(result) == expected
        milvus_client.drop_collection("ar2")

    def test_divide_in_filter(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "ar3")
        result = milvus_client.query(
            "ar3",
            filter="int_val / 10 >= 5",
            output_fields=["id", "int_val"],
        )
        for r in result:
            assert r["int_val"] / 10 >= 5
        expected = sum(1 for r in rows if r["int_val"] / 10 >= 5)
        assert len(result) == expected
        milvus_client.drop_collection("ar3")

    def test_arithmetic_in_search_filter(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "ar4")
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "ar4", data=q, limit=50,
            filter="score * 100 > 80",
            output_fields=["score"],
        )
        for hit in res[0]:
            assert hit["entity"]["score"] * 100 > 80
        milvus_client.drop_collection("ar4")


# ===========================================================================
# 4. LIKE patterns — prefix, suffix, inner match
# ===========================================================================

class TestLikePatterns:
    def test_like_prefix(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "lk1")
        result = milvus_client.query(
            "lk1",
            filter='title is not null and title like "doc_00%"',
            output_fields=["title"],
        )
        for r in result:
            assert r["title"].startswith("doc_00")
        milvus_client.drop_collection("lk1")

    def test_like_suffix(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "lk2")
        result = milvus_client.query(
            "lk2",
            filter='title is not null and title like "%01"',
            output_fields=["title"],
        )
        for r in result:
            assert r["title"].endswith("01")
        expected = sum(1 for r in rows
                       if r["title"] is not None and r["title"].endswith("01"))
        assert len(result) == expected
        milvus_client.drop_collection("lk2")

    def test_like_inner_match(self, milvus_client):
        """LIKE with inner match. Note: use %05% (no underscore) to
        test pure substring match. SQL LIKE's _ is a single-char
        wildcard, which makes assertions tricky."""
        rows = _create_loaded_nullable(milvus_client, "lk3")
        result = milvus_client.query(
            "lk3",
            filter='title is not null and title like "%0050%"',
            output_fields=["title"],
        )
        for r in result:
            assert "0050" in r["title"]
        expected = sum(1 for r in rows
                       if r["title"] is not None and "0050" in r["title"])
        assert len(result) == expected
        milvus_client.drop_collection("lk3")


# ===========================================================================
# 5. NOT operator
# ===========================================================================

class TestNotOperator:
    def test_not_equals(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "not1")
        result = milvus_client.query(
            "not1",
            filter="not (active == true)",
            output_fields=["id", "active"],
        )
        for r in result:
            assert r["active"] is not True
        expected = sum(1 for r in rows if not r["active"])
        assert len(result) == expected
        milvus_client.drop_collection("not1")

    def test_not_in(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "not2")
        result = milvus_client.query(
            "not2",
            filter="id not in [0, 1, 2, 3, 4]",
            output_fields=["id"],
        )
        for r in result:
            assert r["id"] not in [0, 1, 2, 3, 4]
        assert len(result) == NB - 5
        milvus_client.drop_collection("not2")

    def test_not_compound(self, milvus_client):
        rows = _create_loaded_nullable(milvus_client, "not3")
        result = milvus_client.query(
            "not3",
            filter="not (id < 50 and active == true)",
            output_fields=["id", "active"],
        )
        for r in result:
            assert not (r["id"] < 50 and r["active"] is True)
        expected = sum(1 for r in rows if not (r["id"] < 50 and r["active"] is True))
        assert len(result) == expected
        milvus_client.drop_collection("not3")


# ===========================================================================
# 6. Dynamic field ($meta) access in search filters
# ===========================================================================

class TestDynamicFieldSearch:
    def test_dynamic_field_search_filter(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        milvus_client.create_collection("dyn_s", schema=schema)
        rng = _rng()
        rows = [
            {"id": i, "vec": rng.standard_normal(4).astype(np.float32).tolist(),
             "category": ["tech", "news", "blog"][i % 3],
             "priority": i % 10}
            for i in range(60)
        ]
        milvus_client.insert("dyn_s", rows)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                      metric_type="L2", params={})
        milvus_client.create_index("dyn_s", idx)
        milvus_client.load_collection("dyn_s")

        q = rng.standard_normal((1, 4)).astype(np.float32).tolist()
        res = milvus_client.search(
            "dyn_s", data=q, limit=30,
            filter='$meta["category"] == "tech"',
        )
        for hit in res[0]:
            assert hit["id"] % 3 == 0  # tech = id % 3 == 0
        milvus_client.drop_collection("dyn_s")

    def test_dynamic_field_query(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        milvus_client.create_collection("dyn_q", schema=schema)
        rows = [
            {"id": i, "vec": [float(i)] * 4, "level": i % 5}
            for i in range(50)
        ]
        milvus_client.insert("dyn_q", rows)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                      metric_type="L2", params={})
        milvus_client.create_index("dyn_q", idx)
        milvus_client.load_collection("dyn_q")

        result = milvus_client.query(
            "dyn_q", filter='$meta["level"] > 3',
            output_fields=["id"],
        )
        for r in result:
            assert r["id"] % 5 == 4  # level 4 = id % 5 == 4
        assert len(result) == 10  # 10 rows where level == 4
        milvus_client.drop_collection("dyn_q")


# ===========================================================================
# 7. Full lifecycle — insert → search → query → delete → verify empty
# ===========================================================================

class TestFullLifecycle:
    def test_insert_search_delete_verify(self, milvus_client):
        """Adapted from test_milvus_client_e2e.py: full CRUD lifecycle
        including deletion verification via both query and search."""
        rows = _create_loaded_nullable(milvus_client, "e2e")

        # Search
        q = _rng().standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search("e2e", data=q, limit=10)
        assert len(res[0]) == 10

        # Query with various filters
        result = milvus_client.query(
            "e2e", filter="active == true",
            output_fields=["id", "active"],
        )
        assert all(r["active"] is True for r in result)

        # Delete all rows
        milvus_client.delete("e2e", filter="id >= 0")

        # Verify empty via query
        result = milvus_client.query(
            "e2e", filter="id >= 0", output_fields=["id"],
        )
        assert len(result) == 0

        # Verify empty via search
        res = milvus_client.search("e2e", data=q, limit=10)
        assert len(res[0]) == 0

        milvus_client.drop_collection("e2e")

    def test_large_batch_insert_search(self, milvus_client):
        """500 rows — exercises multi-flush + compaction boundaries."""
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tag", DataType.VARCHAR, max_length=32)
        milvus_client.create_collection("large", schema=schema)

        rng = _rng()
        data = [
            {"id": i, "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
             "tag": f"t{i % 10}"}
            for i in range(500)
        ]
        milvus_client.insert("large", data)
        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW",
                      metric_type="COSINE", params={"M": 16})
        milvus_client.create_index("large", idx)
        milvus_client.load_collection("large")

        # All rows inserted
        result = milvus_client.query(
            "large", filter="id >= 0", output_fields=["id"],
        )
        assert len(result) == 500

        # Search with filter
        q = rng.standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "large", data=q, limit=50,
            filter='tag == "t5"',
            output_fields=["tag"],
        )
        for hit in res[0]:
            assert hit["entity"]["tag"] == "t5"

        milvus_client.drop_collection("large")
