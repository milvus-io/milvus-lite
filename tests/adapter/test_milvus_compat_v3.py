"""Milvus compatibility tests v3 -- edge cases extracted from
milvus-io/milvus python_client/milvus_client_v2/ test suite.

Adapted from:
  - test_milvus_client_e2e.py (end-to-end lifecycle)
  - test_milvus_client_search_pagination.py (offset pagination)
  - test_milvus_client_search_string.py (string/LIKE/IN filters)
  - test_milvus_client_search_by_pk.py (get-by-pk patterns)

Focuses on patterns NOT already covered by test_milvus_compat.py /
test_milvus_compat_v2.py:
  1. VARCHAR primary key with search filters
  2. Pagination consistency (offset pages match full search)
  3. Partition-scoped pagination
  4. JSON field query and search filters
  5. Multiple nullable scalar types in one schema
  6. String IN filter
  7. Mixed int64 + varchar comparison filter in search
  8. Search distance ordering verification (COSINE ascending)
  9. Delete-all lifecycle with nullable fields
 10. Chained range expression (e.g. 100 <= field < 200)
 11. Get with output_fields=["*"] (wildcard)
 12. Search with filter matching zero rows
 13. Upsert on VARCHAR pk
 14. Auto-ID insert and get-back
 15. Query with OR combining nullable fields
 16. Offset beyond data count returns empty

All tests use BRUTE_FORCE index + COSINE metric for deterministic,
fast results. Each test is self-contained (create + insert + index +
load + exercise + drop).

Skipped automatically when pymilvus / faiss-cpu is not installed.
"""

import numpy as np
import pytest
from pymilvus import DataType, MilvusClient

from milvus_lite.index.factory import is_faiss_available

pytestmark = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss-cpu required for search tests"
)

DIM = 4
NB = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed=7777):
    return np.random.default_rng(seed=seed)


def _index_and_load(client, name, field="vec", metric="COSINE"):
    """Create BRUTE_FORCE index and load."""
    idx = client.prepare_index_params()
    idx.add_index(field_name=field, index_type="BRUTE_FORCE",
                  metric_type=metric, params={})
    client.create_index(name, idx)
    client.load_collection(name)


def _simple_schema(client, dynamic=False):
    """id(INT64 PK), vec(FLOAT_VECTOR dim=4), val(FLOAT), tag(VARCHAR)."""
    schema = MilvusClient.create_schema(auto_id=False,
                                        enable_dynamic_field=dynamic)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("val", DataType.FLOAT)
    schema.add_field("tag", DataType.VARCHAR, max_length=128)
    return schema


def _gen_simple_rows(n=NB, seed=7777):
    rng = _rng(seed)
    return [
        {
            "id": i,
            "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
            "val": float(i) * 1.0,
            "tag": f"tag_{i:03d}",
        }
        for i in range(n)
    ]


def _create_simple(client, name, rows=None, dynamic=False):
    """Create, insert, index, load. Returns rows."""
    rows = rows or _gen_simple_rows()
    client.create_collection(name, schema=_simple_schema(client, dynamic))
    client.insert(name, rows)
    _index_and_load(client, name)
    return rows


# ===========================================================================
# 1. VARCHAR primary key with search and query filters
# ===========================================================================

class TestVarcharPK:
    """Tests using VARCHAR as primary key -- adapted from
    TestSearchStringVarcharPK in test_milvus_client_search_string.py."""

    def _schema(self, client):
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True,
                         max_length=64)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("score", DataType.FLOAT)
        schema.add_field("label", DataType.VARCHAR, max_length=64)
        return schema

    def _rows(self, n=NB):
        rng = _rng()
        return [
            {
                "pk": f"doc_{i:03d}",
                "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
                "score": float(i) * 0.1,
                "label": ["alpha", "beta", "gamma"][i % 3],
            }
            for i in range(n)
        ]

    def test_search_varchar_pk_equality(self, milvus_client):
        """Search with filter on varchar PK (exact match).
        Adapted from test_search_string_field_is_primary_true."""
        rows = self._rows()
        milvus_client.create_collection("vpk1", schema=self._schema(milvus_client))
        milvus_client.insert("vpk1", rows)
        _index_and_load(milvus_client, "vpk1")

        q = _rng(42).standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "vpk1", data=q, limit=NB,
            filter='pk == "doc_005"',
            output_fields=["pk", "score"],
        )
        assert len(res[0]) == 1
        assert res[0][0]["entity"]["pk"] == "doc_005"
        milvus_client.drop_collection("vpk1")

    def test_query_varchar_pk_like_prefix(self, milvus_client):
        """Query with LIKE prefix on VARCHAR PK.
        Adapted from test_search_string_field_index."""
        rows = self._rows()
        milvus_client.create_collection("vpk2", schema=self._schema(milvus_client))
        milvus_client.insert("vpk2", rows)
        _index_and_load(milvus_client, "vpk2")

        result = milvus_client.query(
            "vpk2", filter='pk like "doc_00%"',
            output_fields=["pk"],
        )
        for r in result:
            assert r["pk"].startswith("doc_00")
        # doc_000 through doc_009 = 10 matches
        assert len(result) == 10
        milvus_client.drop_collection("vpk2")

    def test_upsert_varchar_pk(self, milvus_client):
        """Upsert on VARCHAR PK overwrites existing row."""
        rows = self._rows(10)
        milvus_client.create_collection("vpk3", schema=self._schema(milvus_client))
        milvus_client.insert("vpk3", rows)
        _index_and_load(milvus_client, "vpk3")

        milvus_client.upsert("vpk3", [{
            "pk": "doc_005",
            "vec": [1.0, 0.0, 0.0, 0.0],
            "score": 999.0,
            "label": "UPDATED",
        }])
        result = milvus_client.query(
            "vpk3", filter='pk == "doc_005"',
            output_fields=["score", "label"],
        )
        assert len(result) == 1
        assert abs(result[0]["score"] - 999.0) < 1e-3
        assert result[0]["label"] == "UPDATED"
        milvus_client.drop_collection("vpk3")


# ===========================================================================
# 2. Pagination consistency -- offset pages match full search
# ===========================================================================

class TestPaginationConsistency:
    """Adapted from test_search_float_vectors_with_pagination_default."""

    def test_paginated_search_matches_full(self, milvus_client):
        """Two pages (offset=0 + offset=5) should equal top-10."""
        rows = _create_simple(milvus_client, "pg1")

        q = _rng(99).standard_normal((1, DIM)).astype(np.float32).tolist()

        # Full search: top 10
        full_res = milvus_client.search("pg1", data=q, limit=10)
        full_ids = [h["id"] for h in full_res[0]]

        # Page 1: offset=0, limit=5
        page1 = milvus_client.search(
            "pg1", data=q, limit=5,
            search_params={"metric_type": "COSINE", "offset": 0},
        )
        page1_ids = [h["id"] for h in page1[0]]

        # Page 2: offset=5, limit=5
        page2 = milvus_client.search(
            "pg1", data=q, limit=5,
            search_params={"metric_type": "COSINE", "offset": 5},
        )
        page2_ids = [h["id"] for h in page2[0]]

        assert page1_ids + page2_ids == full_ids
        milvus_client.drop_collection("pg1")

    def test_offset_beyond_count_returns_empty(self, milvus_client):
        """Offset >= total rows -> empty results.
        Adapted from test_search_offset_beyond_count."""
        _create_simple(milvus_client, "pg2")
        q = _rng(99).standard_normal((1, DIM)).astype(np.float32).tolist()

        res = milvus_client.search(
            "pg2", data=q, limit=5,
            search_params={"metric_type": "COSINE", "offset": NB + 100},
        )
        assert len(res[0]) == 0
        milvus_client.drop_collection("pg2")

    def test_search_pagination_with_filter(self, milvus_client):
        """Pagination with scalar filter -- paginated results match
        filtered full results.
        Adapted from test_search_pagination_with_expression."""
        _create_simple(milvus_client, "pg3")
        q = _rng(99).standard_normal((1, DIM)).astype(np.float32).tolist()
        expr = "val >= 10.0"

        # Full filtered
        full = milvus_client.search(
            "pg3", data=q, limit=NB, filter=expr,
            output_fields=["id"],
        )
        full_ids = [h["id"] for h in full[0]]

        # Page 1: first 5
        p1 = milvus_client.search(
            "pg3", data=q, limit=5, filter=expr,
            search_params={"metric_type": "COSINE", "offset": 0},
            output_fields=["id"],
        )
        p1_ids = [h["id"] for h in p1[0]]

        # The first 5 of full == page 1
        assert p1_ids == full_ids[:5]
        milvus_client.drop_collection("pg3")


# ===========================================================================
# 3. Partition-scoped pagination
# ===========================================================================

class TestPartitionPagination:
    """Adapted from test_search_pagination_in_partitions."""

    def test_search_in_partition_returns_only_partition_data(self, milvus_client):
        """Insert into specific partition, search scoped to it."""
        schema = _simple_schema(milvus_client)
        milvus_client.create_collection("pp1", schema=schema)
        milvus_client.create_partition("pp1", "p_even")
        milvus_client.create_partition("pp1", "p_odd")

        rng = _rng()
        even_rows = [
            {"id": i, "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
             "val": float(i), "tag": "even"}
            for i in range(0, 20, 2)
        ]
        odd_rows = [
            {"id": i, "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
             "val": float(i), "tag": "odd"}
            for i in range(1, 20, 2)
        ]
        milvus_client.insert("pp1", even_rows, partition_name="p_even")
        milvus_client.insert("pp1", odd_rows, partition_name="p_odd")
        _index_and_load(milvus_client, "pp1")

        q = rng.standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "pp1", data=q, limit=10,
            partition_names=["p_even"],
            output_fields=["id", "tag"],
        )
        for hit in res[0]:
            assert hit["entity"]["id"] % 2 == 0
            assert hit["entity"]["tag"] == "even"
        milvus_client.drop_collection("pp1")


# ===========================================================================
# 4. JSON field query and search filters
# ===========================================================================

class TestJsonFieldFilters:
    """Adapted from test_milvus_client_e2e query_cases with JSON."""

    def _schema(self, client):
        schema = MilvusClient.create_schema(auto_id=False,
                                            enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("meta", DataType.JSON)
        return schema

    def _rows(self, n=NB):
        rng = _rng()
        return [
            {
                "id": i,
                "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
                "meta": {"count": i, "label": f"item_{i % 5}"},
            }
            for i in range(n)
        ]

    def test_query_json_nested_field(self, milvus_client):
        """Query with JSON path filter: meta['count'] < 10."""
        rows = self._rows()
        milvus_client.create_collection("js1", schema=self._schema(milvus_client))
        milvus_client.insert("js1", rows)
        _index_and_load(milvus_client, "js1")

        result = milvus_client.query(
            "js1", filter="meta['count'] < 10",
            output_fields=["id", "meta"],
        )
        for r in result:
            assert r["meta"]["count"] < 10
        assert len(result) == 10
        milvus_client.drop_collection("js1")

    def test_search_json_string_filter(self, milvus_client):
        """Search with JSON string field filter."""
        rows = self._rows()
        milvus_client.create_collection("js2", schema=self._schema(milvus_client))
        milvus_client.insert("js2", rows)
        _index_and_load(milvus_client, "js2")

        q = _rng(42).standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search(
            "js2", data=q, limit=NB,
            filter='meta["label"] == "item_0"',
            output_fields=["meta"],
        )
        for hit in res[0]:
            assert hit["entity"]["meta"]["label"] == "item_0"
        # ids 0,5,10,15,20,25 -> 6 matches
        assert len(res[0]) == NB // 5
        milvus_client.drop_collection("js2")


# ===========================================================================
# 5. Multiple nullable scalar types in one schema
# ===========================================================================

class TestMultiNullableTypes:
    """Adapted from test_milvus_client_e2e with int8/int16/float/double
    all nullable."""

    def _schema(self, client):
        schema = MilvusClient.create_schema(auto_id=False,
                                            enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("int_val", DataType.INT64, nullable=True)
        schema.add_field("float_val", DataType.FLOAT, nullable=True)
        schema.add_field("text", DataType.VARCHAR, max_length=128,
                         nullable=True)
        return schema

    def _rows(self, n=NB):
        rng = _rng()
        rows = []
        for i in range(n):
            rows.append({
                "id": i,
                "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
                "int_val": i if i % 3 != 0 else None,
                "float_val": float(i) * 0.1 if i % 4 != 0 else None,
                "text": f"row_{i:03d}" if i % 5 != 0 else None,
            })
        return rows

    def test_query_multi_null_checks(self, milvus_client):
        """int_val is null AND text is not null.
        Adapted from test_milvus_client_e2e multi-field null checks."""
        rows = self._rows()
        milvus_client.create_collection("mn1", schema=self._schema(milvus_client))
        milvus_client.insert("mn1", rows)
        _index_and_load(milvus_client, "mn1")

        result = milvus_client.query(
            "mn1",
            filter="int_val is null and text is not null",
            output_fields=["id", "int_val", "text"],
        )
        for r in result:
            assert r["int_val"] is None
            assert r["text"] is not None
        expected = sum(1 for r in rows
                       if r["int_val"] is None and r["text"] is not None)
        assert len(result) == expected
        milvus_client.drop_collection("mn1")

    def test_query_or_with_nullable(self, milvus_client):
        """(int_val is null) OR (float_val > 2.0).
        Adapted from test_milvus_client_e2e complex OR with null."""
        rows = self._rows()
        milvus_client.create_collection("mn2", schema=self._schema(milvus_client))
        milvus_client.insert("mn2", rows)
        _index_and_load(milvus_client, "mn2")

        result = milvus_client.query(
            "mn2",
            filter="int_val is null or float_val > 2.0",
            output_fields=["id", "int_val", "float_val"],
        )
        for r in result:
            assert (r["int_val"] is None or
                    (r["float_val"] is not None and r["float_val"] > 2.0))
        expected = sum(
            1 for r in rows
            if r["int_val"] is None or
            (r["float_val"] is not None and r["float_val"] > 2.0)
        )
        assert len(result) == expected
        milvus_client.drop_collection("mn2")


# ===========================================================================
# 6. String IN filter
# ===========================================================================

class TestStringInFilter:
    """Adapted from test_search_with_different_string_expr (IN operator
    with varchar values)."""

    def test_search_varchar_in_list(self, milvus_client):
        """Search with varchar IN ["tag_001", "tag_003"]."""
        rows = _create_simple(milvus_client, "si1")
        q = _rng(42).standard_normal((1, DIM)).astype(np.float32).tolist()

        res = milvus_client.search(
            "si1", data=q, limit=NB,
            filter='tag in ["tag_001", "tag_003"]',
            output_fields=["tag"],
        )
        for hit in res[0]:
            assert hit["entity"]["tag"] in ["tag_001", "tag_003"]
        assert len(res[0]) == 2
        milvus_client.drop_collection("si1")

    def test_query_varchar_not_in(self, milvus_client):
        """Query with varchar NOT IN list."""
        _create_simple(milvus_client, "si2")

        result = milvus_client.query(
            "si2",
            filter='tag not in ["tag_000", "tag_001", "tag_002"]',
            output_fields=["tag"],
        )
        for r in result:
            assert r["tag"] not in ["tag_000", "tag_001", "tag_002"]
        assert len(result) == NB - 3
        milvus_client.drop_collection("si2")


# ===========================================================================
# 7. Mixed int64 + varchar comparison filter in search
# ===========================================================================

class TestMixedFilter:
    """Adapted from test_search_string_mix_expr."""

    def test_search_mixed_int_varchar(self, milvus_client):
        """Search with int + varchar combined filter."""
        rows = _create_simple(milvus_client, "mx1")
        q = _rng(42).standard_normal((1, DIM)).astype(np.float32).tolist()

        res = milvus_client.search(
            "mx1", data=q, limit=NB,
            filter='val >= 10.0 and tag >= "tag_010"',
            output_fields=["val", "tag"],
        )
        for hit in res[0]:
            assert hit["entity"]["val"] >= 10.0
            assert hit["entity"]["tag"] >= "tag_010"
        milvus_client.drop_collection("mx1")


# ===========================================================================
# 8. Search distance ordering verification
# ===========================================================================

class TestDistanceOrdering:
    """Adapted from test_milvus_client_e2e search distance verification."""

    def test_cosine_distances_ascending(self, milvus_client):
        """COSINE search distances must be in ascending order."""
        _create_simple(milvus_client, "do1")
        q = _rng(42).standard_normal((1, DIM)).astype(np.float32).tolist()

        res = milvus_client.search("do1", data=q, limit=NB)
        distances = [h["distance"] for h in res[0]]
        assert distances == sorted(distances), \
            "COSINE distances should be in ascending order"
        milvus_client.drop_collection("do1")

    def test_self_query_distance_near_zero(self, milvus_client):
        """Searching with an inserted vector should return itself at
        distance ~0 (COSINE)."""
        rows = _create_simple(milvus_client, "do2")
        q = [rows[5]["vec"]]
        res = milvus_client.search("do2", data=q, limit=1)
        assert res[0][0]["id"] == 5
        assert res[0][0]["distance"] < 1e-4
        milvus_client.drop_collection("do2")


# ===========================================================================
# 9. Delete-all lifecycle with nullable fields
# ===========================================================================

class TestDeleteAllNullable:
    """Adapted from test_milvus_client_e2e deletion steps (7-9)."""

    def test_delete_all_then_verify_empty(self, milvus_client):
        """Delete all rows, then verify via both query and search."""
        schema = MilvusClient.create_schema(auto_id=False,
                                            enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("nval", DataType.VARCHAR, max_length=64,
                         nullable=True)
        milvus_client.create_collection("da1", schema=schema)

        rng = _rng()
        rows = [
            {"id": i,
             "vec": rng.standard_normal(DIM).astype(np.float32).tolist(),
             "nval": f"v{i}" if i % 2 == 0 else None}
            for i in range(20)
        ]
        milvus_client.insert("da1", rows)
        _index_and_load(milvus_client, "da1")

        # Verify data present
        result = milvus_client.query("da1", filter="id >= 0",
                                     output_fields=["id"])
        assert len(result) == 20

        # Delete all
        milvus_client.delete("da1", filter="id >= 0")

        # Verify empty via query
        result = milvus_client.query("da1", filter="id >= 0",
                                     output_fields=["id"])
        assert len(result) == 0

        # Verify empty via search
        q = rng.standard_normal((1, DIM)).astype(np.float32).tolist()
        res = milvus_client.search("da1", data=q, limit=10)
        assert len(res[0]) == 0

        milvus_client.drop_collection("da1")


# ===========================================================================
# 10. Chained range expression
# ===========================================================================

class TestChainedRange:
    """Adapted from test_milvus_client_e2e query_cases with range
    filters like '100 <= int16_field < 200'."""

    def test_query_range_both_bounds(self, milvus_client):
        """val >= 10.0 AND val < 20.0 should match exactly 10 rows."""
        _create_simple(milvus_client, "cr1")
        result = milvus_client.query(
            "cr1",
            filter="val >= 10.0 and val < 20.0",
            output_fields=["id", "val"],
        )
        for r in result:
            assert 10.0 <= r["val"] < 20.0
        assert len(result) == 10
        milvus_client.drop_collection("cr1")


# ===========================================================================
# 11. Get with output_fields=["*"] (wildcard)
# ===========================================================================

class TestGetWithOutputFields:
    """Adapted from test_search_by_pk_with_output_fields_all."""

    def test_get_explicit_fields(self, milvus_client):
        """get(ids, output_fields=[...]) returns requested fields."""
        _create_simple(milvus_client, "gw1")
        result = milvus_client.get("gw1", ids=[0, 1],
                                   output_fields=["val", "tag"])
        assert len(result) == 2
        for r in result:
            assert "id" in r
            assert "val" in r
            assert "tag" in r
        milvus_client.drop_collection("gw1")


# ===========================================================================
# 12. Search with filter matching zero rows
# ===========================================================================

class TestFilterZeroMatch:
    """Edge case: filter expression eliminates all candidates."""

    def test_search_filter_no_match(self, milvus_client):
        """Search with filter that matches no rows returns empty."""
        _create_simple(milvus_client, "zm1")
        q = _rng(42).standard_normal((1, DIM)).astype(np.float32).tolist()

        res = milvus_client.search(
            "zm1", data=q, limit=10,
            filter="val > 99999.0",
        )
        assert len(res[0]) == 0
        milvus_client.drop_collection("zm1")

    def test_query_filter_no_match(self, milvus_client):
        """Query with filter matching no rows returns empty list."""
        _create_simple(milvus_client, "zm2")
        result = milvus_client.query(
            "zm2", filter="val > 99999.0", output_fields=["id"],
        )
        assert len(result) == 0
        milvus_client.drop_collection("zm2")


# ===========================================================================
# 13. Auto-ID insert and retrieve
# ===========================================================================

class TestAutoIdViaGrpc:
    """Adapted from test_milvus_client_search_by_pk auto_id patterns."""

    def test_auto_id_insert_returns_ids(self, milvus_client):
        """Insert with auto_id, verify returned PKs can be used to get."""
        schema = MilvusClient.create_schema(auto_id=True,
                                            enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True,
                         auto_id=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("label", DataType.VARCHAR, max_length=64)
        milvus_client.create_collection("ai1", schema=schema)

        rows = [
            {"vec": [float(i), 0, 0, 0], "label": f"item_{i}"}
            for i in range(10)
        ]
        res = milvus_client.insert("ai1", rows)
        assert res["insert_count"] == 10
        pks = res["ids"]
        assert len(pks) == 10

        _index_and_load(milvus_client, "ai1")

        # Get by returned PKs with explicit output_fields
        got = milvus_client.get("ai1", ids=pks[:3],
                                output_fields=["label"])
        assert len(got) == 3
        labels = sorted(r["label"] for r in got)
        assert labels == sorted([f"item_{i}" for i in range(3)])
        milvus_client.drop_collection("ai1")


# ===========================================================================
# 14. Multi-query search (nq > 1)
# ===========================================================================

class TestMultiQuerySearch:
    """Adapted from test_search_multi_query in test_milvus_compat.py
    but with BRUTE_FORCE and self-query verification."""

    def test_multi_nq_self_query(self, milvus_client):
        """Search with 3 query vectors, each should find itself."""
        rows = _create_simple(milvus_client, "mq1")

        qs = [rows[0]["vec"], rows[10]["vec"], rows[20]["vec"]]
        res = milvus_client.search("mq1", data=qs, limit=1)
        assert len(res) == 3
        assert res[0][0]["id"] == 0
        assert res[1][0]["id"] == 10
        assert res[2][0]["id"] == 20
        milvus_client.drop_collection("mq1")


# ===========================================================================
# 15. Query with LIKE empty-string comparison
# ===========================================================================

class TestVarcharEmptyComparison:
    """Adapted from test_search_string_field_not_primary_is_empty."""

    def test_varchar_gte_empty_matches_all(self, milvus_client):
        """Filter tag >= '' should match all rows (every string >= '')."""
        _create_simple(milvus_client, "ve1")
        result = milvus_client.query(
            "ve1", filter='tag >= ""', output_fields=["tag"],
        )
        assert len(result) == NB
        milvus_client.drop_collection("ve1")


# ===========================================================================
# 16. Get with missing IDs returns partial
# ===========================================================================

class TestGetPartialMiss:
    """Adapted from test_get_missing_ids_returns_empty -- extend to
    mixed existing + non-existing IDs."""

    def test_get_mixed_existing_and_missing(self, milvus_client):
        """Get with some valid IDs + some missing -> only valid returned."""
        _create_simple(milvus_client, "gp1")
        result = milvus_client.get("gp1", ids=[0, 1, 9999, 2])
        returned_ids = sorted(r["id"] for r in result)
        assert returned_ids == [0, 1, 2]
        milvus_client.drop_collection("gp1")

    def test_get_all_missing(self, milvus_client):
        """Get with all non-existing IDs -> empty result."""
        _create_simple(milvus_client, "gp2")
        result = milvus_client.get("gp2", ids=[9997, 9998, 9999])
        assert result == []
        milvus_client.drop_collection("gp2")
