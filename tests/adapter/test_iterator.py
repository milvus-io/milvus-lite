"""Phase 16 — Iterator tests.

pymilvus query_iterator and search_iterator are client-side
implementations that wrap regular Query/Search RPCs. These tests
verify the underlying RPCs support the patterns iterators use.

Covers:
1. query_iterator: paginate all records in batches
2. query_iterator with filter
3. query_iterator collects all records without duplicates
4. search_iterator: paginate search results in batches
5. Engine: query(expr=None) returns all records
"""

import tempfile

import pytest

from pymilvus import DataType, MilvusClient


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------

class TestQueryAllRecords:
    """query(expr=None) and query(expr='') return all records."""

    def test_query_none_returns_all(self):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.engine.collection import Collection

        with tempfile.TemporaryDirectory() as d:
            schema = CollectionSchema(fields=[
                FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
                FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
            ])
            col = Collection(name="t", data_dir=d, schema=schema)
            col.insert([
                {"id": i, "vec": [float(i), 0, 0, 0]}
                for i in range(10)
            ])
            results = col.query(None)
            assert len(results) == 10

    def test_query_empty_string_returns_all(self):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.engine.collection import Collection

        with tempfile.TemporaryDirectory() as d:
            schema = CollectionSchema(fields=[
                FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
                FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
            ])
            col = Collection(name="t", data_dir=d, schema=schema)
            col.insert([
                {"id": i, "vec": [float(i), 0, 0, 0]}
                for i in range(5)
            ])
            results = col.query("")
            assert len(results) == 5

    def test_query_none_with_limit(self):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.engine.collection import Collection

        with tempfile.TemporaryDirectory() as d:
            schema = CollectionSchema(fields=[
                FieldSchema(name="id", dtype=LDT.INT64, is_primary=True),
                FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
            ])
            col = Collection(name="t", data_dir=d, schema=schema)
            col.insert([
                {"id": i, "vec": [float(i), 0, 0, 0]}
                for i in range(20)
            ])
            results = col.query(None, limit=5)
            assert len(results) == 5


# ---------------------------------------------------------------------------
# gRPC integration tests
# ---------------------------------------------------------------------------

def _setup_collection(client, name, n=20):
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("category", DataType.INT64)
    client.create_collection(name, schema=schema)
    client.insert(name, [
        {"id": i, "vec": [float(i % 4), float((i+1) % 4),
                          float((i+2) % 4), float((i+3) % 4)],
         "category": i % 3}
        for i in range(n)
    ])
    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    client.create_index(name, idx)
    client.load_collection(name)


def test_query_iterator_all_records(milvus_client):
    """query_iterator paginates through all records."""
    _setup_collection(milvus_client, "qi_all", n=25)

    it = milvus_client.query_iterator(
        "qi_all", batch_size=7, output_fields=["id"],
    )
    all_ids = set()
    batch_count = 0
    while True:
        batch = it.next()
        if not batch:
            break
        batch_count += 1
        for r in batch:
            all_ids.add(r["id"])
    it.close()

    assert len(all_ids) == 25  # no duplicates, all collected
    assert batch_count >= 4    # ceil(25/7) = 4 batches

    milvus_client.drop_collection("qi_all")


def test_query_iterator_with_filter(milvus_client):
    """query_iterator with filter only returns matching records."""
    _setup_collection(milvus_client, "qi_filt", n=30)

    it = milvus_client.query_iterator(
        "qi_filt", batch_size=5,
        filter="category == 0",
        output_fields=["id", "category"],
    )
    all_rows = []
    while True:
        batch = it.next()
        if not batch:
            break
        all_rows.extend(batch)
    it.close()

    assert len(all_rows) == 10  # 30 records, category 0 = every 3rd
    for r in all_rows:
        assert r["category"] == 0

    milvus_client.drop_collection("qi_filt")


def test_query_iterator_no_duplicates(milvus_client):
    """query_iterator should never return the same pk twice."""
    _setup_collection(milvus_client, "qi_nodup", n=50)

    it = milvus_client.query_iterator(
        "qi_nodup", batch_size=8, output_fields=["id"],
    )
    seen = set()
    while True:
        batch = it.next()
        if not batch:
            break
        for r in batch:
            assert r["id"] not in seen, f"duplicate id: {r['id']}"
            seen.add(r["id"])
    it.close()

    assert len(seen) == 50
    milvus_client.drop_collection("qi_nodup")


def test_search_iterator(milvus_client):
    """search_iterator paginates search results."""
    _setup_collection(milvus_client, "si_basic", n=20)

    it = milvus_client.search_iterator(
        "si_basic",
        data=[[1, 0, 0, 0]],
        batch_size=5,
        limit=15,
        search_params={"metric_type": "COSINE"},
        output_fields=["id"],
    )
    all_ids = []
    while True:
        batch = it.next()
        if not batch:
            break
        all_ids.extend([r["id"] for r in batch])
    it.close()

    assert len(all_ids) > 0
    assert len(all_ids) <= 15

    milvus_client.drop_collection("si_basic")
