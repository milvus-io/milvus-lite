"""Phase 15 — Auto ID tests.

Covers:
1. Engine: auto_id schema validation
2. Engine: insert without pk field, IDs auto-generated
3. Engine: multiple inserts produce strictly increasing IDs
4. Engine: auto-generated IDs survive flush + restart
5. gRPC: pymilvus create_collection(auto_id=True) + insert
6. gRPC: returned IDs are correct
7. gRPC: search/query on auto_id collection
"""

import tempfile

import pytest

from pymilvus import DataType, MilvusClient


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------

class TestAutoIdEngine:
    def _make_collection(self, tmpdir, auto_id=True):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.engine.collection import Collection

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=LDT.INT64, is_primary=True, auto_id=auto_id),
            FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
            FieldSchema(name="label", dtype=LDT.VARCHAR),
        ])
        return Collection(name="test_auto", data_dir=tmpdir, schema=schema)

    def test_insert_without_pk(self):
        """Insert records without pk field — IDs auto-generated."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            pks = col.insert([
                {"vec": [1, 0, 0, 0], "label": "a"},
                {"vec": [0, 1, 0, 0], "label": "b"},
            ])
            assert len(pks) == 2
            assert all(isinstance(pk, int) for pk in pks)
            assert pks[0] < pks[1]  # strictly increasing

    def test_multiple_inserts_increasing(self):
        """Multiple insert calls produce globally increasing IDs."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            pks1 = col.insert([{"vec": [1, 0, 0, 0], "label": "a"}])
            pks2 = col.insert([{"vec": [0, 1, 0, 0], "label": "b"}])
            pks3 = col.insert([{"vec": [0, 0, 1, 0], "label": "c"}])
            all_pks = pks1 + pks2 + pks3
            for i in range(len(all_pks) - 1):
                assert all_pks[i] < all_pks[i + 1]

    def test_get_by_auto_id(self):
        """Can retrieve records by auto-generated ID."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            pks = col.insert([
                {"vec": [1, 0, 0, 0], "label": "hello"},
            ])
            results = col.get(pks)
            assert len(results) == 1
            assert results[0]["label"] == "hello"
            assert results[0]["id"] == pks[0]

    def test_search_on_auto_id_collection(self):
        """Search works on auto_id collection."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            pks = col.insert([
                {"vec": [1, 0, 0, 0], "label": "a"},
                {"vec": [0, 1, 0, 0], "label": "b"},
            ])
            results = col.search(
                [[1, 0, 0, 0]], top_k=2, output_fields=["label"],
            )
            assert len(results[0]) == 2
            assert results[0][0]["id"] in pks

    def test_insert_with_pk_still_works(self):
        """User can still provide pk if they want (overrides auto)."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            pks = col.insert([
                {"id": 999, "vec": [1, 0, 0, 0], "label": "manual"},
            ])
            assert pks == [999]

    def test_flush_preserves_auto_ids(self):
        """Auto-generated IDs survive flush."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            pks = col.insert([
                {"vec": [1, 0, 0, 0], "label": "flushed"},
            ])
            col._trigger_flush()
            results = col.get(pks)
            assert len(results) == 1
            assert results[0]["id"] == pks[0]

    def test_auto_id_only_on_int64(self):
        """auto_id on VARCHAR pk raises error."""
        from milvus_lite.schema.types import (
            CollectionSchema, DataType as LDT, FieldSchema,
        )
        from milvus_lite.exceptions import SchemaValidationError

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=LDT.VARCHAR, is_primary=True, auto_id=True),
            FieldSchema(name="vec", dtype=LDT.FLOAT_VECTOR, dim=4),
        ])
        with pytest.raises(SchemaValidationError, match="INT64"):
            from milvus_lite.engine.collection import Collection
            Collection(name="bad", data_dir="/tmp/bad", schema=schema)

    def test_num_entities_with_auto_id(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([
                {"vec": [1, 0, 0, 0], "label": "a"},
                {"vec": [0, 1, 0, 0], "label": "b"},
                {"vec": [0, 0, 1, 0], "label": "c"},
            ])
            assert col.num_entities == 3


# ---------------------------------------------------------------------------
# gRPC integration tests
# ---------------------------------------------------------------------------

def test_grpc_auto_id_create_and_insert(milvus_client):
    """pymilvus create_collection(auto_id=True) + insert without pk."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("label", DataType.VARCHAR, max_length=100)
    milvus_client.create_collection("auto_id_test", schema=schema)

    result = milvus_client.insert("auto_id_test", [
        {"vec": [1, 0, 0, 0], "label": "first"},
        {"vec": [0, 1, 0, 0], "label": "second"},
        {"vec": [0, 0, 1, 0], "label": "third"},
    ])
    assert result["insert_count"] == 3
    # pymilvus returns the generated IDs in result["ids"]
    ids = result["ids"]
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)

    milvus_client.drop_collection("auto_id_test")


def test_grpc_auto_id_search(milvus_client):
    """Search on auto_id collection via pymilvus."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    milvus_client.create_collection("auto_search", schema=schema)

    milvus_client.insert("auto_search", [
        {"vec": [1, 0, 0, 0]},
        {"vec": [0, 1, 0, 0]},
        {"vec": [0, 0, 1, 0]},
    ])

    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    milvus_client.create_index("auto_search", idx)
    milvus_client.load_collection("auto_search")

    results = milvus_client.search(
        "auto_search", data=[[1, 0, 0, 0]], limit=3,
    )
    assert len(results[0]) == 3

    milvus_client.drop_collection("auto_search")


def test_grpc_auto_id_query(milvus_client):
    """Query on auto_id collection via pymilvus."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("tag", DataType.INT64)
    milvus_client.create_collection("auto_query", schema=schema)

    result = milvus_client.insert("auto_query", [
        {"vec": [1, 0, 0, 0], "tag": 10},
        {"vec": [0, 1, 0, 0], "tag": 20},
    ])
    auto_ids = result["ids"]

    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    milvus_client.create_index("auto_query", idx)
    milvus_client.load_collection("auto_query")

    # Query by filter
    rows = milvus_client.query("auto_query", filter="tag == 10",
                               output_fields=["tag"], limit=10)
    assert len(rows) == 1
    assert rows[0]["tag"] == 10
    assert rows[0]["id"] == auto_ids[0]

    milvus_client.drop_collection("auto_query")


def test_grpc_auto_id_ids_increasing(milvus_client):
    """Multiple inserts produce increasing IDs."""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    milvus_client.create_collection("auto_incr", schema=schema)

    r1 = milvus_client.insert("auto_incr", [{"vec": [1, 0, 0, 0]}])
    r2 = milvus_client.insert("auto_incr", [{"vec": [0, 1, 0, 0]}])
    assert r1["ids"][0] < r2["ids"][0]

    milvus_client.drop_collection("auto_incr")
