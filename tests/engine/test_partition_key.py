"""Partition key — auto-bucket routing tests.

When a schema field has is_partition_key=True, the collection auto-creates
N bucket partitions and routes records by hashing the partition key value.
"""

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.constants import PARTITION_KEY_BUCKET_PREFIX, DEFAULT_NUM_PARTITIONS
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def pk_schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="tenant", dtype=DataType.VARCHAR, max_length=64,
                    is_partition_key=True),
    ])


@pytest.fixture
def pk_int_schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="group_id", dtype=DataType.INT64,
                    is_partition_key=True),
    ])


def test_auto_creates_bucket_partitions(tmp_path, pk_schema):
    col = Collection("test", str(tmp_path / "data"), pk_schema)
    parts = col.list_partitions()
    bucket_parts = [p for p in parts if p.startswith(PARTITION_KEY_BUCKET_PREFIX)]
    assert len(bucket_parts) == DEFAULT_NUM_PARTITIONS
    col.close()


def test_insert_routes_by_partition_key(tmp_path, pk_schema):
    col = Collection("test", str(tmp_path / "data"), pk_schema)
    col.insert([
        {"id": 1, "vec": [1, 0, 0, 0], "tenant": "alpha"},
        {"id": 2, "vec": [0, 1, 0, 0], "tenant": "beta"},
        {"id": 3, "vec": [0, 0, 1, 0], "tenant": "alpha"},
    ])
    col.load()
    # All records accessible via query (searches all partitions)
    res = col.query(expr=None, limit=10)
    assert len(res) == 3
    col.close()


def test_same_key_same_partition(tmp_path, pk_schema):
    """Records with the same partition key value should land in the same bucket."""
    col = Collection("test", str(tmp_path / "data"), pk_schema)
    col.insert([
        {"id": 1, "vec": [1, 0, 0, 0], "tenant": "alpha"},
        {"id": 2, "vec": [0, 1, 0, 0], "tenant": "alpha"},
    ])
    col.flush()
    # Both should be in the same bucket partition
    # Check by querying each bucket
    col.load()
    for i in range(DEFAULT_NUM_PARTITIONS):
        bucket = f"{PARTITION_KEY_BUCKET_PREFIX}{i}"
        res = col.query(expr=None, partition_names=[bucket], limit=10)
        if res:
            ids = {r["id"] for r in res}
            # Both alpha records should be together
            if 1 in ids:
                assert 2 in ids
    col.close()


def test_int64_partition_key(tmp_path, pk_int_schema):
    col = Collection("test", str(tmp_path / "data"), pk_int_schema)
    col.insert([
        {"id": 1, "vec": [1, 0, 0, 0], "group_id": 100},
        {"id": 2, "vec": [0, 1, 0, 0], "group_id": 200},
        {"id": 3, "vec": [0, 0, 1, 0], "group_id": 100},
    ])
    col.load()
    res = col.query(expr=None, limit=10)
    assert len(res) == 3
    col.close()


def test_search_across_all_buckets(tmp_path, pk_schema):
    """Search without partition_names should search all bucket partitions."""
    col = Collection("test", str(tmp_path / "data"), pk_schema)
    col.insert([
        {"id": i, "vec": [float(i % 4 == j) for j in range(4)],
         "tenant": f"t_{i % 5}"}
        for i in range(20)
    ])
    col.load()
    results = col.search([[1, 0, 0, 0]], top_k=5, metric_type="COSINE")
    assert len(results[0]) == 5
    col.close()


def test_manual_create_partition_blocked(tmp_path, pk_schema):
    """Cannot create manual partitions when partition key is set."""
    col = Collection("test", str(tmp_path / "data"), pk_schema)
    with pytest.raises(SchemaValidationError, match="partition key"):
        col.create_partition("custom")
    col.close()


def test_manual_drop_partition_blocked(tmp_path, pk_schema):
    """Cannot drop bucket partitions when partition key is set (issue #18)."""
    col = Collection("test", str(tmp_path / "data"), pk_schema)
    with pytest.raises(SchemaValidationError, match="partition key"):
        col.drop_partition("_pk_0")
    col.close()


def test_recovery_preserves_partitions(tmp_path, pk_schema):
    """Reopen after insert — bucket partitions and data survive."""
    data_dir = str(tmp_path / "data")
    col = Collection("test", data_dir, pk_schema)
    col.insert([
        {"id": 1, "vec": [1, 0, 0, 0], "tenant": "alpha"},
        {"id": 2, "vec": [0, 1, 0, 0], "tenant": "beta"},
    ])
    col.close()

    col2 = Collection("test", data_dir, pk_schema)
    col2.load()
    assert col2.num_entities == 2
    col2.close()


def test_delete_with_partition_key(tmp_path, pk_schema):
    """Delete across partition key buckets."""
    col = Collection("test", str(tmp_path / "data"), pk_schema)
    col.insert([
        {"id": 1, "vec": [1, 0, 0, 0], "tenant": "alpha"},
        {"id": 2, "vec": [0, 1, 0, 0], "tenant": "beta"},
    ])
    col.load()
    col.delete([1])
    res = col.query(expr=None, limit=10)
    assert len(res) == 1
    assert res[0]["id"] == 2
    col.close()


def test_query_with_filter_across_buckets(tmp_path, pk_schema):
    """Filter expression works across all bucket partitions."""
    col = Collection("test", str(tmp_path / "data"), pk_schema)
    col.insert([
        {"id": i, "vec": [0.1] * 4, "tenant": f"t_{i % 3}"}
        for i in range(30)
    ])
    col.load()
    res = col.query(expr='tenant == "t_0"', limit=100)
    assert all(r["tenant"] == "t_0" for r in res)
    assert len(res) == 10
    col.close()
