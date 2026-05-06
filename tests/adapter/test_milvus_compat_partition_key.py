"""Milvus compatibility test suite — partition key.

Migrated from milvus/tests/python_client/milvus_client/
test_milvus_client_partition_key.py.

Covers: creation with INT64/VARCHAR partition key, auto-bucket partitions,
insert + search + query routing, filter on partition key field, describe
collection shows is_partition_key, error cases (multi key, PK as key,
invalid type).
"""

import numpy as np
import pytest

from pymilvus import DataType, MilvusClient


DIM = 16
rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Valid cases
# ---------------------------------------------------------------------------

class TestPartitionKeyValid:

    def test_partition_key_varchar_insert_search_query(self, milvus_client):
        """Create with VARCHAR partition key, insert, search, query."""
        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("int64", DataType.INT64)
        schema.add_field("varchar", DataType.VARCHAR, max_length=128,
                         is_partition_key=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("pk_v_full", schema=schema,
                                        index_params=idx)

        nb = 100
        data = [{"int64": i, "varchar": f"key_{i % 10}",
                 "vec": rng.random(DIM).tolist()} for i in range(nb)]
        milvus_client.insert("pk_v_full", data)
        milvus_client.load_collection("pk_v_full")

        # Search
        res = milvus_client.search("pk_v_full",
                                   data=[rng.random(DIM).tolist()], limit=10)
        assert len(res[0]) == 10

        # Query with filter on partition key field
        res = milvus_client.query("pk_v_full", filter='varchar == "key_0"',
                                  limit=100, output_fields=["varchar"])
        assert len(res) == 10
        assert all(r["varchar"] == "key_0" for r in res)

        # Search with mixed filter (partition key + non-partition key)
        res = milvus_client.search(
            "pk_v_full", data=[rng.random(DIM).tolist()], limit=100,
            filter='int64 in [1,3,5] and varchar in ["key_1","key_3","key_5"]',
            output_fields=["int64", "varchar"],
        )
        for h in res[0]:
            assert h["entity"]["int64"] in [1, 3, 5]
            assert h["entity"]["varchar"] in ["key_1", "key_3", "key_5"]

    def test_partition_key_int64_insert_search(self, milvus_client):
        """Create with INT64 partition key."""
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("pk", DataType.VARCHAR, max_length=64,
                         is_primary=True)
        schema.add_field("group_id", DataType.INT64, is_partition_key=True)
        schema.add_field("name", DataType.VARCHAR, max_length=128)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("pk_i_full", schema=schema,
                                        index_params=idx)

        nb = 100
        data = [{"pk": str(i), "group_id": i % 5,
                 "name": f"item_{i}", "vec": rng.random(DIM).tolist()}
                for i in range(nb)]
        milvus_client.insert("pk_i_full", data)
        milvus_client.load_collection("pk_i_full")

        res = milvus_client.search("pk_i_full",
                                   data=[rng.random(DIM).tolist()], limit=10)
        assert len(res[0]) == 10

        # Filter on partition key
        res = milvus_client.query("pk_i_full", filter="group_id == 0",
                                  limit=100)
        assert len(res) == 20  # 100 / 5 groups

    def test_describe_collection_shows_partition_key(self, milvus_client):
        """DescribeCollection should report is_partition_key."""
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("tenant", DataType.VARCHAR, max_length=64,
                         is_partition_key=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("pk_desc", schema=schema,
                                        index_params=idx)

        desc = milvus_client.describe_collection("pk_desc")
        pk_fields = [f for f in desc["fields"]
                     if f.get("is_partition_key")]
        assert len(pk_fields) == 1
        assert pk_fields[0]["name"] == "tenant"

    def test_partition_key_with_flush_and_search(self, milvus_client):
        """Data survives flush with partition key routing."""
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("tag", DataType.VARCHAR, max_length=64,
                         is_partition_key=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("pk_flush", schema=schema,
                                        index_params=idx)

        data = [{"pk": i, "tag": f"t_{i % 3}",
                 "vec": rng.random(DIM).tolist()} for i in range(30)]
        milvus_client.insert("pk_flush", data)
        milvus_client.flush("pk_flush")
        milvus_client.load_collection("pk_flush")

        res = milvus_client.query("pk_flush", filter="pk >= 0", limit=100)
        assert len(res) == 30

    def test_partition_key_delete_and_query(self, milvus_client):
        """Delete across partition key buckets."""
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("tag", DataType.VARCHAR, max_length=64,
                         is_partition_key=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("pk_del", schema=schema,
                                        index_params=idx)

        milvus_client.insert("pk_del", [
            {"pk": 1, "tag": "a", "vec": [0.1] * DIM},
            {"pk": 2, "tag": "b", "vec": [0.2] * DIM},
            {"pk": 3, "tag": "a", "vec": [0.3] * DIM},
        ])
        milvus_client.load_collection("pk_del")

        milvus_client.delete("pk_del", ids=[1])
        res = milvus_client.query("pk_del", filter="pk >= 0", limit=10)
        assert len(res) == 2
        ids = {r["pk"] for r in res}
        assert 1 not in ids

    def test_partition_key_upsert(self, milvus_client):
        """Upsert with partition key collection."""
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("tag", DataType.VARCHAR, max_length=64,
                         is_partition_key=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", DataType.FLOAT, nullable=True)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("pk_ups", schema=schema,
                                        index_params=idx)

        milvus_client.insert("pk_ups", [
            {"pk": 1, "tag": "a", "vec": [0.1] * DIM, "val": 1.0},
        ])
        milvus_client.upsert("pk_ups", [
            {"pk": 1, "tag": "a", "vec": [0.1] * DIM, "val": 9.0},
        ])
        milvus_client.load_collection("pk_ups")
        res = milvus_client.query("pk_ups", filter="pk == 1",
                                  output_fields=["val"])
        assert res[0]["val"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Invalid cases
# ---------------------------------------------------------------------------

class TestPartitionKeyInvalid:

    def test_partition_key_on_multi_fields_raises(self, milvus_client):
        """Two fields with is_partition_key=True should be rejected."""
        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("f1", DataType.INT64, is_partition_key=True)
        schema.add_field("f2", DataType.VARCHAR, max_length=64,
                         is_partition_key=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        with pytest.raises(Exception):
            milvus_client.create_collection("pk_multi", schema=schema)

    def test_partition_key_on_invalid_type_raises(self, milvus_client):
        """Partition key on FLOAT field should be rejected."""
        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("score", DataType.FLOAT, is_partition_key=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        with pytest.raises(Exception):
            milvus_client.create_collection("pk_float", schema=schema)

    def test_partition_key_on_bool_type_raises(self, milvus_client):
        """Partition key on BOOL field should be rejected."""
        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("flag", DataType.BOOL, is_partition_key=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        with pytest.raises(Exception):
            milvus_client.create_collection("pk_bool", schema=schema)
