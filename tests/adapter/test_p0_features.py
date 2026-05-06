"""P0 compatibility features: aliases, truncate, list_indexes, partition stats."""

from pymilvus import DataType, MilvusClient


def _create_basic_collection(client: MilvusClient, name: str) -> None:
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("tag", DataType.VARCHAR, max_length=32)
    client.create_collection(name, schema=schema)


def test_alias_lifecycle_and_alias_reads(milvus_client):
    _create_basic_collection(milvus_client, "alias_a")
    _create_basic_collection(milvus_client, "alias_b")
    milvus_client.insert("alias_a", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "a"},
    ])
    milvus_client.insert("alias_b", [
        {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "b"},
    ])

    milvus_client.create_alias("alias_a", "current_alias")
    info = milvus_client.describe_alias("current_alias")
    assert info["collection_name"] == "alias_a"

    rows = milvus_client.query(
        "current_alias", filter="id == 1", output_fields=["tag"]
    )
    assert rows == [{"id": 1, "tag": "a"}]

    listed = milvus_client.list_aliases("alias_a")
    assert "current_alias" in listed["aliases"]

    milvus_client.alter_alias("alias_b", "current_alias")
    assert milvus_client.describe_alias("current_alias")["collection_name"] == "alias_b"
    rows = milvus_client.query(
        "current_alias", filter="id == 2", output_fields=["tag"]
    )
    assert rows == [{"id": 2, "tag": "b"}]

    milvus_client.drop_alias("current_alias")
    assert "current_alias" not in milvus_client.list_aliases()["aliases"]


def test_truncate_collection_preserves_schema(milvus_client):
    _create_basic_collection(milvus_client, "to_truncate")
    milvus_client.insert("to_truncate", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "old"},
    ])
    assert milvus_client.get_collection_stats("to_truncate")["row_count"] == 1

    milvus_client.truncate_collection("to_truncate")
    assert milvus_client.get_collection_stats("to_truncate")["row_count"] == 0

    milvus_client.insert("to_truncate", [
        {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "new"},
    ])
    rows = milvus_client.query(
        "to_truncate", filter="id == 2", output_fields=["tag"]
    )
    assert rows == [{"id": 2, "tag": "new"}]


def test_list_indexes_returns_index_names(milvus_client):
    _create_basic_collection(milvus_client, "idx_names")
    idx = milvus_client.prepare_index_params()
    idx.add_index(
        field_name="vec",
        index_type="BRUTE_FORCE",
        metric_type="COSINE",
        params={},
    )
    milvus_client.create_index("idx_names", idx)

    assert milvus_client.list_indexes("idx_names") == ["vec"]


def test_get_partition_stats(milvus_client):
    _create_basic_collection(milvus_client, "part_stats")
    milvus_client.create_partition("part_stats", "archive")
    milvus_client.insert("part_stats", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "default"},
        {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "default"},
    ])
    milvus_client.insert(
        "part_stats",
        [
            {"id": 10, "vec": [0.0, 0.0, 1.0, 0.0], "tag": "archive"},
            {"id": 11, "vec": [0.0, 0.0, 0.0, 1.0], "tag": "archive"},
            {"id": 12, "vec": [1.0, 1.0, 0.0, 0.0], "tag": "archive"},
        ],
        partition_name="archive",
    )

    assert milvus_client.get_partition_stats("part_stats", "_default")["row_count"] == 2
    assert milvus_client.get_partition_stats("part_stats", "archive")["row_count"] == 3
