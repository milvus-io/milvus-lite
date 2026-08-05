"""P0 compatibility features: aliases, truncate, indexes, and statistics."""

import pytest
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException


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


def test_truncate_preserves_metadata_and_clears_mutation_history(milvus_client):
    _create_basic_collection(milvus_client, "truncate_history")
    milvus_client.create_partition("truncate_history", "archive")
    milvus_client.create_alias("truncate_history", "truncate_alias")
    idx = milvus_client.prepare_index_params()
    idx.add_index(
        field_name="vec",
        index_type="BRUTE_FORCE",
        metric_type="COSINE",
        params={},
    )
    milvus_client.create_index("truncate_history", idx)
    milvus_client.alter_collection_properties(
        "truncate_history", {"custom.keep": "yes"}
    )
    milvus_client.insert(
        "truncate_history",
        [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "delete"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "upsert"},
        ],
    )
    milvus_client.upsert(
        "truncate_history",
        [
            {"id": 2, "vec": [0.0, 0.0, 1.0, 0.0], "tag": "updated"},
            {"id": 3, "vec": [0.0, 0.0, 0.0, 1.0], "tag": "new"},
        ],
    )
    milvus_client.delete("truncate_history", ids=[1])

    milvus_client.truncate_collection("truncate_alias")

    assert int(
        milvus_client.get_collection_stats("truncate_history")["row_count"]
    ) == 0
    assert milvus_client.query(
        "truncate_history", filter="id >= 0", output_fields=["id"]
    ) == []
    assert milvus_client.list_indexes("truncate_history") == ["vec"]
    assert "archive" in milvus_client.list_partitions("truncate_history")
    assert (
        milvus_client.describe_alias("truncate_alias")["collection_name"]
        == "truncate_history"
    )
    assert (
        milvus_client.describe_collection("truncate_history")["properties"][
            "custom.keep"
        ]
        == "yes"
    )


def test_truncate_released_collection_then_flush(milvus_client):
    _create_basic_collection(milvus_client, "truncate_released")
    milvus_client.insert(
        "truncate_released",
        [{"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "old"}],
    )
    milvus_client.release_collection("truncate_released")

    milvus_client.truncate_collection("truncate_released")
    milvus_client.flush("truncate_released")

    assert int(
        milvus_client.get_collection_stats("truncate_released")["row_count"]
    ) == 0


def test_truncate_is_database_isolated(milvus_client):
    for database, row_id in (("truncate_db_a", 1), ("truncate_db_b", 2)):
        milvus_client.create_database(database)
        milvus_client.using_database(database)
        _create_basic_collection(milvus_client, "shared")
        milvus_client.insert(
            "shared",
            [
                {
                    "id": row_id,
                    "vec": [1.0, 0.0, 0.0, 0.0],
                    "tag": database,
                }
            ],
        )

    milvus_client.using_database("truncate_db_a")
    milvus_client.truncate_collection("shared")
    assert int(milvus_client.get_collection_stats("shared")["row_count"]) == 0
    milvus_client.using_database("truncate_db_b")
    assert int(milvus_client.get_collection_stats("shared")["row_count"]) == 1


def test_collection_stats_follow_upsert_delete_flush_and_truncate(milvus_client):
    _create_basic_collection(milvus_client, "collection_stats")
    assert int(
        milvus_client.get_collection_stats("collection_stats")["row_count"]
    ) == 0
    milvus_client.insert(
        "collection_stats",
        [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "one"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "two"},
        ],
    )
    milvus_client.upsert(
        "collection_stats",
        [
            {"id": 2, "vec": [0.0, 0.0, 1.0, 0.0], "tag": "updated"},
            {"id": 3, "vec": [0.0, 0.0, 0.0, 1.0], "tag": "three"},
        ],
    )
    assert int(
        milvus_client.get_collection_stats("collection_stats")["row_count"]
    ) == 3
    milvus_client.delete("collection_stats", ids=[1, 999])
    assert int(
        milvus_client.get_collection_stats("collection_stats")["row_count"]
    ) == 2
    milvus_client.flush("collection_stats")
    assert int(
        milvus_client.get_collection_stats("collection_stats")["row_count"]
    ) == 2
    milvus_client.truncate_collection("collection_stats")
    assert int(
        milvus_client.get_collection_stats("collection_stats")["row_count"]
    ) == 0


def test_partition_stats_follow_upsert_delete_and_flush(milvus_client):
    _create_basic_collection(milvus_client, "partition_stats")
    milvus_client.create_partition("partition_stats", "archive")
    milvus_client.insert(
        "partition_stats",
        [{"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "default"}],
    )
    milvus_client.insert(
        "partition_stats",
        [
            {"id": 10, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "old"},
            {"id": 11, "vec": [0.0, 0.0, 1.0, 0.0], "tag": "keep"},
        ],
        partition_name="archive",
    )
    milvus_client.upsert(
        "partition_stats",
        [
            {"id": 10, "vec": [0.0, 0.0, 0.0, 1.0], "tag": "updated"},
            {"id": 12, "vec": [1.0, 1.0, 0.0, 0.0], "tag": "new"},
        ],
        partition_name="archive",
    )
    milvus_client.delete("partition_stats", ids=[11], partition_name="archive")
    assert int(
        milvus_client.get_partition_stats("partition_stats", "_default")[
            "row_count"
        ]
    ) == 1
    assert int(
        milvus_client.get_partition_stats("partition_stats", "archive")[
            "row_count"
        ]
    ) == 2
    milvus_client.flush("partition_stats")
    assert int(
        milvus_client.get_partition_stats("partition_stats", "archive")[
            "row_count"
        ]
    ) == 2


def test_statistics_missing_resources_raise(milvus_client):
    with pytest.raises(MilvusException):
        milvus_client.get_collection_stats("missing")

    _create_basic_collection(milvus_client, "missing_partition")
    with pytest.raises(MilvusException):
        milvus_client.get_partition_stats("missing_partition", "missing")
    milvus_client.create_partition("missing_partition", "gone")
    milvus_client.drop_partition("missing_partition", "gone")
    with pytest.raises(MilvusException):
        milvus_client.get_partition_stats("missing_partition", "gone")
