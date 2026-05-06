"""Phase 10.2 — pymilvus Collection lifecycle integration tests.

The 5 RPCs added in Phase 10.2 (CreateCollection / DropCollection /
HasCollection / DescribeCollection / ShowCollections) drive the
following pymilvus client API surface:

    client.create_collection(name, schema=...)
    client.drop_collection(name)
    client.has_collection(name)
    client.describe_collection(name)
    client.list_collections()

These tests are skipped when pymilvus / grpcio is not installed.
The MVP test path uses MilvusClient with an EXPLICIT schema —
quick-mode `dimension=N` would auto-bundle CreateIndex + LoadCollection
which Phase 10.4 will add.
"""

import pytest

from pymilvus import DataType, MilvusClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema():
    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=False
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("title", DataType.VARCHAR, max_length=128)
    schema.add_field("active", DataType.BOOL)
    return schema


# ---------------------------------------------------------------------------
# create / drop
# ---------------------------------------------------------------------------

def test_create_collection_and_list(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    assert "demo" in milvus_client.list_collections()


def test_create_collection_then_drop(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    assert milvus_client.has_collection("demo")
    milvus_client.drop_collection("demo")
    assert not milvus_client.has_collection("demo")
    assert "demo" not in milvus_client.list_collections()


def test_create_collection_duplicate_raises(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    with pytest.raises(Exception) as exc_info:
        milvus_client.create_collection("demo", schema=_make_schema())
    assert "already exists" in str(exc_info.value).lower()


def test_drop_nonexistent_is_idempotent(milvus_client):
    """Dropping a non-existent collection should silently succeed (Milvus compat)."""
    milvus_client.drop_collection("ghost")  # no error


# ---------------------------------------------------------------------------
# has_collection
# ---------------------------------------------------------------------------

def test_has_collection_returns_true_for_existing(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    assert milvus_client.has_collection("demo") is True


def test_has_collection_returns_false_for_missing(milvus_client):
    """The single most failure-prone path: requires both
    error_code AND code fields on the Status response."""
    assert milvus_client.has_collection("ghost") is False


def test_has_collection_after_drop(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.drop_collection("demo")
    assert milvus_client.has_collection("demo") is False


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------

def test_list_collections_empty(milvus_client):
    assert milvus_client.list_collections() == []


def test_list_collections_multiple_sorted(milvus_client):
    print("xxxxxxxxxx----")
    for name in ["zebra", "alpha", "middle"]:
        milvus_client.create_collection(name, schema=_make_schema())
    names = milvus_client.list_collections()
    print(names)
    assert {"zebra", "alpha", "middle"}.issubset(set(names))
    # Verify the returned list is sorted
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# rename_collection (Issue #11)
# ---------------------------------------------------------------------------

def test_rename_collection(milvus_client):
    milvus_client.create_collection("old_name", schema=_make_schema())
    milvus_client.rename_collection("old_name", "new_name")
    assert not milvus_client.has_collection("old_name")
    assert milvus_client.has_collection("new_name")


def test_rename_collection_preserves_data(milvus_client):
    schema = _make_schema()
    milvus_client.create_collection("src", schema=schema)
    milvus_client.insert("src", [
        {"id": 1, "vec": [1, 0, 0, 0], "title": "hello", "active": True},
    ])
    milvus_client.rename_collection("src", "dst")
    results = milvus_client.query("dst", filter="id == 1", output_fields=["title"])
    assert len(results) == 1
    assert results[0]["title"] == "hello"


# ---------------------------------------------------------------------------
# describe_collection
# ---------------------------------------------------------------------------

def test_describe_collection_returns_schema(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    desc = milvus_client.describe_collection("demo")

    field_names = [f["name"] for f in desc["fields"]]
    assert sorted(field_names) == sorted(["id", "vec", "title", "active"])


def test_describe_collection_field_types(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    desc = milvus_client.describe_collection("demo")

    by_name = {f["name"]: f for f in desc["fields"]}
    # pymilvus DataType is the same enum as schema_pb2.DataType integer
    assert by_name["id"]["type"] == DataType.INT64
    assert by_name["id"]["is_primary"] is True
    assert by_name["vec"]["type"] == DataType.FLOAT_VECTOR
    assert by_name["title"]["type"] == DataType.VARCHAR
    assert by_name["active"]["type"] == DataType.BOOL


def test_describe_collection_vector_dim_preserved(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    desc = milvus_client.describe_collection("demo")
    vec_field = next(f for f in desc["fields"] if f["name"] == "vec")
    assert vec_field["params"]["dim"] == 4


def test_describe_collection_varchar_max_length_preserved(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    desc = milvus_client.describe_collection("demo")
    title_field = next(f for f in desc["fields"] if f["name"] == "title")
    assert title_field["params"]["max_length"] == 128


def test_describe_collection_includes_partitions(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    desc = milvus_client.describe_collection("demo")
    # Default partition count for a fresh collection
    assert desc["num_partitions"] == 1


def test_describe_collection_unknown_raises(milvus_client):
    with pytest.raises(Exception):
        milvus_client.describe_collection("ghost")


# ---------------------------------------------------------------------------
# enable_dynamic_field round-trip
# ---------------------------------------------------------------------------

def test_dynamic_field_round_trip(milvus_client):
    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=True
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=8)

    milvus_client.create_collection("dyn", schema=schema)
    desc = milvus_client.describe_collection("dyn")
    assert desc["enable_dynamic_field"] is True
