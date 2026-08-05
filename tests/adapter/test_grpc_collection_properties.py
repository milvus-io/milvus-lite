"""gRPC coverage for mutable collection properties."""

import pytest
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException


def _schema():
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    return schema


def _create(client, name):
    client.create_collection(name, schema=_schema())


def _properties(client, name):
    return client.describe_collection(name).get("properties", {})


def test_collection_properties_set_replace_and_delete(milvus_client):
    _create(milvus_client, "props")

    milvus_client.alter_collection_properties(
        "props", {"custom.mode": "first", "custom.keep": "yes"}
    )
    assert _properties(milvus_client, "props")["custom.mode"] == "first"

    milvus_client.alter_collection_properties("props", {"custom.mode": "second"})
    props = _properties(milvus_client, "props")
    assert props["custom.mode"] == "second"
    assert props["custom.keep"] == "yes"

    milvus_client.drop_collection_properties("props", ["custom.mode"])
    props = _properties(milvus_client, "props")
    assert "custom.mode" not in props
    assert props["custom.keep"] == "yes"


def test_alter_missing_collection_has_no_side_effect(milvus_client):
    _create(milvus_client, "sentinel")
    milvus_client.alter_collection_properties("sentinel", {"custom.keep": "yes"})

    with pytest.raises(MilvusException):
        milvus_client.alter_collection_properties("missing", {"custom.keep": "no"})

    assert _properties(milvus_client, "sentinel")["custom.keep"] == "yes"
    assert milvus_client.has_collection("sentinel")


def test_collection_properties_via_alias(milvus_client):
    _create(milvus_client, "target")
    milvus_client.create_alias("target", "target_alias")

    milvus_client.alter_collection_properties(
        "target_alias", {"custom.route": "alias"}
    )

    assert _properties(milvus_client, "target")["custom.route"] == "alias"
    assert (
        milvus_client.describe_alias("target_alias")["collection_name"]
        == "target"
    )


def test_collection_properties_are_database_isolated(milvus_client):
    milvus_client.create_database("tenant_a")
    milvus_client.create_database("tenant_b")
    for database, value in (("tenant_a", "a"), ("tenant_b", "b")):
        milvus_client.using_database(database)
        _create(milvus_client, "shared")
        milvus_client.alter_collection_properties(
            "shared", {"custom.owner": value}
        )

    milvus_client.using_database("tenant_a")
    assert _properties(milvus_client, "shared")["custom.owner"] == "a"
    milvus_client.using_database("tenant_b")
    assert _properties(milvus_client, "shared")["custom.owner"] == "b"
