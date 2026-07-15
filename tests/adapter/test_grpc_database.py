"""Database namespace integration tests for the gRPC adapter."""

from uuid import uuid4

import pytest
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException


def _unique_name(prefix):
    return f"{prefix}_{uuid4().hex}"


def _make_schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("title", DataType.VARCHAR, max_length=128)
    return schema


def _create_sentinel_database(milvus_client):
    database_name = _unique_name("sentinel")
    collection_name = _unique_name("docs")
    properties = {"timezone": "UTC", "custom_marker": "keep"}
    record = {
        "id": 4242,
        "vec": [0, 0, 0, 1],
        "title": "database-property-sentinel",
    }
    milvus_client.create_database(database_name, properties=properties)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(collection_name, schema=_make_schema())
    milvus_client.insert(collection_name, [record])
    return database_name, collection_name, record


def _assert_sentinel_database_unchanged(
    milvus_client,
    database_name,
    collection_name,
    record,
    databases_before,
    collections_before,
    properties_before,
):
    assert milvus_client.list_databases() == databases_before
    assert milvus_client.has_collection(collection_name)
    assert milvus_client.list_collections() == collections_before
    assert milvus_client.describe_database(database_name) == properties_before
    query_result = milvus_client.query(
        collection_name,
        filter="id == 4242",
        output_fields=["id", "vec", "title"],
    )
    assert len(query_result) == 1
    assert query_result[0] == record


# Upstream: test_milvus_client_create_drop_database_default
def test_database_lifecycle(milvus_client):
    database_name = _unique_name("tenant")

    milvus_client.create_database(database_name)
    assert database_name in milvus_client.list_databases()

    milvus_client.drop_database(database_name)
    assert database_name not in milvus_client.list_databases()


# Upstream: test_milvus_client_create_database_with_properties
def test_create_database_with_timezone_property(milvus_client):
    database_name = _unique_name("tenant")

    milvus_client.create_database(database_name, properties={"timezone": "UTC"})

    assert milvus_client.describe_database(database_name)["timezone"] == "UTC"


# Upstream: test_milvus_client_alter_database_properties_default
def test_alter_database_timezone_property(milvus_client):
    database_name = _unique_name("tenant")
    milvus_client.create_database(database_name, properties={"timezone": "UTC"})

    milvus_client.alter_database_properties(
        database_name,
        properties={"timezone": "Asia/Shanghai"},
    )

    assert milvus_client.describe_database(database_name)["timezone"] == "Asia/Shanghai"


# Upstream: test_milvus_client_drop_database_properties_default
def test_drop_database_timezone_property(milvus_client):
    database_name = _unique_name("tenant")
    milvus_client.create_database(
        database_name,
        properties={"timezone": "UTC", "custom_marker": "keep"},
    )

    milvus_client.drop_database_properties(database_name, ["timezone"])

    properties = milvus_client.describe_database(database_name)
    assert "timezone" not in properties
    assert properties["custom_marker"] == "keep"

    properties_before = milvus_client.describe_database(database_name)
    milvus_client.drop_database_properties(database_name, ["missing_property"])

    assert milvus_client.describe_database(database_name) == properties_before


# Upstream: test_milvus_client_alter_database_properties_nonexistent_db_name
def test_alter_missing_database_properties_fails_without_changes(milvus_client):
    sentinel_database, collection_name, record = _create_sentinel_database(milvus_client)
    missing_database = _unique_name("missing")
    databases_before = milvus_client.list_databases()
    collections_before = milvus_client.list_collections()
    properties_before = milvus_client.describe_database(sentinel_database)

    with pytest.raises(MilvusException):
        milvus_client.alter_database_properties(
            missing_database,
            properties={"timezone": "Asia/Shanghai"},
        )

    _assert_sentinel_database_unchanged(
        milvus_client,
        sentinel_database,
        collection_name,
        record,
        databases_before,
        collections_before,
        properties_before,
    )


# Upstream: test_milvus_client_drop_database_properties_nonexistent_db_name
def test_drop_missing_database_properties_fails_without_changes(milvus_client):
    sentinel_database, collection_name, record = _create_sentinel_database(milvus_client)
    missing_database = _unique_name("missing")
    databases_before = milvus_client.list_databases()
    collections_before = milvus_client.list_collections()
    properties_before = milvus_client.describe_database(sentinel_database)

    with pytest.raises(MilvusException):
        milvus_client.drop_database_properties(missing_database, ["timezone"])

    _assert_sentinel_database_unchanged(
        milvus_client,
        sentinel_database,
        collection_name,
        record,
        databases_before,
        collections_before,
        properties_before,
    )


# Upstream: test_db_default
def test_default_database_exists(milvus_client):
    assert "default" in milvus_client.list_databases()


# Upstream: test_milvus_client_create_database_name_with_default
def test_create_default_database_fails_and_default_remains(milvus_client):
    databases_before = milvus_client.list_databases()

    with pytest.raises(MilvusException):
        milvus_client.create_database("default")

    assert milvus_client.list_databases() == databases_before
    assert "default" in milvus_client.list_databases()


# Upstream: test_milvus_client_create_database_with_existed_name
def test_create_duplicate_database_fails_and_database_remains(milvus_client):
    database_name = _unique_name("tenant")
    milvus_client.create_database(database_name, properties={"timezone": "UTC"})
    databases_before = milvus_client.list_databases()

    with pytest.raises(MilvusException):
        milvus_client.create_database(database_name)

    assert milvus_client.list_databases() == databases_before
    assert milvus_client.describe_database(database_name)["timezone"] == "UTC"


# Upstream: test_create_db_invalid_name_value, test_drop_db_invalid_name
@pytest.mark.parametrize("database_name", [None, "", ".", "..", "/", "\\", "\x00"])
def test_unsafe_database_name_create_and_drop_fail_without_changes(
    milvus_client,
    database_name,
):
    databases_before = milvus_client.list_databases()

    with pytest.raises((MilvusException, TypeError, ValueError)):
        milvus_client.create_database(database_name)
    assert milvus_client.list_databases() == databases_before

    with pytest.raises((MilvusException, TypeError, ValueError)):
        milvus_client.drop_database(database_name)
    assert milvus_client.list_databases() == databases_before


# Upstream: test_milvus_client_create_database_name_over_max_length
def test_database_name_over_255_bytes_create_fails_without_changes(milvus_client):
    database_name = "a" * 256
    databases_before = milvus_client.list_databases()

    with pytest.raises(MilvusException):
        milvus_client.create_database(database_name)

    assert milvus_client.list_databases() == databases_before


# Upstream: test_milvus_client_use_database_nonexistent_db_name,
# test_milvus_client_describe_database_invalid_db_name,
# test_milvus_client_drop_database_nonexistent_db_name
@pytest.mark.parametrize(
    "operation_name",
    ["using_database", "describe_database", "drop_database"],
)
def test_missing_database_operation_fails_without_changes(
    milvus_client,
    operation_name,
):
    database_name = _unique_name("missing")
    collection_name = _unique_name("docs")
    milvus_client.create_collection(collection_name, schema=_make_schema())
    databases_before = milvus_client.list_databases()
    collections_before = milvus_client.list_collections()

    with pytest.raises(MilvusException):
        getattr(milvus_client, operation_name)(database_name)

    assert milvus_client.list_databases() == databases_before
    assert milvus_client.list_collections() == collections_before
    assert milvus_client.has_collection(collection_name)


# Upstream: test_drop_default_db
def test_drop_default_database_fails_and_default_remains(milvus_client):
    databases_before = milvus_client.list_databases()

    with pytest.raises(MilvusException):
        milvus_client.drop_database("default")

    assert milvus_client.list_databases() == databases_before
    assert "default" in milvus_client.list_databases()


# Upstream: test_milvus_client_drop_database_has_collections
def test_drop_nonempty_database_fails_and_collection_remains(milvus_client):
    database_name = _unique_name("tenant")
    collection_name = _unique_name("docs")
    milvus_client.create_database(database_name)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(collection_name, schema=_make_schema())

    with pytest.raises(MilvusException):
        milvus_client.drop_database(database_name)

    assert database_name in milvus_client.list_databases()
    assert milvus_client.has_collection(collection_name)
    assert collection_name in milvus_client.list_collections()


# Upstream: test_milvus_client_use_database_default
def test_using_database_routes_collection_metadata(milvus_client):
    database_name = _unique_name("tenant")
    collection_name = _unique_name("docs")
    milvus_client.create_database(database_name)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(collection_name, schema=_make_schema())
    assert milvus_client.has_collection(collection_name)
    assert milvus_client.list_collections() == [collection_name]

    milvus_client.using_database("default")
    assert not milvus_client.has_collection(collection_name)
    assert collection_name not in milvus_client.list_collections()


# Upstream: test_create_same_collection_name_different_db
def test_same_collection_name_data_isolated_by_database(milvus_client):
    database_name = _unique_name("tenant")
    collection_name = _unique_name("docs")
    schema = _make_schema()
    milvus_client.create_collection(collection_name, schema=schema)
    milvus_client.insert(
        collection_name,
        [{"id": 1, "vec": [1, 0, 0, 0], "title": "default"}],
    )

    milvus_client.create_database(database_name)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(collection_name, schema=schema)
    milvus_client.insert(
        collection_name,
        [{"id": 1, "vec": [0, 1, 0, 0], "title": "tenant"}],
    )

    tenant_result = milvus_client.query(
        collection_name,
        filter="id == 1",
        output_fields=["title"],
    )
    assert len(tenant_result) == 1
    assert tenant_result[0]["title"] == "tenant"

    milvus_client.using_database("default")
    default_result = milvus_client.query(
        collection_name,
        filter="id == 1",
        output_fields=["title"],
    )
    assert len(default_result) == 1
    assert default_result[0]["title"] == "default"


# Upstream: test_create_collection_name_same_db
def test_collection_name_can_match_database_name(milvus_client):
    database_name = _unique_name("tenant")
    milvus_client.create_database(database_name)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(database_name, schema=_make_schema())

    result = milvus_client.insert(
        database_name,
        [{"id": 1, "vec": [1, 0, 0, 0], "title": "matching-name"}],
    )

    assert result["insert_count"] == 1
    assert milvus_client.list_collections() == [database_name]


# Upstream: test_different_db_same_collection_name
def test_same_collection_name_isolated_between_nondefault_databases(milvus_client):
    first_database = _unique_name("tenant")
    second_database = _unique_name("tenant")
    collection_name = _unique_name("docs")
    first_record = {"id": 1, "vec": [1, 0, 0, 0], "title": "first"}
    second_record = {"id": 1, "vec": [0, 1, 0, 0], "title": "second"}

    milvus_client.create_database(first_database)
    milvus_client.using_database(first_database)
    milvus_client.create_collection(collection_name, schema=_make_schema())
    milvus_client.insert(collection_name, [first_record])

    milvus_client.create_database(second_database)
    milvus_client.using_database(second_database)
    milvus_client.create_collection(collection_name, schema=_make_schema())
    milvus_client.insert(collection_name, [second_record])

    second_result = milvus_client.query(
        collection_name,
        filter="id >= 0",
        output_fields=["id", "vec", "title"],
    )
    assert len(second_result) == 1
    assert second_result[0] == second_record

    milvus_client.using_database(first_database)
    first_result = milvus_client.query(
        collection_name,
        filter="id >= 0",
        output_fields=["id", "vec", "title"],
    )
    assert len(first_result) == 1
    assert first_result[0] == first_record


# Upstream: test_rename_collection_in_new_db
def test_rename_collection_stays_in_current_database(milvus_client):
    database_name = _unique_name("tenant")
    old_collection_name = _unique_name("docs")
    new_collection_name = _unique_name("renamed_docs")
    sentinel_record = {
        "id": 99,
        "vec": [0, 0, 1, 0],
        "title": "default-sentinel",
    }
    milvus_client.create_collection(old_collection_name, schema=_make_schema())
    milvus_client.insert(old_collection_name, [sentinel_record])

    milvus_client.create_database(database_name)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(old_collection_name, schema=_make_schema())

    milvus_client.rename_collection(old_collection_name, new_collection_name)

    assert not milvus_client.has_collection(old_collection_name)
    assert milvus_client.has_collection(new_collection_name)
    assert new_collection_name in milvus_client.list_collections()

    milvus_client.using_database("default")
    assert milvus_client.has_collection(old_collection_name)
    assert not milvus_client.has_collection(new_collection_name)
    assert new_collection_name not in milvus_client.list_collections()
    sentinel_result = milvus_client.query(
        old_collection_name,
        filter="id == 99",
        output_fields=["id", "vec", "title"],
    )
    assert len(sentinel_result) == 1
    assert sentinel_result[0] == sentinel_record


# Upstream: test_query_db, test_search_db
def test_nondefault_database_query_and_search_data_isolated(milvus_client):
    contaminant_database = _unique_name("contaminant")
    database_name = _unique_name("tenant")
    collection_name = _unique_name("docs")
    records = [
        {"id": 1, "vec": [0.9, 0.1, 0, 0], "title": "nearest"},
        {"id": 2, "vec": [0, 1, 0, 0], "title": "other"},
    ]
    milvus_client.create_database(contaminant_database)
    milvus_client.using_database(contaminant_database)
    milvus_client.create_collection(collection_name, schema=_make_schema())
    milvus_client.insert(
        collection_name,
        [{"id": 99, "vec": [1, 0, 0, 0], "title": "contaminant"}],
    )
    milvus_client.load_collection(collection_name)

    milvus_client.create_database(database_name)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(collection_name, schema=_make_schema())
    milvus_client.insert(collection_name, records)

    query_result = milvus_client.query(
        collection_name,
        filter="id >= 0",
        output_fields=["id", "vec", "title"],
    )
    assert len(query_result) == 2
    assert {row["id"] for row in query_result} == {1, 2}

    milvus_client.load_collection(collection_name)
    search_result = milvus_client.search(
        collection_name,
        data=[[1, 0, 0, 0]],
        limit=1,
        output_fields=["title"],
    )
    assert len(search_result) == 1
    assert len(search_result[0]) == 1
    assert search_result[0][0]["id"] == 1
    assert search_result[0][0]["entity"]["title"] == "nearest"

    milvus_client.using_database("default")
    assert not milvus_client.has_collection(collection_name)
    assert collection_name not in milvus_client.list_collections()


# Upstream: test_connect_db
def test_client_db_name_connects_directly_to_database(grpc_server, milvus_client):
    database_name = _unique_name("tenant")
    collection_name = _unique_name("docs")
    milvus_client.create_database(database_name)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(collection_name, schema=_make_schema())

    port, _db = grpc_server
    second_client = MilvusClient(
        uri=f"http://127.0.0.1:{port}",
        db_name=database_name,
    )
    try:
        assert second_client.has_collection(collection_name)
        assert collection_name in second_client.list_collections()
    finally:
        second_client.close()


# Upstream intent: aliases resolve within the selected database namespace.
def test_same_alias_isolated_by_database(milvus_client):
    database_name = _unique_name("tenant")
    default_collection = _unique_name("default_docs")
    tenant_collection = _unique_name("tenant_docs")
    alias = _unique_name("current")
    default_record = {
        "id": 1,
        "vec": [1, 0, 0, 0],
        "title": "default-alias-record",
    }
    tenant_record = {
        "id": 2,
        "vec": [0, 1, 0, 0],
        "title": "tenant-alias-record",
    }

    milvus_client.create_collection(default_collection, schema=_make_schema())
    milvus_client.insert(default_collection, [default_record])
    milvus_client.create_alias(default_collection, alias)

    milvus_client.create_database(database_name)
    milvus_client.using_database(database_name)
    milvus_client.create_collection(tenant_collection, schema=_make_schema())
    milvus_client.insert(tenant_collection, [tenant_record])
    milvus_client.create_alias(tenant_collection, alias)

    assert milvus_client.describe_alias(alias)["collection_name"] == tenant_collection
    assert alias in milvus_client.list_aliases(tenant_collection)["aliases"]
    tenant_result = milvus_client.query(
        alias,
        filter="id >= 0",
        output_fields=["id", "vec", "title"],
    )
    assert len(tenant_result) == 1
    assert tenant_result[0] == tenant_record

    milvus_client.using_database("default")
    assert milvus_client.describe_alias(alias)["collection_name"] == default_collection
    assert alias in milvus_client.list_aliases(default_collection)["aliases"]
    default_result = milvus_client.query(
        alias,
        filter="id >= 0",
        output_fields=["id", "vec", "title"],
    )
    assert len(default_result) == 1
    assert default_result[0] == default_record


# Upstream: test_using_invalid_db
@pytest.mark.parametrize("database_name", [1, (), [], {}, [1, "2"]])
def test_using_database_rejects_non_string_without_changing_default_state(
    milvus_client,
    database_name,
):
    collection_name = _unique_name("default_docs")
    record = {
        "id": 7,
        "vec": [0, 0, 1, 0],
        "title": "default-routing-sentinel",
    }
    milvus_client.using_database("default")
    milvus_client.create_collection(collection_name, schema=_make_schema())
    milvus_client.insert(collection_name, [record])
    databases_before = milvus_client.list_databases()
    collections_before = milvus_client.list_collections()
    query_before = milvus_client.query(
        collection_name,
        filter="id == 7",
        output_fields=["id", "vec", "title"],
    )

    with pytest.raises((MilvusException, TypeError, ValueError)):
        milvus_client.using_database(database_name)

    assert milvus_client.list_databases() == databases_before
    assert milvus_client.list_collections() == collections_before
    assert milvus_client.has_collection(collection_name)
    assert milvus_client.query(
        collection_name,
        filter="id == 7",
        output_fields=["id", "vec", "title"],
    ) == query_before
