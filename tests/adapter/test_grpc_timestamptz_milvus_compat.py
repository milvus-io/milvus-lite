"""MilvusClient TIMESTAMPTZ compatibility tests.

Migrated and reduced from:
``~/Workspace/dev/milvus/tests/python_client/milvus_client/test_milvus_client_timestamptz.py``.

MilvusLite currently implements the core TIMESTAMPTZ surface: schema,
insert/query/search filters, nullable/default values, UTC
normalization, database/collection/request-level timezone properties,
and time_fields extraction. Schema evolution and STL_SORT timestamp
indexes are tracked as follow-up work in the roadmap.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymilvus")
from pymilvus import DataType, MilvusClient  # noqa: E402


DIM = 3
PK = "id"
VEC = "vector"
TS = "timestamp"
_MISSING = object()


def _schema(nullable: bool = True, default_value=_MISSING):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(PK, DataType.INT64, is_primary=True)
    schema.add_field(VEC, DataType.FLOAT_VECTOR, dim=DIM)
    kwargs = {"nullable": nullable}
    if default_value is not _MISSING:
        kwargs["default_value"] = default_value
    schema.add_field(TS, DataType.TIMESTAMPTZ, **kwargs)
    return schema


def _rows():
    return [
        {PK: 0, VEC: [1.0, 2.0, 3.0], TS: "2025-01-01T00:00:00+08:00"},
        {PK: 1, VEC: [4.0, 5.0, 6.0], TS: "2025-01-03T00:00:00Z"},
        {PK: 2, VEC: [7.0, 8.0, 9.0], TS: "2024-02-29T00:00:00+03:00"},
        {PK: 3, VEC: [10.0, 11.0, 12.0], TS: None},
    ]


def _by_id(rows):
    return {row[PK]: row for row in rows}


def _describe_properties(client: MilvusClient, collection: str) -> dict:
    description = client.describe_collection(collection)
    return description.get("properties") or {}


def test_milvus_client_timestamptz_utc_insert_query(milvus_client):
    collection = "ts_utc"
    milvus_client.create_collection(collection, schema=_schema())
    milvus_client.insert(collection, _rows())

    rows = milvus_client.query(
        collection,
        filter=f"{PK} >= 0",
        output_fields=[PK, TS],
    )

    got = _by_id(rows)
    assert got[0][TS] == "2024-12-31T16:00:00Z"
    assert got[1][TS] == "2025-01-03T00:00:00Z"
    assert got[2][TS] == "2024-02-28T21:00:00Z"
    assert got[3][TS] is None


def test_milvus_client_timestamptz_query_operators_and_interval(milvus_client):
    collection = "ts_query"
    milvus_client.create_collection(collection, schema=_schema())
    milvus_client.insert(collection, _rows())

    rows = milvus_client.query(
        collection,
        filter=f"{TS} >= ISO '2025-01-01T00:00:00Z'",
        output_fields=[PK, TS],
    )
    assert sorted(row[PK] for row in rows) == [1]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} <= ISO '2024-12-31T16:00:00Z'",
        output_fields=[PK, TS],
    )
    assert sorted(row[PK] for row in rows) == [0, 2]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} != ISO '2025-01-03T00:00:00Z'",
        output_fields=[PK, TS],
    )
    assert sorted(row[PK] for row in rows) == [0, 2]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} + INTERVAL 'P1D' <= ISO '2025-01-01T16:00:00Z'",
        output_fields=[PK, TS],
    )
    assert sorted(row[PK] for row in rows) == [0, 2]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} is null",
        output_fields=[PK, TS],
    )
    assert rows == [{PK: 3, TS: None}]


def test_milvus_client_timestamptz_default_value(milvus_client):
    collection = "ts_default"
    milvus_client.create_collection(
        collection,
        schema=_schema(default_value="2025-01-01T00:00:00Z"),
    )
    milvus_client.insert(collection, [
        {PK: 0, VEC: [1.0, 2.0, 3.0]},
        {PK: 1, VEC: [4.0, 5.0, 6.0], TS: None},
    ])

    rows = milvus_client.query(collection, filter=f"{PK} >= 0", output_fields=[PK, TS])
    got = _by_id(rows)
    assert got[0][TS] == "2025-01-01T00:00:00Z"
    assert got[1][TS] == "2025-01-01T00:00:00Z"


def test_milvus_client_timestamptz_default_value_uses_collection_timezone(milvus_client):
    collection = "ts_default_timezone"
    milvus_client.create_collection(
        collection,
        schema=_schema(default_value="2025-01-01T00:00:00"),
        properties={"timezone": "Asia/Shanghai"},
    )
    milvus_client.insert(collection, [
        {PK: 0, VEC: [1.0, 2.0, 3.0]},
        {PK: 1, VEC: [4.0, 5.0, 6.0], TS: None},
    ])

    rows = milvus_client.query(collection, filter=f"{PK} >= 0", output_fields=[PK, TS])
    got = _by_id(rows)
    assert got[0][TS] == "2024-12-31T16:00:00Z"
    assert got[1][TS] == "2024-12-31T16:00:00Z"


def test_milvus_client_timestamptz_search_filter(milvus_client):
    collection = "ts_search"
    milvus_client.create_collection(collection, schema=_schema())
    milvus_client.insert(collection, _rows())

    results = milvus_client.search(
        collection,
        data=[[1.0, 2.0, 3.0]],
        limit=10,
        filter=f"{TS} <= ISO '2024-12-31T16:00:00Z'",
        output_fields=[PK, TS],
    )

    assert len(results) == 1
    assert sorted(hit["id"] for hit in results[0]) == [0, 2]
    entities = {hit["id"]: hit["entity"] for hit in results[0]}
    assert entities[0][TS] == "2024-12-31T16:00:00Z"
    assert entities[2][TS] == "2024-02-28T21:00:00Z"


def test_milvus_client_timestamptz_insert_delete_upsert_flush(milvus_client):
    collection = "ts_dml"
    milvus_client.create_collection(collection, schema=_schema())
    milvus_client.insert(collection, _rows())

    milvus_client.delete(collection, filter=f"{PK} < 2")
    milvus_client.flush(collection)
    rows = milvus_client.query(collection, filter=f"{PK} >= 0", output_fields=[PK, TS])
    assert sorted(row[PK] for row in rows) == [2, 3]

    milvus_client.upsert(collection, [
        {PK: 0, VEC: [1.0, 1.0, 1.0], TS: "2030-01-01T00:00:00Z"},
        {PK: 2, VEC: [2.0, 2.0, 2.0], TS: "2030-01-02T00:00:00+08:00"},
    ])
    milvus_client.flush(collection)

    rows = milvus_client.query(collection, filter=f"{PK} >= 0", output_fields=[PK, TS])
    got = _by_id(rows)
    assert sorted(got) == [0, 2, 3]
    assert got[0][TS] == "2030-01-01T00:00:00Z"
    assert got[2][TS] == "2030-01-01T16:00:00Z"
    assert got[3][TS] is None


def test_milvus_client_timestamptz_invalid_time_format(milvus_client):
    collection = "ts_invalid"
    milvus_client.create_collection(collection, schema=_schema())

    bad_values = [
        "invalid_time_format",
        "2025-04-31T00:00:00Z",
        "2025-02-29T00:00:00Z",
        "2025-01-01T00:00:00+24:00",
    ]
    for i, value in enumerate(bad_values):
        with pytest.raises(Exception, match="TIMESTAMPTZ|timestamp|timezone|isoformat|ISO"):
            milvus_client.insert(collection, [{PK: i, VEC: [1.0, 2.0, 3.0], TS: value}])


def test_milvus_client_timestamptz_upstream_edge_cases(milvus_client):
    collection = "ts_edge_cases"
    milvus_client.create_collection(collection, schema=_schema(nullable=False))
    rows = [
        {PK: 0, VEC: [1.0, 2.0, 3.0], TS: "1970-01-01T00:00:00Z"},
        {PK: 1, VEC: [4.0, 5.0, 6.0], TS: "9999-12-31T23:59:59Z"},
        {PK: 2, VEC: [7.0, 8.0, 9.0], TS: "2000-01-01T00:00:00+01:00"},
    ]
    milvus_client.insert(collection, rows)

    returned = milvus_client.query(collection, filter=f"{PK} >= 0", output_fields=[PK, TS])
    got = _by_id(returned)
    assert got[0][TS] == "1970-01-01T00:00:00Z"
    assert got[1][TS] == "9999-12-31T23:59:59Z"
    assert got[2][TS] == "1999-12-31T23:00:00Z"

    rows = milvus_client.query(
        collection,
        filter=f"{TS} > ISO '9000-01-01T00:00:00Z'",
        output_fields=[PK],
    )
    assert rows == [{PK: 1}]


def test_milvus_client_timestamptz_feb_29(milvus_client):
    collection = "ts_feb29"
    milvus_client.create_collection(collection, schema=_schema(nullable=False))
    milvus_client.insert(
        collection,
        [{PK: 10, VEC: [13.0, 14.0, 15.0], TS: "2024-02-29T00:00:00+03:00"}],
    )

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2024-02-28T21:00:00Z'",
        output_fields=[PK, TS],
    )
    assert rows == [{PK: 10, TS: "2024-02-28T21:00:00Z"}]


def test_milvus_client_timestamptz_request_timezone_filter_matrix(milvus_client):
    collection = "ts_request_timezone_matrix"
    milvus_client.create_collection(
        collection,
        schema=_schema(nullable=False),
        properties={"timezone": "Asia/Shanghai"},
    )
    milvus_client.insert(collection, [
        {PK: 0, VEC: [1.0, 2.0, 3.0], TS: "2024-12-31T22:00:00Z"},
        {PK: 1, VEC: [4.0, 5.0, 6.0], TS: "2025-01-01T06:00:00+08:00"},
        {PK: 2, VEC: [7.0, 8.0, 9.0], TS: "2024-12-31T17:00:00Z"},
    ])

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2024-12-31T22:00:00Z'",
        output_fields=[PK],
    )
    assert sorted(row[PK] for row in rows) == [0, 1]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2024-12-31 17:00:00'",
        output_fields=[PK],
        timezone="America/New_York",
    )
    assert sorted(row[PK] for row in rows) == [0, 1]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01 06:00:00'",
        output_fields=[PK],
        timezone="Asia/Shanghai",
    )
    assert sorted(row[PK] for row in rows) == [0, 1]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2024-12-31 17:00:00'",
        output_fields=[PK],
        timezone="Asia/Shanghai",
    )
    assert rows == []


def test_milvus_client_timestamptz_collection_timezone_property(milvus_client):
    collection = "ts_timezone_followup"
    milvus_client.create_collection(
        collection,
        schema=_schema(),
        properties={"timezone": "Asia/Shanghai"},
    )
    assert _describe_properties(milvus_client, collection)["timezone"] == "Asia/Shanghai"

    milvus_client.insert(collection, [{PK: 0, VEC: [1.0, 2.0, 3.0], TS: "2025-01-01T00:00:00"}])
    rows = milvus_client.query(collection, filter=f"{PK} == 0", output_fields=[PK, TS])
    assert rows[0][TS] == "2024-12-31T16:00:00Z"

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK, TS],
        timezone="UTC",
    )
    assert rows == []

    rows = milvus_client.search(
        collection,
        data=[[1.0, 2.0, 3.0]],
        limit=10,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK, TS],
        timezone="Asia/Shanghai",
    )
    assert [hit["id"] for hit in rows[0]] == [0]

    rows = milvus_client.query(
        collection,
        filter=f"{PK} == 0",
        output_fields=[PK, TS],
        timezone="Asia/Shanghai",
        time_fields="year, month, day, hour, minute, second, microsecond",
    )
    assert rows == [{PK: 0, TS: [2025, 1, 1, 0, 0, 0, 0]}]

    rows = milvus_client.search(
        collection,
        data=[[1.0, 2.0, 3.0]],
        limit=10,
        filter=f"{PK} == 0",
        output_fields=[PK, TS],
        timezone="UTC",
        time_fields="year, month, day, hour",
    )
    assert rows[0][0]["entity"][TS] == [2024, 12, 31, 16]


def test_milvus_client_timestamptz_alter_collection_timezone(milvus_client):
    collection = "ts_alter_timezone"
    milvus_client.create_collection(
        collection,
        schema=_schema(),
        properties={"timezone": "UTC"},
    )
    milvus_client.insert(collection, [
        {PK: 0, VEC: [1.0, 2.0, 3.0], TS: "2025-01-01T00:00:00Z"},
        {PK: 1, VEC: [4.0, 5.0, 6.0], TS: "2024-12-31T16:00:00Z"},
    ])

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK],
    )
    assert rows == [{PK: 0}]

    milvus_client.alter_collection_properties(
        collection,
        properties={"timezone": "Asia/Shanghai"},
    )
    assert _describe_properties(milvus_client, collection)["timezone"] == "Asia/Shanghai"

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK],
    )
    assert rows == [{PK: 1}]


def test_milvus_client_timestamptz_alter_collection_timezone_after_insert(milvus_client):
    collection = "ts_alter_timezone_after_insert"
    milvus_client.create_collection(
        collection,
        schema=_schema(nullable=False),
        properties={"timezone": "UTC"},
    )
    milvus_client.insert(collection, [
        {PK: 0, VEC: [1.0, 2.0, 3.0], TS: "2025-01-01T00:00:00"},
    ])

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK],
    )
    assert rows == [{PK: 0}]

    milvus_client.alter_collection_properties(
        collection,
        properties={"timezone": "America/New_York"},
    )
    assert _describe_properties(milvus_client, collection)["timezone"] == "America/New_York"

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK],
    )
    assert rows == []

    milvus_client.insert(collection, [
        {PK: 1, VEC: [4.0, 5.0, 6.0], TS: "2025-01-01T00:00:00"},
    ])
    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK],
    )
    assert rows == [{PK: 1}]


def test_milvus_client_timestamptz_database_timezone_default(milvus_client):
    collection = "ts_database_timezone_default"
    try:
        milvus_client.alter_database_properties(
            "default",
            properties={"timezone": "Asia/Shanghai"},
        )
        assert milvus_client.describe_database("default")["timezone"] == "Asia/Shanghai"

        milvus_client.create_collection(collection, schema=_schema(nullable=False))
        milvus_client.insert(collection, [
            {PK: 0, VEC: [1.0, 2.0, 3.0], TS: "2025-01-01T00:00:00"},
        ])
        rows = milvus_client.query(collection, filter=f"{PK} == 0", output_fields=[PK, TS])
        assert rows == [{PK: 0, TS: "2024-12-31T16:00:00Z"}]

        milvus_client.alter_database_properties(
            "default",
            properties={"timezone": "America/New_York"},
        )
        milvus_client.insert(collection, [
            {PK: 1, VEC: [4.0, 5.0, 6.0], TS: "2025-01-01T00:00:00"},
        ])

        rows = milvus_client.query(
            collection,
            filter=f"{TS} == ISO '2025-01-01T00:00:00'",
            output_fields=[PK],
        )
        assert rows == [{PK: 1}]
    finally:
        milvus_client.drop_database_properties("default", ["timezone"])


def test_milvus_client_timestamptz_different_time_expressions(milvus_client):
    collection = "ts_different_time_expressions"
    milvus_client.create_collection(
        collection,
        schema=_schema(nullable=False),
        properties={"timezone": "Asia/Shanghai"},
    )
    milvus_client.insert(collection, [
        {PK: 40, VEC: [1.0, 2.0, 3.0], TS: "2024-12-31 22:00:00Z"},
        {PK: 41, VEC: [4.0, 5.0, 6.0], TS: "2024-12-31 22:00:00"},
        {PK: 42, VEC: [7.0, 8.0, 9.0], TS: "2024-12-31T22:00:00"},
        {PK: 43, VEC: [10.0, 11.0, 12.0], TS: "2024-12-31T22:00:00+08:00"},
        {PK: 44, VEC: [13.0, 14.0, 15.0], TS: "2024-12-31T22:00:00-08:00"},
        {PK: 45, VEC: [16.0, 17.0, 18.0], TS: "2024-12-31T22:00:00Z"},
        {PK: 46, VEC: [19.0, 20.0, 21.0], TS: "2024-12-31 22:00:00+08:00"},
        {PK: 47, VEC: [22.0, 23.0, 24.0], TS: "2024-12-31 22:00:00-08:00"},
    ])

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2024-12-31 22:00:00'",
        output_fields=[PK],
    )
    assert sorted(row[PK] for row in rows) == [41, 42, 43, 46]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2024-12-31T22:00:00Z'",
        output_fields=[PK],
    )
    assert sorted(row[PK] for row in rows) == [40, 45]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2024-12-31 22:00:00'",
        output_fields=[PK],
        timezone="America/Los_Angeles",
    )
    assert sorted(row[PK] for row in rows) == [44, 47]


def test_milvus_client_timestamptz_different_timezone_query(milvus_client):
    collection = "ts_different_timezone_query"
    milvus_client.create_collection(
        collection,
        schema=_schema(nullable=False),
        properties={"timezone": "UTC"},
    )
    milvus_client.insert(collection, [
        {PK: 50, VEC: [1.0, 2.0, 3.0], TS: "2025-01-01T00:00:00Z"},
        {PK: 51, VEC: [4.0, 5.0, 6.0], TS: "2024-12-31T16:00:00Z"},
        {PK: 52, VEC: [7.0, 8.0, 9.0], TS: "2025-01-01T05:00:00Z"},
    ])

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK],
    )
    assert rows == [{PK: 50}]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK],
        timezone="Asia/Shanghai",
    )
    assert rows == [{PK: 51}]

    rows = milvus_client.query(
        collection,
        filter=f"{TS} == ISO '2025-01-01T00:00:00'",
        output_fields=[PK],
        timezone="America/New_York",
    )
    assert rows == [{PK: 52}]
