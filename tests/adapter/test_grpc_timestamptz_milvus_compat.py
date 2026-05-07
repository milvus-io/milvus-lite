"""MilvusClient TIMESTAMPTZ compatibility tests.

Migrated and reduced from:
``~/Workspace/dev/milvus/tests/python_client/milvus_client/test_milvus_client_timestamptz.py``.

MilvusLite currently implements the core TIMESTAMPTZ surface: schema,
insert/query/search filters, nullable/default values, and UTC
normalization. Milvus timezone properties, schema evolution, and
STL_SORT timestamp indexes are tracked as follow-up work in the roadmap
and intentionally excluded here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymilvus")
from pymilvus import DataType, MilvusClient  # noqa: E402


DIM = 3
PK = "id"
VEC = "vector"
TS = "timestamp"


def _schema(nullable: bool = True, default_value=None):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(PK, DataType.INT64, is_primary=True)
    schema.add_field(VEC, DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(TS, DataType.TIMESTAMPTZ, nullable=nullable, default_value=default_value)
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


@pytest.mark.xfail(reason="MilvusLite does not yet implement timezone properties for naive TIMESTAMPTZ values")
def test_milvus_client_timestamptz_naive_timezone_property_followup(milvus_client):
    collection = "ts_timezone_followup"
    milvus_client.create_collection(
        collection,
        schema=_schema(),
        properties={"timezone": "Asia/Shanghai"},
    )
    milvus_client.insert(collection, [{PK: 0, VEC: [1.0, 2.0, 3.0], TS: "2025-01-01T00:00:00"}])
    rows = milvus_client.query(collection, filter=f"{PK} == 0", output_fields=[PK, TS])
    assert rows[0][TS] == "2024-12-31T16:00:00Z"
