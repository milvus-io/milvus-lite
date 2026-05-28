"""pymilvus GEOMETRY compatibility tests.

Migrated and reduced from Milvus python_client geometry coverage. MilvusLite
currently supports WKT POINT/simple POLYGON plus ST_CONTAINS, ST_WITHIN,
ST_INTERSECTS, ST_DWITHIN, and ST_ISVALID.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymilvus")
from pymilvus import DataType, MilvusClient  # noqa: E402


GEOMETRY = getattr(DataType, "GEOMETRY", None)
pytestmark = pytest.mark.skipif(GEOMETRY is None, reason="pymilvus lacks GEOMETRY DataType")

DIM = 2
PK = "id"
VEC = "vector"
GEO = "geo"
_MISSING = object()


def _schema(nullable: bool = False, default_value=_MISSING):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(PK, DataType.INT64, is_primary=True)
    schema.add_field(VEC, DataType.FLOAT_VECTOR, dim=DIM)
    kwargs = {"nullable": nullable}
    if default_value is not _MISSING:
        kwargs["default_value"] = default_value
    schema.add_field(GEO, GEOMETRY, **kwargs)
    return schema


def _vector_index(client: MilvusClient):
    index = client.prepare_index_params()
    index.add_index(
        field_name=VEC,
        index_type="BRUTE_FORCE",
        metric_type="L2",
        params={},
    )
    return index


def _rows():
    return [
        {PK: 1, VEC: [0.0, 0.0], GEO: "POINT(1 1)"},
        {PK: 2, VEC: [1.0, 0.0], GEO: "POINT(20 20)"},
        {PK: 3, VEC: [0.0, 1.0], GEO: "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"},
    ]


def _by_id(rows):
    return {row[PK]: row for row in rows}


def test_milvus_client_geometry_create_insert_query(milvus_client):
    collection = "geo_basic"
    milvus_client.create_collection(collection, schema=_schema())
    milvus_client.insert(collection, _rows())

    rows = milvus_client.query(
        collection,
        filter="id >= 0",
        output_fields=[PK, GEO],
    )

    got = _by_id(rows)
    assert got[1][GEO] == "POINT(1 1)"
    assert got[2][GEO] == "POINT(20 20)"
    assert got[3][GEO] == "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"


def test_milvus_client_geometry_spatial_predicates(milvus_client):
    collection = "geo_predicates"
    milvus_client.create_collection(collection, schema=_schema())
    milvus_client.insert(collection, _rows())

    rows = milvus_client.query(
        collection,
        filter="ST_WITHIN(geo, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
        output_fields=[PK, GEO],
    )
    assert sorted(row[PK] for row in rows) == [1, 3]

    rows = milvus_client.query(
        collection,
        filter="ST_CONTAINS(geo, 'POINT(5 5)')",
        output_fields=[PK, GEO],
    )
    assert [row[PK] for row in rows] == [3]

    rows = milvus_client.query(
        collection,
        filter="ST_INTERSECTS(geo, 'POINT(0 5)')",
        output_fields=[PK, GEO],
    )
    assert [row[PK] for row in rows] == [3]


def test_milvus_client_geometry_search_filter(milvus_client):
    collection = "geo_search"
    milvus_client.create_collection(collection, schema=_schema())
    milvus_client.insert(collection, _rows())
    milvus_client.create_index(collection, _vector_index(milvus_client))
    milvus_client.load_collection(collection)

    results = milvus_client.search(
        collection,
        data=[[0.0, 0.0]],
        limit=10,
        filter="ST_WITHIN(geo, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
        output_fields=[GEO],
    )

    assert len(results) == 1
    assert sorted(hit[PK] for hit in results[0]) == [1, 3]
    entities = {hit[PK]: hit["entity"] for hit in results[0]}
    assert entities[1][GEO] == "POINT(1 1)"
    assert entities[3][GEO] == "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"


def test_milvus_client_geometry_nullable_and_isvalid(milvus_client):
    collection = "geo_nullable"
    milvus_client.create_collection(collection, schema=_schema(nullable=True))
    milvus_client.insert(collection, [
        {PK: 1, VEC: [0.0, 0.0], GEO: "POINT(1 1)"},
        {PK: 2, VEC: [1.0, 0.0], GEO: None},
    ])

    rows = milvus_client.query(collection, filter="geo is null", output_fields=[PK, GEO])
    assert rows == [{PK: 2, GEO: None}]

    rows = milvus_client.query(collection, filter="ST_ISVALID(geo)", output_fields=[PK, GEO])
    assert rows == [{PK: 1, GEO: "POINT(1 1)"}]


def test_milvus_client_geometry_default_value(milvus_client):
    collection = "geo_default"
    milvus_client.create_collection(collection, schema=_schema(default_value="POINT(0 0)"))
    milvus_client.insert(collection, [
        {PK: 1, VEC: [0.0, 0.0]},
        {PK: 2, VEC: [1.0, 0.0], GEO: None},
        {PK: 3, VEC: [0.0, 1.0], GEO: "POINT(2 2)"},
    ])

    rows = milvus_client.query(collection, filter="id >= 0", output_fields=[PK, GEO])
    got = _by_id(rows)
    assert got[1][GEO] == "POINT(0 0)"
    assert got[2][GEO] == "POINT(0 0)"
    assert got[3][GEO] == "POINT(2 2)"


def test_milvus_client_geometry_dwithin(milvus_client):
    collection = "geo_dwithin"
    milvus_client.create_collection(collection, schema=_schema())
    milvus_client.insert(collection, [
        {PK: 1, VEC: [0.0, 0.0], GEO: "POINT(1 1)"},
        {PK: 2, VEC: [1.0, 0.0], GEO: "POINT(4 5)"},
        {PK: 3, VEC: [0.0, 1.0], GEO: "POINT(10 10)"},
    ])

    rows = milvus_client.query(
        collection,
        filter="ST_DWITHIN(geo, 'POINT(1 1)', 5)",
        output_fields=[PK, GEO],
    )

    assert sorted(row[PK] for row in rows) == [1, 2]


def test_milvus_client_geometry_invalid_wkt_rejected(milvus_client):
    collection = "geo_invalid_wkt"
    milvus_client.create_collection(collection, schema=_schema())

    with pytest.raises(Exception, match="GEOMETRY|WKT|syntax|parse"):
        milvus_client.insert(collection, [{PK: 1, VEC: [0.0, 0.0], GEO: "not-wkt"}])


def test_milvus_client_geometry_filter_on_non_geometry_field_rejected(milvus_client):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(PK, DataType.INT64, is_primary=True)
    schema.add_field(VEC, DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("name", DataType.VARCHAR, max_length=64)
    collection = "geo_wrong_field"
    milvus_client.create_collection(collection, schema=schema)
    milvus_client.insert(collection, [{PK: 1, VEC: [0.0, 0.0], "name": "POINT(1 1)"}])

    with pytest.raises(Exception, match="GEOMETRY|geometry|parse|plan"):
        milvus_client.query(collection, filter="ST_WITHIN(name, 'POINT(1 1)')")
