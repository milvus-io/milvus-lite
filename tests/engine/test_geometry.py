import pytest

from milvus_lite.db import MilvusLite
from milvus_lite.engine.collection import Collection
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.search.filter.exceptions import FilterTypeError


def _schema(nullable=False):
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="shape", dtype=DataType.GEOMETRY, nullable=nullable),
    ])


def test_geometry_query_round_trip(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)"},
        {"id": 2, "vec": [0.2, 0.3], "shape": "POINT(20 20)"},
    ])

    rows = col.query(
        "geometry_within(shape, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
        output_fields=["shape"],
    )

    assert rows == [{"id": 1, "shape": "POINT(1 1)"}]


def test_geometry_st_alias_query(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)"},
        {"id": 2, "vec": [0.2, 0.3], "shape": "POINT(20 20)"},
    ])

    rows = col.query(
        "ST_WITHIN(shape, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
        output_fields=["shape"],
    )

    assert rows == [{"id": 1, "shape": "POINT(1 1)"}]


def test_geometry_search_filter(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([
        {"id": 1, "vec": [0.0, 1.0], "shape": "POINT(20 20)"},
        {"id": 2, "vec": [1.0, 0.0], "shape": "POINT(1 1)"},
    ])

    results = col.search(
        [[1.0, 0.0]],
        top_k=10,
        metric_type="L2",
        expr="ST_WITHIN(shape, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
        output_fields=["shape"],
    )

    assert [[hit["id"] for hit in hits] for hits in results] == [[2]]
    assert results[0][0]["entity"] == {"shape": "POINT(1 1)"}


def test_geometry_search_output_fields_after_flush(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([
        {"id": 1, "vec": [0.0, 1.0], "shape": "POINT(20 20)"},
        {"id": 2, "vec": [1.0, 0.0], "shape": "POINT(1 1)"},
    ])
    col.flush()

    results = col.search(
        [[1.0, 0.0]],
        top_k=1,
        metric_type="L2",
        output_fields=["shape"],
    )

    assert results[0][0]["id"] == 2
    assert results[0][0]["entity"] == {"shape": "POINT(1 1)"}


def test_geometry_multiple_fields(tmp_path):
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="origin", dtype=DataType.GEOMETRY),
        FieldSchema(name="area", dtype=DataType.GEOMETRY),
    ])
    col = Collection("geo", str(tmp_path / "geo"), schema)
    col.insert([
        {
            "id": 1,
            "vec": [0.1, 0.2],
            "origin": "POINT(1 1)",
            "area": "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
        },
        {
            "id": 2,
            "vec": [0.2, 0.3],
            "origin": "POINT(20 20)",
            "area": "POLYGON((15 15, 30 15, 30 30, 15 30, 15 15))",
        },
    ])

    rows = col.query(
        "ST_CONTAINS(area, 'POINT(5 5)')",
        output_fields=["origin", "area"],
    )

    assert rows == [{
        "id": 1,
        "origin": "POINT(1 1)",
        "area": "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
    }]


def test_geometry_nullable_field(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema(nullable=True))
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)"},
        {"id": 2, "vec": [0.2, 0.3], "shape": None},
    ])

    rows = col.query(
        "ST_WITHIN(shape, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
        output_fields=["shape"],
    )

    assert rows == [{"id": 1, "shape": "POINT(1 1)"}]
    assert col.get([2])[0]["shape"] is None


def test_geometry_combined_scalar_and_spatial_filter(tmp_path):
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="shape", dtype=DataType.GEOMETRY),
        FieldSchema(name="city", dtype=DataType.VARCHAR),
    ])
    col = Collection("geo", str(tmp_path / "geo"), schema)
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)", "city": "shanghai"},
        {"id": 2, "vec": [0.2, 0.3], "shape": "POINT(2 2)", "city": "beijing"},
        {"id": 3, "vec": [0.3, 0.4], "shape": "POINT(20 20)", "city": "shanghai"},
    ])

    rows = col.query(
        "city == 'shanghai' and ST_WITHIN(shape, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
        output_fields=["shape", "city"],
    )

    assert rows == [{"id": 1, "shape": "POINT(1 1)", "city": "shanghai"}]


def test_geometry_dwithin_query(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)"},
        {"id": 2, "vec": [0.2, 0.3], "shape": "POINT(4 5)"},
        {"id": 3, "vec": [0.3, 0.4], "shape": "POINT(10 10)"},
    ])

    rows = col.query("ST_DWITHIN(shape, 'POINT(1 1)', 5)", output_fields=["shape"])

    assert rows == [
        {"id": 1, "shape": "POINT(1 1)"},
        {"id": 2, "shape": "POINT(4 5)"},
    ]


def test_geometry_dwithin_polygon_boundary(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))"},
        {"id": 2, "vec": [0.2, 0.3], "shape": "POLYGON((10 10, 12 10, 12 12, 10 12, 10 10))"},
    ])

    rows = col.query("ST_DWITHIN(shape, 'POINT(4 1)', 2)", output_fields=["shape"])

    assert rows == [{
        "id": 1,
        "shape": "POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))",
    }]


def test_geometry_isvalid_query(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema(nullable=True))
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)"},
        {"id": 2, "vec": [0.2, 0.3], "shape": None},
    ])

    rows = col.query("ST_ISVALID(shape)", output_fields=["shape"])

    assert rows == [{"id": 1, "shape": "POINT(1 1)"}]


def test_geometry_upsert_and_delete(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POINT(20 20)"},
        {"id": 2, "vec": [0.2, 0.3], "shape": "POINT(5 5)"},
    ])
    col.upsert([{"id": 1, "shape": "POINT(1 1)"}])
    col.delete([2])

    rows = col.query(
        "ST_WITHIN(shape, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
        output_fields=["shape"],
    )

    assert rows == [{"id": 1, "shape": "POINT(1 1)"}]


def test_geometry_default_value(tmp_path):
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="shape", dtype=DataType.GEOMETRY, default_value="POINT(0 0)"),
    ])
    col = Collection("geo", str(tmp_path / "geo"), schema)
    col.insert([{"id": 1, "vec": [0.1, 0.2]}])

    rows = col.query(output_fields=["shape"])

    assert rows == [{"id": 1, "shape": "POINT(0 0)"}]


def test_geometry_invalid_wkt_rejected(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())

    with pytest.raises(SchemaValidationError, match="GEOMETRY"):
        col.insert([{"id": 1, "vec": [0.1, 0.2], "shape": "not-wkt"}])


def test_geometry_query_after_db_reopen(tmp_path):
    data_dir = str(tmp_path / "db")
    db = MilvusLite(data_dir)
    col = db.create_collection("geo", _schema())
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)"},
        {"id": 2, "vec": [0.2, 0.3], "shape": "POINT(20 20)"},
    ])
    col.flush()
    db.close()

    reopened = MilvusLite(data_dir)
    try:
        col = reopened.get_collection("geo")
        rows = col.query(
            "ST_WITHIN(shape, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')",
            output_fields=["shape"],
        )
    finally:
        reopened.close()

    assert rows == [{"id": 1, "shape": "POINT(1 1)"}]


def test_geometry_filter_on_non_geometry_field_rejected(tmp_path):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([{"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)"}])

    with pytest.raises(FilterTypeError, match="GEOMETRY field"):
        col.query("ST_WITHIN(id, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')")


@pytest.mark.parametrize(
    "expr",
    [
        "ST_DWITHIN(shape, 'POINT(1 1)', -1)",
        "ST_DWITHIN(shape, 'POINT(1 1)')",
        "ST_DWITHIN(shape, 'POINT(1 1)', 'far')",
    ],
)
def test_geometry_dwithin_invalid_parameters_rejected(tmp_path, expr):
    col = Collection("geo", str(tmp_path / "geo"), _schema())
    col.insert([{"id": 1, "vec": [0.1, 0.2], "shape": "POINT(1 1)"}])

    with pytest.raises(Exception):
        col.query(expr)
