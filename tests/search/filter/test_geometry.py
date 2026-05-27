import pytest
import pyarrow as pa

from milvus_lite.schema.geometry import (
    geometry_contains,
    geometry_dwithin,
    geometry_intersects,
    geometry_is_valid,
    geometry_within,
    validate_geometry_wkt,
)
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.search.filter import compile_expr, evaluate, parse_expr
from milvus_lite.search.filter.exceptions import FilterTypeError
from milvus_lite.search.filter.ast import GeometryDWithinOp, GeometryIsValidOp, GeometryOp


def _schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="shape", dtype=DataType.GEOMETRY),
    ])


def test_geometry_helpers_point_polygon():
    square = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"
    assert geometry_contains(square, "POINT(5 5)") is True
    assert geometry_contains(square, "POINT(15 5)") is False
    assert geometry_within("POINT(5 5)", square) is True
    assert geometry_intersects(square, "POINT(0 5)") is True


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("POINT(1 1)", "POINT(1 1)", True),
        ("POINT(1 1)", "POINT(2 2)", False),
        ("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", "POINT(5 5)", True),
        ("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", "POINT(0 5)", True),
        ("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", "POINT(15 5)", False),
        (
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            True,
        ),
        (
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))",
            True,
        ),
    ],
)
def test_geometry_contains_truth_table(left, right, expected):
    assert geometry_contains(left, right) is expected


def test_geometry_contains_concave_polygon_rejects_crossing_inner_polygon():
    outer = "POLYGON((0 0, 4 0, 4 4, 3 4, 3 1, 1 1, 1 4, 0 4, 0 0))"
    crossing = "POLYGON((0.5 0.5, 3.5 0.5, 3.5 3.5, 0.5 3.5, 0.5 0.5))"
    contained = "POLYGON((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))"

    assert geometry_contains(outer, crossing) is False
    assert geometry_contains(outer, contained) is True


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("POINT(5 5)", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", True),
        ("POINT(0 5)", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", True),
        ("POINT(15 5)", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", False),
    ],
)
def test_geometry_within_truth_table(left, right, expected):
    assert geometry_within(left, right) is expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("POINT(1 1)", "POINT(1 1)", True),
        ("POINT(1 1)", "POINT(2 2)", False),
        ("POINT(0 5)", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", True),
        (
            "POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))",
            True,
        ),
        (
            "POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON((3 3, 4 3, 4 4, 3 4, 3 3))",
            False,
        ),
    ],
)
def test_geometry_intersects_truth_table(left, right, expected):
    assert geometry_intersects(left, right) is expected


@pytest.mark.parametrize(
    ("left", "right", "distance", "expected"),
    [
        ("POINT(1 1)", "POINT(4 5)", 5, True),
        ("POINT(1 1)", "POINT(4 5)", 4.999, False),
        ("POINT(1 1)", "POINT(1 1)", 0, True),
        ("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))", "POINT(4 1)", 2, True),
        ("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))", "POINT(4.1 1)", 2, False),
    ],
)
def test_geometry_dwithin_truth_table(left, right, distance, expected):
    assert geometry_dwithin(left, right, distance) is expected


def test_geometry_is_valid_rejects_non_finite_and_self_intersecting_wkt():
    assert geometry_is_valid("POINT(1 2)") is True
    assert geometry_is_valid("POINT(nan 2)") is False
    assert geometry_is_valid("POINT(inf 2)") is False
    assert geometry_is_valid("POLYGON((0 0, 2 2, 0 2, 2 0, 0 0))") is False


@pytest.mark.parametrize(
    "wkt",
    [
        "LINESTRING(0 0, 1 1)",
        "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)))",
        "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 1))",
        "POINT Z (1 2 3)",
    ],
)
def test_geometry_rejects_unsupported_wkt_subset(wkt):
    assert geometry_is_valid(wkt) is False
    with pytest.raises(Exception):
        validate_geometry_wkt(wkt)


def test_parse_geometry_predicate():
    expr = parse_expr("geometry_contains(shape, 'POINT(1 2)')")
    assert isinstance(expr, GeometryOp)
    assert expr.op == "geometry_contains"
    assert expr.field.name == "shape"
    assert expr.geometry.value == "POINT(1 2)"


def test_parse_st_geometry_aliases():
    aliases = {
        "ST_CONTAINS": "geometry_contains",
        "ST_WITHIN": "geometry_within",
        "ST_INTERSECTS": "geometry_intersects",
    }
    for alias, op in aliases.items():
        expr = parse_expr(f"{alias}(shape, 'POINT(1 2)')")
        assert isinstance(expr, GeometryOp)
        assert expr.op == op
        assert expr.field.name == "shape"
        assert expr.geometry.value == "POINT(1 2)"


def test_parse_st_isvalid():
    expr = parse_expr("ST_ISVALID(shape)")
    assert isinstance(expr, GeometryIsValidOp)
    assert expr.field.name == "shape"


def test_parse_st_dwithin():
    expr = parse_expr("ST_DWITHIN(shape, 'POINT(1 2)', 3.5)")
    assert isinstance(expr, GeometryDWithinOp)
    assert expr.field.name == "shape"
    assert expr.geometry.value == "POINT(1 2)"
    assert expr.distance.value == 3.5


def test_geometry_filter_uses_python_backend():
    compiled = compile_expr(
        parse_expr("geometry_contains(shape, 'POINT(5 5)')"),
        _schema(),
        "geometry_contains(shape, 'POINT(5 5)')",
    )

    assert compiled.backend == "python"


def test_geometry_filter_rejects_invalid_wkt_literal():
    with pytest.raises(FilterTypeError, match="valid WKT"):
        compile_expr(
            parse_expr("geometry_contains(shape, 'not-wkt')"),
            _schema(),
            "geometry_contains(shape, 'not-wkt')",
        )


def test_geometry_filter_evaluates():
    schema = _schema()
    table = pa.table({
        "id": [1, 2],
        "vec": [[0.1, 0.2], [0.2, 0.3]],
        "shape": [
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))",
        ],
    })
    compiled = compile_expr(
        parse_expr("geometry_contains(shape, 'POINT(5 5)')"),
        schema,
        "geometry_contains(shape, 'POINT(5 5)')",
    )

    result = evaluate(compiled, table)

    assert result.to_pylist() == [True, False]


def test_geometry_isvalid_filter_evaluates():
    schema = _schema()
    table = pa.table({
        "id": [1, 2],
        "vec": [[0.1, 0.2], [0.2, 0.3]],
        "shape": ["POINT(1 2)", "not-wkt"],
    })
    compiled = compile_expr(
        parse_expr("ST_ISVALID(shape)"),
        schema,
        "ST_ISVALID(shape)",
    )

    result = evaluate(compiled, table)

    assert result.to_pylist() == [True, False]


def test_geometry_dwithin_filter_evaluates():
    schema = _schema()
    table = pa.table({
        "id": [1, 2, 3],
        "vec": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
        "shape": ["POINT(1 1)", "POINT(4 5)", "POINT(10 10)"],
    })
    compiled = compile_expr(
        parse_expr("ST_DWITHIN(shape, 'POINT(1 1)', 5)"),
        schema,
        "ST_DWITHIN(shape, 'POINT(1 1)', 5)",
    )

    result = evaluate(compiled, table)

    assert result.to_pylist() == [True, True, False]
