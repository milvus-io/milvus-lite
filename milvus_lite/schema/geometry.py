"""Geometry WKT helpers for the supported spatial filter subset."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from milvus_lite.exceptions import SchemaValidationError


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Polygon:
    points: tuple[Point, ...]


Geometry = Point | Polygon

_POINT_RE = re.compile(r"^\s*POINT\s*\(\s*([^\s,()]+)\s+([^\s,()]+)\s*\)\s*$", re.IGNORECASE)
_POLYGON_RE = re.compile(r"^\s*POLYGON\s*\(\s*\(\s*(.*?)\s*\)\s*\)\s*$", re.IGNORECASE | re.DOTALL)


def validate_geometry_wkt(value: str) -> str:
    parse_wkt(value)
    return value


def geometry_is_valid(value: str) -> bool:
    try:
        parse_wkt(value)
    except SchemaValidationError:
        return False
    return True


def parse_wkt(value: object) -> Geometry:
    if not isinstance(value, str):
        raise SchemaValidationError("GEOMETRY value must be a WKT string")
    point_match = _POINT_RE.match(value)
    if point_match:
        return Point(_parse_float(point_match.group(1), value), _parse_float(point_match.group(2), value))
    polygon_match = _POLYGON_RE.match(value)
    if polygon_match:
        return _parse_polygon(polygon_match.group(1), value)
    raise SchemaValidationError(
        "GEOMETRY value must be WKT POINT(x y) or POLYGON((x y, ...))"
    )


def geometry_contains(left: str, right: str) -> bool:
    a = parse_wkt(left)
    b = parse_wkt(right)
    if isinstance(a, Polygon) and isinstance(b, Point):
        return _point_in_polygon(b, a, include_boundary=True)
    if isinstance(a, Point) and isinstance(b, Point):
        return a == b
    if isinstance(a, Polygon) and isinstance(b, Polygon):
        return all(_point_in_polygon(p, a, include_boundary=True) for p in b.points[:-1])
    return False


def geometry_within(left: str, right: str) -> bool:
    return geometry_contains(right, left)


def geometry_intersects(left: str, right: str) -> bool:
    a = parse_wkt(left)
    b = parse_wkt(right)
    if isinstance(a, Point) and isinstance(b, Point):
        return a == b
    if isinstance(a, Polygon) and isinstance(b, Point):
        return _point_in_polygon(b, a, include_boundary=True)
    if isinstance(a, Point) and isinstance(b, Polygon):
        return _point_in_polygon(a, b, include_boundary=True)
    if isinstance(a, Polygon) and isinstance(b, Polygon):
        return _polygons_intersect(a, b)
    return False


def geometry_dwithin(left: str, right: str, distance: float) -> bool:
    if distance < 0:
        raise SchemaValidationError("ST_DWITHIN distance must be non-negative")
    return _geometry_distance(parse_wkt(left), parse_wkt(right)) <= distance + 1e-9


def _geometry_distance(a: Geometry, b: Geometry) -> float:
    if geometry_intersects_wkt(a, b):
        return 0.0
    if isinstance(a, Point) and isinstance(b, Point):
        return _point_distance(a, b)
    if isinstance(a, Point) and isinstance(b, Polygon):
        return _point_polygon_distance(a, b)
    if isinstance(a, Polygon) and isinstance(b, Point):
        return _point_polygon_distance(b, a)
    if isinstance(a, Polygon) and isinstance(b, Polygon):
        return _polygon_polygon_distance(a, b)
    return math.inf


def geometry_intersects_wkt(a: Geometry, b: Geometry) -> bool:
    if isinstance(a, Point) and isinstance(b, Point):
        return a == b
    if isinstance(a, Polygon) and isinstance(b, Point):
        return _point_in_polygon(b, a, include_boundary=True)
    if isinstance(a, Point) and isinstance(b, Polygon):
        return _point_in_polygon(a, b, include_boundary=True)
    if isinstance(a, Polygon) and isinstance(b, Polygon):
        return _polygons_intersect(a, b)
    return False


def _parse_polygon(body: str, original: str) -> Polygon:
    points: list[Point] = []
    for part in body.split(","):
        coords = part.strip().split()
        if len(coords) != 2:
            raise SchemaValidationError(
                "GEOMETRY POLYGON coordinates must be pairs of x y values"
            )
        points.append(Point(_parse_float(coords[0], original), _parse_float(coords[1], original)))
    if len(points) < 4:
        raise SchemaValidationError("GEOMETRY POLYGON requires at least 4 coordinate pairs")
    if points[0] != points[-1]:
        raise SchemaValidationError("GEOMETRY POLYGON ring must be closed")
    polygon = Polygon(tuple(points))
    _validate_simple_polygon(polygon)
    return polygon


def _parse_float(raw: str, original: str) -> float:
    try:
        value = float(raw)
    except ValueError as e:
        raise SchemaValidationError(f"GEOMETRY value {original!r} has non-numeric coordinate {raw!r}") from e
    if not math.isfinite(value):
        raise SchemaValidationError(f"GEOMETRY value {original!r} has non-finite coordinate {raw!r}")
    return value


def _validate_simple_polygon(polygon: Polygon) -> None:
    edges = list(zip(polygon.points, polygon.points[1:]))
    last = len(edges) - 1
    for i, (a1, a2) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            b1, b2 = edges[j]
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == last:
                continue
            if _segments_intersect(a1, a2, b1, b2):
                raise SchemaValidationError("GEOMETRY POLYGON ring must not self-intersect")


def _point_in_polygon(point: Point, polygon: Polygon, include_boundary: bool) -> bool:
    inside = False
    pts = polygon.points
    j = len(pts) - 1
    for i, pi in enumerate(pts):
        pj = pts[j]
        if _point_on_segment(point, pj, pi):
            return include_boundary
        if (pi.y > point.y) != (pj.y > point.y):
            x_intersect = (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x
            if point.x < x_intersect:
                inside = not inside
        j = i
    return inside


def _point_on_segment(p: Point, a: Point, b: Point) -> bool:
    cross = (p.y - a.y) * (b.x - a.x) - (p.x - a.x) * (b.y - a.y)
    if abs(cross) > 1e-9:
        return False
    return min(a.x, b.x) - 1e-9 <= p.x <= max(a.x, b.x) + 1e-9 and min(a.y, b.y) - 1e-9 <= p.y <= max(a.y, b.y) + 1e-9


def _point_distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx = b.x - a.x
    dy = b.y - a.y
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return _point_distance(p, a)
    t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projected = Point(a.x + t * dx, a.y + t * dy)
    return _point_distance(p, projected)


def _point_polygon_distance(point: Point, polygon: Polygon) -> float:
    if _point_in_polygon(point, polygon, include_boundary=True):
        return 0.0
    return min(
        _point_segment_distance(point, a, b)
        for a, b in zip(polygon.points, polygon.points[1:])
    )


def _segment_segment_distance(a1: Point, a2: Point, b1: Point, b2: Point) -> float:
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        _point_segment_distance(a1, b1, b2),
        _point_segment_distance(a2, b1, b2),
        _point_segment_distance(b1, a1, a2),
        _point_segment_distance(b2, a1, a2),
    )


def _polygon_polygon_distance(a: Polygon, b: Polygon) -> float:
    return min(
        _segment_segment_distance(a1, a2, b1, b2)
        for a1, a2 in zip(a.points, a.points[1:])
        for b1, b2 in zip(b.points, b.points[1:])
    )


def _polygons_intersect(a: Polygon, b: Polygon) -> bool:
    a_edges = list(zip(a.points, a.points[1:]))
    b_edges = list(zip(b.points, b.points[1:]))
    for a1, a2 in a_edges:
        for b1, b2 in b_edges:
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return _point_in_polygon(a.points[0], b, include_boundary=True) or _point_in_polygon(b.points[0], a, include_boundary=True)


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    d1 = _orientation(a1, a2, b1)
    d2 = _orientation(a1, a2, b2)
    d3 = _orientation(b1, b2, a1)
    d4 = _orientation(b1, b2, a2)
    if d1 == 0 and _point_on_segment(b1, a1, a2):
        return True
    if d2 == 0 and _point_on_segment(b2, a1, a2):
        return True
    if d3 == 0 and _point_on_segment(a1, b1, b2):
        return True
    if d4 == 0 and _point_on_segment(a2, b1, b2):
        return True
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    if abs(value) <= 1e-9:
        return 0
    return 1 if value > 0 else -1
