"""TIMESTAMPTZ collection-level behavior."""

from datetime import datetime, timezone

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.timestamptz import micros_to_utc_datetime
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def _schema() -> CollectionSchema:
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="tsz", dtype=DataType.TIMESTAMPTZ, nullable=True),
    ])


def test_timestamptz_query_and_restart(tmp_path):
    data_dir = str(tmp_path / "data")
    col = Collection("events", data_dir, _schema())
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "tsz": "2025-01-01T00:00:00+08:00"},
        {"id": 2, "vec": [0.2, 0.3], "tsz": "2025-01-03T00:00:00Z"},
        {"id": 3, "vec": [0.3, 0.4], "tsz": None},
    ])

    rows = col.query(
        "tsz > ISO '2025-01-02T00:00:00Z'",
        output_fields=["id", "tsz"],
    )
    assert [r["id"] for r in rows] == [2]
    assert rows[0]["tsz"] == datetime(2025, 1, 3, tzinfo=timezone.utc)

    rows = col.query(
        "tsz + INTERVAL 'P1D' <= ISO '2025-01-02T16:00:00Z'",
        output_fields=["id"],
    )
    assert [r["id"] for r in rows] == [1]
    col.close()

    reopened = Collection("events", data_dir, _schema())
    got = reopened.get([1], output_fields=["id", "tsz"])
    assert got[0]["tsz"] == micros_to_utc_datetime("2025-01-01T00:00:00+08:00")
    reopened.close()
