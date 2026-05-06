"""Upsert with partial update — merge new fields onto existing records."""

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT, nullable=True),
    ])


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    yield c
    c.close()


def test_upsert_partial_updates_existing_record(col):
    """Upsert with only some fields should merge onto the old record."""
    col.insert([{"id": 1, "vec": [1, 0, 0, 0], "title": "hello", "score": 0.5}])
    col.load()

    # Partial upsert: only change title, keep vec and score
    col.upsert([{"id": 1, "title": "world"}])

    result = col.get([1])[0]
    assert result["title"] == "world"
    assert result["vec"] == [1.0, 0.0, 0.0, 0.0]
    assert result["score"] == pytest.approx(0.5)


def test_upsert_partial_new_record_needs_all_fields(col):
    """Upsert a new pk — must provide all required fields."""
    col.insert([{"id": 1, "vec": [1, 0, 0, 0]}])
    col.load()

    # New pk with all fields works
    col.upsert([{"id": 2, "vec": [0, 1, 0, 0], "title": "new"}])
    result = col.get([2])[0]
    assert result["title"] == "new"


def test_upsert_partial_after_flush(col):
    """Partial upsert should read old records from flushed segments."""
    col.insert([{"id": 1, "vec": [1, 0, 0, 0], "title": "original", "score": 1.0}])
    col.flush()
    col.load()

    col.upsert([{"id": 1, "score": 9.9}])
    result = col.get([1])[0]
    assert result["title"] == "original"
    assert result["score"] == pytest.approx(9.9)


def test_upsert_partial_multiple_records(col):
    """Mix of existing and new records in one upsert call."""
    col.insert([
        {"id": 1, "vec": [1, 0, 0, 0], "title": "a", "score": 1.0},
        {"id": 2, "vec": [0, 1, 0, 0], "title": "b", "score": 2.0},
    ])
    col.load()

    col.upsert([
        {"id": 1, "title": "a-updated"},          # partial update
        {"id": 3, "vec": [0, 0, 1, 0], "title": "c"},  # new record
    ])

    r1 = col.get([1])[0]
    assert r1["title"] == "a-updated"
    assert r1["vec"] == [1.0, 0.0, 0.0, 0.0]

    r3 = col.get([3])[0]
    assert r3["title"] == "c"


def test_upsert_full_record_overwrites(col):
    """Upsert with all fields behaves like a full replace."""
    col.insert([{"id": 1, "vec": [1, 0, 0, 0], "title": "old", "score": 1.0}])
    col.load()

    col.upsert([{"id": 1, "vec": [0, 0, 0, 1], "title": "new", "score": 9.0}])
    result = col.get([1])[0]
    assert result["vec"] == [0.0, 0.0, 0.0, 1.0]
    assert result["title"] == "new"
    assert result["score"] == pytest.approx(9.0)


def test_upsert_partial_with_dynamic_fields(tmp_path):
    """Partial upsert should preserve dynamic fields from old record."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ],
        enable_dynamic_field=True,
    )
    col = Collection("test", str(tmp_path / "data"), schema)
    col.insert([{"id": 1, "vec": [1, 0, 0, 0], "color": "red", "tag": "a"}])
    col.load()

    # Partial upsert: update color, keep tag
    col.upsert([{"id": 1, "color": "blue"}])
    result = col.get([1])[0]
    assert result["color"] == "blue"
    assert result["tag"] == "a"
    assert result["vec"] == [1.0, 0.0, 0.0, 0.0]
    col.close()
