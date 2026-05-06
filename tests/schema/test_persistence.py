"""Tests for schema/persistence.py"""

import json
import os

import pytest

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.schema.persistence import load_schema, save_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_schema() -> CollectionSchema:
    """A schema covering all DataTypes we care about persisting."""
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
            FieldSchema(name="b", dtype=DataType.BOOL),
            FieldSchema(name="i8", dtype=DataType.INT8),
            FieldSchema(name="i16", dtype=DataType.INT16),
            FieldSchema(name="i32", dtype=DataType.INT32),
            FieldSchema(name="i64", dtype=DataType.INT64),
            FieldSchema(name="f32", dtype=DataType.FLOAT),
            FieldSchema(name="f64", dtype=DataType.DOUBLE),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True, default_value="(none)"),
        ],
        version=3,
        enable_dynamic_field=True,
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    schema = _full_schema()
    path = str(tmp_path / "schema.json")
    save_schema(schema, "my_collection", path)

    name, loaded = load_schema(path)
    assert name == "my_collection"
    assert loaded.version == schema.version
    assert loaded.enable_dynamic_field == schema.enable_dynamic_field
    assert len(loaded.fields) == len(schema.fields)
    for src, dst in zip(schema.fields, loaded.fields):
        assert src.name == dst.name
        assert src.dtype == dst.dtype
        assert src.is_primary == dst.is_primary
        assert src.dim == dst.dim
        assert src.max_length == dst.max_length
        assert src.nullable == dst.nullable
        assert src.default_value == dst.default_value


def test_save_creates_parent_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    path = str(nested / "schema.json")
    save_schema(_full_schema(), "x", path)
    assert os.path.exists(path)


def test_save_atomic_no_tmp_left_behind(tmp_path):
    path = str(tmp_path / "schema.json")
    save_schema(_full_schema(), "x", path)
    assert not os.path.exists(path + ".tmp")


def test_save_overwrites_existing(tmp_path):
    path = str(tmp_path / "schema.json")
    save_schema(_full_schema(), "first", path)
    save_schema(_full_schema(), "second", path)
    name, _ = load_schema(path)
    assert name == "second"


# ---------------------------------------------------------------------------
# Load failures
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_schema(str(tmp_path / "nope.json"))


def test_load_corrupt_json_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ this is not json")
    with pytest.raises(SchemaValidationError, match="not valid JSON"):
        load_schema(str(path))


def test_load_root_not_object_raises(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(SchemaValidationError, match="root must be an object"):
        load_schema(str(path))


def test_load_missing_key_raises(tmp_path):
    path = tmp_path / "missing.json"
    path.write_text(json.dumps({"version": 1, "fields": []}))  # no collection_name
    with pytest.raises(SchemaValidationError, match="missing key"):
        load_schema(str(path))


def test_load_unknown_dtype_raises(tmp_path):
    path = tmp_path / "weird.json"
    path.write_text(json.dumps({
        "collection_name": "x",
        "version": 1,
        "fields": [
            {"name": "id", "dtype": "complex_number", "is_primary": True},
        ],
    }))
    with pytest.raises(SchemaValidationError, match="unknown dtype"):
        load_schema(str(path))


def test_load_field_not_object_raises(tmp_path):
    path = tmp_path / "bad_field.json"
    path.write_text(json.dumps({
        "collection_name": "x",
        "version": 1,
        "fields": ["just a string"],
    }))
    with pytest.raises(SchemaValidationError, match="must be an object"):
        load_schema(str(path))


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def test_save_is_deterministic(tmp_path):
    """Two saves of the same schema must produce byte-identical files."""
    schema = _full_schema()
    p1 = str(tmp_path / "a.json")
    p2 = str(tmp_path / "b.json")
    save_schema(schema, "x", p1)
    save_schema(schema, "x", p2)
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        assert f1.read() == f2.read()
