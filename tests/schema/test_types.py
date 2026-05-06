"""Tests for schema/types.py — DataType enum, FieldSchema, CollectionSchema, TYPE_MAP."""

import pyarrow as pa

from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    TYPE_MAP,
)


# ---------------------------------------------------------------------------
# DataType enum
# ---------------------------------------------------------------------------

def test_datatype_values():
    assert DataType.BOOL.value == "bool"
    assert DataType.INT64.value == "int64"
    assert DataType.FLOAT_VECTOR.value == "float_vector"
    assert DataType.VARCHAR.value == "varchar"
    assert DataType.JSON.value == "json"


def test_datatype_members():
    expected = {
        "BOOL", "INT8", "INT16", "INT32", "INT64",
        "FLOAT", "DOUBLE", "VARCHAR", "JSON", "ARRAY",
        "FLOAT_VECTOR", "SPARSE_FLOAT_VECTOR",
    }
    assert set(DataType.__members__.keys()) == expected


# ---------------------------------------------------------------------------
# FieldSchema
# ---------------------------------------------------------------------------

def test_field_schema_defaults():
    f = FieldSchema(name="id", dtype=DataType.INT64)
    assert f.name == "id"
    assert f.dtype == DataType.INT64
    assert f.is_primary is False
    assert f.dim is None
    assert f.max_length is None
    assert f.nullable is False
    assert f.default_value is None


def test_field_schema_primary_key():
    f = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True)
    assert f.is_primary is True


def test_field_schema_vector():
    f = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128)
    assert f.dim == 128


def test_field_schema_varchar():
    f = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256)
    assert f.max_length == 256


# ---------------------------------------------------------------------------
# CollectionSchema
# ---------------------------------------------------------------------------

def test_collection_schema_defaults():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]
    schema = CollectionSchema(fields=fields)
    assert schema.version == 1
    assert schema.enable_dynamic_field is False
    assert len(schema.fields) == 2


def test_collection_schema_dynamic():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
    assert schema.enable_dynamic_field is True


# ---------------------------------------------------------------------------
# TYPE_MAP
# ---------------------------------------------------------------------------

def test_type_map_scalar_types():
    assert TYPE_MAP[DataType.BOOL] == pa.bool_()
    assert TYPE_MAP[DataType.INT8] == pa.int8()
    assert TYPE_MAP[DataType.INT16] == pa.int16()
    assert TYPE_MAP[DataType.INT32] == pa.int32()
    assert TYPE_MAP[DataType.INT64] == pa.int64()
    assert TYPE_MAP[DataType.FLOAT] == pa.float32()
    assert TYPE_MAP[DataType.DOUBLE] == pa.float64()
    assert TYPE_MAP[DataType.VARCHAR] == pa.string()
    assert TYPE_MAP[DataType.JSON] == pa.string()


def test_type_map_vector_is_none():
    assert TYPE_MAP[DataType.FLOAT_VECTOR] is None
