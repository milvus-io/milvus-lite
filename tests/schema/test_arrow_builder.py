"""Tests for schema/arrow_builder.py — 4 schema builders + helper functions."""

import pytest
import pyarrow as pa

from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.schema.arrow_builder import (
    build_data_schema,
    build_delta_schema,
    build_wal_data_schema,
    build_wal_delta_schema,
    get_primary_field,
    get_vector_field,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


@pytest.fixture
def dynamic_schema():
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ],
        enable_dynamic_field=True,
    )


@pytest.fixture
def varchar_pk_schema():
    return CollectionSchema(fields=[
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3),
    ])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_get_primary_field(basic_schema):
    pk = get_primary_field(basic_schema)
    assert pk.name == "id"
    assert pk.is_primary is True


def test_get_vector_field(basic_schema):
    vf = get_vector_field(basic_schema)
    assert vf.name == "vec"
    assert vf.dtype == DataType.FLOAT_VECTOR


def test_get_primary_field_missing():
    schema = CollectionSchema(fields=[
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ])
    with pytest.raises(ValueError, match="no primary key"):
        get_primary_field(schema)


def test_get_vector_field_missing():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    ])
    assert get_vector_field(schema) is None


# ---------------------------------------------------------------------------
# build_data_schema
# ---------------------------------------------------------------------------

def test_build_data_schema_columns(basic_schema):
    arrow = build_data_schema(basic_schema)
    names = arrow.names
    assert names[0] == "_seq"
    assert "id" in names
    assert "vec" in names
    assert "title" in names
    assert "$meta" not in names  # dynamic field disabled


def test_build_data_schema_seq_type(basic_schema):
    arrow = build_data_schema(basic_schema)
    assert arrow.field("_seq").type == pa.uint64()
    assert arrow.field("_seq").nullable is False


def test_build_data_schema_vector_type(basic_schema):
    arrow = build_data_schema(basic_schema)
    vec_type = arrow.field("vec").type
    assert pa.types.is_fixed_size_list(vec_type)
    assert vec_type.list_size == 4
    assert vec_type.value_type == pa.float32()


def test_build_data_schema_with_dynamic(dynamic_schema):
    arrow = build_data_schema(dynamic_schema)
    assert "$meta" in arrow.names
    assert arrow.field("$meta").type == pa.string()
    assert arrow.field("$meta").nullable is True


def test_build_data_schema_no_partition(basic_schema):
    arrow = build_data_schema(basic_schema)
    assert "_partition" not in arrow.names


# ---------------------------------------------------------------------------
# build_delta_schema
# ---------------------------------------------------------------------------

def test_build_delta_schema_int64_pk(basic_schema):
    arrow = build_delta_schema(basic_schema)
    assert arrow.names == ["id", "_seq"]
    assert arrow.field("id").type == pa.int64()
    assert arrow.field("_seq").type == pa.uint64()


def test_build_delta_schema_varchar_pk(varchar_pk_schema):
    arrow = build_delta_schema(varchar_pk_schema)
    assert arrow.names == ["doc_id", "_seq"]
    assert arrow.field("doc_id").type == pa.string()


# ---------------------------------------------------------------------------
# build_wal_data_schema
# ---------------------------------------------------------------------------

def test_build_wal_data_schema_has_partition(basic_schema):
    arrow = build_wal_data_schema(basic_schema)
    assert "_partition" in arrow.names
    assert arrow.field("_partition").type == pa.string()


def test_build_wal_data_schema_column_order(basic_schema):
    arrow = build_wal_data_schema(basic_schema)
    names = arrow.names
    assert names[0] == "_seq"
    assert names[1] == "_partition"
    # User fields follow
    assert "id" in names
    assert "vec" in names


def test_build_wal_data_schema_with_dynamic(dynamic_schema):
    arrow = build_wal_data_schema(dynamic_schema)
    assert "$meta" in arrow.names
    assert "_partition" in arrow.names


# ---------------------------------------------------------------------------
# build_wal_delta_schema
# ---------------------------------------------------------------------------

def test_build_wal_delta_schema_columns(basic_schema):
    arrow = build_wal_delta_schema(basic_schema)
    assert arrow.names == ["id", "_seq", "_partition"]
    assert arrow.field("id").type == pa.int64()
    assert arrow.field("_seq").type == pa.uint64()
    assert arrow.field("_partition").type == pa.string()


def test_build_wal_delta_schema_varchar_pk(varchar_pk_schema):
    arrow = build_wal_delta_schema(varchar_pk_schema)
    assert arrow.names == ["doc_id", "_seq", "_partition"]
    assert arrow.field("doc_id").type == pa.string()
