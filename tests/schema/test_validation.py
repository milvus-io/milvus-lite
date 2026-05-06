"""Tests for schema/validation.py"""

import json

import pytest

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)
from milvus_lite.schema.validation import (
    RESERVED_FIELD_NAMES,
    separate_dynamic_fields,
    validate_record,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _basic_schema(**kwargs) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
            FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
            FieldSchema(name="score", dtype=DataType.FLOAT),
        ],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

def test_valid_schema_ok():
    validate_schema(_basic_schema())


def test_schema_rejects_rerank_function():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ],
        functions=[
            Function(
                name="rerank_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["text"],
                output_field_names=[],
                params={"provider": "cohere"},
            )
        ],
    )

    with pytest.raises(SchemaValidationError, match="not supported in collection schema"):
        validate_schema(schema)


def test_no_fields():
    with pytest.raises(SchemaValidationError, match="no fields"):
        validate_schema(CollectionSchema(fields=[]))


def test_empty_field_name():
    with pytest.raises(SchemaValidationError, match="must not be empty"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]))


@pytest.mark.parametrize("reserved", sorted(RESERVED_FIELD_NAMES))
def test_reserved_field_name(reserved):
    with pytest.raises(SchemaValidationError, match="reserved"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name=reserved, dtype=DataType.VARCHAR),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]))


def test_duplicate_field_names():
    with pytest.raises(SchemaValidationError, match="duplicate"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="x", dtype=DataType.INT64),
            FieldSchema(name="x", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]))


def test_no_primary_key():
    with pytest.raises(SchemaValidationError, match="no primary key"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]))


def test_multiple_primary_keys():
    with pytest.raises(SchemaValidationError, match="exactly one primary key"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="a", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="b", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]))


def test_primary_key_wrong_dtype():
    with pytest.raises(SchemaValidationError, match="VARCHAR or INT64"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.FLOAT, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]))


def test_primary_key_nullable():
    with pytest.raises(SchemaValidationError, match="must not be nullable"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]))


def test_no_vector_field():
    with pytest.raises(SchemaValidationError, match="no vector field"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        ]))


def test_multiple_vector_fields():
    """Multiple FLOAT_VECTOR fields are allowed (multi-vector support)."""
    validate_schema(CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="v1", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="v2", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]))


@pytest.mark.parametrize("bad_dim", [None, 0, -1])
def test_vector_invalid_dim(bad_dim):
    with pytest.raises(SchemaValidationError, match="dim"):
        validate_schema(CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=bad_dim),
        ]))


def test_int64_pk_ok():
    validate_schema(CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]))


# ---------------------------------------------------------------------------
# validate_record
# ---------------------------------------------------------------------------

def test_valid_record_ok():
    schema = _basic_schema()
    rec = {"id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4], "title": "hi", "score": 0.5}
    validate_record(rec, schema)


def test_record_not_dict():
    with pytest.raises(SchemaValidationError, match="must be a dict"):
        validate_record(["id", "doc1"], _basic_schema())


def test_record_missing_pk():
    with pytest.raises(SchemaValidationError, match="primary key"):
        validate_record(
            {"vec": [0.1, 0.2, 0.3, 0.4], "title": "x", "score": 0.5},
            _basic_schema(),
        )


def test_record_pk_none():
    with pytest.raises(SchemaValidationError, match="primary key"):
        validate_record(
            {"id": None, "vec": [0.1, 0.2, 0.3, 0.4], "title": "x", "score": 0.5},
            _basic_schema(),
        )


def test_record_missing_vector():
    with pytest.raises(SchemaValidationError, match="vector field"):
        validate_record(
            {"id": "doc1", "title": "x", "score": 0.5},
            _basic_schema(),
        )


def test_record_wrong_vector_dim():
    with pytest.raises(SchemaValidationError, match="dim 4"):
        validate_record(
            {"id": "doc1", "vec": [0.1, 0.2], "title": "x", "score": 0.5},
            _basic_schema(),
        )


def test_record_vector_non_numeric():
    with pytest.raises(SchemaValidationError, match="numeric"):
        validate_record(
            {"id": "doc1", "vec": [0.1, "x", 0.3, 0.4], "title": "x", "score": 0.5},
            _basic_schema(),
        )


def test_record_vector_must_be_list():
    with pytest.raises(SchemaValidationError, match="list/tuple"):
        validate_record(
            {"id": "doc1", "vec": "not a vector", "title": "x", "score": 0.5},
            _basic_schema(),
        )


def test_record_field_wrong_type():
    with pytest.raises(SchemaValidationError, match="dtype"):
        validate_record(
            {"id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4], "title": "x", "score": "not a number"},
            _basic_schema(),
        )


def test_record_nullable_missing_ok():
    rec = {"id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4], "score": 0.5}  # title missing, nullable
    validate_record(rec, _basic_schema())


def test_record_non_nullable_missing_raises():
    rec = {"id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4], "title": "x"}  # score missing
    with pytest.raises(SchemaValidationError, match="missing"):
        validate_record(rec, _basic_schema())


def test_record_extra_field_without_dynamic():
    rec = {
        "id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4],
        "title": "x", "score": 0.5,
        "unknown": "extra",
    }
    with pytest.raises(SchemaValidationError, match="not in schema"):
        validate_record(rec, _basic_schema())


def test_record_extra_field_with_dynamic_ok():
    rec = {
        "id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4],
        "title": "x", "score": 0.5,
        "category": "blog",
    }
    validate_record(rec, _basic_schema(enable_dynamic_field=True))


def test_record_int_overflow():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="n", dtype=DataType.INT8),
    ])
    with pytest.raises(SchemaValidationError, match="dtype"):
        validate_record(
            {"id": "x", "vec": [0.1, 0.2], "n": 999},
            schema,
        )


def test_record_bool_not_int():
    """bool should NOT validate as INT64."""
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="n", dtype=DataType.INT64),
    ])
    with pytest.raises(SchemaValidationError, match="dtype"):
        validate_record({"id": "x", "vec": [0.1, 0.2], "n": True}, schema)


# ---------------------------------------------------------------------------
# separate_dynamic_fields
# ---------------------------------------------------------------------------

def test_separate_no_extras():
    schema = _basic_schema()
    rec = {"id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4], "title": "x", "score": 0.5}
    schema_part, meta = separate_dynamic_fields(rec, schema)
    assert schema_part == rec
    assert meta is None


def test_separate_with_extras():
    schema = _basic_schema(enable_dynamic_field=True)
    rec = {
        "id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4],
        "title": "x", "score": 0.5,
        "category": "blog", "lang": "en",
    }
    schema_part, meta = separate_dynamic_fields(rec, schema)
    assert set(schema_part.keys()) == {"id", "vec", "title", "score"}
    assert meta is not None
    parsed = json.loads(meta)
    assert parsed == {"category": "blog", "lang": "en"}


def test_separate_extras_without_dynamic_raises():
    rec = {"id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4], "title": "x", "score": 0.5, "x": 1}
    with pytest.raises(SchemaValidationError, match="not in schema"):
        separate_dynamic_fields(rec, _basic_schema())


def test_separate_fills_nullable_default_none():
    schema = _basic_schema()
    rec = {"id": "doc1", "vec": [0.1, 0.2, 0.3, 0.4], "score": 0.5}  # title missing
    schema_part, _ = separate_dynamic_fields(rec, schema)
    assert schema_part["title"] is None


def test_separate_fills_default_value():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="kind", dtype=DataType.VARCHAR, default_value="unknown"),
    ])
    rec = {"id": "x", "vec": [0.1, 0.2]}
    schema_part, _ = separate_dynamic_fields(rec, schema)
    assert schema_part["kind"] == "unknown"


def test_separate_meta_is_deterministic():
    """meta JSON must be sorted by key so two equal records produce equal meta."""
    schema = _basic_schema(enable_dynamic_field=True)
    rec_a = {"id": "x", "vec": [0.1, 0.2, 0.3, 0.4], "title": "t", "score": 0.5,
             "b": 1, "a": 2}
    rec_b = {"id": "x", "vec": [0.1, 0.2, 0.3, 0.4], "title": "t", "score": 0.5,
             "a": 2, "b": 1}
    _, meta_a = separate_dynamic_fields(rec_a, schema)
    _, meta_b = separate_dynamic_fields(rec_b, schema)
    assert meta_a == meta_b


# ---------------------------------------------------------------------------
# validate_record — default_value filling (Issue #14)
# ---------------------------------------------------------------------------

def test_validate_record_fills_default_for_missing_field():
    """Missing field with default_value should be filled, not rejected."""
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32,
                    default_value="active"),
    ])
    rec = {"id": 1, "vec": [0.1, 0.2]}
    validate_record(rec, schema)  # should not raise
    assert rec["status"] == "active"


def test_validate_record_fills_default_for_none_value():
    """Field explicitly set to None with default_value should be filled."""
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="count", dtype=DataType.INT64, default_value=0),
    ])
    rec = {"id": 1, "vec": [0.1, 0.2], "count": None}
    validate_record(rec, schema)
    assert rec["count"] == 0


def test_validate_record_keeps_explicit_value_over_default():
    """Explicit value should not be overwritten by default_value."""
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32,
                    default_value="active"),
    ])
    rec = {"id": 1, "vec": [0.1, 0.2], "status": "inactive"}
    validate_record(rec, schema)
    assert rec["status"] == "inactive"
