"""Phase 10.3 — records translator unit tests.

Bidirectional FieldData ↔ records round-trip plus edge cases:
    - all supported scalar types (Bool, Int8/16/32/64, Float, Double,
      VarChar, JSON)
    - FLOAT_VECTOR dim slicing
    - nullable fields via valid_data
    - length mismatches → SchemaValidationError
    - unsupported types → SchemaValidationError

The tests build FieldData protos directly (rather than going through
the gRPC server) so failures point at the translator instead of the
RPC plumbing.
"""

import json

import pytest
from pymilvus.grpc_gen import schema_pb2

from milvus_lite.adapter.grpc.translators.records import (
    fields_data_to_records,
    records_to_fields_data,
)
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


# ---------------------------------------------------------------------------
# Helpers — build FieldData from a Python column
# ---------------------------------------------------------------------------

def _scalar_fd(name: str, type_int: int, slot: str, values, valid=None):
    fd = schema_pb2.FieldData()
    fd.field_name = name
    fd.type = type_int
    sub = getattr(fd.scalars, slot)
    sub.data.extend(values)
    if valid is not None:
        fd.valid_data.extend(valid)
    return fd


def _vector_fd(name: str, dim: int, vectors):
    fd = schema_pb2.FieldData()
    fd.field_name = name
    fd.type = 101  # FloatVector
    fd.vectors.dim = dim
    flat = []
    for v in vectors:
        flat.extend(v)
    fd.vectors.float_vector.data.extend(flat)
    return fd


# ---------------------------------------------------------------------------
# fields_data_to_records — basic happy path
# ---------------------------------------------------------------------------

def test_decode_int64_pk_and_float_vector():
    fields_data = [
        _scalar_fd("id", 5, "long_data", [1, 2, 3]),
        _vector_fd("vec", 2, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    ]
    records = fields_data_to_records(fields_data, num_rows=3)
    assert len(records) == 3
    assert records[0] == {"id": 1, "vec": [1.0, 2.0]}
    assert records[2] == {"id": 3, "vec": [5.0, 6.0]}


def test_decode_all_scalar_types():
    fields_data = [
        _scalar_fd("b",   1,  "bool_data",   [True, False, True]),
        _scalar_fd("i8",  2,  "int_data",    [1, 2, 3]),
        _scalar_fd("i16", 3,  "int_data",    [10, 20, 30]),
        _scalar_fd("i32", 4,  "int_data",    [100, 200, 300]),
        _scalar_fd("i64", 5,  "long_data",   [1000, 2000, 3000]),
        _scalar_fd("f",   10, "float_data",  [0.1, 0.2, 0.3]),
        _scalar_fd("d",   11, "double_data", [1.1, 2.2, 3.3]),
        _scalar_fd("s",   21, "string_data", ["a", "b", "c"]),
    ]
    records = fields_data_to_records(fields_data, num_rows=3)
    assert records[0]["b"] is True
    assert records[1]["i64"] == 2000
    assert records[2]["s"] == "c"
    assert abs(records[0]["f"] - 0.1) < 1e-6


def test_decode_json_field():
    fd = schema_pb2.FieldData()
    fd.field_name = "data"
    fd.type = 23  # JSON
    fd.scalars.json_data.data.extend([
        b'{"k": 1}',
        b'{"k": "two"}',
        b'[1, 2, 3]',
    ])
    records = fields_data_to_records([fd], num_rows=3)
    assert records[0]["data"] == {"k": 1}
    assert records[1]["data"] == {"k": "two"}
    assert records[2]["data"] == [1, 2, 3]


def test_decode_empty_request():
    assert fields_data_to_records([], num_rows=0) == []


def test_decode_vector_dim_slicing_correct():
    """Flat 12-element float buffer with dim=4 → 3 rows of 4 floats."""
    fd = schema_pb2.FieldData()
    fd.field_name = "vec"
    fd.type = 101
    fd.vectors.dim = 4
    fd.vectors.float_vector.data.extend([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    ])
    records = fields_data_to_records([fd], num_rows=3)
    assert records[0]["vec"] == [1.0, 2.0, 3.0, 4.0]
    assert records[1]["vec"] == [5.0, 6.0, 7.0, 8.0]
    assert records[2]["vec"] == [9.0, 10.0, 11.0, 12.0]


# ---------------------------------------------------------------------------
# Nullable fields via valid_data
# ---------------------------------------------------------------------------

def test_decode_nullable_field_with_valid_data_mask():
    fd = _scalar_fd(
        "title", 21, "string_data",
        ["a", "x", "c", "x", "e"],
        valid=[True, False, True, False, True],
    )
    records = fields_data_to_records([fd], num_rows=5)
    assert [r["title"] for r in records] == ["a", None, "c", None, "e"]


def test_decode_valid_data_length_mismatch_raises():
    fd = _scalar_fd(
        "title", 21, "string_data",
        ["a", "b", "c"],
        valid=[True, False],  # length 2, num_rows 3
    )
    with pytest.raises(SchemaValidationError, match="valid_data length"):
        fields_data_to_records([fd], num_rows=3)


# ---------------------------------------------------------------------------
# Error / validation paths
# ---------------------------------------------------------------------------

def test_decode_unsupported_vector_type_raises():
    fd = schema_pb2.FieldData()
    fd.field_name = "binvec"
    fd.type = 100  # BinaryVector
    fd.vectors.dim = 16
    with pytest.raises(SchemaValidationError, match="BinaryVector"):
        fields_data_to_records([fd], num_rows=1)


def test_decode_unsupported_scalar_type_raises():
    fd = schema_pb2.FieldData()
    fd.field_name = "geo"
    fd.type = 24  # Geometry
    fd.scalars.bytes_data.data.extend([b""])
    with pytest.raises(SchemaValidationError, match="Geometry"):
        fields_data_to_records([fd], num_rows=1)


def test_decode_length_mismatch_raises():
    fd = _scalar_fd("id", 5, "long_data", [1, 2])  # 2 elements
    with pytest.raises(SchemaValidationError, match="rows"):
        fields_data_to_records([fd], num_rows=3)


def test_decode_vector_total_length_mismatch_raises():
    fd = schema_pb2.FieldData()
    fd.field_name = "vec"
    fd.type = 101
    fd.vectors.dim = 4
    fd.vectors.float_vector.data.extend([1, 2, 3, 4, 5, 6])  # only 1.5 rows
    with pytest.raises(SchemaValidationError, match="float_vector data has"):
        fields_data_to_records([fd], num_rows=2)


# ---------------------------------------------------------------------------
# records_to_fields_data — happy path
# ---------------------------------------------------------------------------

def _schema_basic():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64, nullable=True),
        FieldSchema(name="active", dtype=DataType.BOOL),
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ])


def test_encode_basic_round_trip():
    schema = _schema_basic()
    records_in = [
        {"id": 1, "vec": [1.0, 2.0, 3.0, 4.0], "title": "a", "active": True, "score": 0.5},
        {"id": 2, "vec": [5.0, 6.0, 7.0, 8.0], "title": "b", "active": False, "score": 0.9},
    ]
    fields_data = records_to_fields_data(records_in, schema)

    # Decode back
    records_out = fields_data_to_records(fields_data, num_rows=2)
    assert records_out[0]["id"] == 1
    assert records_out[0]["vec"] == [1.0, 2.0, 3.0, 4.0]
    assert records_out[0]["title"] == "a"
    assert records_out[0]["active"] is True
    assert abs(records_out[0]["score"] - 0.5) < 1e-6
    assert records_out[1]["title"] == "b"


def test_encode_output_fields_subset():
    schema = _schema_basic()
    records_in = [
        {"id": 1, "vec": [1, 2, 3, 4], "title": "a", "active": True, "score": 0.5},
    ]
    fields_data = records_to_fields_data(
        records_in, schema, output_fields=["title"]
    )
    # pk is always included; title is in the explicit list
    field_names = [fd.field_name for fd in fields_data]
    assert "id" in field_names
    assert "title" in field_names
    assert "vec" not in field_names
    assert "active" not in field_names


def test_encode_output_fields_none_emits_all():
    schema = _schema_basic()
    records_in = [
        {"id": 1, "vec": [1, 2, 3, 4], "title": "a", "active": True, "score": 0.5},
    ]
    fields_data = records_to_fields_data(records_in, schema, output_fields=None)
    field_names = sorted([fd.field_name for fd in fields_data])
    assert field_names == sorted(["id", "vec", "title", "active", "score"])


def test_encode_with_null_value():
    schema = _schema_basic()
    records_in = [
        {"id": 1, "vec": [1, 2, 3, 4], "title": "a", "active": True, "score": 0.5},
        {"id": 2, "vec": [5, 6, 7, 8], "title": None, "active": False, "score": 0.9},
    ]
    fields_data = records_to_fields_data(records_in, schema)

    # The title FieldData should have valid_data with [True, False]
    title_fd = next(fd for fd in fields_data if fd.field_name == "title")
    assert list(title_fd.valid_data) == [True, False]

    # Round-trip preserves None
    records_out = fields_data_to_records(fields_data, num_rows=2)
    assert records_out[1]["title"] is None
    assert records_out[0]["title"] == "a"


def test_encode_json_field_round_trip():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="meta", dtype=DataType.JSON),
    ])
    records_in = [
        {"id": 1, "vec": [1.0, 2.0], "meta": {"k": "v", "n": 5}},
        {"id": 2, "vec": [3.0, 4.0], "meta": [1, 2, 3]},
    ]
    fields_data = records_to_fields_data(records_in, schema)
    records_out = fields_data_to_records(fields_data, num_rows=2)
    assert records_out[0]["meta"] == {"k": "v", "n": 5}
    assert records_out[1]["meta"] == [1, 2, 3]


def test_encode_empty_records_returns_empty_field_data_per_field():
    """Empty result set still emits one FieldData per emitted field
    (with empty data), so pymilvus's query client doesn't raise
    "No fields returned" — that's a quirk of pymilvus's response
    parser, not us."""
    schema = _schema_basic()
    fields_data = records_to_fields_data([], schema)
    field_names = sorted(fd.field_name for fd in fields_data)
    # All schema fields are emitted
    assert field_names == sorted(["id", "vec", "title", "active", "score"])
    # Each FieldData has zero rows worth of data
    for fd in fields_data:
        if fd.HasField("scalars"):
            scalars = fd.scalars
            for slot in ("bool_data", "int_data", "long_data",
                         "float_data", "double_data", "string_data"):
                sub = getattr(scalars, slot)
                assert len(list(sub.data)) == 0
        elif fd.HasField("vectors"):
            assert len(list(fd.vectors.float_vector.data)) == 0


def test_encode_vector_wrong_length_raises():
    schema = _schema_basic()
    records_in = [
        {"id": 1, "vec": [1, 2, 3], "title": "x", "active": True, "score": 0.5},  # dim 3 != 4
    ]
    with pytest.raises(SchemaValidationError, match="length"):
        records_to_fields_data(records_in, schema)
