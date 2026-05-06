"""Phase 10.2 — schema translator unit tests.

Tight focus: just the proto ↔ milvus_lite conversion in
milvus_lite.adapter.grpc.translators.schema. The Collection lifecycle
tests in test_grpc_collection_lifecycle.py exercise the same code
path through pymilvus + the gRPC server, but those run a full
end-to-end stack and are slower. These unit tests pin specific
edge cases.
"""

import pytest
from pymilvus.grpc_gen import schema_pb2

from milvus_lite.adapter.grpc.translators.schema import (
    milvus_lite_to_milvus_schema,
    milvus_to_milvus_lite_schema,
)
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


# ---------------------------------------------------------------------------
# Round-trip happy path — every supported type
# ---------------------------------------------------------------------------

def test_round_trip_all_supported_types():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="active", dtype=DataType.BOOL),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="ratio", dtype=DataType.DOUBLE),
        FieldSchema(name="i8", dtype=DataType.INT8),
        FieldSchema(name="i16", dtype=DataType.INT16),
        FieldSchema(name="i32", dtype=DataType.INT32),
        FieldSchema(name="data", dtype=DataType.JSON),
    ])
    proto = milvus_lite_to_milvus_schema("test", schema)
    decoded = milvus_to_milvus_lite_schema(proto)

    # Names + types preserved
    by_name = {f.name: f for f in decoded.fields}
    for f in schema.fields:
        assert f.name in by_name
        assert by_name[f.name].dtype == f.dtype
        assert by_name[f.name].is_primary == f.is_primary

    # vector dim preserved
    assert by_name["vec"].dim == 128
    # varchar max_length preserved
    assert by_name["title"].max_length == 512


def test_round_trip_dynamic_field_flag():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ],
        enable_dynamic_field=True,
    )
    proto = milvus_lite_to_milvus_schema("d", schema)
    decoded = milvus_to_milvus_lite_schema(proto)
    assert decoded.enable_dynamic_field is True


def test_round_trip_nullable():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64, nullable=True),
    ])
    proto = milvus_lite_to_milvus_schema("n", schema)
    decoded = milvus_to_milvus_lite_schema(proto)
    by_name = {f.name: f for f in decoded.fields}
    assert by_name["title"].nullable is True


# ---------------------------------------------------------------------------
# Direct proto → milvus_lite (skipping the round-trip)
# ---------------------------------------------------------------------------

def test_decode_int64_pk_field():
    proto = schema_pb2.CollectionSchema()
    proto.name = "t"
    f = proto.fields.add()
    f.name = "id"
    f.is_primary_key = True
    f.data_type = 5  # Int64
    v = proto.fields.add()
    v.name = "v"
    v.data_type = 101  # FloatVector
    kv = v.type_params.add()
    kv.key = "dim"
    kv.value = "4"

    decoded = milvus_to_milvus_lite_schema(proto)
    assert decoded.fields[0].name == "id"
    assert decoded.fields[0].dtype == DataType.INT64
    assert decoded.fields[0].is_primary is True


def test_decode_float_vector_dim():
    proto = schema_pb2.CollectionSchema()
    f = proto.fields.add()
    f.name = "vec"
    f.data_type = 101  # FloatVector
    kv = f.type_params.add()
    kv.key = "dim"
    kv.value = "256"
    # need a primary
    pk = proto.fields.add()
    pk.name = "id"
    pk.is_primary_key = True
    pk.data_type = 5

    decoded = milvus_to_milvus_lite_schema(proto)
    by_name = {f.name: f for f in decoded.fields}
    assert by_name["vec"].dim == 256


# ---------------------------------------------------------------------------
# Validation / errors
# ---------------------------------------------------------------------------

def test_decode_unsupported_type_raises():
    proto = schema_pb2.CollectionSchema()
    proto.name = "x"
    pk = proto.fields.add()
    pk.name = "id"
    pk.is_primary_key = True
    pk.data_type = 5
    bad = proto.fields.add()
    bad.name = "binvec"
    bad.data_type = 100  # BinaryVector — not supported

    with pytest.raises(SchemaValidationError, match="BinaryVector"):
        milvus_to_milvus_lite_schema(proto)


def test_decode_float_vector_missing_dim_raises():
    proto = schema_pb2.CollectionSchema()
    pk = proto.fields.add()
    pk.name = "id"
    pk.is_primary_key = True
    pk.data_type = 5
    f = proto.fields.add()
    f.name = "vec"
    f.data_type = 101  # FloatVector but no dim type_param

    with pytest.raises(SchemaValidationError, match="dim"):
        milvus_to_milvus_lite_schema(proto)


def test_decode_float_vector_non_integer_dim_raises():
    proto = schema_pb2.CollectionSchema()
    pk = proto.fields.add()
    pk.name = "id"
    pk.is_primary_key = True
    pk.data_type = 5
    f = proto.fields.add()
    f.name = "vec"
    f.data_type = 101
    kv = f.type_params.add()
    kv.key = "dim"
    kv.value = "not-a-number"

    with pytest.raises(SchemaValidationError, match="dim"):
        milvus_to_milvus_lite_schema(proto)


# ---------------------------------------------------------------------------
# Encode error path
# ---------------------------------------------------------------------------

def test_encode_float_vector_without_dim_raises():
    """A MilvusLite FieldSchema with FLOAT_VECTOR but no dim should
    have already been rejected by validate_schema, but if it slips
    through the encoder must reject it loudly."""
    # Bypass FieldSchema's own validation by mutating after the fact
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ])
    object.__setattr__(schema.fields[1], "dim", None)
    with pytest.raises(SchemaValidationError, match="dim"):
        milvus_lite_to_milvus_schema("x", schema)


# ---------------------------------------------------------------------------
# default_value round-trip (Issue #13)
# ---------------------------------------------------------------------------

def test_round_trip_default_value_varchar():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32,
                    default_value="active"),
    ])
    proto = milvus_lite_to_milvus_schema("dv", schema)
    decoded = milvus_to_milvus_lite_schema(proto)
    by_name = {f.name: f for f in decoded.fields}
    assert by_name["status"].default_value == "active"


def test_round_trip_default_value_int64():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="count", dtype=DataType.INT64, default_value=0),
    ])
    proto = milvus_lite_to_milvus_schema("dv", schema)
    decoded = milvus_to_milvus_lite_schema(proto)
    by_name = {f.name: f for f in decoded.fields}
    assert by_name["count"].default_value == 0


def test_round_trip_default_value_bool():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="flag", dtype=DataType.BOOL, default_value=True),
    ])
    proto = milvus_lite_to_milvus_schema("dv", schema)
    decoded = milvus_to_milvus_lite_schema(proto)
    by_name = {f.name: f for f in decoded.fields}
    assert by_name["flag"].default_value is True


def test_round_trip_default_value_float():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="score", dtype=DataType.FLOAT, default_value=0.5),
    ])
    proto = milvus_lite_to_milvus_schema("dv", schema)
    decoded = milvus_to_milvus_lite_schema(proto)
    by_name = {f.name: f for f in decoded.fields}
    assert abs(by_name["score"].default_value - 0.5) < 1e-6


def test_round_trip_no_default_value():
    """Fields without default_value should decode to None."""
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64),
    ])
    proto = milvus_lite_to_milvus_schema("dv", schema)
    decoded = milvus_to_milvus_lite_schema(proto)
    by_name = {f.name: f for f in decoded.fields}
    assert by_name["title"].default_value is None
