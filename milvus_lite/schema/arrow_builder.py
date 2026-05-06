"""Arrow Schema builders for the four schema variants.

- data schema:      _seq + user fields + [$meta]           (Parquet data files)
- delta schema:     {pk} + _seq                            (Parquet delta files)
- wal_data schema:  _seq + _partition + user fields + [$meta]  (WAL data files)
- wal_delta schema: {pk} + _seq + _partition               (WAL delta files)
"""

from __future__ import annotations

from typing import Optional

import pyarrow as pa

from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema, TYPE_MAP


def get_primary_field(schema: CollectionSchema) -> FieldSchema:
    """Return the primary-key FieldSchema."""
    for f in schema.fields:
        if f.is_primary:
            return f
    raise ValueError("Schema has no primary key field")


def get_vector_field(schema: CollectionSchema) -> Optional[FieldSchema]:
    """Return the first dense vector FieldSchema, or None if none exists.

    Only returns FLOAT_VECTOR fields.  SPARSE_FLOAT_VECTOR has a
    completely different storage layout (pa.binary()) and must not be
    treated as a dense vector column.
    """
    for f in schema.fields:
        if f.dtype == DataType.FLOAT_VECTOR:
            return f
    return None


def _arrow_type(field: FieldSchema) -> pa.DataType:
    """Resolve a FieldSchema to its PyArrow type."""
    if field.dtype == DataType.FLOAT_VECTOR:
        if field.dim is None:
            raise ValueError(f"FLOAT_VECTOR field '{field.name}' requires dim")
        return pa.list_(pa.float32(), field.dim)
    if field.dtype == DataType.ARRAY:
        if field.element_type is None:
            raise ValueError(f"ARRAY field '{field.name}' requires element_type")
        elem_arrow = TYPE_MAP.get(field.element_type)
        if elem_arrow is None:
            raise ValueError(f"ARRAY field '{field.name}' has unsupported element_type {field.element_type}")
        return pa.list_(elem_arrow)
    if field.dtype == DataType.SPARSE_FLOAT_VECTOR:
        return pa.binary()
    return TYPE_MAP[field.dtype]


def _user_arrow_fields(schema: CollectionSchema) -> list[pa.Field]:
    """Build the list of PyArrow fields for user-defined columns."""
    fields: list[pa.Field] = []
    for f in schema.fields:
        fields.append(pa.field(f.name, _arrow_type(f), nullable=f.nullable))
    return fields


def build_data_schema(schema: CollectionSchema) -> pa.Schema:
    """Data Parquet schema: _seq(uint64) + user fields + [$meta(string)]."""
    arrow_fields: list[pa.Field] = [pa.field("_seq", pa.uint64(), nullable=False)]
    arrow_fields.extend(_user_arrow_fields(schema))
    if schema.enable_dynamic_field:
        arrow_fields.append(pa.field("$meta", pa.string(), nullable=True))
    return pa.schema(arrow_fields)


def build_delta_schema(schema: CollectionSchema) -> pa.Schema:
    """Delta Parquet schema: {pk}(pk_type) + _seq(uint64)."""
    pk = get_primary_field(schema)
    return pa.schema([
        pa.field(pk.name, _arrow_type(pk), nullable=False),
        pa.field("_seq", pa.uint64(), nullable=False),
    ])


def build_wal_data_schema(schema: CollectionSchema) -> pa.Schema:
    """WAL data schema: _seq(uint64) + _partition(string) + user fields + [$meta]."""
    arrow_fields: list[pa.Field] = [
        pa.field("_seq", pa.uint64(), nullable=False),
        pa.field("_partition", pa.string(), nullable=False),
    ]
    arrow_fields.extend(_user_arrow_fields(schema))
    if schema.enable_dynamic_field:
        arrow_fields.append(pa.field("$meta", pa.string(), nullable=True))
    return pa.schema(arrow_fields)


def build_wal_delta_schema(schema: CollectionSchema) -> pa.Schema:
    """WAL delta schema: {pk}(pk_type) + _seq(uint64) + _partition(string)."""
    pk = get_primary_field(schema)
    return pa.schema([
        pa.field(pk.name, _arrow_type(pk), nullable=False),
        pa.field("_seq", pa.uint64(), nullable=False),
        pa.field("_partition", pa.string(), nullable=False),
    ])
