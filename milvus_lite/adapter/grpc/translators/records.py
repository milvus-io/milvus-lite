"""FieldData ↔ records translation.

This is the most error-prone translator in Phase 10 — Milvus's
``InsertRequest.fields_data`` is COLUMNAR (one FieldData per field,
each carrying a length-N value array), while MilvusLite engine takes
ROW-WISE list[dict]. The two views must be transposed bidirectionally
without losing any field, type, or null information.

Phase 10.3 supported types (matches translators/schema.py):
    Bool / Int8 / Int16 / Int32 / Int64 / Float / Double / VarChar
    JSON / FloatVector

Unsupported (raise UnsupportedFieldTypeError):
    BinaryVector / Float16Vector / BFloat16Vector / SparseFloatVector
    Int8Vector / Array / Geometry / Text / Timestamptz / ArrayOfVector

Two functions:

    fields_data_to_records(fields_data, num_rows)
        list[FieldData] + num_rows → list[dict]
        Used by Insert / Upsert RPCs.

    records_to_fields_data(records, schema, output_fields)
        list[dict] + CollectionSchema → list[FieldData]
        Used by Query / Get / Search response builders.

valid_data semantics: Milvus uses ``FieldData.valid_data`` (a parallel
bool array) to mark per-row null values for nullable fields. We
translate this to Python ``None`` in the records form, and rebuild
the valid_data array on the way back.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pymilvus.grpc_gen import schema_pb2

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import CollectionSchema, DataType


# ── Milvus DataType enum (int) → name and category. We use the int
# values directly to avoid importing pymilvus.DataType (the Python
# enum that wraps the same numbers).

# Scalar types we handle, mapped to the ScalarField oneof slot name.
_SCALAR_TYPE_TO_SLOT: Dict[int, str] = {
    1:  "bool_data",     # Bool
    2:  "int_data",      # Int8 (uses IntArray)
    3:  "int_data",      # Int16
    4:  "int_data",      # Int32
    5:  "long_data",     # Int64
    10: "float_data",    # Float
    11: "double_data",   # Double
    21: "string_data",   # VarChar
    23: "json_data",     # JSON
}

_VECTOR_TYPES = frozenset({100, 101, 102, 103, 104, 105})  # Binary/Float/F16/BF16/Sparse/Int8


def _milvus_type_name(dtype_int: int) -> str:
    """Pretty name for error messages."""
    names = {
        1: "Bool", 2: "Int8", 3: "Int16", 4: "Int32", 5: "Int64",
        10: "Float", 11: "Double", 20: "String", 21: "VarChar",
        22: "Array", 23: "JSON", 24: "Geometry", 25: "Text",
        100: "BinaryVector", 101: "FloatVector",
        102: "Float16Vector", 103: "BFloat16Vector",
        104: "SparseFloatVector", 105: "Int8Vector",
    }
    return names.get(dtype_int, f"Unknown({dtype_int})")


# ── Milvus → records (Insert path) ──────────────────────────────────

def fields_data_to_records(
    fields_data,
    num_rows: int,
) -> List[Dict[str, Any]]:
    """Transpose Milvus columnar fields_data into row-wise records.

    Args:
        fields_data: iterable of FieldData proto messages
        num_rows: declared row count from InsertRequest.num_rows.
            We use this as the authoritative length and validate every
            FieldData against it.

    Returns:
        List of length num_rows. Each element is a dict mapping
        field_name → Python value (or None for null entries).

    Raises:
        SchemaValidationError: any FieldData length mismatches num_rows
            or uses an unsupported type
    """
    if num_rows == 0:
        return []

    records: List[Dict[str, Any]] = [{} for _ in range(num_rows)]

    for fd in fields_data:
        column = _extract_column(fd, num_rows)

        # Dynamic field handling: pymilvus packs dynamic fields into a
        # single FieldData with is_dynamic=True (or field_name="$meta")
        # and type=JSON. The JSON value is a dict like {"level": 5}.
        # We unpack those keys into the top-level record so the
        # engine's separate_dynamic_fields can re-pack them into $meta
        # without double-nesting.
        if fd.is_dynamic or fd.field_name == "$meta":
            for i in range(num_rows):
                val = column[i]
                if isinstance(val, dict):
                    records[i].update(val)
                # If it's a string (pre-serialized JSON), try to parse it.
                elif isinstance(val, str):
                    try:
                        parsed = __import__("json").loads(val)
                        if isinstance(parsed, dict):
                            records[i].update(parsed)
                    except (ValueError, TypeError):
                        pass
            continue

        for i in range(num_rows):
            records[i][fd.field_name] = column[i]

    return records


def _extract_column(fd, num_rows: int) -> List[Any]:
    """Pull a single FieldData out as a length-num_rows Python list.

    Handles the scalar/vector dispatch, validates length, and
    overlays valid_data nulls if present.

    Nullable encoding (pymilvus convention):
        pymilvus sends nullable fields in COMPACT form — the scalar
        data array contains ONLY the non-null values (length = count
        of True in valid_data), while valid_data has full num_rows
        length. We expand the compact array back to full length by
        inserting None at positions where valid_data is False.
    """
    dtype_int = int(fd.type)

    is_vector = False
    if fd.HasField("scalars"):
        column = _extract_scalar_column(fd, dtype_int)
    elif fd.HasField("vectors"):
        column = _extract_vector_column(fd, dtype_int, num_rows)
        is_vector = True
    else:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} has neither scalars nor vectors"
        )

    # Apply valid_data null mask if present (nullable fields).
    # Skip for vector columns — _extract_vector_column already handles valid_data.
    valid_list = list(fd.valid_data) if not is_vector else []
    if valid_list:
        if len(valid_list) != num_rows:
            raise SchemaValidationError(
                f"FieldData {fd.field_name!r} valid_data length "
                f"{len(valid_list)} != num_rows {num_rows}"
            )
        n_valid = sum(1 for v in valid_list if v)
        if len(column) == n_valid:
            # Compact form: expand by interleaving Nones.
            expanded: List[Any] = []
            val_iter = iter(column)
            for v in valid_list:
                if v:
                    expanded.append(next(val_iter))
                else:
                    expanded.append(None)
            column = expanded
        elif len(column) == num_rows:
            # Full form: overlay nulls on existing values.
            column = [v if valid_list[i] else None for i, v in enumerate(column)]
        else:
            raise SchemaValidationError(
                f"FieldData {fd.field_name!r} has {len(column)} values "
                f"but valid_data expects {n_valid} non-null or {num_rows} total"
            )
    else:
        # No valid_data — column must have full num_rows length.
        if len(column) != num_rows:
            raise SchemaValidationError(
                f"FieldData {fd.field_name!r} has {len(column)} rows, "
                f"expected {num_rows}"
            )

    return column


def _extract_scalar_column(fd, dtype_int: int) -> List[Any]:
    scalars = fd.scalars

    # Array type (22) — special handling via array_data slot
    if dtype_int == 22:
        return _extract_array_column(fd)

    if dtype_int not in _SCALAR_TYPE_TO_SLOT:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} uses scalar type "
            f"{_milvus_type_name(dtype_int)} which MilvusLite does not support"
        )

    slot = _SCALAR_TYPE_TO_SLOT[dtype_int]
    sub = getattr(scalars, slot)
    raw = list(sub.data)

    if dtype_int == 23:  # JSON
        # JSON values arrive as bytes; decode + parse each.
        out = []
        for b in raw:
            if isinstance(b, bytes):
                b = b.decode("utf-8")
            try:
                out.append(json.loads(b))
            except (json.JSONDecodeError, ValueError):
                # Tolerate malformed JSON: pass through as a string.
                out.append(b)
        return out

    return raw


def _extract_array_column(fd) -> List[Any]:
    """Extract Array column — each row is a Python list."""
    array_data = fd.scalars.array_data
    column: List[Any] = []
    for row_sf in array_data.data:
        # Each row_sf is a ScalarField containing one element type
        if row_sf.HasField("long_data"):
            column.append(list(row_sf.long_data.data))
        elif row_sf.HasField("int_data"):
            column.append(list(row_sf.int_data.data))
        elif row_sf.HasField("float_data"):
            column.append(list(row_sf.float_data.data))
        elif row_sf.HasField("double_data"):
            column.append(list(row_sf.double_data.data))
        elif row_sf.HasField("string_data"):
            column.append(list(row_sf.string_data.data))
        elif row_sf.HasField("bool_data"):
            column.append(list(row_sf.bool_data.data))
        else:
            column.append([])
    return column


def _extract_vector_column(fd, dtype_int: int, num_rows: int) -> List[Any]:
    vectors = fd.vectors

    if dtype_int == 104:  # SparseFloatVector
        return _extract_sparse_vector_column(fd, num_rows)

    if dtype_int != 101:  # FloatVector
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} uses vector type "
            f"{_milvus_type_name(dtype_int)} which MilvusLite does not "
            f"support (supported: FloatVector, SparseFloatVector)"
        )

    if not vectors.HasField("float_vector"):
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} declared FloatVector but no "
            f"float_vector data is set"
        )

    dim = int(vectors.dim)
    if dim <= 0:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} has invalid dim {dim}"
        )

    flat = list(vectors.float_vector.data)
    expected_full = num_rows * dim

    if len(flat) == expected_full:
        # Full form: all rows present.
        return [flat[i * dim:(i + 1) * dim] for i in range(num_rows)]

    # Compact form: nullable vector — flat contains only non-null rows.
    # Check that len(flat) matches n_valid * dim.
    valid_list = list(fd.valid_data)
    if valid_list:
        n_valid = sum(1 for v in valid_list if v)
        if len(flat) == n_valid * dim:
            # Expand by slicing non-null vectors and inserting None.
            column: List[Any] = []
            val_idx = 0
            for v in valid_list:
                if v:
                    column.append(flat[val_idx * dim:(val_idx + 1) * dim])
                    val_idx += 1
                else:
                    column.append(None)
            return column

    raise SchemaValidationError(
        f"FieldData {fd.field_name!r} float_vector data has "
        f"{len(flat)} elements, expected num_rows({num_rows}) * dim({dim}) "
        f"= {expected_full}"
    )


def _extract_sparse_vector_column(fd, num_rows: int) -> List[Any]:
    """Extract SparseFloatVector column as list of dict[int, float]."""
    from milvus_lite.analyzer.sparse import bytes_to_sparse

    vectors = fd.vectors
    if not vectors.HasField("sparse_float_vector"):
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} declared SparseFloatVector but no "
            f"sparse_float_vector data is set"
        )

    sfa = vectors.sparse_float_vector
    column: List[Any] = []
    for content_bytes in sfa.contents:
        column.append(bytes_to_sparse(content_bytes))

    if len(column) != num_rows:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} sparse_float_vector has "
            f"{len(column)} rows, expected {num_rows}"
        )
    return column


# ── records → Milvus (Query / Get / Search response path) ───────────

def records_to_fields_data(
    records: List[Dict[str, Any]],
    schema: CollectionSchema,
    output_fields: Optional[List[str]] = None,
) -> List:
    """Build columnar FieldData list from row-wise records.

    Args:
        records: list of dicts (engine output)
        schema: source CollectionSchema — needed for per-field type info
        output_fields: optional whitelist; None → emit every schema field.
            Pk is always emitted.

    Returns:
        List of FieldData proto messages, one per emitted field. When
        *records* is empty, the FieldData list still contains one entry
        per emitted field with an empty data slot — pymilvus's query
        client raises "No fields returned" if we send back an entirely
        empty fields_data list.
    """
    pk_field = next((f for f in schema.fields if f.is_primary), None)
    pk_name = pk_field.name if pk_field else None

    if output_fields is None:
        emit_names = [f.name for f in schema.fields]
    else:
        emit = set(output_fields)
        if pk_name:
            emit.add(pk_name)
        # Preserve schema order for determinism.
        emit_names = [f.name for f in schema.fields if f.name in emit]

    field_by_name = {f.name: f for f in schema.fields}

    fields_data: List = []
    for fname in emit_names:
        fschema = field_by_name[fname]
        column = [r.get(fname) for r in records]
        fd = _build_field_data(fname, fschema, column)
        fields_data.append(fd)

    # Emit $meta JSON column for dynamic fields.
    # Milvus returns a single $meta FieldData (type=JSON, is_dynamic=True)
    # containing all dynamic field values. pymilvus unpacks individual
    # fields on the client side, preserving original types (int/float/
    # bool/list/dict).
    if schema.enable_dynamic_field:
        import json
        schema_names = {f.name for f in schema.fields}
        meta_column: list[bytes] = []
        for r in records:
            meta_dict = {
                k: v for k, v in r.items()
                if k not in schema_names and k != "$meta"
            }
            meta_column.append(json.dumps(meta_dict).encode("utf-8"))
        fd = schema_pb2.FieldData()
        fd.field_name = "$meta"
        fd.type = 23  # JSON
        fd.is_dynamic = True
        fd.scalars.json_data.data.extend(meta_column)
        fields_data.append(fd)

    return fields_data


def _build_field_data(name, fschema, column):
    """Build one FieldData proto from a (name, FieldSchema, column) triple.

    Handles per-type oneof slot population, valid_data null mask
    construction, and float_vector flattening.
    """
    fd = schema_pb2.FieldData()
    fd.field_name = name

    dtype = fschema.dtype

    # Build the null-aware column: replace None with a type-appropriate
    # default and emit a parallel valid_data list.
    has_nulls = any(v is None for v in column)
    if has_nulls:
        valid = [v is not None for v in column]
        fd.valid_data.extend(valid)

    if dtype == DataType.FLOAT_VECTOR:
        fd.type = 101  # FloatVector
        dim = int(fschema.dim or 0)
        if dim <= 0:
            raise SchemaValidationError(
                f"FLOAT_VECTOR field {name!r} has missing or invalid dim"
            )
        fd.vectors.dim = dim
        flat: List[float] = []
        zero_row = [0.0] * dim
        for v in column:
            if v is None:
                flat.extend(zero_row)  # placeholder; valid_data marks it null
            else:
                if len(v) != dim:
                    raise SchemaValidationError(
                        f"vector value for {name!r} has length {len(v)}, "
                        f"expected dim {dim}"
                    )
                flat.extend(float(x) for x in v)
        fd.vectors.float_vector.data.extend(flat)
        return fd

    if dtype == DataType.SPARSE_FLOAT_VECTOR:
        from milvus_lite.analyzer.sparse import sparse_to_bytes, bytes_to_sparse
        fd.type = 104  # SparseFloatVector
        max_dim = 0
        for v in column:
            if v is None:
                fd.vectors.sparse_float_vector.contents.append(b"")
            elif isinstance(v, bytes):
                fd.vectors.sparse_float_vector.contents.append(v)
                sv = bytes_to_sparse(v)
                if sv:
                    max_dim = max(max_dim, max(sv.keys()) + 1)
            elif isinstance(v, dict):
                fd.vectors.sparse_float_vector.contents.append(sparse_to_bytes(v))
                if v:
                    max_dim = max(max_dim, max(v.keys()) + 1)
            else:
                fd.vectors.sparse_float_vector.contents.append(b"")
        fd.vectors.sparse_float_vector.dim = max_dim
        return fd

    if dtype == DataType.ARRAY:
        fd.type = 22  # Array
        # Determine element Milvus type
        elem_type = fschema.element_type
        elem_milvus = _LITEVECDB_TO_MILVUS_INT.get(elem_type, 5) if elem_type else 5
        _ELEM_SLOT = {
            1: "bool_data", 2: "int_data", 3: "int_data", 4: "int_data",
            5: "long_data", 10: "float_data", 11: "double_data", 21: "string_data",
        }
        elem_slot = _ELEM_SLOT.get(elem_milvus, "long_data")
        fd.scalars.array_data.element_type = elem_milvus
        for v in column:
            row_sf = fd.scalars.array_data.data.add()
            arr_vals = v if v is not None else []
            if isinstance(arr_vals, (list, tuple)):
                getattr(row_sf, elem_slot).data.extend(arr_vals)
        return fd

    # Scalar types
    milvus_type_int = _LITEVECDB_TO_MILVUS_INT.get(dtype)
    if milvus_type_int is None:
        raise SchemaValidationError(
            f"field {name!r} has type {dtype.name} which has no Milvus "
            f"FieldData equivalent"
        )

    fd.type = milvus_type_int
    slot = _SCALAR_TYPE_TO_SLOT[milvus_type_int]
    sub = getattr(fd.scalars, slot)

    if dtype == DataType.JSON:
        # Encode each value as bytes JSON
        encoded = []
        for v in column:
            if v is None:
                encoded.append(b"null")  # placeholder
            elif isinstance(v, (str, bytes)):
                # Already-serialized: pass through (decode if str)
                encoded.append(v.encode("utf-8") if isinstance(v, str) else v)
            else:
                encoded.append(json.dumps(v).encode("utf-8"))
        sub.data.extend(encoded)
        return fd

    # Numeric / bool / string scalar
    out = []
    for v in column:
        if v is None:
            out.append(_default_for(dtype))
        else:
            out.append(_coerce_for(dtype, v))
    sub.data.extend(out)
    return fd


# milvus_lite.DataType → Milvus enum int
_LITEVECDB_TO_MILVUS_INT: Dict[DataType, int] = {
    DataType.BOOL:    1,
    DataType.INT8:    2,
    DataType.INT16:   3,
    DataType.INT32:   4,
    DataType.INT64:   5,
    DataType.FLOAT:   10,
    DataType.DOUBLE:  11,
    DataType.VARCHAR: 21,
    DataType.ARRAY:   22,
    DataType.JSON:    23,
    DataType.FLOAT_VECTOR: 101,
    DataType.SPARSE_FLOAT_VECTOR: 104,
}


def _default_for(dtype: DataType) -> Any:
    """Type-appropriate placeholder for null entries. The valid_data
    array marks them as null on the wire."""
    if dtype == DataType.BOOL:
        return False
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        return 0
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return 0.0
    if dtype == DataType.VARCHAR:
        return ""
    return None


def _coerce_for(dtype: DataType, v: Any) -> Any:
    """Coerce a Python value into the type its proto slot expects."""
    if dtype == DataType.BOOL:
        return bool(v)
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        return int(v)
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return float(v)
    if dtype == DataType.VARCHAR:
        return str(v)
    return v
