"""Schema and record validation."""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)
from milvus_lite.schema.timestamptz import parse_timestamptz

# Reserved column names — users may not name fields these.
#
# `iso`/`interval` mirror Milvus Plan.g4: they are lexer keywords for
# TIMESTAMPTZ literals, not identifiers.
RESERVED_FIELD_NAMES = frozenset({
    "_seq",
    "_partition",
    "$meta",
    "iso",
    "ISO",
    "interval",
    "INTERVAL",
})
_CASE_INSENSITIVE_RESERVED_FIELD_NAMES = frozenset({"iso", "interval"})

# pk dtype must be one of these.
_PK_ALLOWED_DTYPES = frozenset({DataType.VARCHAR, DataType.INT64})

# Per-DataType Python-type predicate. We accept the canonical Python types
# plus a couple of widening cases (int → float, bool → int handled below).
_DTYPE_PYTHON_CHECK = {
    DataType.BOOL: lambda v: isinstance(v, bool),
    DataType.INT8: lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2**7) <= v < 2**7,
    DataType.INT16: lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2**15) <= v < 2**15,
    DataType.INT32: lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2**31) <= v < 2**31,
    DataType.INT64: lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2**63) <= v < 2**63,
    DataType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    DataType.DOUBLE: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    DataType.VARCHAR: lambda v: isinstance(v, str),
    DataType.JSON: lambda v: isinstance(v, (dict, list, str, int, float, bool)) or v is None,
    DataType.TIMESTAMPTZ: lambda v: isinstance(v, int) and not isinstance(v, bool),
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(schema: CollectionSchema) -> None:
    """Validate a CollectionSchema definition.

    Rules:
    - exactly one is_primary=True field; dtype is VARCHAR or INT64
    - at least one vector field (FLOAT_VECTOR or SPARSE_FLOAT_VECTOR)
    - at most one FLOAT_VECTOR field
    - FLOAT_VECTOR field must have dim > 0
    - primary key field must not be nullable
    - field names must be unique
    - field names must not collide with reserved names
    - BM25 function: input must be VARCHAR with enable_analyzer,
      output must be SPARSE_FLOAT_VECTOR
    """
    if not schema.fields:
        raise SchemaValidationError("schema has no fields")

    seen_names: set[str] = set()
    pk_fields: list[FieldSchema] = []
    float_vector_fields: list[FieldSchema] = []
    all_vector_fields: list[FieldSchema] = []
    field_by_name: dict[str, FieldSchema] = {}

    for f in schema.fields:
        if not f.name:
            raise SchemaValidationError("field name must not be empty")
        if (
            f.name in RESERVED_FIELD_NAMES
            or f.name.casefold() in _CASE_INSENSITIVE_RESERVED_FIELD_NAMES
        ):
            raise SchemaValidationError(
                f"field name {f.name!r} is reserved (one of {sorted(RESERVED_FIELD_NAMES)})"
            )
        if f.name in seen_names:
            raise SchemaValidationError(f"duplicate field name: {f.name!r}")
        seen_names.add(f.name)
        field_by_name[f.name] = f

        if f.is_primary:
            pk_fields.append(f)
        if f.dtype == DataType.FLOAT_VECTOR:
            float_vector_fields.append(f)
            all_vector_fields.append(f)
            if f.dim is None or f.dim <= 0:
                raise SchemaValidationError(
                    f"vector field {f.name!r} requires dim > 0, got {f.dim}"
                )
        if f.dtype == DataType.SPARSE_FLOAT_VECTOR:
            all_vector_fields.append(f)
        if f.dtype == DataType.ARRAY:
            if f.element_type is None:
                raise SchemaValidationError(
                    f"ARRAY field {f.name!r} requires element_type"
                )
        if f.dtype == DataType.TIMESTAMPTZ and f.default_value is not None:
            f.default_value = parse_timestamptz(f.default_value)

    if len(pk_fields) == 0:
        raise SchemaValidationError("schema has no primary key field")
    if len(pk_fields) > 1:
        names = [f.name for f in pk_fields]
        raise SchemaValidationError(
            f"schema must have exactly one primary key field, got {names}"
        )
    pk = pk_fields[0]
    if pk.dtype not in _PK_ALLOWED_DTYPES:
        raise SchemaValidationError(
            f"primary key {pk.name!r} must be VARCHAR or INT64, got {pk.dtype}"
        )
    if pk.nullable:
        raise SchemaValidationError(
            f"primary key {pk.name!r} must not be nullable"
        )
    if pk.auto_id and pk.dtype != DataType.INT64:
        raise SchemaValidationError(
            f"auto_id primary key {pk.name!r} must be INT64, got {pk.dtype.name}"
        )

    if len(all_vector_fields) == 0:
        raise SchemaValidationError(
            "schema has no vector field (FLOAT_VECTOR or SPARSE_FLOAT_VECTOR)"
        )

    # Partition key validation
    pk_key_fields = [f for f in schema.fields if f.is_partition_key]
    if len(pk_key_fields) > 1:
        names = [f.name for f in pk_key_fields]
        raise SchemaValidationError(
            f"at most one field can be partition key, got {names}"
        )
    if pk_key_fields:
        pkf = pk_key_fields[0]
        if pkf.is_primary:
            raise SchemaValidationError(
                f"partition key {pkf.name!r} must not be the primary key"
            )
        if pkf.dtype not in (DataType.INT64, DataType.VARCHAR):
            raise SchemaValidationError(
                f"partition key {pkf.name!r} must be INT64 or VARCHAR, "
                f"got {pkf.dtype.name}"
            )

    # Validate functions
    func_names_seen: set = set()
    func_outputs_seen: set = set()
    for func in schema.functions:
        if func.name in func_names_seen:
            raise SchemaValidationError(
                f"duplicate function name {func.name!r}"
            )
        func_names_seen.add(func.name)
        for out_name in func.output_field_names:
            if out_name in func_outputs_seen:
                raise SchemaValidationError(
                    f"function output field {out_name!r} is used by multiple functions"
                )
            func_outputs_seen.add(out_name)
        _validate_function(func, field_by_name)


# ---------------------------------------------------------------------------
# Function validation
# ---------------------------------------------------------------------------

def _validate_function(func: Function, field_by_name: dict[str, FieldSchema]) -> None:
    """Validate a single Function definition against its schema fields."""
    if func.function_type == FunctionType.BM25:
        # Input: exactly one VARCHAR field with enable_analyzer=True
        if len(func.input_field_names) != 1:
            raise SchemaValidationError(
                f"BM25 function {func.name!r} requires exactly one input field"
            )
        in_name = func.input_field_names[0]
        in_field = field_by_name.get(in_name)
        if in_field is None:
            raise SchemaValidationError(
                f"BM25 function {func.name!r} input field {in_name!r} not found in schema"
            )
        if in_field.dtype != DataType.VARCHAR:
            raise SchemaValidationError(
                f"BM25 function {func.name!r} input field {in_name!r} must be VARCHAR, "
                f"got {in_field.dtype.name}"
            )
        if not in_field.enable_analyzer:
            raise SchemaValidationError(
                f"BM25 function {func.name!r} input field {in_name!r} "
                f"must have enable_analyzer=True"
            )

        # Output: exactly one SPARSE_FLOAT_VECTOR field
        if len(func.output_field_names) != 1:
            raise SchemaValidationError(
                f"BM25 function {func.name!r} requires exactly one output field"
            )
        out_name = func.output_field_names[0]
        out_field = field_by_name.get(out_name)
        if out_field is None:
            raise SchemaValidationError(
                f"BM25 function {func.name!r} output field {out_name!r} not found in schema"
            )
        if out_field.dtype != DataType.SPARSE_FLOAT_VECTOR:
            raise SchemaValidationError(
                f"BM25 function {func.name!r} output field {out_name!r} must be "
                f"SPARSE_FLOAT_VECTOR, got {out_field.dtype.name}"
            )
    elif func.function_type == FunctionType.TEXT_EMBEDDING:
        # Input: exactly one VARCHAR field
        if len(func.input_field_names) != 1:
            raise SchemaValidationError(
                f"TEXT_EMBEDDING function {func.name!r} requires exactly one input field"
            )
        in_name = func.input_field_names[0]
        in_field = field_by_name.get(in_name)
        if in_field is None:
            raise SchemaValidationError(
                f"TEXT_EMBEDDING function {func.name!r} input field {in_name!r} "
                f"not found in schema"
            )
        if in_field.dtype != DataType.VARCHAR:
            raise SchemaValidationError(
                f"TEXT_EMBEDDING function {func.name!r} input field {in_name!r} "
                f"must be VARCHAR, got {in_field.dtype.name}"
            )

        # Output: exactly one FLOAT_VECTOR field
        if len(func.output_field_names) != 1:
            raise SchemaValidationError(
                f"TEXT_EMBEDDING function {func.name!r} requires exactly one output field"
            )
        out_name = func.output_field_names[0]
        out_field = field_by_name.get(out_name)
        if out_field is None:
            raise SchemaValidationError(
                f"TEXT_EMBEDDING function {func.name!r} output field {out_name!r} "
                f"not found in schema"
            )
        if out_field.dtype != DataType.FLOAT_VECTOR:
            raise SchemaValidationError(
                f"TEXT_EMBEDDING function {func.name!r} output field {out_name!r} "
                f"must be FLOAT_VECTOR, got {out_field.dtype.name}"
            )
    elif func.function_type == FunctionType.RERANK:
        raise SchemaValidationError(
            f"RERANK function {func.name!r} is not supported in collection schema; "
            "use request-level function_score/ranker instead"
        )
    else:
        raise SchemaValidationError(
            f"unknown function type {func.function_type!r} for function {func.name!r}"
        )


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------

def _function_output_field_names(schema: CollectionSchema) -> frozenset[str]:
    """Return the set of field names that are auto-generated by functions."""
    names: set[str] = set()
    for func in schema.functions:
        names.update(func.output_field_names)
    # Also include fields explicitly marked
    for f in schema.fields:
        if f.is_function_output:
            names.add(f.name)
    return frozenset(names)


def validate_record(record: dict, schema: CollectionSchema) -> None:
    """Validate a single record dict against the schema.

    Rules:
    - pk field present and non-None
    - FLOAT_VECTOR field present and len(vector) == field.dim, every element numeric
    - SPARSE_FLOAT_VECTOR fields that are function outputs are skipped
      (auto-generated by engine); user-provided sparse vectors are validated
    - declared field values match their dtype
    - non-nullable fields are not None
    - if enable_dynamic_field=False, no fields outside the schema
    """
    if not isinstance(record, dict):
        raise SchemaValidationError(
            f"record must be a dict, got {type(record).__name__}"
        )

    # Fill default values for missing or None fields before validation.
    import copy
    for f in schema.fields:
        if f.default_value is not None:
            if f.name not in record or record[f.name] is None:
                record[f.name] = copy.deepcopy(f.default_value)

    schema_field_names = {f.name for f in schema.fields}
    pk = _find_pk(schema)
    func_output_names = _function_output_field_names(schema)

    # pk presence (skip check when auto_id — engine generates it)
    if not pk.auto_id:
        if pk.name not in record or record[pk.name] is None:
            raise SchemaValidationError(
                f"primary key {pk.name!r} missing or None"
            )

    # FLOAT_VECTOR presence + shape
    float_vec = _find_float_vector(schema)
    if float_vec is not None:
        vec_val = record.get(float_vec.name)
        if vec_val is None or float_vec.name not in record:
            if not float_vec.nullable:
                raise SchemaValidationError(
                    f"vector field {float_vec.name!r} missing or None"
                )
            # nullable vector — None is valid, skip shape check
        else:
            if not isinstance(vec_val, (list, tuple)):
                raise SchemaValidationError(
                    f"vector field {float_vec.name!r} must be list/tuple, "
                    f"got {type(vec_val).__name__}"
                )
            if len(vec_val) != float_vec.dim:
                raise SchemaValidationError(
                    f"vector field {float_vec.name!r} expected dim {float_vec.dim}, "
                    f"got {len(vec_val)}"
                )
            for i, x in enumerate(vec_val):
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    raise SchemaValidationError(
                        f"vector field {float_vec.name!r}[{i}] must be numeric, "
                        f"got {type(x).__name__}"
                    )

    # per-field type / nullability
    for f in schema.fields:
        if f.dtype == DataType.FLOAT_VECTOR:
            continue  # already checked above
        if f.dtype == DataType.ARRAY:
            if f.name in record and record[f.name] is not None:
                val = record[f.name]
                if not isinstance(val, (list, tuple)):
                    raise SchemaValidationError(
                        f"ARRAY field {f.name!r} must be list/tuple, "
                        f"got {type(val).__name__}"
                    )
                if f.max_capacity is not None and len(val) > f.max_capacity:
                    raise SchemaValidationError(
                        f"ARRAY field {f.name!r} has {len(val)} elements, "
                        f"exceeds max_capacity={f.max_capacity}"
                    )
            elif not f.nullable and f.default_value is None:
                raise SchemaValidationError(
                    f"ARRAY field {f.name!r} missing and not nullable / no default"
                )
            continue
        if f.dtype == DataType.SPARSE_FLOAT_VECTOR:
            # Function output fields: skip — engine auto-generates them.
            if f.name in func_output_names:
                continue
            # User-provided sparse vectors: validate if present.
            if f.name in record and record[f.name] is not None:
                _validate_sparse_vector(f.name, record[f.name])
            elif not f.nullable and f.default_value is None:
                raise SchemaValidationError(
                    f"sparse vector field {f.name!r} missing and not nullable / no default"
                )
            continue
        if f.name not in record:
            if f.nullable or f.default_value is not None:
                continue
            # Function output fields can be absent
            if f.name in func_output_names:
                continue
            raise SchemaValidationError(
                f"field {f.name!r} missing and not nullable / no default"
            )
        value = record[f.name]
        if value is None:
            if not f.nullable:
                raise SchemaValidationError(
                    f"field {f.name!r} is None but not nullable"
                )
            continue
        if f.dtype == DataType.TIMESTAMPTZ:
            try:
                record[f.name] = parse_timestamptz(value)
            except SchemaValidationError as e:
                raise SchemaValidationError(
                    f"field {f.name!r} value {value!r} does not match dtype {f.dtype}"
                ) from e
            value = record[f.name]
        # VARCHAR max_length check
        if f.dtype == DataType.VARCHAR and f.max_length is not None:
            if isinstance(value, str) and len(value) > f.max_length:
                raise SchemaValidationError(
                    f"VARCHAR field {f.name!r} value length {len(value)} "
                    f"exceeds max_length={f.max_length}"
                )
        check = _DTYPE_PYTHON_CHECK.get(f.dtype)
        if check is None:
            continue
        if not check(value):
            raise SchemaValidationError(
                f"field {f.name!r} value {value!r} does not match dtype {f.dtype}"
            )

    # dynamic-field policy
    if not schema.enable_dynamic_field:
        extras = set(record.keys()) - schema_field_names
        if extras:
            raise SchemaValidationError(
                f"record has fields {sorted(extras)} not in schema "
                f"(enable_dynamic_field is False)"
            )


# ---------------------------------------------------------------------------
# Dynamic field separation
# ---------------------------------------------------------------------------

def separate_dynamic_fields(
    record: dict, schema: CollectionSchema
) -> Tuple[dict, Optional[str]]:
    """Split a record into (schema_fields, meta_json).

    schema_fields contains only fields declared in the schema, with default
    values filled in for missing nullable fields.

    meta_json is a JSON string of the extra fields, or None if no extras.

    Raises SchemaValidationError if enable_dynamic_field=False and there
    are extra fields.
    """
    if not isinstance(record, dict):
        raise SchemaValidationError(
            f"record must be a dict, got {type(record).__name__}"
        )

    schema_field_names = {f.name for f in schema.fields}

    schema_part: dict = {}
    extras: dict = {}
    for key, value in record.items():
        if key in schema_field_names:
            schema_part[key] = value
        else:
            extras[key] = value

    # Fill defaults for missing schema fields.
    for f in schema.fields:
        if f.name not in schema_part:
            if f.default_value is not None:
                schema_part[f.name] = f.default_value
            elif f.nullable:
                schema_part[f.name] = None
            # else: leave missing — validate_record will have caught it

    if extras and not schema.enable_dynamic_field:
        raise SchemaValidationError(
            f"record has fields {sorted(extras)} not in schema "
            f"(enable_dynamic_field is False)"
        )

    meta_json = json.dumps(extras, sort_keys=True) if extras else None
    return schema_part, meta_json


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_pk(schema: CollectionSchema) -> FieldSchema:
    for f in schema.fields:
        if f.is_primary:
            return f
    raise SchemaValidationError("schema has no primary key field")


def _find_float_vector(schema: CollectionSchema) -> Optional[FieldSchema]:
    """Return the FLOAT_VECTOR field, or None if only sparse vectors exist."""
    for f in schema.fields:
        if f.dtype == DataType.FLOAT_VECTOR:
            return f
    return None


def _validate_sparse_vector(field_name: str, value: Any) -> None:
    """Validate a user-provided sparse vector value.

    Expected format: dict[int, float] where keys are non-negative integers
    (term IDs) and values are float scores.
    """
    if not isinstance(value, dict):
        raise SchemaValidationError(
            f"sparse vector field {field_name!r} must be a dict, "
            f"got {type(value).__name__}"
        )
    for k, v in value.items():
        if not isinstance(k, int) or isinstance(k, bool):
            raise SchemaValidationError(
                f"sparse vector field {field_name!r} key {k!r} must be int"
            )
        if k < 0:
            raise SchemaValidationError(
                f"sparse vector field {field_name!r} key {k} must be non-negative"
            )
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise SchemaValidationError(
                f"sparse vector field {field_name!r} value for key {k} "
                f"must be numeric, got {type(v).__name__}"
            )
