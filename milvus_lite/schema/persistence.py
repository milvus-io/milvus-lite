"""Schema serialization to / from JSON.

Format (one schema = one JSON file):

    {
      "collection_name": "...",
      "version": 1,
      "enable_dynamic_field": false,
      "fields": [
        {
          "name": "id",
          "dtype": "varchar",
          "is_primary": true,
          "dim": null,
          "max_length": null,
          "nullable": false,
          "default_value": null
        },
        ...
      ]
    }

Atomic write: dump to ``path + ".tmp"`` then ``os.rename`` over the target.
"""

from __future__ import annotations

import json
import os
from typing import Any, Tuple

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)


SCHEMA_FORMAT_VERSION = 2


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_schema(
    schema: CollectionSchema,
    collection_name: str,
    path: str,
) -> None:
    """Serialize *schema* to *path* atomically (write-tmp + rename)."""
    payload: dict[str, Any] = {
        "collection_name": collection_name,
        "schema_format_version": SCHEMA_FORMAT_VERSION,
        "version": schema.version,
        "enable_dynamic_field": schema.enable_dynamic_field,
        "fields": [_field_to_dict(f) for f in schema.fields],
    }
    if schema.functions:
        payload["functions"] = [_function_to_dict(fn) for fn in schema.functions]
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_schema(path: str) -> Tuple[str, CollectionSchema]:
    """Load (collection_name, CollectionSchema) from *path*.

    Raises:
        FileNotFoundError: file does not exist
        SchemaValidationError: JSON malformed or missing required keys
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"schema file {path!r} is not valid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise SchemaValidationError(f"schema file {path!r} root must be an object")

    try:
        collection_name = payload["collection_name"]
        version = payload["version"]
        enable_dynamic = payload.get("enable_dynamic_field", False)
        fields_raw = payload["fields"]
    except KeyError as e:
        raise SchemaValidationError(
            f"schema file {path!r} missing key {e.args[0]!r}"
        ) from e

    if not isinstance(fields_raw, list):
        raise SchemaValidationError(
            f"schema file {path!r} 'fields' must be a list"
        )

    fields = [_field_from_dict(d, path) for d in fields_raw]

    functions_raw = payload.get("functions", [])
    functions = [_function_from_dict(fd, path) for fd in functions_raw]

    schema = CollectionSchema(
        fields=fields,
        version=int(version),
        enable_dynamic_field=bool(enable_dynamic),
        functions=functions,
    )
    return str(collection_name), schema


# ---------------------------------------------------------------------------
# Field <-> dict
# ---------------------------------------------------------------------------

def _field_to_dict(f: FieldSchema) -> dict:
    d: dict[str, Any] = {
        "name": f.name,
        "dtype": f.dtype.value,
        "is_primary": f.is_primary,
        "auto_id": f.auto_id,
        "dim": f.dim,
        "max_length": f.max_length,
        "element_type": f.element_type.value if f.element_type else None,
        "max_capacity": f.max_capacity,
        "nullable": f.nullable,
        "default_value": f.default_value,
    }
    # Only persist FTS attributes when non-default to reduce JSON size
    if f.enable_analyzer:
        d["enable_analyzer"] = True
    if f.analyzer_params:
        d["analyzer_params"] = f.analyzer_params
    if f.enable_match:
        d["enable_match"] = True
    if f.is_function_output:
        d["is_function_output"] = True
    if f.is_partition_key:
        d["is_partition_key"] = True
    return d


def _field_from_dict(d: Any, source: str) -> FieldSchema:
    if not isinstance(d, dict):
        raise SchemaValidationError(
            f"schema file {source!r} field entry must be an object, got {type(d).__name__}"
        )
    try:
        name = d["name"]
        dtype_str = d["dtype"]
    except KeyError as e:
        raise SchemaValidationError(
            f"schema file {source!r} field missing key {e.args[0]!r}"
        ) from e

    try:
        dtype = DataType(dtype_str)
    except ValueError as e:
        raise SchemaValidationError(
            f"schema file {source!r} unknown dtype {dtype_str!r}"
        ) from e

    return FieldSchema(
        name=str(name),
        dtype=dtype,
        is_primary=bool(d.get("is_primary", False)),
        auto_id=bool(d.get("auto_id", False)),
        dim=d.get("dim"),
        max_length=d.get("max_length"),
        element_type=DataType(d["element_type"]) if d.get("element_type") else None,
        max_capacity=d.get("max_capacity"),
        nullable=bool(d.get("nullable", False)),
        default_value=d.get("default_value"),
        enable_analyzer=bool(d.get("enable_analyzer", False)),
        analyzer_params=d.get("analyzer_params"),
        enable_match=bool(d.get("enable_match", False)),
        is_function_output=bool(d.get("is_function_output", False)),
        is_partition_key=bool(d.get("is_partition_key", False)),
    )


def _function_to_dict(fn: Function) -> dict:
    return {
        "name": fn.name,
        "function_type": int(fn.function_type),
        "input_field_names": fn.input_field_names,
        "output_field_names": fn.output_field_names,
        "params": fn.params,
    }


def _function_from_dict(d: Any, source: str) -> Function:
    if not isinstance(d, dict):
        raise SchemaValidationError(
            f"schema file {source!r} function entry must be an object"
        )
    try:
        name = d["name"]
        ft_int = d["function_type"]
    except KeyError as e:
        raise SchemaValidationError(
            f"schema file {source!r} function missing key {e.args[0]!r}"
        ) from e
    try:
        ft = FunctionType(int(ft_int))
    except ValueError as e:
        raise SchemaValidationError(
            f"schema file {source!r} unknown function_type {ft_int!r}"
        ) from e
    return Function(
        name=str(name),
        function_type=ft,
        input_field_names=list(d.get("input_field_names", [])),
        output_field_names=list(d.get("output_field_names", [])),
        params=dict(d.get("params", {})),
    )
