"""Unified output projection for schema and dynamic fields."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, Optional, Tuple

from milvus_lite.schema.types import CollectionSchema, DataType


_INTERNAL_FIELDS = frozenset({"_seq", "_partition", "$meta"})
_VECTOR_TYPES = frozenset({DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR})
_API_KINDS = frozenset({"search", "query", "get"})


@dataclass(frozen=True)
class ProjectionPlan:
    """Normalized schema/dynamic projection semantics for one read API."""

    requested_fields: Optional[Tuple[str, ...]]
    engine_fields: Optional[Tuple[str, ...]]
    response_schema_fields: Tuple[str, ...]
    explicit_dynamic_fields: Tuple[str, ...]
    include_all_dynamic: bool
    include_raw_meta: bool
    api_kind: str


def build_projection_plan(
    output_fields: Optional[Iterable[str]],
    schema: CollectionSchema,
    *,
    api_kind: str,
) -> ProjectionPlan:
    """Normalize user output fields without changing API-specific defaults."""
    if api_kind not in _API_KINDS:
        raise ValueError(f"unknown projection api_kind {api_kind!r}")

    schema_names = tuple(field.name for field in schema.fields)
    schema_name_set = set(schema_names)

    if output_fields is None:
        if api_kind == "search":
            response_schema = tuple(
                field.name
                for field in schema.fields
                if not field.is_primary and field.dtype not in _VECTOR_TYPES
            )
        else:
            response_schema = schema_names
        include_dynamic = schema.enable_dynamic_field
        return ProjectionPlan(
            requested_fields=None,
            engine_fields=None,
            response_schema_fields=response_schema,
            explicit_dynamic_fields=(),
            include_all_dynamic=include_dynamic,
            include_raw_meta=include_dynamic,
            api_kind=api_kind,
        )

    requested = tuple(dict.fromkeys(output_fields))
    include_star = "*" in requested
    include_meta = "$meta" in requested
    include_all_dynamic = schema.enable_dynamic_field and (
        include_star or include_meta
    )

    if include_star:
        response_schema = schema_names
    else:
        requested_set = set(requested)
        response_schema = tuple(
            name for name in schema_names if name in requested_set
        )

    explicit_dynamic = tuple(
        name
        for name in requested
        if name not in schema_name_set and name not in {"*", "$meta"}
    ) if schema.enable_dynamic_field else ()

    include_raw_meta = include_all_dynamic or bool(explicit_dynamic)
    engine_fields_list = list(response_schema)
    if include_raw_meta:
        engine_fields_list.append("$meta")

    return ProjectionPlan(
        requested_fields=requested,
        engine_fields=tuple(dict.fromkeys(engine_fields_list)),
        response_schema_fields=response_schema,
        explicit_dynamic_fields=explicit_dynamic,
        include_all_dynamic=include_all_dynamic,
        include_raw_meta=include_raw_meta,
        api_kind=api_kind,
    )


def decode_meta(raw: object) -> Dict[str, Any]:
    """Decode a physical `$meta` value, tolerating malformed old data."""
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def extract_dynamic_fields(
    record: Dict[str, Any],
    schema: CollectionSchema,
) -> Dict[str, Any]:
    """Merge raw `$meta` with already-unpacked top-level dynamic fields."""
    schema_names = {field.name for field in schema.fields}
    dynamic = decode_meta(record.get("$meta"))
    dynamic.update({
        key: value
        for key, value in record.items()
        if key not in schema_names and key not in _INTERNAL_FIELDS
    })
    for name in schema_names:
        dynamic.pop(name, None)
    return dynamic


def project_record(
    record: Dict[str, Any],
    schema: CollectionSchema,
    plan: ProjectionPlan,
) -> Dict[str, Any]:
    """Project one logical record according to a normalized plan."""
    dynamic = extract_dynamic_fields(record, schema)
    if plan.include_all_dynamic:
        result: Dict[str, Any] = dict(dynamic)
    else:
        result = {
            name: dynamic[name]
            for name in plan.explicit_dynamic_fields
            if name in dynamic
        }

    for name in plan.response_schema_fields:
        if plan.api_kind == "search" and _is_primary(schema, name):
            continue
        if name in record:
            result[name] = record[name]

    if plan.api_kind != "search":
        pk_name = next(field.name for field in schema.fields if field.is_primary)
        if pk_name in record:
            result[pk_name] = record[pk_name]
    result.pop("$meta", None)
    return result


def projection_output_fields(
    plan: ProjectionPlan,
    schema: CollectionSchema,
    *,
    include_primary: bool,
) -> Tuple[str, ...]:
    """Return protocol-visible field names for a normalized plan."""
    primary_names = {field.name for field in schema.fields if field.is_primary}
    names = [
        name
        for name in plan.response_schema_fields
        if include_primary or name not in primary_names
    ]
    if plan.include_all_dynamic:
        names.append("$meta")
    else:
        names.extend(plan.explicit_dynamic_fields)
    return tuple(dict.fromkeys(names))


def _is_primary(schema: CollectionSchema, field_name: str) -> bool:
    return any(
        field.name == field_name and field.is_primary
        for field in schema.fields
    )
