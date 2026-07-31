# Search Default Projection Compatibility Design

## Goal

Restore Milvus-compatible Search output when `output_fields` is omitted. The
default response must preserve all non-primary schema fields, including vector
fields, and must preserve dynamic fields through `$meta` when dynamic fields are
enabled.

## Root Cause

`build_projection_plan()` currently gives Search a special default that removes
primary-key and vector fields. Removing the primary key from the entity is
appropriate because Search returns it through the hit ID, but removing vector
fields changes the observable Milvus API behavior. Function Chain execution
exposes this mismatch because its final response projection reuses the Search
projection plan.

## Design

When `output_fields is None` and `api_kind == "search"`, the projection plan
will include every non-primary schema field. This keeps the primary key in the
Search hit ID while retaining scalar and vector entity fields. If the collection
has dynamic fields enabled, the plan will continue to set `include_all_dynamic`
and `include_raw_meta`, and the gRPC result will declare `$meta` so PyMilvus can
unpack dynamic values.

Explicit `output_fields`, Query, and Get projection behavior will not change.
Function Chain will not receive a special-case workaround; it will inherit the
correct ordinary Search projection.

## Testing

- Update the projection unit test to require vector fields in the default Search
  projection.
- Update the gRPC result regression test to require the vector field and `$meta`
  in default output fields.
- Run the existing Function Chain limit test that currently fails because the
  vector field is missing.
- Run projection, gRPC result, Function Chain, and the regular non-slow test
  suite.
