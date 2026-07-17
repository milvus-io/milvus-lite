"""Unit tests for unified schema/dynamic output projection."""

import pytest

from milvus_lite.engine.projection import (
    build_projection_plan,
    decode_meta,
    project_record,
    projection_output_fields,
)
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(
        fields=[
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2),
        ],
        enable_dynamic_field=True,
    )


def test_star_includes_all_schema_and_dynamic(schema):
    plan = build_projection_plan(["*"], schema, api_kind="query")
    assert plan.response_schema_fields == ("pk", "text", "vector")
    assert plan.include_all_dynamic is True
    assert plan.include_raw_meta is True


def test_meta_only_includes_all_dynamic(schema):
    plan = build_projection_plan(["$meta"], schema, api_kind="search")
    assert plan.response_schema_fields == ()
    assert plan.include_all_dynamic is True
    assert plan.include_raw_meta is True


def test_explicit_dynamic_field(schema):
    plan = build_projection_plan(["text", "page"], schema, api_kind="query")
    assert plan.response_schema_fields == ("text",)
    assert plan.explicit_dynamic_fields == ("page",)
    assert plan.include_all_dynamic is False


def test_empty_projection_is_id_only(schema):
    query_plan = build_projection_plan([], schema, api_kind="query")
    search_plan = build_projection_plan([], schema, api_kind="search")
    record = {"pk": 1, "text": "foo", "page": 2}
    assert project_record(record, schema, query_plan) == {"pk": 1}
    assert project_record(record, schema, search_plan) == {}


def test_project_record_merges_raw_and_top_level_dynamic(schema):
    plan = build_projection_plan(["$meta"], schema, api_kind="query")
    record = {
        "pk": 1,
        "$meta": '{"page": 1, "source": "raw"}',
        "source": "top-level",
    }
    assert project_record(record, schema, plan) == {
        "pk": 1,
        "page": 1,
        "source": "top-level",
    }


def test_schema_field_wins_over_dynamic_collision(schema):
    plan = build_projection_plan(["*"], schema, api_kind="query")
    record = {
        "pk": 1,
        "text": "schema",
        "vector": [0.0, 0.0],
        "$meta": '{"text": "dynamic", "page": 1}',
    }
    projected = project_record(record, schema, plan)
    assert projected["text"] == "schema"
    assert projected["page"] == 1


@pytest.mark.parametrize("raw", [None, "", "not-json", "[]", 123])
def test_malformed_or_non_object_meta_is_empty(raw):
    assert decode_meta(raw) == {}


def test_default_projection_includes_dynamic_fields(schema):
    query_plan = build_projection_plan(None, schema, api_kind="query")
    search_plan = build_projection_plan(None, schema, api_kind="search")

    assert query_plan.include_all_dynamic is True
    assert query_plan.response_schema_fields == ("pk", "text", "vector")
    assert search_plan.include_all_dynamic is True
    assert search_plan.response_schema_fields == ("text",)


def test_projection_deduplicates_requested_fields(schema):
    plan = build_projection_plan(
        ["text", "page", "text", "page", "$meta"],
        schema,
        api_kind="query",
    )

    assert plan.requested_fields == ("text", "page", "$meta")
    assert plan.response_schema_fields == ("text",)
    assert plan.explicit_dynamic_fields == ("page",)
    assert plan.engine_fields == ("text", "$meta")


def test_unknown_fields_are_ignored_when_dynamic_is_disabled(schema):
    schema.enable_dynamic_field = False
    plan = build_projection_plan(
        ["text", "unknown", "$meta"], schema, api_kind="query"
    )

    assert plan.response_schema_fields == ("text",)
    assert plan.explicit_dynamic_fields == ()
    assert plan.include_all_dynamic is False
    assert plan.include_raw_meta is False


def test_projection_output_fields_normalizes_star_and_meta(schema):
    plan = build_projection_plan(["*"], schema, api_kind="search")

    assert projection_output_fields(
        plan, schema, include_primary=False
    ) == ("text", "vector", "$meta")
    assert projection_output_fields(
        plan, schema, include_primary=True
    ) == ("pk", "text", "vector", "$meta")


def test_decode_meta_accepts_predecoded_dict():
    raw = {"page": 1, "nested": {"value": True}}

    decoded = decode_meta(raw)

    assert decoded == raw
    assert decoded is not raw


def test_invalid_projection_api_kind_is_rejected(schema):
    with pytest.raises(ValueError, match="unknown projection api_kind"):
        build_projection_plan(["*"], schema, api_kind="invalid")
