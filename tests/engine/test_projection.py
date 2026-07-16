"""Unit tests for unified schema/dynamic output projection."""

import pytest

from milvus_lite.engine.projection import (
    build_projection_plan,
    decode_meta,
    project_record,
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
