"""Regression tests for gRPC search-result encoding."""

import pytest

from milvus_lite.adapter.grpc.translators.result import build_search_result_data
from milvus_lite.engine.projection import build_projection_plan
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def test_default_dynamic_search_declares_meta_output_field():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2),
            FieldSchema(name="popularity", dtype=DataType.FLOAT),
        ],
        enable_dynamic_field=True,
    )
    projection_plan = build_projection_plan(None, schema, api_kind="search")

    result = build_search_result_data(
        results=[[
            {
                "id": 1,
                "distance": 1.0,
                "entity": {
                    "vector": [1.0, 0.0],
                    "popularity": 2.0,
                    "dynamic_tag": "first",
                },
            }
        ]],
        schema=schema,
        top_k=1,
        pk_name="id",
        projection_plan=projection_plan,
    )

    assert list(result.output_fields) == ["vector", "popularity", "$meta"]
    assert [field.field_name for field in result.fields_data] == ["vector", "popularity", "$meta"]
    assert result.fields_data[-1].field_name == "$meta"
    assert result.fields_data[-1].is_dynamic is True


@pytest.mark.parametrize("with_plan", [False, True])
@pytest.mark.parametrize("pk_type,pk", [(DataType.INT64, 9), (DataType.VARCHAR, "doc9")])
@pytest.mark.parametrize("output_fields,expected", [
    (None, ["vector", "label"]),
    ([], []),
    (["vector"], ["vector"]),
    (["pk", "label"], ["pk", "label"]),
    (["pk"], ["pk"]),
    (["*"], ["pk", "vector", "label"]),
])
def test_search_fields_follow_projection_without_forcing_primary(with_plan, pk_type, pk, output_fields, expected):
    schema = CollectionSchema(fields=[
        FieldSchema("pk", pk_type, is_primary=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
        FieldSchema("label", DataType.VARCHAR),
    ])
    result = build_search_result_data(
        [[{"id": pk, "distance": 0.5, "entity": {"vector": [1.0, 0.0], "label": "a"}}]],
        schema, top_k=1, pk_name="pk", output_fields=output_fields,
        projection_plan=build_projection_plan(output_fields, schema, api_kind="search") if with_plan else None,
    )

    ids = result.ids.int_id.data if pk_type == DataType.INT64 else result.ids.str_id.data
    assert list(ids) == [pk]
    assert result.primary_field_name == "pk"
    assert [field.field_name for field in result.fields_data] == expected
    assert list(result.output_fields) == expected
    if "pk" in expected:
        pk_data = next(field for field in result.fields_data if field.field_name == "pk")
        values = pk_data.scalars.long_data.data if pk_type == DataType.INT64 else pk_data.scalars.string_data.data
        assert list(values) == [pk]


def test_empty_search_result_does_not_add_primary_column():
    schema = CollectionSchema(fields=[
        FieldSchema("pk", DataType.INT64, is_primary=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
    ])
    result = build_search_result_data([[]], schema, top_k=5, pk_name="pk", output_fields=["vector"])
    assert list(result.ids.int_id.data) == []
    assert list(result.topks) == [0]
    assert [field.field_name for field in result.fields_data] == ["vector"]
    assert list(result.fields_data[0].vectors.float_vector.data) == []
