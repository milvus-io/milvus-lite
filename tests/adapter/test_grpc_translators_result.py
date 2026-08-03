"""Regression tests for gRPC search-result encoding."""

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
    assert result.fields_data[-1].field_name == "$meta"
    assert result.fields_data[-1].is_dynamic is True
