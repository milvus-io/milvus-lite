"""Tests for build_ingestion_chain."""

from milvus_lite.function.builder import build_ingestion_chain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.map_op import MapOp
from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)


def _bm25_schema():
    """Schema with a BM25 function: text -> sparse_vec."""
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=1024,
                enable_analyzer=True,
            ),
            FieldSchema(
                name="sparse_vec",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                is_function_output=True,
            ),
        ],
        functions=[
            Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse_vec"],
            ),
        ],
    )


def _field_by_name(schema):
    return {f.name: f for f in schema.fields}


def test_build_ingestion_chain_bm25():
    schema = _bm25_schema()
    chain = build_ingestion_chain(schema, _field_by_name(schema))
    assert chain is not None
    assert len(chain.operators) == 1
    assert isinstance(chain.operators[0], MapOp)
    assert chain.operators[0].expr.name == "bm25"


def test_build_ingestion_chain_no_functions():
    schema = CollectionSchema(
        fields=[FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)]
    )
    chain = build_ingestion_chain(schema, _field_by_name(schema))
    assert chain is None


def test_ingestion_chain_bm25_end_to_end():
    """Build chain and execute on sample records."""
    schema = _bm25_schema()
    chain = build_ingestion_chain(schema, _field_by_name(schema))

    records = [
        {"id": 1, "text": "hello world"},
        {"id": 2, "text": "foo bar"},
        {"id": 3, "text": None},
    ]
    df = DataFrame.from_records(records)
    chain.execute(df)

    # sparse_vec should be injected
    assert "sparse_vec" in records[0]
    assert isinstance(records[0]["sparse_vec"], dict)
    assert len(records[0]["sparse_vec"]) > 0  # "hello" + "world" -> 2 terms

    assert "sparse_vec" in records[1]
    assert len(records[1]["sparse_vec"]) > 0

    # None text -> empty sparse vec
    assert records[2]["sparse_vec"] == {}
