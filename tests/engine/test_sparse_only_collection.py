"""Issue #12 — sparse-only collections (no FLOAT_VECTOR) must not crash
on flush, close, or drop.

Root cause: get_vector_field() used to fall back to SPARSE_FLOAT_VECTOR,
but Segment.load()'s _extract_vector_array() assumed FixedSizeListType.
Now get_vector_field() returns None for sparse-only schemas, and the
downstream code handles that gracefully.
"""

import pytest

from milvus_lite.db import MilvusLite
from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)


@pytest.fixture
def sparse_only_schema():
    """Schema with SPARSE_FLOAT_VECTOR and no FLOAT_VECTOR."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=1024,
                enable_analyzer=True,
                analyzer_params={"tokenizer": "standard"},
            ),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
        functions=[
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            ),
        ],
    )
    return schema


def test_sparse_only_collection_vector_name_is_none(tmp_path, sparse_only_schema):
    """_vector_name should be None for a sparse-only collection."""
    col = Collection("sparse_col", str(tmp_path / "data"), sparse_only_schema)
    assert col._vector_name is None
    col.close()


def test_sparse_only_insert_and_close(tmp_path, sparse_only_schema):
    """Insert into a sparse-only collection and close without crash."""
    col = Collection("sparse_col", str(tmp_path / "data"), sparse_only_schema)
    col.insert([{"text": "hello world"}])
    # close() triggers flush when MemTable is non-empty — this is the
    # exact path that crashed before the fix.
    col.close()


def test_sparse_only_flush(tmp_path, sparse_only_schema):
    """Explicit flush on a sparse-only collection must not crash."""
    col = Collection("sparse_col", str(tmp_path / "data"), sparse_only_schema)
    col.insert([{"text": "hello world"}, {"text": "foo bar"}])
    col.flush()
    col.close()


def test_sparse_only_drop_collection(tmp_path, sparse_only_schema):
    """DropCollection on a sparse-only collection must succeed (issue #12)."""
    db = MilvusLite(str(tmp_path / "db"))
    db.create_collection("fts_test", sparse_only_schema)
    col = db.get_collection("fts_test")
    col.insert([{"text": "hello world"}])
    # This used to crash with:
    #   ValueError: vector column must be FixedSizeList, got binary
    db.drop_collection("fts_test")
    assert not db.has_collection("fts_test")
    db.close()


def test_sparse_only_query(tmp_path, sparse_only_schema):
    """Scalar query on a sparse-only collection must work."""
    col = Collection("sparse_col", str(tmp_path / "data"), sparse_only_schema)
    col.insert([{"text": "hello world"}, {"text": "foo bar"}])
    col.load()
    results = col.query(expr=None, output_fields=["text"])
    assert len(results) == 2
    col.close()


def test_sparse_only_num_entities(tmp_path, sparse_only_schema):
    """num_entities must work for sparse-only collections."""
    col = Collection("sparse_col", str(tmp_path / "data"), sparse_only_schema)
    col.insert([{"text": "hello"}, {"text": "world"}])
    col.load()
    assert col.num_entities == 2
    col.close()


def test_sparse_only_recovery(tmp_path, sparse_only_schema):
    """Reopen a sparse-only collection after insert + close."""
    data_dir = str(tmp_path / "data")
    col = Collection("sparse_col", data_dir, sparse_only_schema)
    col.insert([{"text": "hello world"}])
    col.close()

    # Reopen — recovery must handle sparse-only segments
    col2 = Collection("sparse_col", data_dir, sparse_only_schema)
    col2.load()
    assert col2.num_entities == 1
    col2.close()
