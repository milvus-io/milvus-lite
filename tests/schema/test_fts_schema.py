"""Phase 11.1 — Full Text Search schema extensions.

Tests for:
- DataType.SPARSE_FLOAT_VECTOR
- Function / FunctionType classes
- FieldSchema new attributes (enable_analyzer, analyzer_params, enable_match, is_function_output)
- Schema validation with BM25 functions
- Arrow schema builder with sparse vector fields
- Schema persistence (save/load) with functions
"""

import json
import os
import tempfile

import pyarrow as pa
import pytest

from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    TYPE_MAP,
)
from milvus_lite.schema.validation import validate_schema, validate_record
from milvus_lite.schema.arrow_builder import (
    _arrow_type,
    build_data_schema,
    build_wal_data_schema,
    get_vector_field,
)
from milvus_lite.schema.persistence import save_schema, load_schema
from milvus_lite.exceptions import SchemaValidationError


# ---------------------------------------------------------------------------
# DataType.SPARSE_FLOAT_VECTOR
# ---------------------------------------------------------------------------

def test_sparse_float_vector_type():
    assert DataType.SPARSE_FLOAT_VECTOR.value == "sparse_float_vector"


def test_sparse_float_vector_in_type_map():
    assert TYPE_MAP[DataType.SPARSE_FLOAT_VECTOR] == pa.binary()


# ---------------------------------------------------------------------------
# FunctionType / Function
# ---------------------------------------------------------------------------

def test_function_type_bm25():
    assert FunctionType.BM25 == 1
    assert int(FunctionType.BM25) == 1


def test_function_creation():
    fn = Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_emb"],
    )
    assert fn.name == "bm25_fn"
    assert fn.function_type == FunctionType.BM25
    assert fn.input_field_names == ["text"]
    assert fn.output_field_names == ["sparse_emb"]
    assert fn.params == {}


def test_function_with_params():
    fn = Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_emb"],
        params={"bm25_k1": 1.5, "bm25_b": 0.75},
    )
    assert fn.params["bm25_k1"] == 1.5


# ---------------------------------------------------------------------------
# FieldSchema new attributes
# ---------------------------------------------------------------------------

def test_field_schema_fts_defaults():
    f = FieldSchema(name="text", dtype=DataType.VARCHAR)
    assert f.enable_analyzer is False
    assert f.analyzer_params is None
    assert f.enable_match is False
    assert f.is_function_output is False


def test_field_schema_fts_attributes():
    f = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "standard"},
        enable_match=True,
    )
    assert f.enable_analyzer is True
    assert f.analyzer_params == {"tokenizer": "standard"}
    assert f.enable_match is True


def test_field_schema_function_output():
    f = FieldSchema(
        name="sparse_emb",
        dtype=DataType.SPARSE_FLOAT_VECTOR,
        is_function_output=True,
    )
    assert f.is_function_output is True


# ---------------------------------------------------------------------------
# CollectionSchema with functions
# ---------------------------------------------------------------------------

def test_collection_schema_functions_default():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ])
    assert schema.functions == []


def _bm25_schema():
    """Helper: schema with BM25 function."""
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(
                name="text", dtype=DataType.VARCHAR,
                enable_analyzer=True,
                analyzer_params={"tokenizer": "standard"},
                enable_match=True,
            ),
            FieldSchema(
                name="sparse_emb", dtype=DataType.SPARSE_FLOAT_VECTOR,
                is_function_output=True,
            ),
        ],
        functions=[
            Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse_emb"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Schema validation — vector field rules
# ---------------------------------------------------------------------------

def test_sparse_only_schema_validates():
    """Schema with only SPARSE_FLOAT_VECTOR (no FLOAT_VECTOR) is valid."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(
                name="text", dtype=DataType.VARCHAR,
                enable_analyzer=True,
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
    validate_schema(schema)  # should not raise


def test_mixed_vector_schema_validates():
    """Schema with both FLOAT_VECTOR and SPARSE_FLOAT_VECTOR is valid."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, enable_analyzer=True),
            FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=128),
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
    validate_schema(schema)  # should not raise


# ---------------------------------------------------------------------------
# Schema validation — BM25 function rules
# ---------------------------------------------------------------------------

def test_bm25_validates():
    validate_schema(_bm25_schema())  # should not raise


def test_bm25_input_not_varchar():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="num", dtype=DataType.INT64, enable_analyzer=True),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
        functions=[
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["num"],
                output_field_names=["sparse"],
            ),
        ],
    )
    with pytest.raises(SchemaValidationError, match="must be VARCHAR"):
        validate_schema(schema)


def test_bm25_input_no_analyzer():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR),  # no enable_analyzer
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
    with pytest.raises(SchemaValidationError, match="enable_analyzer"):
        validate_schema(schema)


def test_bm25_output_not_sparse():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, enable_analyzer=True),
            FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=4),
        ],
        functions=[
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["dense"],
            ),
        ],
    )
    with pytest.raises(SchemaValidationError, match="SPARSE_FLOAT_VECTOR"):
        validate_schema(schema)


def test_bm25_input_field_not_found():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
        functions=[
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["missing"],
                output_field_names=["sparse"],
            ),
        ],
    )
    with pytest.raises(SchemaValidationError, match="not found"):
        validate_schema(schema)


def test_bm25_multiple_inputs_rejected():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="t1", dtype=DataType.VARCHAR, enable_analyzer=True),
            FieldSchema(name="t2", dtype=DataType.VARCHAR, enable_analyzer=True),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
        functions=[
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["t1", "t2"],
                output_field_names=["sparse"],
            ),
        ],
    )
    with pytest.raises(SchemaValidationError, match="exactly one input"):
        validate_schema(schema)


# ---------------------------------------------------------------------------
# Record validation — sparse vectors
# ---------------------------------------------------------------------------

def test_validate_record_function_output_skipped():
    """Function output fields should not be required in user-provided records."""
    schema = _bm25_schema()
    record = {"id": 1, "text": "hello world"}
    validate_record(record, schema)  # should not raise


def test_validate_record_sparse_vector_user_provided():
    """User-provided sparse vectors are validated."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
    )
    # Valid sparse vector
    record = {"id": 1, "sv": {0: 1.0, 5: 0.5}}
    validate_record(record, schema)  # should not raise


def test_validate_record_sparse_vector_bad_key():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
    )
    with pytest.raises(SchemaValidationError, match="must be int"):
        validate_record({"id": 1, "sv": {"bad": 1.0}}, schema)


def test_validate_record_sparse_vector_negative_key():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
    )
    with pytest.raises(SchemaValidationError, match="non-negative"):
        validate_record({"id": 1, "sv": {-1: 1.0}}, schema)


def test_validate_record_sparse_vector_bad_value():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
    )
    with pytest.raises(SchemaValidationError, match="must be numeric"):
        validate_record({"id": 1, "sv": {0: "bad"}}, schema)


# ---------------------------------------------------------------------------
# Arrow schema builder
# ---------------------------------------------------------------------------

def test_arrow_type_sparse():
    f = FieldSchema(name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR)
    assert _arrow_type(f) == pa.binary()


def test_build_data_schema_with_sparse():
    schema = _bm25_schema()
    arrow_schema = build_data_schema(schema)
    # Should have: _seq, id, text, sparse_emb
    assert "_seq" in arrow_schema.names
    assert "sparse_emb" in arrow_schema.names
    assert arrow_schema.field("sparse_emb").type == pa.binary()


def test_build_wal_data_schema_with_sparse():
    schema = _bm25_schema()
    arrow_schema = build_wal_data_schema(schema)
    assert "_seq" in arrow_schema.names
    assert "_partition" in arrow_schema.names
    assert "sparse_emb" in arrow_schema.names


def test_get_vector_field_sparse_only():
    """get_vector_field returns None when only SPARSE_FLOAT_VECTOR exists."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
    )
    assert get_vector_field(schema) is None


def test_get_vector_field_prefers_float():
    """get_vector_field returns FLOAT_VECTOR, ignoring SPARSE_FLOAT_VECTOR."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dv", dtype=DataType.FLOAT_VECTOR, dim=4),
        ],
    )
    vf = get_vector_field(schema)
    assert vf.name == "dv"


# ---------------------------------------------------------------------------
# Schema persistence
# ---------------------------------------------------------------------------

def test_save_load_bm25_schema():
    """Round-trip save/load of schema with BM25 function."""
    schema = _bm25_schema()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "schema.json")
        save_schema(schema, "test_col", path)

        name, loaded = load_schema(path)
        assert name == "test_col"

        # Fields
        assert len(loaded.fields) == 3
        text_f = next(f for f in loaded.fields if f.name == "text")
        assert text_f.enable_analyzer is True
        assert text_f.analyzer_params == {"tokenizer": "standard"}
        assert text_f.enable_match is True

        sparse_f = next(f for f in loaded.fields if f.name == "sparse_emb")
        assert sparse_f.dtype == DataType.SPARSE_FLOAT_VECTOR
        assert sparse_f.is_function_output is True

        # Functions
        assert len(loaded.functions) == 1
        fn = loaded.functions[0]
        assert fn.name == "bm25_fn"
        assert fn.function_type == FunctionType.BM25
        assert fn.input_field_names == ["text"]
        assert fn.output_field_names == ["sparse_emb"]


def test_load_v1_schema_compat():
    """v1 schema files (no functions/FTS attrs) load without error."""
    v1_payload = {
        "collection_name": "old_col",
        "schema_format_version": 1,
        "version": 1,
        "enable_dynamic_field": False,
        "fields": [
            {"name": "id", "dtype": "int64", "is_primary": True,
             "dim": None, "max_length": None, "nullable": False,
             "default_value": None},
            {"name": "vec", "dtype": "float_vector", "is_primary": False,
             "dim": 4, "max_length": None, "nullable": False,
             "default_value": None},
        ],
    }

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "schema.json")
        with open(path, "w") as f:
            json.dump(v1_payload, f)

        name, loaded = load_schema(path)
        assert name == "old_col"
        assert len(loaded.fields) == 2
        assert loaded.functions == []
        # FTS attributes default to False/None
        assert loaded.fields[0].enable_analyzer is False
        assert loaded.fields[0].analyzer_params is None


def test_save_load_function_params():
    """BM25 function params (k1, b) persist correctly."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, enable_analyzer=True),
            FieldSchema(name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR,
                        is_function_output=True),
        ],
        functions=[
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sv"],
                params={"bm25_k1": 1.5, "bm25_b": 0.75},
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "schema.json")
        save_schema(schema, "test", path)
        _, loaded = load_schema(path)
        fn = loaded.functions[0]
        assert fn.params["bm25_k1"] == 1.5
        assert fn.params["bm25_b"] == 0.75
