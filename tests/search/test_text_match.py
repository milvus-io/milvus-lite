"""Phase 11.6 — text_match filter tests.

Covers:
- Parser: text_match(field, 'tokens') syntax
- Python backend: TextMatchOp evaluation
- Engine integration: text_match in query() and search() with filter
"""

import tempfile

import pytest

from milvus_lite.search.filter.parser import parse_expr
from milvus_lite.search.filter.ast import TextMatchOp, FieldRef, StringLit
from milvus_lite.search.filter.semantic import compile_expr
from milvus_lite.search.filter.eval.python_backend import _eval_row
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestTextMatchParser:
    def test_basic_parse(self):
        ast = parse_expr("text_match(title, 'hello world')")
        assert isinstance(ast, TextMatchOp)
        assert isinstance(ast.field, FieldRef)
        assert ast.field.name == "title"
        assert isinstance(ast.query, StringLit)
        assert ast.query.value == "hello world"

    def test_single_token(self):
        ast = parse_expr("text_match(name, 'python')")
        assert isinstance(ast, TextMatchOp)
        assert ast.query.value == "python"

    def test_double_quoted_query(self):
        ast = parse_expr('text_match(text, "machine learning")')
        assert isinstance(ast, TextMatchOp)
        assert ast.query.value == "machine learning"

    def test_combined_with_and(self):
        from milvus_lite.search.filter.ast import And
        ast = parse_expr("text_match(text, 'hello') and id > 5")
        assert isinstance(ast, And)
        assert isinstance(ast.operands[0], TextMatchOp)

    def test_combined_with_or(self):
        from milvus_lite.search.filter.ast import Or
        ast = parse_expr("text_match(text, 'hello') or text_match(text, 'world')")
        assert isinstance(ast, Or)
        assert isinstance(ast.operands[0], TextMatchOp)
        assert isinstance(ast.operands[1], TextMatchOp)

    def test_unknown_function_error(self):
        from milvus_lite.search.filter.exceptions import FilterParseError
        with pytest.raises(FilterParseError, match="unknown function"):
            parse_expr("unknown_func(x, 'y')")


# ---------------------------------------------------------------------------
# Python backend evaluation
# ---------------------------------------------------------------------------

class TestTextMatchEval:
    def test_single_token_match(self):
        ast = parse_expr("text_match(text, 'hello')")
        row = {"text": "hello world foo"}
        assert _eval_row(ast, row) is True

    def test_single_token_no_match(self):
        ast = parse_expr("text_match(text, 'missing')")
        row = {"text": "hello world"}
        assert _eval_row(ast, row) is False

    def test_multi_token_or_logic(self):
        """Multiple query tokens use OR logic."""
        ast = parse_expr("text_match(text, 'hello missing')")
        row = {"text": "hello world"}
        # "hello" matches, "missing" doesn't → OR → True
        assert _eval_row(ast, row) is True

    def test_case_insensitive(self):
        ast = parse_expr("text_match(text, 'HELLO')")
        row = {"text": "Hello World"}
        assert _eval_row(ast, row) is True

    def test_null_field_returns_false(self):
        ast = parse_expr("text_match(text, 'hello')")
        row = {"text": None}
        assert _eval_row(ast, row) is False

    def test_missing_field_returns_false(self):
        ast = parse_expr("text_match(text, 'hello')")
        row = {}
        assert _eval_row(ast, row) is False

    def test_empty_query_returns_false(self):
        ast = parse_expr("text_match(text, '')")
        row = {"text": "hello world"}
        assert _eval_row(ast, row) is False

    def test_punctuation_ignored(self):
        ast = parse_expr("text_match(text, 'hello')")
        row = {"text": "Hello, World!"}
        assert _eval_row(ast, row) is True


# ---------------------------------------------------------------------------
# Semantic compile
# ---------------------------------------------------------------------------

class TestTextMatchSemantic:
    def test_compiles(self):
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ])
        ast = parse_expr("text_match(text, 'hello')")
        compiled = compile_expr(ast, schema, source="text_match(text, 'hello')")
        assert compiled is not None
        assert isinstance(compiled.ast, TextMatchOp)
        # Should use python or hybrid backend (not arrow)
        assert compiled.backend in ("python", "hybrid")


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestTextMatchEngine:
    def _make_collection(self, tmpdir):
        from milvus_lite.schema.types import Function, FunctionType
        from milvus_lite.engine.collection import Collection

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(
                    name="text", dtype=DataType.VARCHAR,
                    enable_analyzer=True,
                    analyzer_params={"tokenizer": "standard"},
                    enable_match=True,
                ),
                FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
                FieldSchema(
                    name="sv", dtype=DataType.SPARSE_FLOAT_VECTOR,
                    is_function_output=True,
                ),
            ],
            functions=[
                Function(
                    name="bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["sv"],
                ),
            ],
        )
        col = Collection(name="test", data_dir=tmpdir, schema=schema)
        col.insert([
            {"id": 1, "text": "python programming language", "vec": [1, 0, 0, 0]},
            {"id": 2, "text": "java programming language", "vec": [0, 1, 0, 0]},
            {"id": 3, "text": "machine learning algorithms", "vec": [0, 0, 1, 0]},
            {"id": 4, "text": "deep learning neural networks", "vec": [0, 0, 0, 1]},
        ])
        return col

    def test_query_with_text_match(self):
        """text_match in query() filters correctly."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.query(
                expr="text_match(text, 'python')",
                output_fields=["text"],
            )
            assert len(results) == 1
            assert results[0]["text"] == "python programming language"

    def test_query_multi_token(self):
        """Multi-token text_match uses OR logic."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.query(
                expr="text_match(text, 'python java')",
                output_fields=["text"],
            )
            ids = {r["id"] for r in results}
            assert ids == {1, 2}

    def test_text_match_combined_with_scalar(self):
        """text_match combined with scalar filter."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.query(
                expr="text_match(text, 'learning') and id >= 3",
                output_fields=["id"],
            )
            ids = {r["id"] for r in results}
            assert ids == {3, 4}

    def test_text_match_with_dense_search(self):
        """text_match as filter in dense vector search."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=[[1, 0, 0, 0]],
                top_k=10,
                metric_type="COSINE",
                expr="text_match(text, 'programming')",
            )
            # Only docs 1,2 have "programming"
            hit_ids = [h["id"] for h in results[0]]
            assert set(hit_ids) == {1, 2}

    def test_text_match_with_bm25_search(self):
        """text_match as filter in BM25 search."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            results = col.search(
                query_vectors=["learning"],
                top_k=10,
                metric_type="BM25",
                anns_field="sv",
                expr="text_match(text, 'deep')",
            )
            # Only doc 4 has both "learning" (BM25 hit) and "deep" (filter)
            assert len(results[0]) == 1
            assert results[0][0]["id"] == 4
