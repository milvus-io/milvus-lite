"""Phase 11.3 — Sparse vector encoding/decoding and BM25 auto-generation.

Covers:
- sparse_to_bytes / bytes_to_sparse round-trip
- compute_tf term frequency computation
- Engine insert with BM25 function auto-generating sparse vectors
- WAL + flush with sparse vector columns
"""

import os
import struct
import tempfile

import pyarrow as pa
import pytest

from milvus_lite.analyzer.sparse import bytes_to_sparse, compute_tf, sparse_to_bytes
from milvus_lite.analyzer.hash import term_to_id


# ---------------------------------------------------------------------------
# sparse_to_bytes / bytes_to_sparse
# ---------------------------------------------------------------------------

class TestSparseCodec:
    def test_empty(self):
        assert sparse_to_bytes({}) == b""
        assert bytes_to_sparse(b"") == {}

    def test_round_trip_single(self):
        sv = {42: 1.5}
        b = sparse_to_bytes(sv)
        assert len(b) == 8  # 1 pair * 8 bytes
        assert bytes_to_sparse(b) == sv

    def test_round_trip_multiple(self):
        sv = {0: 1.0, 5: 0.5, 100: 2.0, 999: 0.1}
        b = sparse_to_bytes(sv)
        assert len(b) == 32  # 4 pairs * 8 bytes
        result = bytes_to_sparse(b)
        assert len(result) == 4
        for k, v in sv.items():
            assert abs(result[k] - v) < 1e-6

    def test_sorted_by_index(self):
        sv = {100: 1.0, 0: 2.0, 50: 3.0}
        b = sparse_to_bytes(sv)
        # First pair should be index 0
        idx0 = struct.unpack_from("<I", b, 0)[0]
        idx1 = struct.unpack_from("<I", b, 8)[0]
        idx2 = struct.unpack_from("<I", b, 16)[0]
        assert idx0 < idx1 < idx2

    def test_large_index(self):
        sv = {2**32 - 1: 1.0}
        b = sparse_to_bytes(sv)
        result = bytes_to_sparse(b)
        assert 2**32 - 1 in result

    def test_zero_value(self):
        sv = {5: 0.0}
        result = bytes_to_sparse(sparse_to_bytes(sv))
        assert result[5] == 0.0

    def test_negative_value(self):
        sv = {5: -1.5}
        result = bytes_to_sparse(sparse_to_bytes(sv))
        assert abs(result[5] - (-1.5)) < 1e-6


# ---------------------------------------------------------------------------
# compute_tf
# ---------------------------------------------------------------------------

class TestComputeTF:
    def test_empty(self):
        assert compute_tf([]) == {}

    def test_single_term(self):
        result = compute_tf([42])
        assert result == {42: 1.0}

    def test_repeated_terms(self):
        result = compute_tf([1, 2, 1, 1, 2])
        assert result == {1: 3.0, 2: 2.0}

    def test_all_unique(self):
        result = compute_tf([10, 20, 30])
        assert result == {10: 1.0, 20: 1.0, 30: 1.0}

    def test_values_are_float(self):
        result = compute_tf([1, 1])
        assert isinstance(result[1], float)


# ---------------------------------------------------------------------------
# Engine integration: BM25 function auto-generation
# ---------------------------------------------------------------------------

class TestBM25AutoGeneration:
    def _make_collection(self, tmpdir):
        """Create a collection with FLOAT_VECTOR + BM25 SPARSE_FLOAT_VECTOR.

        Both vector types coexist — this mirrors real use cases where
        users have dense embeddings for semantic search and BM25 for
        keyword search.
        """
        from milvus_lite.schema.types import (
            CollectionSchema, DataType, FieldSchema,
            Function, FunctionType,
        )
        from milvus_lite.engine.collection import Collection

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(
                    name="text", dtype=DataType.VARCHAR,
                    enable_analyzer=True,
                    analyzer_params={"tokenizer": "standard"},
                ),
                FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
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
        return Collection(name="test_bm25", data_dir=tmpdir, schema=schema)

    def _record(self, id, text):
        return {"id": id, "text": text, "vec": [1.0, 0.0, 0.0, 0.0]}

    def test_insert_auto_generates_sparse(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            pks = col.insert([
                self._record(1, "hello world"),
                self._record(2, "machine learning algorithms"),
            ])
            assert pks == [1, 2]

    def test_sparse_vector_content(self):
        """Verify the auto-generated sparse vector has correct term hashes."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            records = [self._record(1, "hello world")]

            # Use ingestion chain (replaces removed _apply_bm25_functions)
            from milvus_lite.function.dataframe import DataFrame
            col._ingestion_chain.execute(DataFrame.from_records(records))
            sv = records[0]["sparse_emb"]
            assert isinstance(sv, dict)
            assert term_to_id("hello") in sv
            assert term_to_id("world") in sv
            assert sv[term_to_id("hello")] == 1.0
            assert sv[term_to_id("world")] == 1.0

    def test_repeated_terms_tf(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            records = [self._record(1, "test test test")]
            from milvus_lite.function.dataframe import DataFrame
            col._ingestion_chain.execute(DataFrame.from_records(records))
            sv = records[0]["sparse_emb"]
            assert sv[term_to_id("test")] == 3.0

    def test_empty_text(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            records = [self._record(1, "")]
            from milvus_lite.function.dataframe import DataFrame
            col._ingestion_chain.execute(DataFrame.from_records(records))
            assert records[0]["sparse_emb"] == {}

    def test_none_text_nullable(self):
        from milvus_lite.schema.types import (
            CollectionSchema, DataType, FieldSchema,
            Function, FunctionType,
        )
        from milvus_lite.engine.collection import Collection

        with tempfile.TemporaryDirectory() as d:
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(
                        name="text", dtype=DataType.VARCHAR,
                        enable_analyzer=True, nullable=True,
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
            col = Collection(name="test", data_dir=d, schema=schema)
            records = [{"id": 1, "text": None, "vec": [1.0, 0.0, 0.0, 0.0]}]
            from milvus_lite.function.dataframe import DataFrame
            col._ingestion_chain.execute(DataFrame.from_records(records))
            assert records[0]["sv"] == {}

    def test_flush_preserves_sparse(self):
        """Insert + flush → sparse vector in Parquet."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([
                self._record(1, "hello world"),
                self._record(2, "foo bar baz"),
            ])
            col._trigger_flush()

            # Read the flushed parquet file
            data_dir = os.path.join(d, "partitions", "_default", "data")
            parquet_files = [f for f in os.listdir(data_dir)
                            if f.startswith("data_") and f.endswith(".parquet")]
            assert len(parquet_files) == 1

            table = pa.parquet.read_table(os.path.join(data_dir, parquet_files[0]))
            assert "sparse_emb" in table.column_names

            # Verify sparse_emb column type is binary
            col_data = table.column("sparse_emb")
            assert col_data.type == pa.binary()

            # Verify content decodes correctly
            for val in col_data:
                b = val.as_py()
                sv = bytes_to_sparse(b)
                assert isinstance(sv, dict)
                assert len(sv) > 0  # "hello world" → 2 terms, "foo bar baz" → 3

    def test_get_returns_sparse(self):
        """get() returns the sparse vector field."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([self._record(1, "hello world")])

            results = col.get([1])
            assert len(results) == 1
            sv_raw = results[0]["sparse_emb"]
            assert sv_raw is not None

    def test_multiple_inserts(self):
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([self._record(1, "first document")])
            col.insert([self._record(2, "second document")])
            col.insert([self._record(3, "third one")])
            assert col.num_entities == 3
