"""Phase 11.5 — SparseInvertedIndex + BM25 search tests.

Covers:
- SparseInvertedIndex build/search correctness
- BM25 scoring formula verification
- save/load round-trip
- Engine end-to-end: insert text → BM25 search → correct results
- Search with scalar filter expression
- Search after flush
"""

import math
import os
import tempfile

import numpy as np
import pytest

from milvus_lite.analyzer.hash import term_to_id
from milvus_lite.analyzer.sparse import compute_tf
from milvus_lite.index.sparse_inverted import SparseInvertedIndex


# ---------------------------------------------------------------------------
# SparseInvertedIndex unit tests
# ---------------------------------------------------------------------------

class TestSparseInvertedIndex:
    def test_build_and_search_basic(self):
        """Basic BM25: query term present in some docs."""
        # 3 docs with term hashes
        t_hello = 100
        t_world = 200
        t_foo = 300
        sparse_vectors = [
            {t_hello: 2.0, t_world: 1.0},   # doc 0: "hello hello world"
            {t_foo: 1.0},                     # doc 1: "foo"
            {t_hello: 1.0, t_foo: 1.0},       # doc 2: "hello foo"
        ]
        idx = SparseInvertedIndex(k1=1.5, b=0.75)
        idx.build(sparse_vectors)

        assert idx.doc_count == 3

        # Search for "hello"
        query = [{t_hello: 1.0}]
        ids, dists = idx.search(query, top_k=3)

        assert ids.shape == (1, 3)
        assert dists.shape == (1, 3)

        # Doc 0 has tf=2 for hello, doc 2 has tf=1
        # Doc 0 should rank higher (more occurrences of query term)
        valid_ids = [int(ids[0, j]) for j in range(3) if ids[0, j] >= 0]
        assert 0 in valid_ids  # doc 0 has "hello"
        assert 2 in valid_ids  # doc 2 has "hello"
        assert 1 not in valid_ids  # doc 1 doesn't have "hello"

        # Distance is negative BM25 score (smaller = more similar)
        assert dists[0, 0] < dists[0, 1]  # best match first

    def test_empty_index(self):
        idx = SparseInvertedIndex()
        idx.build([])
        ids, dists = idx.search([{1: 1.0}], top_k=5)
        assert ids.shape == (1, 5)
        assert all(ids[0, j] == -1 for j in range(5))

    def test_empty_query(self):
        idx = SparseInvertedIndex()
        idx.build([{1: 1.0}])
        ids, dists = idx.search([{}], top_k=5)
        assert all(ids[0, j] == -1 for j in range(5))

    def test_valid_mask(self):
        """valid_mask excludes documents during build."""
        sparse_vectors = [
            {1: 1.0},  # doc 0
            {1: 1.0},  # doc 1 (will be masked)
            {1: 1.0},  # doc 2
        ]
        mask = np.array([True, False, True])
        idx = SparseInvertedIndex()
        idx.build(sparse_vectors, valid_mask=mask)
        assert idx.doc_count == 2

        ids, dists = idx.search([{1: 1.0}], top_k=5)
        valid_results = [int(ids[0, j]) for j in range(5) if ids[0, j] >= 0]
        assert 1 not in valid_results

    def test_bm25_score_manual(self):
        """Manually verify BM25 formula for a simple case."""
        t = 42
        # Single doc, single term with tf=1
        sparse_vectors = [{t: 1.0}]
        idx = SparseInvertedIndex(k1=1.2, b=0.75)
        idx.build(sparse_vectors)

        # Manual BM25 calculation:
        # N=1, df=1, avgdl=1, dl=1, tf=1
        # IDF = log((1 - 1 + 0.5)/(1 + 0.5) + 1) = log(0.5/1.5 + 1) = log(1.333)
        # tf_norm = 1*(1.2+1)/(1+1.2*(1-0.75+0.75*1/1)) = 2.2/(1+1.2) = 1.0
        # score = IDF * tf_norm
        expected_idf = math.log((1 - 1 + 0.5) / (1 + 0.5) + 1.0)
        expected_tf_norm = (1.0 * 2.2) / (1.0 + 1.2)
        expected_score = expected_idf * expected_tf_norm

        ids, dists = idx.search([{t: 1.0}], top_k=1)
        assert ids[0, 0] == 0
        actual_score = -float(dists[0, 0])
        assert abs(actual_score - expected_score) < 1e-5

    def test_multi_term_query(self):
        """Query with multiple terms accumulates scores."""
        t1, t2, t3 = 10, 20, 30
        sparse_vectors = [
            {t1: 1.0, t2: 1.0},  # doc 0: matches both query terms
            {t1: 1.0},            # doc 1: matches one query term
            {t3: 1.0},            # doc 2: matches no query terms
        ]
        idx = SparseInvertedIndex()
        idx.build(sparse_vectors)

        # Query for both t1 and t2
        ids, dists = idx.search([{t1: 1.0, t2: 1.0}], top_k=3)
        valid = [(int(ids[0, j]), float(dists[0, j])) for j in range(3) if ids[0, j] >= 0]

        # Doc 0 should rank first (matches both terms)
        assert valid[0][0] == 0
        # Doc 1 should rank second (matches one term)
        assert valid[1][0] == 1

    def test_multiple_queries(self):
        """Search with multiple queries at once."""
        t1, t2 = 10, 20
        sparse_vectors = [{t1: 1.0}, {t2: 1.0}]
        idx = SparseInvertedIndex()
        idx.build(sparse_vectors)

        queries = [{t1: 1.0}, {t2: 1.0}]
        ids, dists = idx.search(queries, top_k=2)
        assert ids.shape == (2, 2)
        # First query should find doc 0
        assert ids[0, 0] == 0
        # Second query should find doc 1
        assert ids[1, 0] == 1

    def test_save_load(self):
        sparse_vectors = [{1: 2.0, 2: 1.0}, {2: 3.0}, {1: 1.0}]
        idx = SparseInvertedIndex(k1=1.8, b=0.6)
        idx.build(sparse_vectors)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            loaded = SparseInvertedIndex.load(path)
            assert loaded.doc_count == 3
            assert loaded.k1 == 1.8
            assert loaded.b == 0.6

            # Search results should match
            q = [{1: 1.0}]
            ids1, d1 = idx.search(q, top_k=3)
            ids2, d2 = loaded.search(q, top_k=3)
            np.testing.assert_array_equal(ids1, ids2)
            np.testing.assert_allclose(d1, d2, atol=1e-5)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Engine end-to-end BM25 search
# ---------------------------------------------------------------------------

class TestBM25EndToEnd:
    def _make_collection(self, tmpdir):
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

    def test_basic_bm25_search(self):
        """Insert documents, search by text, verify relevance ordering."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([
                self._record(1, "machine learning algorithms"),
                self._record(2, "deep learning neural networks"),
                self._record(3, "database systems and storage"),
                self._record(4, "machine learning and deep learning"),
            ])

            # Search for "machine learning" — should rank docs 1,4 above 2,3
            query_tf = compute_tf([term_to_id("machine"), term_to_id("learning")])
            results = col.search(
                query_vectors=[query_tf],
                top_k=4,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            assert len(results) == 1
            hits = results[0]
            assert len(hits) > 0

            hit_ids = [h["id"] for h in hits]
            # Doc 1 and 4 contain "machine" and "learning"
            assert hit_ids[0] in (1, 4)

    def test_text_query_search(self):
        """Search using text string directly (auto-tokenized)."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([
                self._record(1, "python programming language"),
                self._record(2, "java programming language"),
                self._record(3, "machine learning algorithms"),
            ])

            results = col.search(
                query_vectors=["python"],
                top_k=3,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            assert len(results) == 1
            hits = results[0]
            # Only doc 1 contains "python"
            assert len(hits) >= 1
            assert hits[0]["id"] == 1

    def test_no_match(self):
        """Query with terms not in any document returns empty."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([self._record(1, "hello world")])

            query_tf = compute_tf([term_to_id("nonexistent")])
            results = col.search(
                query_vectors=[query_tf],
                top_k=5,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            assert results == [[]]

    def test_search_after_flush(self):
        """BM25 search works on flushed segments."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([
                self._record(1, "hello world"),
                self._record(2, "hello there"),
            ])
            col._trigger_flush()

            results = col.search(
                query_vectors=["hello"],
                top_k=5,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            assert len(results[0]) == 2

    def test_search_with_filter(self):
        """BM25 search respects scalar filter expression."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([
                self._record(1, "hello world"),
                self._record(2, "hello there"),
                self._record(3, "hello again"),
            ])

            results = col.search(
                query_vectors=["hello"],
                top_k=5,
                metric_type="BM25",
                anns_field="sparse_emb",
                expr="id >= 2",
            )
            hit_ids = [h["id"] for h in results[0]]
            assert 1 not in hit_ids
            assert 2 in hit_ids
            assert 3 in hit_ids

    def test_search_output_fields(self):
        """output_fields controls which fields appear in entity."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([self._record(1, "hello world")])

            results = col.search(
                query_vectors=["hello"],
                top_k=1,
                metric_type="BM25",
                anns_field="sparse_emb",
                output_fields=["text"],
            )
            hit = results[0][0]
            assert "text" in hit["entity"]
            assert hit["entity"]["text"] == "hello world"

    def test_upsert_dedup(self):
        """BM25 search respects upsert dedup (latest seq wins)."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([self._record(1, "old text about cats")])
            col.insert([self._record(1, "new text about dogs")])

            # Search for "dogs" should find the updated doc
            results = col.search(
                query_vectors=["dogs"],
                top_k=5,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            assert len(results[0]) == 1
            assert results[0][0]["id"] == 1

            # Search for "cats" should NOT find it (overwritten)
            results = col.search(
                query_vectors=["cats"],
                top_k=5,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            assert results == [[]]

    def test_delete_excluded(self):
        """Deleted documents are excluded from BM25 search."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([
                self._record(1, "hello world"),
                self._record(2, "hello there"),
            ])
            col.delete([1])

            results = col.search(
                query_vectors=["hello"],
                top_k=5,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            hit_ids = [h["id"] for h in results[0]]
            assert 1 not in hit_ids
            assert 2 in hit_ids

    def test_multiple_queries(self):
        """Multiple query vectors in a single search call."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([
                self._record(1, "python programming"),
                self._record(2, "java programming"),
            ])

            results = col.search(
                query_vectors=["python", "java"],
                top_k=2,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            assert len(results) == 2
            assert results[0][0]["id"] == 1  # "python" → doc 1
            assert results[1][0]["id"] == 2  # "java" → doc 2

    def test_distance_is_negative_score(self):
        """BM25 distance is negative score (smaller = more relevant)."""
        with tempfile.TemporaryDirectory() as d:
            col = self._make_collection(d)
            col.insert([self._record(1, "hello hello hello")])

            results = col.search(
                query_vectors=["hello"],
                top_k=1,
                metric_type="BM25",
                anns_field="sparse_emb",
            )
            dist = results[0][0]["distance"]
            assert dist < 0  # negative BM25 score
