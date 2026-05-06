"""Tests for search/executor.py — top-k correctness."""

import numpy as np
import pyarrow as pa
import pytest

from milvus_lite.schema.arrow_builder import build_delta_schema
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.search.distance import METRIC_COSINE, METRIC_IP, METRIC_L2
from milvus_lite.search.executor import execute_search
from milvus_lite.storage.delta_index import DeltaIndex


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=3),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


def _candidates_4_records():
    pks = ["a", "b", "c", "d"]
    seqs = np.array([1, 2, 3, 4], dtype=np.uint64)
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    records = [
        {"id": "a", "vec": [1.0, 0.0, 0.0], "title": "ta"},
        {"id": "b", "vec": [0.0, 1.0, 0.0], "title": "tb"},
        {"id": "c", "vec": [0.0, 0.0, 1.0], "title": "tc"},
        {"id": "d", "vec": [1.0, 1.0, 0.0], "title": "td"},
    ]
    return pks, seqs, vectors, records


def _empty_index() -> DeltaIndex:
    return DeltaIndex("id")


# ---------------------------------------------------------------------------
# Empty cases
# ---------------------------------------------------------------------------

def test_search_empty_candidates():
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=[],
        all_seqs=np.zeros(0, dtype=np.uint64),
        all_vectors=np.zeros((0, 3), dtype=np.float32),
        all_records=[],
        delta_index=_empty_index(),
        top_k=10,
        metric_type=METRIC_COSINE,
        pk_field="id",
        vector_field="vec",
    )
    assert results == [[]]


def test_search_single_query_basic():
    pks, seqs, vectors, records = _candidates_4_records()
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=pks,
        all_seqs=seqs,
        all_vectors=vectors,
        all_records=records,
        delta_index=_empty_index(),
        top_k=10,
        metric_type=METRIC_COSINE,
        pk_field="id",
        vector_field="vec",
    )
    assert len(results) == 1
    [hits] = results
    assert len(hits) == 4
    # 'a' should be the closest (identical vector → distance 0)
    assert hits[0]["id"] == "a"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_search_top_k_smaller_than_n():
    pks, seqs, vectors, records = _candidates_4_records()
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=pks, all_seqs=seqs, all_vectors=vectors, all_records=records,
        delta_index=_empty_index(),
        top_k=2,
        metric_type=METRIC_COSINE,
        pk_field="id", vector_field="vec",
    )
    [hits] = results
    assert len(hits) == 2
    assert hits[0]["id"] == "a"  # closest
    # Distances must be ascending
    assert hits[0]["distance"] <= hits[1]["distance"]


def test_search_result_dict_shape():
    pks, seqs, vectors, records = _candidates_4_records()
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=pks, all_seqs=seqs, all_vectors=vectors, all_records=records,
        delta_index=_empty_index(),
        top_k=1,
        metric_type=METRIC_COSINE,
        pk_field="id", vector_field="vec",
    )
    [hits] = results
    assert len(hits) == 1
    hit = hits[0]
    assert set(hit.keys()) == {"id", "distance", "entity"}
    assert hit["id"] == "a"
    assert hit["entity"] == {"title": "ta"}  # no id, no vec
    assert isinstance(hit["distance"], float)


def test_search_metric_l2():
    pks, seqs, vectors, records = _candidates_4_records()
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=pks, all_seqs=seqs, all_vectors=vectors, all_records=records,
        delta_index=_empty_index(),
        top_k=4,
        metric_type=METRIC_L2,
        pk_field="id", vector_field="vec",
    )
    [hits] = results
    assert hits[0]["id"] == "a"  # distance 0
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_search_metric_ip():
    pks, seqs, vectors, records = _candidates_4_records()
    q = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=pks, all_seqs=seqs, all_vectors=vectors, all_records=records,
        delta_index=_empty_index(),
        top_k=2,
        metric_type=METRIC_IP,
        pk_field="id", vector_field="vec",
    )
    [hits] = results
    # IP: a→2, b→0, c→0, d→2.  Top-2 should be {a, d}.
    top_ids = {h["id"] for h in hits}
    assert top_ids == {"a", "d"}


# ---------------------------------------------------------------------------
# Multiple queries (nq > 1)
# ---------------------------------------------------------------------------

def test_search_multiple_queries():
    pks, seqs, vectors, records = _candidates_4_records()
    q = np.array([
        [1.0, 0.0, 0.0],   # closest to 'a'
        [0.0, 1.0, 0.0],   # closest to 'b'
        [0.0, 0.0, 1.0],   # closest to 'c'
    ], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=pks, all_seqs=seqs, all_vectors=vectors, all_records=records,
        delta_index=_empty_index(),
        top_k=1,
        metric_type=METRIC_COSINE,
        pk_field="id", vector_field="vec",
    )
    assert len(results) == 3
    assert results[0][0]["id"] == "a"
    assert results[1][0]["id"] == "b"
    assert results[2][0]["id"] == "c"


# ---------------------------------------------------------------------------
# Dedup + tombstone via bitmap
# ---------------------------------------------------------------------------

def test_search_dedup_keeps_max_seq():
    """Two rows with the same pk — only the larger seq should appear."""
    pks = ["x", "x"]
    seqs = np.array([1, 5], dtype=np.uint64)
    vectors = np.array([
        [1.0, 0.0, 0.0],   # old
        [0.0, 1.0, 0.0],   # new
    ], dtype=np.float32)
    records = [
        {"id": "x", "vec": [1.0, 0.0, 0.0], "title": "old"},
        {"id": "x", "vec": [0.0, 1.0, 0.0], "title": "new"},
    ]
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=pks, all_seqs=seqs, all_vectors=vectors, all_records=records,
        delta_index=_empty_index(),
        top_k=10,
        metric_type=METRIC_COSINE,
        pk_field="id", vector_field="vec",
    )
    [hits] = results
    assert len(hits) == 1
    # The newer (seq=5) record wins, even though the older one is closer
    # to the query.
    assert hits[0]["entity"]["title"] == "new"


def test_search_tombstone_filters(schema):
    pks, seqs, vectors, records = _candidates_4_records()
    idx = DeltaIndex("id")
    idx.add_batch(pa.RecordBatch.from_pydict(
        {"id": ["a"], "_seq": [10]}, schema=build_delta_schema(schema),
    ))
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = execute_search(
        query_vectors=q,
        all_pks=pks, all_seqs=seqs, all_vectors=vectors, all_records=records,
        delta_index=idx,
        top_k=10,
        metric_type=METRIC_COSINE,
        pk_field="id", vector_field="vec",
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert "a" not in ids
    assert ids == {"b", "c", "d"}


# ---------------------------------------------------------------------------
# Brute-force comparison — the gold standard
# ---------------------------------------------------------------------------

def test_search_matches_numpy_brute_force():
    """Random vectors, compare top-k against direct numpy computation."""
    rng = np.random.default_rng(42)
    n = 100
    dim = 8
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    pks = [f"doc_{i}" for i in range(n)]
    seqs = np.arange(n, dtype=np.uint64) + 1
    records = [
        {"id": pks[i], "vec": vectors[i].tolist(), "title": f"t{i}"}
        for i in range(n)
    ]
    q = rng.standard_normal((1, dim)).astype(np.float32)

    results = execute_search(
        query_vectors=q,
        all_pks=pks, all_seqs=seqs, all_vectors=vectors, all_records=records,
        delta_index=_empty_index(),
        top_k=5,
        metric_type=METRIC_L2,
        pk_field="id", vector_field="vec",
    )
    [hits] = results

    # Direct numpy: compute all distances, take 5 smallest indices.
    dists = np.linalg.norm(vectors - q[0], axis=1)
    expected_top5 = np.argsort(dists)[:5]
    expected_ids = [pks[i] for i in expected_top5]

    actual_ids = [h["id"] for h in hits]
    assert actual_ids == expected_ids
