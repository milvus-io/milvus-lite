"""FaissHnswIndex unit tests.

These tests are skipped automatically when faiss-cpu is not installed,
so they don't break CI on bare environments. The differential test in
test_index_differential.py is the architectural safety net — if FAISS
breaks any of the MilvusLite invariants (metric symbol, padding,
local_id space) the differential test catches it.
"""

import os

import numpy as np
import pytest

from milvus_lite.index.factory import is_faiss_available

pytestmark = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss-cpu is not installed"
)

# These imports are inside the skipif because the module imports faiss
# at top level — we don't want to error at collection time when faiss
# is missing.
if is_faiss_available():
    from milvus_lite.index.faiss_hnsw import FaissHnswIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vectors() -> np.ndarray:
    np.random.seed(0)
    return np.random.randn(100, 8).astype(np.float32)


@pytest.fixture
def queries() -> np.ndarray:
    np.random.seed(1)
    return np.random.randn(5, 8).astype(np.float32)


HNSW_PARAMS = {"M": 16, "efConstruction": 200}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_build_basic(vectors):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    assert idx.metric == "L2"
    assert idx.num_vectors == 100
    assert idx.dim == 8
    assert idx.index_type == "HNSW"


@pytest.mark.parametrize("metric", ["L2", "IP", "COSINE"])
def test_build_each_metric(vectors, metric):
    idx = FaissHnswIndex.build(vectors, metric, HNSW_PARAMS)
    assert idx.metric == metric
    assert idx.num_vectors == 100


def test_build_default_params(vectors):
    idx = FaissHnswIndex.build(vectors, "L2")
    assert idx.num_vectors == 100


def test_build_unsupported_metric_raises(vectors):
    with pytest.raises(ValueError, match="metric"):
        FaissHnswIndex.build(vectors, "JACCARD", HNSW_PARAMS)


def test_build_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        FaissHnswIndex.build(np.zeros((8,), dtype=np.float32), "L2")


# ---------------------------------------------------------------------------
# Search — basic + shape + padding
# ---------------------------------------------------------------------------

def test_search_returns_correct_shapes(vectors, queries):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    ids, dists = idx.search(queries, top_k=5)
    assert ids.shape == (5, 5)
    assert dists.shape == (5, 5)
    assert ids.dtype == np.int64
    assert dists.dtype == np.float32


def test_search_returns_self_as_top1_l2(vectors):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    for i in range(0, 100, 10):
        ids, dists = idx.search(vectors[i:i + 1], top_k=1)
        assert ids[0, 0] == i, f"row {i} self-match failed"
        assert dists[0, 0] < 1e-3, f"row {i} self-distance too large"


def test_search_returns_self_as_top1_cosine(vectors):
    idx = FaissHnswIndex.build(vectors, "COSINE", HNSW_PARAMS)
    for i in range(0, 100, 10):
        ids, dists = idx.search(vectors[i:i + 1], top_k=1)
        assert ids[0, 0] == i
        assert dists[0, 0] < 1e-3


def test_search_results_sorted_ascending(vectors, queries):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    _, dists = idx.search(queries, top_k=10)
    for q in range(queries.shape[0]):
        for i in range(9):
            if dists[q, i + 1] != np.inf:
                assert dists[q, i] <= dists[q, i + 1] + 1e-6


def test_search_top_k_larger_than_n_pads(vectors, queries):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    ids, dists = idx.search(queries, top_k=200)
    # First (≤100) entries should be valid, the rest padded
    for q in range(queries.shape[0]):
        valid = ids[q] >= 0
        assert valid.sum() <= 100
        # Padding slots are -1 / inf
        assert (ids[q, valid.sum():] == -1).all()


def test_search_empty_index_returns_padding():
    empty = np.zeros((0, 4), dtype=np.float32)
    idx = FaissHnswIndex.build(empty, "L2", HNSW_PARAMS)
    ids, dists = idx.search(np.ones((2, 4), dtype=np.float32), top_k=3)
    assert ids.shape == (2, 3)
    assert (ids == -1).all()
    assert np.isinf(dists).all()


def test_search_top_k_zero_returns_empty(vectors, queries):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    ids, dists = idx.search(queries, top_k=0)
    assert ids.shape == (5, 0)
    assert dists.shape == (5, 0)


# ---------------------------------------------------------------------------
# valid_mask via IDSelectorBatch
# ---------------------------------------------------------------------------

def test_search_valid_mask_filters_correctly(vectors, queries):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    mask = np.zeros(100, dtype=bool)
    allowed = [0, 5, 10, 15, 20]
    mask[allowed] = True
    ids, _ = idx.search(queries, top_k=5, valid_mask=mask)
    for q in range(queries.shape[0]):
        for j in range(5):
            lid = int(ids[q, j])
            if lid >= 0:
                assert lid in allowed, (
                    f"masked-out id {lid} appeared in result for query {q}"
                )


def test_search_all_masked_returns_padding(vectors, queries):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    mask = np.zeros(100, dtype=bool)
    ids, dists = idx.search(queries, top_k=3, valid_mask=mask)
    assert (ids == -1).all()
    assert np.isinf(dists).all()


def test_search_valid_mask_local_ids_are_global(vectors):
    """When valid_mask removes some rows, returned local_ids must be
    indices into the full original array — same contract as
    BruteForceIndex."""
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    mask = np.zeros(100, dtype=bool)
    mask[[7, 33, 71]] = True
    # Self query for vector at index 33
    ids, dists = idx.search(vectors[33:34], top_k=1, valid_mask=mask)
    assert ids[0, 0] == 33
    assert dists[0, 0] < 1e-3


def test_search_valid_mask_wrong_length_raises(vectors, queries):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    bad_mask = np.zeros(50, dtype=bool)  # 100 expected
    with pytest.raises(ValueError, match="valid_mask length"):
        idx.search(queries, top_k=3, valid_mask=bad_mask)


# ---------------------------------------------------------------------------
# Search params (ef)
# ---------------------------------------------------------------------------

def test_search_params_ef_overrides_default(vectors, queries):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    # Just verify it doesn't crash and returns valid shape;
    # ef quality vs latency tradeoff is a benchmark concern.
    ids, _ = idx.search(queries, top_k=5, params={"ef": 32})
    assert ids.shape == (5, 5)
    ids, _ = idx.search(queries, top_k=5, params={"ef": 200})
    assert ids.shape == (5, 5)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path, vectors):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    path = str(tmp_path / "seg.hnsw.idx")
    idx.save(path)

    assert os.path.exists(path)

    loaded = FaissHnswIndex.load(path, metric="L2", dim=8)
    assert loaded.num_vectors == 100
    assert loaded.dim == 8
    assert loaded.metric == "L2"

    # Same query → same top-k ids and distances
    q = np.random.RandomState(42).randn(3, 8).astype(np.float32)
    a_ids, a_dists = idx.search(q, top_k=5)
    b_ids, b_dists = loaded.search(q, top_k=5)
    assert (a_ids == b_ids).all()
    assert np.allclose(a_dists, b_dists, atol=1e-5)


def test_save_load_cosine(tmp_path, vectors):
    idx = FaissHnswIndex.build(vectors, "COSINE", HNSW_PARAMS)
    path = str(tmp_path / "seg.hnsw.idx")
    idx.save(path)
    loaded = FaissHnswIndex.load(path, metric="COSINE", dim=8)
    assert loaded.metric == "COSINE"

    q = np.random.RandomState(42).randn(3, 8).astype(np.float32)
    a_ids, a_dists = idx.search(q, top_k=5)
    b_ids, b_dists = loaded.search(q, top_k=5)
    assert (a_ids == b_ids).all()
    assert np.allclose(a_dists, b_dists, atol=1e-5)


def test_load_dim_mismatch_raises(tmp_path, vectors):
    idx = FaissHnswIndex.build(vectors, "L2", HNSW_PARAMS)
    path = str(tmp_path / "seg.hnsw.idx")
    idx.save(path)
    with pytest.raises(ValueError, match="dim"):
        FaissHnswIndex.load(path, metric="L2", dim=16)
