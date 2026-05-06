"""BruteForceIndex unit tests.

Validates the VectorIndex contract on the brute-force implementation:
- build / search / save / load round-trips
- valid_mask semantics (None vs explicit + cross-checks)
- top-k padding with -1 / +inf
- distance-convention parity across COSINE / L2 / IP
- shape validation
"""

import os

import numpy as np
import pytest

from milvus_lite.index.brute_force import BruteForceIndex
from milvus_lite.index.protocol import VectorIndex
from milvus_lite.search.distance import compute_distances


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vectors() -> np.ndarray:
    """8 vectors in a 4-dim space, deterministic."""
    np.random.seed(42)
    return np.random.randn(8, 4).astype(np.float32)


@pytest.fixture
def queries() -> np.ndarray:
    np.random.seed(7)
    return np.random.randn(3, 4).astype(np.float32)


# ---------------------------------------------------------------------------
# Construction + protocol contract
# ---------------------------------------------------------------------------

def test_brute_force_is_a_vector_index():
    assert issubclass(BruteForceIndex, VectorIndex) or hasattr(BruteForceIndex, "search")


def test_build_basic(vectors):
    idx = BruteForceIndex.build(vectors, "L2")
    assert idx.metric == "L2"
    assert idx.num_vectors == 8
    assert idx.dim == 4
    assert idx.index_type == "BRUTE_FORCE"


def test_build_casts_to_float32(vectors):
    f64 = vectors.astype(np.float64)
    idx = BruteForceIndex.build(f64, "L2")
    # Internal vectors should be float32 (or at least search must work).
    out_ids, out_dists = idx.search(vectors[:1].astype(np.float32), top_k=3)
    assert out_dists.dtype == np.float32


def test_build_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        BruteForceIndex.build(np.zeros((8,), dtype=np.float32), "L2")


# ---------------------------------------------------------------------------
# Search — basic shapes and content
# ---------------------------------------------------------------------------

def test_search_returns_correct_shapes(vectors, queries):
    idx = BruteForceIndex.build(vectors, "L2")
    ids, dists = idx.search(queries, top_k=5)
    assert ids.shape == (3, 5)
    assert dists.shape == (3, 5)
    assert ids.dtype == np.int64
    assert dists.dtype == np.float32


def test_search_top_1_finds_exact_self_match(vectors):
    """Querying with one of the candidate vectors must return that
    vector at top-1 with distance close to 0 for L2."""
    idx = BruteForceIndex.build(vectors, "L2")
    q = vectors[3:4].copy()
    ids, dists = idx.search(q, top_k=1)
    assert ids[0, 0] == 3
    assert dists[0, 0] < 1e-5


def test_search_distances_match_compute_distances(vectors, queries):
    """The distances returned by .search should equal what
    compute_distances would compute on the same (query, vector) pair."""
    idx = BruteForceIndex.build(vectors, "COSINE")
    ids, dists = idx.search(queries, top_k=8)
    for q in range(queries.shape[0]):
        for j in range(8):
            local_id = int(ids[q, j])
            ref = compute_distances(queries[q], vectors[local_id:local_id + 1], "COSINE")[0]
            assert abs(float(dists[q, j]) - float(ref)) < 1e-5


def test_search_results_sorted_ascending(vectors, queries):
    idx = BruteForceIndex.build(vectors, "L2")
    _, dists = idx.search(queries, top_k=8)
    for q in range(queries.shape[0]):
        row = dists[q]
        # Strictly non-decreasing.
        assert all(row[i] <= row[i + 1] for i in range(len(row) - 1))


def test_search_1d_query_is_treated_as_single(vectors):
    idx = BruteForceIndex.build(vectors, "L2")
    q = vectors[0]  # 1-D
    ids, dists = idx.search(q, top_k=3)
    assert ids.shape == (1, 3)
    assert ids[0, 0] == 0


# ---------------------------------------------------------------------------
# valid_mask semantics
# ---------------------------------------------------------------------------

def test_search_valid_mask_skips_excluded_rows(vectors, queries):
    idx = BruteForceIndex.build(vectors, "L2")

    # Mask out rows 0, 2, 4 — only 1, 3, 5, 6, 7 are valid.
    mask = np.ones(8, dtype=bool)
    mask[[0, 2, 4]] = False

    ids, _ = idx.search(queries, top_k=5, valid_mask=mask)
    valid_set = set(np.flatnonzero(mask).tolist())
    for q in range(queries.shape[0]):
        for j in range(5):
            lid = int(ids[q, j])
            if lid >= 0:
                assert lid in valid_set, f"masked-out id {lid} appeared in result"


def test_search_local_ids_use_original_indices(vectors):
    """When valid_mask removes some rows, the returned local_ids must
    be indices into the FULL original array, not into the post-mask
    compacted array. Otherwise the segment can't translate back to pk."""
    idx = BruteForceIndex.build(vectors, "L2")
    mask = np.zeros(8, dtype=bool)
    mask[[2, 5, 7]] = True

    # Query with vectors[5] — should match itself at top-1.
    q = vectors[5:6].copy()
    ids, dists = idx.search(q, top_k=1, valid_mask=mask)
    assert ids[0, 0] == 5
    assert dists[0, 0] < 1e-5


def test_search_all_masked_returns_padding(vectors, queries):
    idx = BruteForceIndex.build(vectors, "L2")
    mask = np.zeros(8, dtype=bool)
    ids, dists = idx.search(queries, top_k=4, valid_mask=mask)
    assert (ids == -1).all()
    assert np.isinf(dists).all()


def test_search_mask_wrong_length_raises(vectors, queries):
    idx = BruteForceIndex.build(vectors, "L2")
    bad_mask = np.zeros(7, dtype=bool)  # wrong length
    with pytest.raises(ValueError, match="valid_mask length"):
        idx.search(queries, top_k=3, valid_mask=bad_mask)


# ---------------------------------------------------------------------------
# top-k padding behavior
# ---------------------------------------------------------------------------

def test_search_top_k_larger_than_n_pads(vectors, queries):
    idx = BruteForceIndex.build(vectors, "L2")
    ids, dists = idx.search(queries, top_k=20)
    assert ids.shape == (3, 20)
    # First 8 are valid, the rest must be padding.
    for q in range(3):
        assert (ids[q, :8] >= 0).all()
        assert (ids[q, 8:] == -1).all()
        assert np.isinf(dists[q, 8:]).all()


def test_search_top_k_zero_returns_empty_padding(vectors, queries):
    idx = BruteForceIndex.build(vectors, "L2")
    ids, dists = idx.search(queries, top_k=0)
    assert ids.shape == (3, 0)
    assert dists.shape == (3, 0)


def test_search_empty_index_returns_padding():
    idx = BruteForceIndex.build(np.zeros((0, 4), dtype=np.float32), "L2")
    ids, dists = idx.search(np.ones((2, 4), dtype=np.float32), top_k=3)
    assert ids.shape == (2, 3)
    assert (ids == -1).all()
    assert np.isinf(dists).all()


# ---------------------------------------------------------------------------
# Metric coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["COSINE", "L2"])
def test_search_self_match_top_1(vectors, metric):
    """For COSINE and L2, querying with one of the candidate vectors must
    return that vector at top-1. (IP is unbounded — a longer vector
    pointing in the same direction can outscore a self-match, so IP
    self-match is not a valid invariant.)"""
    idx = BruteForceIndex.build(vectors, metric)
    for i in range(vectors.shape[0]):
        ids, _ = idx.search(vectors[i:i + 1], top_k=1)
        assert ids[0, 0] == i, f"metric={metric} row={i}"


def test_search_ip_returns_negative_dot_for_self(vectors):
    """For IP, the distance returned is -dot(q, v); querying with a
    candidate against itself yields -||v||² which is its own distance,
    even if that's not the global minimum."""
    idx = BruteForceIndex.build(vectors, "IP")
    # Search top-N so we always include the self entry.
    n = vectors.shape[0]
    for i in range(n):
        ids, dists = idx.search(vectors[i:i + 1], top_k=n)
        positions = {int(local_id): k for k, local_id in enumerate(ids[0]) if local_id >= 0}
        assert i in positions
        expected = -float(np.dot(vectors[i], vectors[i]))
        assert abs(float(dists[0, positions[i]]) - expected) < 1e-4


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path, vectors):
    idx = BruteForceIndex.build(vectors, "L2")
    path = str(tmp_path / "seg.brute_force.idx")
    idx.save(path)

    # numpy may store with a .npy extension
    assert os.path.exists(path) or os.path.exists(path + ".npy")

    loaded = BruteForceIndex.load(path, metric="L2", dim=4)
    assert loaded.num_vectors == 8
    assert loaded.dim == 4
    assert loaded.metric == "L2"

    # Same search results before/after.
    q = vectors[2:5]
    a_ids, a_dists = idx.search(q, top_k=3)
    b_ids, b_dists = loaded.search(q, top_k=3)
    assert (a_ids == b_ids).all()
    assert np.allclose(a_dists, b_dists, atol=1e-5)


def test_load_dim_mismatch_raises(tmp_path, vectors):
    idx = BruteForceIndex.build(vectors, "L2")
    path = str(tmp_path / "seg.brute_force.idx")
    idx.save(path)
    with pytest.raises(ValueError, match="dim"):
        BruteForceIndex.load(path, metric="L2", dim=8)
