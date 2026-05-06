"""Differential test: FaissIvfFlatIndex vs BruteForceIndex.

Same structure as test_index_differential.py (HNSW), adapted for
IVF_FLAT's nlist/nprobe parameters and slightly relaxed recall
thresholds (IVF is approximate with limited nprobe).

Skipped automatically when faiss-cpu is not installed.
"""

import numpy as np
import pytest

from milvus_lite.index.brute_force import BruteForceIndex
from milvus_lite.index.factory import is_faiss_available

pytestmark = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss-cpu is not installed"
)

if is_faiss_available():
    from milvus_lite.index.faiss_ivf_flat import FaissIvfFlatIndex


# Use high nprobe for recall tests to get near-exact results.
IVF_BUILD_PARAMS = {"nlist": 16}
IVF_SEARCH_PARAMS = {"nprobe": 16}  # nprobe == nlist → exhaustive


def _gen_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Recall@10 across all (dim, n, metric) combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
@pytest.mark.parametrize("dim,n", [(4, 200), (32, 2000), (128, 2000)])
def test_recall_at_10_vs_brute_force(metric, dim, n):
    vectors = _gen_vectors(n, dim, seed=42)
    queries = _gen_vectors(20, dim, seed=7)

    brute = BruteForceIndex.build(vectors, metric)
    ivf_idx = FaissIvfFlatIndex.build(vectors, metric, IVF_BUILD_PARAMS)

    brute_ids, _ = brute.search(queries, top_k=10)
    ivf_ids, _ = ivf_idx.search(queries, top_k=10, params=IVF_SEARCH_PARAMS)

    total_recall = 0.0
    for q in range(20):
        brute_set = {int(i) for i in brute_ids[q] if i >= 0}
        ivf_set = {int(i) for i in ivf_ids[q] if i >= 0}
        intersect = len(brute_set & ivf_set)
        denom = len(brute_set) or 1
        total_recall += intersect / denom

    avg_recall = total_recall / 20
    assert avg_recall >= 0.90, (
        f"recall@10 = {avg_recall:.3f} < 0.90 for metric={metric} dim={dim} n={n}"
    )


# ---------------------------------------------------------------------------
# Distance value parity for matching IDs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
def test_distance_value_parity(metric):
    """For IDs that appear in BOTH the IVF and brute-force top-k,
    the reported distance values must agree within 1e-3 relative error."""
    vectors = _gen_vectors(500, 16, seed=11)
    queries = _gen_vectors(15, 16, seed=22)

    brute = BruteForceIndex.build(vectors, metric)
    ivf_idx = FaissIvfFlatIndex.build(vectors, metric, IVF_BUILD_PARAMS)

    brute_ids, brute_dists = brute.search(queries, top_k=50)
    ivf_ids, ivf_dists = ivf_idx.search(queries, top_k=50, params=IVF_SEARCH_PARAMS)

    mismatches = []
    for q in range(15):
        brute_map = {
            int(brute_ids[q, j]): float(brute_dists[q, j])
            for j in range(50) if brute_ids[q, j] >= 0
        }
        ivf_map = {
            int(ivf_ids[q, j]): float(ivf_dists[q, j])
            for j in range(50) if ivf_ids[q, j] >= 0
        }
        common = set(brute_map.keys()) & set(ivf_map.keys())
        for pid in common:
            bd = brute_map[pid]
            fd = ivf_map[pid]
            tol = 1e-3 + 1e-3 * abs(bd)
            if abs(bd - fd) > tol:
                mismatches.append((q, pid, bd, fd))

    assert not mismatches, (
        f"distance mismatch for metric={metric}: {mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# valid_mask parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["COSINE", "L2"])
def test_valid_mask_parity(metric):
    """When given the same valid_mask, IVF and BruteForce should return
    the same top-k IDs and only consider rows where mask is True."""
    vectors = _gen_vectors(2000, 16, seed=33)
    queries = _gen_vectors(10, 16, seed=44)

    rng = np.random.default_rng(55)
    mask = rng.random(2000) < 0.7

    brute = BruteForceIndex.build(vectors, metric)
    ivf_idx = FaissIvfFlatIndex.build(vectors, metric, IVF_BUILD_PARAMS)

    brute_ids, _ = brute.search(queries, top_k=10, valid_mask=mask)
    ivf_ids, _ = ivf_idx.search(queries, top_k=10, valid_mask=mask, params=IVF_SEARCH_PARAMS)

    valid_set = set(np.flatnonzero(mask).tolist())

    # Every returned id must be in the valid set (no leakage)
    for q in range(10):
        for j in range(10):
            for ids in (brute_ids, ivf_ids):
                lid = int(ids[q, j])
                if lid >= 0:
                    assert lid in valid_set

    # Recall on masked subset
    total_recall = 0.0
    for q in range(10):
        bset = {int(i) for i in brute_ids[q] if i >= 0}
        fset = {int(i) for i in ivf_ids[q] if i >= 0}
        denom = len(bset) or 1
        total_recall += len(bset & fset) / denom
    avg_recall = total_recall / 10
    assert avg_recall >= 0.85, f"masked recall@10 = {avg_recall:.3f}"


# ---------------------------------------------------------------------------
# Self-match guarantee on top-1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["L2", "COSINE"])
def test_self_match_top_1(metric):
    """Querying with an indexed vector should return that vector at rank 1."""
    vectors = _gen_vectors(200, 16, seed=66)
    ivf_idx = FaissIvfFlatIndex.build(vectors, metric, IVF_BUILD_PARAMS)
    for i in range(0, 200, 10):
        ids, _ = ivf_idx.search(vectors[i:i + 1], top_k=1, params=IVF_SEARCH_PARAMS)
        assert ids[0, 0] == i, f"metric={metric} self-match failed for row {i}"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    """Save and reload should produce identical search results."""
    vectors = _gen_vectors(300, 16, seed=77)
    queries = _gen_vectors(5, 16, seed=88)

    ivf_idx = FaissIvfFlatIndex.build(vectors, "COSINE", IVF_BUILD_PARAMS)
    ids_before, dists_before = ivf_idx.search(queries, top_k=5, params=IVF_SEARCH_PARAMS)

    path = str(tmp_path / "test.ivf_flat.idx")
    ivf_idx.save(path)
    loaded = FaissIvfFlatIndex.load(path, "COSINE", 16)

    ids_after, dists_after = loaded.search(queries, top_k=5, params=IVF_SEARCH_PARAMS)

    np.testing.assert_array_equal(ids_before, ids_after)
    np.testing.assert_allclose(dists_before, dists_after, rtol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_index():
    """Empty index should return padding."""
    vectors = np.empty((0, 8), dtype=np.float32)
    ivf_idx = FaissIvfFlatIndex.build(vectors, "L2", {"nlist": 1})
    queries = _gen_vectors(3, 8, seed=99)
    ids, dists = ivf_idx.search(queries, top_k=5, params={"nprobe": 1})
    assert ids.shape == (3, 5)
    assert np.all(ids == -1)
    assert np.all(np.isinf(dists))


def test_small_dataset_nlist_clamp():
    """When N < nlist, nlist should be clamped down without error."""
    vectors = _gen_vectors(10, 8, seed=100)
    # nlist=128 > N=10, should clamp to 10
    ivf_idx = FaissIvfFlatIndex.build(vectors, "L2", {"nlist": 128})
    ids, dists = ivf_idx.search(vectors[:1], top_k=3, params={"nprobe": 10})
    assert ids[0, 0] == 0  # self-match
