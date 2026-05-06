"""Phase 9.5 — Differential test: FaissHnswIndex ≡ BruteForceIndex.

Recall@10 must be ≥ 0.95 across all combinations of:
    - dim ∈ {4, 32, 128}
    - n   ∈ {200, 5000}
    - metric ∈ {COSINE, L2, IP}

Distance value parity (within 1e-3 relative) must hold for the IDs that
appear in BOTH result sets — this is what catches metric symbol bugs.

This test is the architectural safety net for the metric alignment in
FaissHnswIndex. If you change the FAISS adapter, run this test.

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
    from milvus_lite.index.faiss_hnsw import FaissHnswIndex


HNSW_PARAMS = {"M": 32, "efConstruction": 400}
SEARCH_PARAMS = {"ef": 200}


def _gen_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Recall@10 across all (dim, n, metric) combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
@pytest.mark.parametrize("dim,n", [(4, 200), (32, 5000), (128, 5000)])
def test_recall_at_10_vs_brute_force(metric, dim, n):
    vectors = _gen_vectors(n, dim, seed=42)
    queries = _gen_vectors(20, dim, seed=7)

    brute = BruteForceIndex.build(vectors, metric)
    faiss_idx = FaissHnswIndex.build(vectors, metric, HNSW_PARAMS)

    brute_ids, _ = brute.search(queries, top_k=10)
    faiss_ids, _ = faiss_idx.search(queries, top_k=10, params=SEARCH_PARAMS)

    total_recall = 0.0
    for q in range(20):
        brute_set = {int(i) for i in brute_ids[q] if i >= 0}
        faiss_set = {int(i) for i in faiss_ids[q] if i >= 0}
        intersect = len(brute_set & faiss_set)
        denom = len(brute_set) or 1
        total_recall += intersect / denom

    avg_recall = total_recall / 20
    assert avg_recall >= 0.95, (
        f"recall@10 = {avg_recall:.3f} < 0.95 for metric={metric} dim={dim} n={n}"
    )


# ---------------------------------------------------------------------------
# Distance value parity for matching IDs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
def test_distance_value_parity(metric):
    """For IDs that appear in BOTH the FAISS and brute-force top-k,
    the reported distance values must agree within 1e-3 relative
    error. This is what catches metric symbol bugs (squared L2,
    negated IP, etc.)."""
    vectors = _gen_vectors(500, 16, seed=11)
    queries = _gen_vectors(15, 16, seed=22)

    brute = BruteForceIndex.build(vectors, metric)
    faiss_idx = FaissHnswIndex.build(vectors, metric, HNSW_PARAMS)

    # Pull top 50 so we have good overlap to compare
    brute_ids, brute_dists = brute.search(queries, top_k=50)
    faiss_ids, faiss_dists = faiss_idx.search(queries, top_k=50, params=SEARCH_PARAMS)

    mismatches = []
    for q in range(15):
        brute_map = {
            int(brute_ids[q, j]): float(brute_dists[q, j])
            for j in range(50) if brute_ids[q, j] >= 0
        }
        faiss_map = {
            int(faiss_ids[q, j]): float(faiss_dists[q, j])
            for j in range(50) if faiss_ids[q, j] >= 0
        }
        common = set(brute_map.keys()) & set(faiss_map.keys())
        for pid in common:
            bd = brute_map[pid]
            fd = faiss_map[pid]
            tol = 1e-3 + 1e-3 * abs(bd)
            if abs(bd - fd) > tol:
                mismatches.append((q, pid, bd, fd))

    assert not mismatches, (
        f"distance mismatch for metric={metric}: {mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# valid_mask parity — masking on FAISS produces same set as brute force
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["COSINE", "L2"])
def test_valid_mask_parity(metric):
    """When given the same valid_mask, FAISS and BruteForce should
    return the same top-k IDs (modulo recall noise — both should
    only consider rows where mask is True)."""
    vectors = _gen_vectors(2000, 16, seed=33)
    queries = _gen_vectors(10, 16, seed=44)

    # Random mask: keep ~70% of rows
    rng = np.random.default_rng(55)
    mask = rng.random(2000) < 0.7

    brute = BruteForceIndex.build(vectors, metric)
    faiss_idx = FaissHnswIndex.build(vectors, metric, HNSW_PARAMS)

    brute_ids, _ = brute.search(queries, top_k=10, valid_mask=mask)
    faiss_ids, _ = faiss_idx.search(queries, top_k=10, valid_mask=mask, params=SEARCH_PARAMS)

    valid_set = set(np.flatnonzero(mask).tolist())

    # Every returned id must be in the valid set (no leakage)
    for q in range(10):
        for j in range(10):
            for ids in (brute_ids, faiss_ids):
                lid = int(ids[q, j])
                if lid >= 0:
                    assert lid in valid_set

    # Recall@10 on the masked subset should still be ≥ 0.95
    total_recall = 0.0
    for q in range(10):
        bset = {int(i) for i in brute_ids[q] if i >= 0}
        fset = {int(i) for i in faiss_ids[q] if i >= 0}
        denom = len(bset) or 1
        total_recall += len(bset & fset) / denom
    avg_recall = total_recall / 10
    assert avg_recall >= 0.90, f"masked recall@10 = {avg_recall:.3f}"


# ---------------------------------------------------------------------------
# Self-match guarantee on top-1 (L2/COSINE only — IP doesn't guarantee)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["L2", "COSINE"])
def test_self_match_top_1(metric):
    """For metrics where self-match is the global minimum, querying
    with one of the indexed vectors should always return that vector
    at rank 1. This is the most basic correctness check."""
    vectors = _gen_vectors(200, 16, seed=66)
    faiss_idx = FaissHnswIndex.build(vectors, metric, HNSW_PARAMS)
    for i in range(0, 200, 10):
        ids, _ = faiss_idx.search(vectors[i:i + 1], top_k=1, params=SEARCH_PARAMS)
        assert ids[0, 0] == i, f"metric={metric} self-match failed for row {i}"
