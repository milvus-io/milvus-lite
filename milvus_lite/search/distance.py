"""Vector distance functions, NumPy-only.

All functions return distances such that **smaller = more similar**, so
that the search executor can uniformly take the top-k smallest. This is
why ``ip_distance`` returns the negated dot product — IP is "larger is
more similar" by nature, but we negate so the executor can stay agnostic.

Each function supports two shapes:

    single query:  query (dim,)        candidates (n, dim)  → (n,)
    batch query:   queries (nq, dim)   candidates (n, dim)  → (nq, n)

Dispatch is via the dimensionality of *query*, not a separate function,
so callers don't have to know which is which.
"""

from __future__ import annotations

import numpy as np


METRIC_COSINE = "COSINE"
METRIC_L2 = "L2"
METRIC_IP = "IP"

SUPPORTED_METRICS = (METRIC_COSINE, METRIC_L2, METRIC_IP)


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------

def cosine_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - cosine_similarity, in [0, 2]. Smaller = closer."""
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.size == 0:
        return _empty_result(query, candidates)

    c_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    # Avoid division by zero — zero-vector candidates produce nan, which we
    # remap to 1.0 (cosine distance for "no information").
    c_norms_safe = np.where(c_norms == 0, 1.0, c_norms)
    c_unit = candidates / c_norms_safe

    if query.ndim == 1:
        q_norm = np.linalg.norm(query)
        q_unit = query if q_norm == 0 else query / q_norm
        sims = c_unit @ q_unit
    else:
        q_norms = np.linalg.norm(query, axis=1, keepdims=True)
        q_norms_safe = np.where(q_norms == 0, 1.0, q_norms)
        q_unit = query / q_norms_safe
        sims = q_unit @ c_unit.T  # (nq, n)

    return np.maximum(1.0 - sims, 0.0)


# ---------------------------------------------------------------------------
# L2 (Euclidean)
# ---------------------------------------------------------------------------

def l2_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """L2 (Euclidean) distance. Smaller = closer."""
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.size == 0:
        return _empty_result(query, candidates)

    if query.ndim == 1:
        diff = candidates - query
        return np.linalg.norm(diff, axis=1)
    else:
        # (nq, 1, dim) - (1, n, dim) → (nq, n, dim) → (nq, n)
        diff = query[:, None, :] - candidates[None, :, :]
        return np.linalg.norm(diff, axis=2)


# ---------------------------------------------------------------------------
# Inner Product
# ---------------------------------------------------------------------------

def ip_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Inner product distance = -dot(q, c). Smaller = closer (so executors
    can take top-k smallest uniformly)."""
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.size == 0:
        return _empty_result(query, candidates)

    if query.ndim == 1:
        return -(candidates @ query)
    else:
        return -(query @ candidates.T)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_distances(
    query: np.ndarray,
    candidates: np.ndarray,
    metric_type: str,
) -> np.ndarray:
    """Dispatch to the metric-specific function."""
    if metric_type == METRIC_COSINE:
        return cosine_distance(query, candidates)
    if metric_type == METRIC_L2:
        return l2_distance(query, candidates)
    if metric_type == METRIC_IP:
        return ip_distance(query, candidates)
    raise ValueError(
        f"unsupported metric_type {metric_type!r}; expected one of {SUPPORTED_METRICS}"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _empty_result(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Return the right-shaped empty distance array."""
    if query.ndim == 1:
        return np.zeros((0,), dtype=np.float32)
    return np.zeros((query.shape[0], 0), dtype=np.float32)
