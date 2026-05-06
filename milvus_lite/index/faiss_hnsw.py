"""FaissHnswIndex — FAISS HNSW per-segment index (Phase 9.5).

The default ANN backend for MilvusLite. Wraps ``faiss.IndexHNSWFlat`` and
normalizes the metric output to MilvusLite's "smaller = more similar"
convention so callers can swap brute-force ↔ FAISS without changing
any downstream code.

Key design points:

1. **Metric symbol alignment** is the most error-prone part. FAISS
   internal conventions:
       - METRIC_L2: returns squared L2 distance
       - METRIC_INNER_PRODUCT: returns positive dot product
                               (larger = more similar)

   MilvusLite conventions (matches BruteForceIndex):
       - L2:     raw L2 distance (sqrt of squared)
       - IP:     -dot(q, v) (negated so smaller = more similar)
       - COSINE: 1 - dot(q_norm, v_norm)

   The differential test in tests/index/test_index_differential.py
   will catch any drift in either direction. Treat that test as the
   ground truth — if it passes, metric alignment is correct.

2. **Cosine via normalized IP**: FAISS has no native cosine metric.
   We L2-normalize vectors at build time and queries at search time,
   then run inner product. ``1 - dot(unit_q, unit_v)`` matches
   BruteForceIndex's ``cosine_distance`` exactly.

3. **valid_mask via IDSelectorBatch**: pass the indices where mask
   is True as a sorted int64 array. This is the simplest IDSelector
   variant and works on every FAISS ≥ 1.7.4. For very dense masks,
   IDSelectorBitmap would save memory, but the bitmap requires
   little-endian packbits which adds a fragility surface; we save it
   for a future optimization.

4. **Padding semantics**: FAISS itself returns -1 in the id slots and
   FLT_MAX (~3.4e38) in the distance slots when fewer than top_k
   results are available. We pass -1 through unchanged but normalize
   FLT_MAX → +inf so the protocol's "padding sentinel" is uniform
   across all VectorIndex implementations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Hard import — this module is only loaded by the factory after
# is_faiss_available() returned True, so the import is safe at module
# top-level. Don't soft-import: if someone imports faiss_hnsw directly
# without faiss installed, an ImportError is the right answer.
import faiss

from milvus_lite.index.protocol import VectorIndex


class FaissHnswIndex(VectorIndex):
    """FAISS HNSW per-segment index.

    Construction-time params (from IndexSpec.build_params):
        M:               int, default 16    — HNSW graph degree
        efConstruction:  int, default 200   — build-time exploration

    Search-time params (from search() params arg):
        ef:              int, default 64    — search-time exploration
    """

    index_type: str = "HNSW"

    __slots__ = ("_index", "metric", "num_vectors", "dim")

    # ── Construction ────────────────────────────────────────────

    def __init__(
        self,
        faiss_index: faiss.Index,
        metric: str,
        num_vectors: int,
        dim: int,
    ) -> None:
        self._index = faiss_index
        self.metric = metric
        self.num_vectors = num_vectors
        self.dim = dim

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        metric: str,
        params: Optional[dict] = None,
    ) -> "FaissHnswIndex":
        """Construct a fresh HNSW index.

        For COSINE, vectors are L2-normalized before being added.
        Search will normalize queries the same way at runtime.
        """
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D (N, dim), got shape {vectors.shape}"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)

        n, dim = int(vectors.shape[0]), int(vectors.shape[1])
        params = params or {}
        M = int(params.get("M", 16))
        ef_construction = int(params.get("efConstruction", 200))

        if metric == "L2":
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        elif metric == "IP":
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        elif metric == "COSINE":
            # Cosine = inner product of L2-normalized vectors.
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            vectors = _l2_normalize(vectors)
        else:
            raise ValueError(
                f"unsupported metric {metric!r}; expected COSINE / L2 / IP"
            )

        index.hnsw.efConstruction = ef_construction

        if n > 0:
            # FAISS requires contiguous float32 arrays.
            vectors_c = np.ascontiguousarray(vectors, dtype=np.float32)
            index.add(vectors_c)

        return cls(index, metric, n, dim)

    # ── Query ───────────────────────────────────────────────────

    def search(
        self,
        queries: np.ndarray,
        top_k: int,
        valid_mask: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k search with optional pre-filter.

        valid_mask=None means "all rows valid". When provided, the
        IDSelectorBatch path is used so HNSW skips excluded ids
        DURING graph traversal, not after.

        Output shape is always (nq, top_k); padding with -1 / +inf
        when fewer than top_k valid candidates exist.
        """
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32, copy=False)

        nq = int(queries.shape[0])

        result_ids = np.full((nq, top_k), -1, dtype=np.int64)
        result_dists = np.full((nq, top_k), np.inf, dtype=np.float32)

        if self.num_vectors == 0 or top_k <= 0:
            return result_ids, result_dists

        if self.metric == "COSINE":
            queries = _l2_normalize(queries)

        # Search-time params
        params = params or {}
        ef = int(params.get("ef", 64))
        # hnsw is a struct on the underlying index; setting efSearch
        # is the modern API for the search-time exploration knob.
        self._index.hnsw.efSearch = ef

        queries_c = np.ascontiguousarray(queries, dtype=np.float32)

        if valid_mask is not None:
            if valid_mask.shape[0] != self.num_vectors:
                raise ValueError(
                    f"valid_mask length {valid_mask.shape[0]} != num_vectors "
                    f"{self.num_vectors}"
                )
            valid_indices = np.flatnonzero(valid_mask).astype(np.int64)
            if valid_indices.size == 0:
                return result_ids, result_dists
            sel = faiss.IDSelectorBatch(valid_indices)
            sp = faiss.SearchParametersHNSW(sel=sel)
            faiss_dists, faiss_ids = self._index.search(queries_c, top_k, params=sp)
        else:
            faiss_dists, faiss_ids = self._index.search(queries_c, top_k)

        # Normalize FAISS output → MilvusLite convention.
        for q in range(nq):
            for j in range(top_k):
                fid = int(faiss_ids[q, j])
                if fid < 0:
                    # Padding from FAISS; leave our own -1/+inf in place.
                    continue
                fdist = float(faiss_dists[q, j])
                # FAISS returns FLT_MAX (~3.4e38) in some padding cases
                # even with non-negative ids; coerce to inf for consistency.
                if fdist >= 1e30:
                    continue
                result_ids[q, j] = fid
                result_dists[q, j] = self._normalize_distance(fdist)

        return result_ids, result_dists

    def _normalize_distance(self, faiss_dist: float) -> float:
        """Convert a single FAISS-internal distance to MilvusLite's
        "smaller = more similar" convention. See module docstring for
        the per-metric reasoning."""
        if self.metric == "L2":
            # FAISS METRIC_L2 returns SQUARED L2.
            return float(np.sqrt(max(faiss_dist, 0.0)))
        if self.metric == "IP":
            # FAISS METRIC_INNER_PRODUCT returns positive dot;
            # we want -dot.
            return -float(faiss_dist)
        if self.metric == "COSINE":
            # We added L2-normalized vectors and queries; FAISS
            # returns dot(q_norm, v_norm). cosine_distance = 1 - dot.
            return 1.0 - float(faiss_dist)
        return float(faiss_dist)

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist via faiss.write_index. The on-disk format is
        FAISS-internal binary; do not assume it's portable across
        FAISS major versions."""
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str, metric: str, dim: int) -> "FaissHnswIndex":
        """Reload via faiss.read_index. *metric* and *dim* are
        supplied externally because they're semantic properties of
        the MilvusLite segment, not stored in the FAISS file."""
        index = faiss.read_index(path)
        n = int(index.ntotal)
        if index.d != dim:
            raise ValueError(
                f"loaded HNSW index dim {index.d} != expected dim {dim}"
            )
        return cls(index, metric, n, dim)


# ── Helpers ─────────────────────────────────────────────────────────

def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row. Zero-norm rows are left as-is to avoid
    NaN propagation (matches BruteForceIndex's cosine handling)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return vectors / safe
