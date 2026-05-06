"""BruteForceIndex — NumPy reference implementation of VectorIndex.

Three roles, all long-lived:

1. **Differential test baseline** for FAISS implementations. Phase 9.5
   diff-tests `FaissHnswIndex` against this for distance parity and
   recall@k ≥ 0.95.

2. **Fallback** when faiss-cpu is not installed. Without this users
   would have no vector index at all and the engine would fail to
   serve `Collection.search`. The factory routes to BruteForceIndex
   automatically when an HNSW request is made on a system without
   faiss.

3. **Implementation of choice for small segments** below
   ``INDEX_BUILD_THRESHOLD``. Building an HNSW index on 100 vectors
   costs more than just running brute force on every query, so the
   factory will prefer brute force for tiny segments.

Storage format: ``np.save`` of the float32 vector array. The .npy
header carries dtype + shape so reload doesn't need explicit metadata.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from milvus_lite.index.protocol import VectorIndex
from milvus_lite.search.distance import compute_distances


class BruteForceIndex(VectorIndex):
    """NumPy brute-force per-segment index.

    Stores a (potentially shared) reference to the segment's vector
    matrix. Search is O(N * dim) per query — vectorized via NumPy so
    it's still fast for N up to a few tens of thousands.
    """

    index_type: str = "BRUTE_FORCE"

    __slots__ = ("_vectors", "metric", "num_vectors", "dim")

    def __init__(self, vectors: np.ndarray, metric: str) -> None:
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D (N, dim), got shape {vectors.shape}"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        self._vectors = vectors
        self.metric = metric
        self.num_vectors = int(vectors.shape[0])
        self.dim = int(vectors.shape[1])

    # ── Construction ────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        metric: str,
        params: Optional[dict] = None,  # noqa: ARG003 — protocol contract
    ) -> "BruteForceIndex":
        """Build an index. *params* is unused (brute force has no knobs)."""
        return cls(vectors, metric)

    # ── Query ───────────────────────────────────────────────────

    def search(
        self,
        queries: np.ndarray,
        top_k: int,
        valid_mask: Optional[np.ndarray] = None,
        params: Optional[dict] = None,  # noqa: ARG002 — protocol contract
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k nearest neighbors per query, NumPy brute force.

        valid_mask = None means "all rows valid". When provided, only
        rows where mask is True are considered — the corresponding
        local_ids are picked from the unmasked space, so callers can
        use them to index back into the original (full) array.

        Output is always shaped (nq, top_k); padding with -1 / +inf
        when there are fewer than top_k valid candidates lets callers
        merge results from multiple sources by sorting on distance
        without special-casing.
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

        # Translate valid_mask into a list of original-row indices.
        if valid_mask is None:
            valid_indices = np.arange(self.num_vectors, dtype=np.int64)
        else:
            if valid_mask.shape[0] != self.num_vectors:
                raise ValueError(
                    f"valid_mask length {valid_mask.shape[0]} != num_vectors "
                    f"{self.num_vectors}"
                )
            valid_indices = np.flatnonzero(valid_mask).astype(np.int64)

        if valid_indices.size == 0:
            return result_ids, result_dists

        valid_vectors = self._vectors[valid_indices]
        effective_k = min(top_k, valid_indices.size)

        # compute_distances supports batch (nq, dim) → (nq, n) layout.
        all_dists = compute_distances(queries, valid_vectors, self.metric)
        # Defensive: in case a metric impl returned 1-D for nq=1.
        if all_dists.ndim == 1:
            all_dists = all_dists.reshape(1, -1)

        for q in range(nq):
            dists = all_dists[q]
            if effective_k < valid_indices.size:
                part = np.argpartition(dists, effective_k - 1)[:effective_k]
                order = part[np.argsort(dists[part])]
            else:
                order = np.argsort(dists)

            for j, local_idx in enumerate(order[:effective_k]):
                result_ids[q, j] = valid_indices[int(local_idx)]
                result_dists[q, j] = float(dists[int(local_idx)])

        return result_ids, result_dists

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save vectors to *path* in NPY format.

        Uses an explicit file handle (np.save would otherwise auto-
        append .npy if the path lacks that extension, which would break
        Phase 9.4's strict ``<segment_stem>.<index_type>.idx`` naming
        convention).

        The NPY header carries dtype + shape so the load path doesn't
        need explicit dim/metric — only metric is supplied externally
        because it's a semantic property, not stored data.
        """
        with open(path, "wb") as f:
            np.save(f, self._vectors, allow_pickle=False)

    @classmethod
    def load(cls, path: str, metric: str, dim: int) -> "BruteForceIndex":
        """Reload a previously saved BruteForceIndex.

        *dim* is validated against the on-disk shape; mismatch is a
        loud error (not a silent reshape) since it indicates the .idx
        file is paired with the wrong segment.
        """
        with open(path, "rb") as f:
            vectors = np.load(f, allow_pickle=False)
        if vectors.ndim != 2:
            raise ValueError(
                f"loaded index has shape {vectors.shape}, expected (N, dim)"
            )
        if vectors.shape[1] != dim:
            raise ValueError(
                f"loaded index dim {vectors.shape[1]} != expected dim {dim}"
            )
        return cls(vectors, metric)
