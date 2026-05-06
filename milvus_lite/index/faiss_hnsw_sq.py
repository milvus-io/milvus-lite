"""FaissHnswSqIndex — FAISS HNSW with 8-bit scalar quantization.

Same HNSW graph as FaissHnswIndex but vectors are 8-bit scalar
quantized (~4x compression vs float32). Trades a small recall hit
for much lower memory, especially useful for large segments.

Design mirrors FaissHnswIndex (same metric alignment, cosine via
normalized IP, IDSelectorBatch, distance normalization). Differs
only in the underlying FAISS class:
``IndexHNSWSQ(dim, QT_8bit, M, metric)`` + ``train()`` before add.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import faiss

from milvus_lite.index.protocol import VectorIndex


class FaissHnswSqIndex(VectorIndex):
    """FAISS HNSW + 8-bit scalar quantizer.

    Construction-time params (from IndexSpec.build_params):
        M:               int, default 16    — HNSW graph degree
        efConstruction:  int, default 200   — build-time exploration

    Search-time params (from search() params arg):
        ef:              int, default 64    — search-time exploration
    """

    index_type: str = "HNSW_SQ"

    __slots__ = ("_index", "metric", "num_vectors", "dim")

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
    ) -> "FaissHnswSqIndex":
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
            faiss_metric = faiss.METRIC_L2
        elif metric == "IP":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == "COSINE":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
            vectors = _l2_normalize(vectors)
        else:
            raise ValueError(
                f"unsupported metric {metric!r}; expected COSINE / L2 / IP"
            )

        index = faiss.IndexHNSWSQ(
            dim, faiss.ScalarQuantizer.QT_8bit, M, faiss_metric,
        )
        index.hnsw.efConstruction = ef_construction

        if n > 0:
            vectors_c = np.ascontiguousarray(vectors, dtype=np.float32)
            # HNSWSQ needs a train() pass to learn SQ min/max per dim.
            index.train(vectors_c)
            index.add(vectors_c)

        return cls(index, metric, n, dim)

    def search(
        self,
        queries: np.ndarray,
        top_k: int,
        valid_mask: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        params = params or {}
        ef = int(params.get("ef", 64))
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

        for q in range(nq):
            for j in range(top_k):
                fid = int(faiss_ids[q, j])
                if fid < 0:
                    continue
                fdist = float(faiss_dists[q, j])
                if fdist >= 1e30:
                    continue
                result_ids[q, j] = fid
                result_dists[q, j] = self._normalize_distance(fdist)

        return result_ids, result_dists

    def _normalize_distance(self, faiss_dist: float) -> float:
        if self.metric == "L2":
            return float(np.sqrt(max(faiss_dist, 0.0)))
        if self.metric == "IP":
            return -float(faiss_dist)
        if self.metric == "COSINE":
            return 1.0 - float(faiss_dist)
        return float(faiss_dist)

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str, metric: str, dim: int) -> "FaissHnswSqIndex":
        index = faiss.read_index(path)
        n = int(index.ntotal)
        if index.d != dim:
            raise ValueError(
                f"loaded HNSW_SQ index dim {index.d} != expected dim {dim}"
            )
        return cls(index, metric, n, dim)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return vectors / safe
