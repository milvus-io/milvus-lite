"""FaissIvfSq8Index — FAISS IVF with 8-bit scalar quantization.

IVF partitions vectors into ``nlist`` Voronoi cells; within each cell
each vector is stored as 8-bit scalar-quantized values (~4x compression
vs float32). Trades a small recall hit for much lower memory.

Design mirrors FaissIvfFlatIndex — same metric alignment, COSINE via
normalized IP, nlist clamping, IDSelectorBatch for valid_mask,
distance normalization. Only the underlying FAISS class differs:
``IndexIVFScalarQuantizer(quantizer, dim, nlist, QT_8bit, metric)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import faiss

from milvus_lite.index.protocol import VectorIndex


class FaissIvfSq8Index(VectorIndex):
    """FAISS IVF + 8-bit scalar quantizer.

    Construction-time params (from IndexSpec.build_params):
        nlist:  int, default 128  — number of Voronoi cells

    Search-time params (from search() params arg):
        nprobe: int, default 10   — number of cells to scan
    """

    index_type: str = "IVF_SQ8"

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
    ) -> "FaissIvfSq8Index":
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D (N, dim), got shape {vectors.shape}"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)

        n, dim = int(vectors.shape[0]), int(vectors.shape[1])
        params = params or {}
        nlist = int(params.get("nlist", 128))

        if n > 0:
            nlist = min(nlist, n)

        if metric == "COSINE":
            vectors = _l2_normalize(vectors)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == "IP":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == "L2":
            faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError(
                f"unsupported metric {metric!r}; expected COSINE / L2 / IP"
            )

        quantizer = faiss.IndexFlat(dim, faiss_metric)
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist,
            faiss.ScalarQuantizer.QT_8bit, faiss_metric,
        )

        if n > 0:
            vectors_c = np.ascontiguousarray(vectors, dtype=np.float32)
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
        nprobe = int(params.get("nprobe", 10))
        self._index.nprobe = nprobe

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
            sp = faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
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
    def load(cls, path: str, metric: str, dim: int) -> "FaissIvfSq8Index":
        index = faiss.read_index(path)
        n = int(index.ntotal)
        if index.d != dim:
            raise ValueError(
                f"loaded IVF_SQ8 index dim {index.d} != expected dim {dim}"
            )
        return cls(index, metric, n, dim)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return vectors / safe
