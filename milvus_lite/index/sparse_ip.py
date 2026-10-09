"""Exact inner-product retrieval over raw sparse embedding weights."""

from __future__ import annotations

import heapq
import json

from milvus_lite.index.sparse_common import empty_results, validate_mask
from milvus_lite.schema.sparse import normalize_sparse_vector


class SparseIpIndex:
    """Immutable-after-build postings; scores are independent of corpus statistics."""

    def __init__(self) -> None:
        self.num_rows = 0
        self._posting_lists: dict[int, list[tuple[int, float]]] = {}

    def build(self, sparse_vectors: list[dict[int, float]], valid_mask=None) -> None:
        num_rows = len(sparse_vectors)
        mask = validate_mask(valid_mask, num_rows)
        postings: dict[int, list[tuple[int, float]]] = {}
        for local_id, vector in enumerate(sparse_vectors):
            vector = normalize_sparse_vector(vector)
            if mask is not None and not mask[local_id]:
                continue
            for dimension, weight in vector.items():
                postings.setdefault(dimension, []).append((local_id, weight))
        self.num_rows = num_rows
        self._posting_lists = postings

    def search(self, query_sparse_vectors: list[dict[int, float]], top_k: int, valid_mask=None):
        queries = [normalize_sparse_vector(query) for query in query_sparse_vectors]
        return self._search_prepared(queries, top_k, valid_mask=valid_mask)

    def _search_prepared(self, queries: list[dict[int, float]], top_k: int, valid_mask=None):
        """Score read-only, validated float32 queries prepared once by Collection.

        Standalone callers use search(). Mask and top-k checks remain enabled
        for every call; only query normalization is skipped.
        """
        mask = validate_mask(valid_mask, self.num_rows)
        ids, distances = empty_results(len(queries), top_k)
        if top_k == 0:
            return ids, distances
        for qi, query in enumerate(queries):
            scores: dict[int, float] = {}
            for dimension, query_weight in query.items():
                for local_id, weight in self._posting_lists.get(dimension, ()):
                    if mask is None or mask[local_id]:
                        scores[local_id] = scores.get(local_id, 0.0) + query_weight * weight
            best = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
            for rank, (local_id, score) in enumerate(best):
                ids[qi, rank] = local_id
                distances[qi, rank] = -score
        return ids, distances

    def save(self, path: str) -> None:
        with open(path, "w") as stream:
            json.dump({
                "metric_type": "IP",
                "num_rows": self.num_rows,
                "posting_lists": self._posting_lists,
            }, stream)

    @classmethod
    def load(cls, path: str) -> "SparseIpIndex":
        with open(path) as stream:
            data = json.load(stream)
        if data.get("metric_type") != "IP":
            raise ValueError("sparse IP index requires persisted metric_type='IP'")
        num_rows = data["num_rows"]
        if not isinstance(num_rows, int) or isinstance(num_rows, bool) or num_rows < 0:
            raise ValueError("invalid sparse index row count")
        rows = [{} for _ in range(num_rows)]
        for dimension, postings in data["posting_lists"].items():
            dimension = int(dimension)
            for local_id, weight in postings:
                if (not isinstance(local_id, int) or isinstance(local_id, bool)
                        or not 0 <= local_id < num_rows or dimension in rows[local_id]):
                    raise ValueError("invalid sparse index posting row ID")
                rows[local_id][dimension] = weight
        index = cls()
        index.build(rows)
        return index

    @property
    def index_type(self) -> str:
        return "SPARSE_INVERTED_INDEX"
