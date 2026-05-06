"""SparseInvertedIndex — per-segment BM25 inverted index.

Builds posting lists from sparse vectors (term_hash → TF) and
evaluates BM25 scoring at query time using segment-local statistics.

Unlike the dense VectorIndex protocol, SparseInvertedIndex works
with sparse vector dicts (not numpy arrays). The search API returns
results in the same (local_ids, distances) tuple format for
compatibility with the engine merge logic.

BM25 formula:
    score(D, Q) = Σ IDF(qi) · f(qi,D)·(k1+1) / (f(qi,D) + k1·(1-b+b·|D|/avgdl))
    where IDF(qi) = log((N - df + 0.5) / (df + 0.5) + 1)

Distance convention: distance = -bm25_score (smaller = more similar,
matching VectorIndex protocol). Zero-score documents get distance 0.0.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SparseInvertedIndex:
    """Per-segment inverted index for BM25 scoring.

    Attributes:
        doc_count: number of valid documents in the segment
        avgdl: average document length (token count)
        k1: BM25 saturation parameter (default 1.5)
        b: BM25 length normalization parameter (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count: int = 0
        self.avgdl: float = 0.0
        # term_hash → list of (local_id, tf)
        self._posting_lists: Dict[int, List[Tuple[int, float]]] = {}
        # term_hash → document frequency (number of docs containing term)
        self._df: Dict[int, int] = {}
        # per-doc token count
        self._doc_lengths: np.ndarray = np.array([], dtype=np.float32)

    def build(
        self,
        sparse_vectors: List[Dict[int, float]],
        valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Build the inverted index from sparse vectors.

        Args:
            sparse_vectors: list of N dicts, each mapping term_hash → TF.
            valid_mask: optional bool array of length N. False rows are
                skipped (deleted or deduped).
        """
        n = len(sparse_vectors)
        self._posting_lists = {}
        self._df = {}
        doc_lengths = np.zeros(n, dtype=np.float32)

        valid_count = 0
        total_length = 0.0

        for local_id in range(n):
            if valid_mask is not None and not valid_mask[local_id]:
                continue
            sv = sparse_vectors[local_id]
            if not sv:
                valid_count += 1
                continue

            doc_len = sum(sv.values())
            doc_lengths[local_id] = doc_len
            total_length += doc_len
            valid_count += 1

            for term_hash, tf in sv.items():
                if term_hash not in self._posting_lists:
                    self._posting_lists[term_hash] = []
                    self._df[term_hash] = 0
                self._posting_lists[term_hash].append((local_id, tf))
                self._df[term_hash] += 1

        self.doc_count = valid_count
        self.avgdl = total_length / valid_count if valid_count > 0 else 0.0
        self._doc_lengths = doc_lengths

    def search(
        self,
        query_sparse_vectors: List[Dict[int, float]],
        top_k: int,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """BM25 search over the inverted index.

        Args:
            query_sparse_vectors: list of nq query vectors, each a dict
                mapping term_hash → weight (typically 1.0 for each query term).
            top_k: requested k per query.
            valid_mask: optional bool mask to further restrict candidates
                (e.g., from scalar filter). Applied on top of the build-time mask.

        Raises:
            ValueError: if valid_mask length does not match the number of
                documents in the index.

        Returns:
            (local_ids, distances), each shape (nq, top_k).
            distances = -bm25_score (smaller = more relevant).
            Padded with -1 / +inf for missing slots.
        """
        if valid_mask is not None and len(valid_mask) != len(self._doc_lengths):
            raise ValueError(
                f"valid_mask length ({len(valid_mask)}) != num_docs "
                f"({len(self._doc_lengths)})"
            )

        nq = len(query_sparse_vectors)
        all_ids = np.full((nq, top_k), -1, dtype=np.int64)
        all_dists = np.full((nq, top_k), float("inf"), dtype=np.float32)

        if self.doc_count == 0:
            return all_ids, all_dists

        N = self.doc_count
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for qi in range(nq):
            query_terms = query_sparse_vectors[qi]
            if not query_terms:
                continue

            scores: Dict[int, float] = {}

            for term_hash, query_weight in query_terms.items():
                posting = self._posting_lists.get(term_hash)
                if not posting:
                    continue

                df = self._df[term_hash]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

                for local_id, tf in posting:
                    if valid_mask is not None and not valid_mask[local_id]:
                        continue
                    dl = self._doc_lengths[local_id]
                    tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
                    scores[local_id] = scores.get(local_id, 0.0) + idf * tf_norm * query_weight

            if not scores:
                continue

            # Top-k by score (descending)
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            k_actual = min(top_k, len(sorted_items))
            for j in range(k_actual):
                local_id, score = sorted_items[j]
                all_ids[qi, j] = local_id
                all_dists[qi, j] = -score  # negate: smaller = more similar

        return all_ids, all_dists

    def save(self, path: str) -> None:
        """Persist the index to a JSON file."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avgdl": self.avgdl,
            "doc_lengths": self._doc_lengths.tolist(),
            "posting_lists": {
                str(k): v for k, v in self._posting_lists.items()
            },
            "df": {str(k): v for k, v in self._df.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, k1: float = 1.5, b: float = 0.75) -> "SparseInvertedIndex":
        """Load a previously saved index."""
        with open(path, "r") as f:
            data = json.load(f)
        idx = cls(k1=data.get("k1", k1), b=data.get("b", b))
        idx.doc_count = data["doc_count"]
        idx.avgdl = data["avgdl"]
        idx._doc_lengths = np.array(data["doc_lengths"], dtype=np.float32)
        idx._posting_lists = {
            int(k): [tuple(pair) for pair in v]
            for k, v in data["posting_lists"].items()
        }
        idx._df = {int(k): v for k, v in data["df"].items()}
        return idx

    @property
    def index_type(self) -> str:
        return "SPARSE_INVERTED_INDEX"
