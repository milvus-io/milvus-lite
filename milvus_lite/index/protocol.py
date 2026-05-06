"""VectorIndex protocol — abstract per-segment vector index.

A VectorIndex is the unit of vector retrieval in Phase 9. It is bound
1:1 to a Segment (each immutable data Parquet file gets one .idx
sidecar). Indexes are immutable: there is no add() or remove() — any
mutation goes through "drop old segment + build new segment + build
new index", driven by compaction.

Implementations:
    - BruteForceIndex (Phase 9.2)  — NumPy. Long-lived first-class
      implementation: differential test baseline + faiss-not-installed
      fallback + small-segment chosen impl.
    - FaissHnswIndex (Phase 9.5)   — FAISS HNSW + IDSelectorBitmap.

Distance convention is uniform regardless of metric: **smaller = more
similar**. FAISS implementations must normalize their internal output
to match (L2 → raw L2 not squared, IP → negated, cosine → 1 - dot of
normalized vectors). This invariant lets the upper-layer executor stay
metric-agnostic and is enforced by differential tests.

local_id space: each VectorIndex sees vectors numbered 0..N-1 where N
is the source segment's row count. Mapping local_id back to a primary
key is the Segment's responsibility, NOT the index's. This keeps index
implementations completely schema-agnostic and pk-type-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class VectorIndex(ABC):
    """Abstract per-segment vector index.

    Subclasses set `metric`, `num_vectors`, `dim` in their constructor
    or `build` classmethod.
    """

    __slots__ = ()

    metric: str       # "COSINE" | "L2" | "IP"
    num_vectors: int
    dim: int

    # ── Construction ────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def build(
        cls,
        vectors: np.ndarray,        # (N, dim) float32
        metric: str,                # "COSINE" | "L2" | "IP"
        params: Optional[dict] = None,
    ) -> "VectorIndex":
        """Build a fresh index from a vector matrix.

        Local id of each row is its position in *vectors* (0..N-1).
        Implementations may copy the array or hold a reference; either
        is acceptable as long as subsequent calls to .search produce
        results consistent with the original snapshot.
        """

    # ── Query ───────────────────────────────────────────────────

    @abstractmethod
    def search(
        self,
        queries: np.ndarray,        # (nq, dim) float32
        top_k: int,
        valid_mask: Optional[np.ndarray] = None,  # (num_vectors,) bool
        params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k nearest neighbors per query.

        Args:
            queries: (nq, dim) float32 query vectors
            top_k: requested k
            valid_mask: optional bool array of length num_vectors. False
                means "skip this row" — used to combine the bitmap
                pipeline (dedup + tombstones + scalar filter) with the
                index lookup *before* distance computation, not after.
                None = all rows valid.
            params: implementation-specific tuning (e.g. ef for HNSW,
                nprobe for IVF). None = use the index's defaults.

        Returns:
            (local_ids, distances), each shape (nq, top_k).
            - local_ids: int64. Padded with -1 if fewer than top_k
              valid candidates exist.
            - distances: float32. Padded with +inf for the same slots.
              Distance convention: smaller = more similar, regardless
              of metric. The padding is chosen so that result merging
              across multiple sources can sort by distance and let
              -1 entries fall to the bottom naturally.
        """

    # ── Persistence ─────────────────────────────────────────────

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to *path*. Format is implementation-defined.

        The caller is responsible for choosing the path (typically
        ``indexes/<segment_stem>.<index_type>.idx``). The index file
        becomes a sidecar of its source data Parquet — see Phase 9.4.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str, metric: str, dim: int) -> "VectorIndex":
        """Reload a previously-saved index from *path*.

        *metric* and *dim* are passed in because some on-disk formats
        (notably FAISS HNSW) don't preserve them and need to be told
        explicitly. The caller knows them from IndexSpec + Segment.
        """

    # ── Identity ────────────────────────────────────────────────

    @property
    @abstractmethod
    def index_type(self) -> str:
        """A short string tag like 'BRUTE_FORCE' / 'HNSW' / 'IVF_FLAT'.

        Used in .idx filenames and in describe_index responses.
        Conventionally uppercase to match Milvus's index_type strings.
        """
