"""Index factory — single dispatch point for build / load.

Routes IndexSpec.index_type to the right VectorIndex implementation
and handles the optional faiss-cpu dependency gracefully.

Routing rules:
    BRUTE_FORCE → BruteForceIndex (no extra dependency)
    HNSW        → FaissHnswIndex (requires faiss-cpu)
    IVF_FLAT    → FaissIvfFlatIndex (requires faiss-cpu)
    IVF_SQ8/PQ  → reserved for future phases (NotImplementedError)

When faiss-cpu is not installed, requesting an HNSW index raises
``IndexBackendUnavailableError`` with a clear "install faiss-cpu" hint.
The user can fall back to BRUTE_FORCE in their create_index call to keep
going without faiss.

Why a separate factory module:
    - Keeps Segment / Collection free of conditional faiss imports
    - Centralizes the version compatibility surface
    - Phase 9.6+ can add new index_type cases without touching
      anything below the factory layer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from milvus_lite.exceptions import IndexBackendUnavailableError
from milvus_lite.index.brute_force import BruteForceIndex

if TYPE_CHECKING:
    from milvus_lite.index.protocol import VectorIndex
    from milvus_lite.index.spec import IndexSpec


# ── Optional FAISS detection ────────────────────────────────────────

try:
    import faiss as _faiss  # noqa: F401
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# Index types that REQUIRE faiss-cpu. BRUTE_FORCE is intentionally NOT
# in this set — it always works.
_FAISS_INDEX_TYPES = frozenset({
    "HNSW", "HNSW_SQ",
    "IVF_FLAT", "IVF_SQ8", "IVF_PQ",
})
KNOWN_VECTOR_INDEX_TYPES = frozenset({
    "AUTOINDEX",
    "BRUTE_FORCE",
    "FLAT",
    "SPARSE_INVERTED_INDEX",
}) | _FAISS_INDEX_TYPES


def is_faiss_available() -> bool:
    """Whether faiss-cpu is importable in this process.

    Phase 10 gRPC adapter (or any caller wanting to know "can I
    actually create an HNSW?") can use this for capability discovery.
    """
    return _FAISS_AVAILABLE


# ── Public dispatch ─────────────────────────────────────────────────

def build_index_from_spec(
    spec: "IndexSpec",
    vectors: np.ndarray,
) -> "VectorIndex":
    """Build a fresh VectorIndex from raw vectors.

    Args:
        spec: IndexSpec with index_type, metric_type, build_params
        vectors: (N, dim) float32 vector matrix

    Raises:
        IndexBackendUnavailableError: requested index_type needs faiss
            but faiss-cpu isn't installed
        NotImplementedError: index_type is reserved for a future phase
        ValueError: index_type is unrecognized
    """
    index_type = spec.index_type
    if index_type in ("BRUTE_FORCE", "FLAT"):
        return BruteForceIndex.build(vectors, spec.metric_type, spec.build_params)
    if index_type == "AUTOINDEX":
        # Use HNSW when faiss is available, otherwise fall back to BruteForce.
        # Load uses try/except to detect actual format on disk.
        if _FAISS_AVAILABLE:
            from milvus_lite.index.faiss_hnsw import FaissHnswIndex
            return FaissHnswIndex.build(vectors, spec.metric_type, spec.build_params)
        return BruteForceIndex.build(vectors, spec.metric_type, spec.build_params)
    if index_type in _FAISS_INDEX_TYPES:
        _require_faiss(index_type)
        if index_type == "HNSW":
            from milvus_lite.index.faiss_hnsw import FaissHnswIndex
            return FaissHnswIndex.build(vectors, spec.metric_type, spec.build_params)
        if index_type == "IVF_FLAT":
            from milvus_lite.index.faiss_ivf_flat import FaissIvfFlatIndex
            return FaissIvfFlatIndex.build(vectors, spec.metric_type, spec.build_params)
        if index_type == "IVF_SQ8":
            from milvus_lite.index.faiss_ivf_sq8 import FaissIvfSq8Index
            return FaissIvfSq8Index.build(vectors, spec.metric_type, spec.build_params)
        if index_type == "HNSW_SQ":
            from milvus_lite.index.faiss_hnsw_sq import FaissHnswSqIndex
            return FaissHnswSqIndex.build(vectors, spec.metric_type, spec.build_params)
        raise NotImplementedError(
            f"index_type={index_type!r} is reserved for a future phase; "
            f"supported: HNSW, HNSW_SQ, IVF_FLAT, IVF_SQ8, BRUTE_FORCE"
        )
    raise ValueError(f"unknown index_type: {index_type!r}")


def load_index_from_spec(
    spec: "IndexSpec",
    path: str,
    dim: int,
) -> "VectorIndex":
    """Reload a VectorIndex from disk based on spec.index_type.

    Same routing rules as build_index_from_spec; see there for the
    error semantics.
    """
    index_type = spec.index_type
    if index_type in ("BRUTE_FORCE", "FLAT"):
        return BruteForceIndex.load(path, spec.metric_type, dim)
    if index_type == "AUTOINDEX":
        # Detect actual format on disk by trying FAISS first, then BruteForce.
        # This handles the case where faiss availability changed since build time.
        if _FAISS_AVAILABLE:
            try:
                from milvus_lite.index.faiss_hnsw import FaissHnswIndex
                return FaissHnswIndex.load(path, spec.metric_type, dim)
            except Exception:
                return BruteForceIndex.load(path, spec.metric_type, dim)
        try:
            return BruteForceIndex.load(path, spec.metric_type, dim)
        except Exception:
            raise IndexBackendUnavailableError(
                f"AUTOINDEX file at {path!r} appears to be a FAISS index, "
                f"but faiss-cpu is not installed. Install with: "
                f"`pip install faiss-cpu`"
            )
    if index_type in _FAISS_INDEX_TYPES:
        _require_faiss(index_type)
        if index_type == "HNSW":
            from milvus_lite.index.faiss_hnsw import FaissHnswIndex
            return FaissHnswIndex.load(path, spec.metric_type, dim)
        if index_type == "IVF_FLAT":
            from milvus_lite.index.faiss_ivf_flat import FaissIvfFlatIndex
            return FaissIvfFlatIndex.load(path, spec.metric_type, dim)
        if index_type == "IVF_SQ8":
            from milvus_lite.index.faiss_ivf_sq8 import FaissIvfSq8Index
            return FaissIvfSq8Index.load(path, spec.metric_type, dim)
        if index_type == "HNSW_SQ":
            from milvus_lite.index.faiss_hnsw_sq import FaissHnswSqIndex
            return FaissHnswSqIndex.load(path, spec.metric_type, dim)
        raise NotImplementedError(
            f"index_type={index_type!r} is reserved for a future phase"
        )
    raise ValueError(f"unknown index_type: {index_type!r}")


# ── Internals ───────────────────────────────────────────────────────

def _require_faiss(index_type: str) -> None:
    if not _FAISS_AVAILABLE:
        raise IndexBackendUnavailableError(
            f"index_type={index_type!r} requires faiss-cpu, which is not "
            f"installed. Install it with `pip install faiss-cpu` (or use "
            f"index_type='BRUTE_FORCE' to fall back without faiss)."
        )
