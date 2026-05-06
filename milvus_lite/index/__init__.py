"""Vector index subsystem (Phase 9).

Each VectorIndex is bound 1:1 to a Segment (one .idx sidecar per data
Parquet file). Indexes are immutable — compaction creates a new merged
segment + a new index, dropping the old ones.

Public exports:
    VectorIndex     — abstract protocol (index/protocol.py)
    IndexSpec       — frozen dataclass that travels through Manifest
    BruteForceIndex — NumPy implementation. Long-lived first-class:
                      differential test baseline + faiss-not-installed
                      fallback + small-segment chosen impl.
    FaissHnswIndex  — FAISS HNSW (Phase 9.5). Only exported if
                      faiss-cpu is installed; otherwise it's omitted
                      from __all__ but the factory still routes
                      requests through it (raising
                      IndexBackendUnavailableError if missing).
    build_index_from_spec / load_index_from_spec — factory dispatch
"""

from milvus_lite.index.brute_force import BruteForceIndex
from milvus_lite.index.factory import (
    build_index_from_spec,
    is_faiss_available,
    load_index_from_spec,
)
from milvus_lite.index.protocol import VectorIndex
from milvus_lite.index.spec import IndexSpec

__all__ = [
    "VectorIndex",
    "BruteForceIndex",
    "IndexSpec",
    "build_index_from_spec",
    "load_index_from_spec",
    "is_faiss_available",
]

if is_faiss_available():
    from milvus_lite.index.faiss_hnsw import FaissHnswIndex  # noqa: F401
    from milvus_lite.index.faiss_ivf_flat import FaissIvfFlatIndex  # noqa: F401
    __all__.extend(["FaissHnswIndex", "FaissIvfFlatIndex"])
