"""IndexSpec — the per-Collection index configuration record.

Persisted on the Manifest (Phase 9.3) and consumed by every Segment to
decide which VectorIndex implementation to build (Phase 9.4-9.5).

Mirrors Milvus's IndexParams structure so the Phase 10 gRPC adapter can
do a near-1:1 KeyValuePair mapping.

Why frozen:
    - Persisted on Manifest → must hash safely + survive snapshots
    - Shared across multiple Segments simultaneously → no surprise
      mutation if one segment edits its copy
    - Mirrors the Phase-8 CompiledExpr / FieldSchema convention

Why build_params is a typed dict and not a typed dataclass per index_type:
    Different index_types have wildly different parameter spaces (HNSW
    has M / efConstruction; IVF has nlist / nprobe; OPQ has yet more).
    A typed dataclass per type would force a Union of N variants and
    cascading isinstance checks. dict + impl-internal validation is
    lighter and aligns directly with Milvus's KeyValuePair representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict


@dataclass(frozen=True)
class IndexSpec:
    """A vector field's index configuration.

    Attributes:
        field_name: name of the vector field this index covers. Must
            match a FLOAT_VECTOR field in the CollectionSchema.
        index_type: tag like "BRUTE_FORCE" / "HNSW" / "IVF_FLAT".
            Conventionally uppercase, matching Milvus's index_type
            string set.
        metric_type: "COSINE" / "L2" / "IP".
        build_params: implementation-specific knobs. For HNSW:
            ``{"M": 16, "efConstruction": 200}``. For BRUTE_FORCE: ``{}``.
        search_params: implementation-specific search-time defaults.
            For HNSW: ``{"ef": 64}``. Search RPC may override per call.
    """

    field_name: str
    index_type: str
    metric_type: str
    build_params: Dict[str, Any]
    search_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclass post-init validation. Cheap, surfaces obvious
        # errors at create_index time rather than at load() time.
        if not isinstance(self.field_name, str) or not self.field_name:
            raise ValueError(f"field_name must be a non-empty string, got {self.field_name!r}")
        if not isinstance(self.index_type, str) or not self.index_type:
            raise ValueError(f"index_type must be a non-empty string, got {self.index_type!r}")
        if self.metric_type not in ("COSINE", "L2", "IP", "BM25"):
            raise ValueError(
                f"metric_type must be one of COSINE/L2/IP/BM25, got {self.metric_type!r}"
            )
        if not isinstance(self.build_params, (dict, MappingProxyType)):
            raise TypeError(f"build_params must be a dict, got {type(self.build_params).__name__}")
        if not isinstance(self.search_params, (dict, MappingProxyType)):
            raise TypeError(f"search_params must be a dict, got {type(self.search_params).__name__}")
        # Freeze mutable dicts to prevent accidental mutation of shared state
        if isinstance(self.build_params, dict):
            object.__setattr__(self, "build_params", MappingProxyType(self.build_params))
        if isinstance(self.search_params, dict):
            object.__setattr__(self, "search_params", MappingProxyType(self.search_params))

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict for Manifest persistence."""
        return {
            "field_name": self.field_name,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "build_params": dict(self.build_params),
            "search_params": dict(self.search_params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IndexSpec":
        """Inverse of to_dict; tolerates missing optional fields."""
        return cls(
            field_name=d["field_name"],
            index_type=d["index_type"],
            metric_type=d["metric_type"],
            build_params=dict(d.get("build_params") or {}),
            search_params=dict(d.get("search_params") or {}),
        )
