# Deep Design: Vector Index Subsystem (Phase 9)

## 1. Overview

MilvusLite Phase 9 introduces vector indexing, upgrading the retrieval path of `Collection.search` from NumPy brute-force scanning to ANN (Approximate Nearest Neighbor) retrieval. **The default implementation is FAISS HNSW**, while BruteForceIndex is retained as a differential baseline + dependency-free fallback.

**Why now**:
- Phase 8's scalar filter system established the `bitmap pipeline + filter_mask` abstraction, and FAISS's `IDSelectorBitmap` is naturally isomorphic to it — Phase 9 connects this entire pipeline in one shot
- The project is positioned as a "local version of Milvus," and the core demand from pymilvus users is "fast top-k." A "local Milvus" without ANN indexing gives early users the wrong first impression
- The storage layer (MVP.md §10) reserved invariants for index integration from the start: **data files are immutable + contain no delete markers**, enabling indexes to be "built once, never modified"

**Why FAISS instead of hnswlib / USearch**:
- `IDSelectorBitmap` naturally aligns with Phase 8's `valid_mask`, natively supporting pre-filter (not post-filter)
- The index family aligns with Milvus (HNSW / IVF_FLAT / IVF_SQ8 / IVF_PQ are all FAISS names), so users migrating from pymilvus don't need to change `index_params`
- Best extensibility for the future (quantized indexes, hybrid indexes, GPU paths)
- Risk: macOS arm64 wheel has historically had issues (faiss-cpu 1.7.4+ is stable now), mitigated via "FAISS is an optional extra + BruteForce fallback"

---

## 2. Architecture Decisions

### 2.1 Index Binding Level: Segment-level (Decided)

**Decision: Each data Parquet file corresponds to an independent VectorIndex file, bound 1:1.**

Candidate approach comparison:

| Dimension | Segment-level (Selected) | Collection-level |
|---|---|---|
| Alignment with LSM immutable architecture | Perfect — segment is immutable → index is also immutable → never need to "delete a vector" | Poor — global graph must "incrementally update + delete," but FAISS HNSW doesn't support true deletion |
| Incremental update cost | Very low — only need to build one new index after a new segment flush | Medium — each flush requires add_items; HNSW supports it but requires locks + possible resize |
| Coordination with compaction | Natural — old segment deleted → old index file deleted; new segment written → new index built. 1:1 correspondence | Painful — after deleting N segments, must remove N batches of pks from the global graph; HNSW can only mark-deleted, space isn't reclaimed, recall drifts; ultimately forced to periodically "fully rebuild the global index" |
| Recall | Slightly lower than global graph with many segments ("per-segment top-k then merge"), actual loss < 5% | Theoretically optimal |
| Memory usage | Slightly higher (each segment's graph duplicates some auxiliary data) | Slightly lower |
| Coexistence with brute-force fallback | Natural — small segments don't build an index, search uses brute-force directly; other segments use the index | Unnatural — memtable always bypasses the global graph |
| Consistency with actual Milvus architecture | ✅ Milvus itself builds indexes per-segment | ✗ |

**Decision rationale**:
1. The LSM-Tree's immutable segment invariant is the project's fundamental architectural advantage; segment-level index means Phase 9 introduces no new mutable state
2. FAISS HNSW doesn't support true deletion, so the global index approach would be constrained by this limitation; segment-level perfectly avoids it
3. The "local version of Milvus" positioning favors mirroring Milvus's own architectural decisions

### 2.2 Index Library Selection: FAISS-cpu (Decided)

| Dimension | hnswlib | **FAISS-cpu** (Selected) | USearch |
|---|---|---|---|
| Dependency size | ~2 MB | ~30-80 MB (macOS arm64 wheel is mature) | ~3 MB |
| Index types | HNSW only | HNSW / IVF_FLAT / IVF_SQ8 / IVF_PQ / OPQ / Flat / ... | HNSW only |
| Pre-filter (IDSelector) | No callback support | **`IDSelectorBitmap` natively supported, isomorphic with bitmap pipeline** | Supported but less mature than FAISS |
| Metric | cosine / l2 / ip | Complete | Complete |
| Maintenance activity | Low (author has been less responsive since 2023) | High (Meta official) | High (unum-cloud) |
| Semantic alignment with Milvus | Partial | **Complete — Milvus index_params directly reusable** | Partial |
| Phase 9 MVP pick | ❌ | ✅ | ❌ |

**Risks and mitigations**:
- **macOS arm64 wheel**: Stable as of late 2024, but installation failure requires downgrade — `try: import faiss` failure auto-falls back to BruteForceIndex
- **HNSW doesn't support true deletion**: Architecturally avoided via segment-level + immutable design
- **FAISS is slower than brute-force on small segments**: Segments below `INDEX_BUILD_THRESHOLD` (default 10000 rows) don't build an index; search uses brute-force

### 2.3 Dual Role of BruteForceIndex (Decided)

`BruteForceIndex` is not a temporary placeholder implementation, but a **long-term retained** first-class citizen:

1. **Zero-dependency fallback**: Users who don't install faiss-cpu can still use MilvusLite (limited performance but full functionality)
2. **Differential test baseline**: `tests/index/test_index_differential.py` uses BruteForceIndex as ground truth to verify FaissHnswIndex recall@10 ≥ 0.95
3. **Actual choice for small segments**: Segments below the threshold actually use it

The design cost of this decision: the `VectorIndex` protocol must be generic enough to accommodate both brute-force and ANN paradigms. The protocol design should first satisfy brute-force (simplest), then let the ANN implementation "stretch" the interface.

### 2.4 load / release State Machine (Decided to Introduce in Phase 9.3)

**Decision: Introduce an explicit `_load_state` state machine in Phase 9.3**, rather than waiting for the Phase 10 gRPC adapter layer.

States:

```
                  ┌──────────┐
                  │ released │  ◄── initial / Collection just opened / explicit release()
                  └─────┬────┘
                        │ load()
                        ▼
                  ┌──────────┐
                  │ loading  │  ◄── building/loading index files in progress
                  └─────┬────┘
                        │ all segments ready
                        ▼
                  ┌──────────┐
                  │  loaded  │  ◄── search available
                  └──────────┘
```

**Behavior**:
- `Collection.search` / `query` / `get` raise `CollectionNotLoadedError` when not in `loaded` state
- `Collection.insert` / `delete` don't require loaded state (write path doesn't depend on indexes)
- After restart, defaults to `released`; must explicitly call `load()` (aligned with Milvus behavior)
- Collections without IndexSpec are also allowed to `load()` — `load_state` still enters `loaded`, but segments don't build any index; search uses brute-force (aligned with Milvus's "collection without index can still load + search" behavior)

**Why not defer to Phase 10**:
- pymilvus users are already accustomed to `load_collection / release_collection`; Phase 10 must have something to map to
- The state machine itself is only a few dozen lines of code, but defining the semantics early has a decisive impact on how the Phase 10 servicer is written
- Phase 9.4's index persistence and load mechanism are tightly coupled; splitting into two phases would actually be more trouble

---

## 3. VectorIndex Abstraction

### 3.1 Protocol Definition

```python
# milvus_lite/index/protocol.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

class VectorIndex(ABC):
    """Abstract interface for any per-segment vector index implementation.

    Implementations: BruteForceIndex (NumPy), FaissHnswIndex (FAISS HNSW),
    FaissIvfFlatIndex, FaissIvfSq8Index, FaissHnswSqIndex,
    SparseInvertedIndex (BM25 full-text search)

    Lifetime: build → save → load → search → close. After close, all
    methods raise. Indexes are immutable — there is no add/remove after
    build. Compaction creates a new index for the merged segment instead.
    """

    metric: str    # "COSINE" | "L2" | "IP"
    num_vectors: int
    dim: int

    @classmethod
    @abstractmethod
    def build(
        cls,
        vectors: np.ndarray,        # (N, dim) float32
        metric: str,                # "COSINE" | "L2" | "IP"
        params: dict,               # implementation-specific
    ) -> "VectorIndex":
        """Construct a fresh index from a set of vectors. The local id
        of each vector is its row index 0..N-1; mapping back to pk is
        the Segment's responsibility, not the index's."""

    @abstractmethod
    def search(
        self,
        queries: np.ndarray,            # (nq, dim) float32
        top_k: int,
        valid_mask: Optional[np.ndarray] = None,  # (num_vectors,) bool
        params: Optional[dict] = None,  # impl-specific (efSearch, nprobe, ...)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (local_ids, distances), each shape (nq, top_k).

        valid_mask is the bitmap pipeline output AFTER dedup + tombstone
        + scalar filter. The index uses it to skip excluded rows DURING
        search (not after — this is the whole point of using FAISS
        IDSelectorBitmap instead of post-filtering).

        distances are returned in the canonical "smaller is more similar"
        convention regardless of metric:
            - L2:    raw L2 distance (NOT squared L2 like FAISS internal)
            - IP:    -dot(q, v)  (negated so smaller = more similar)
            - COSINE: 1 - dot(q_norm, v_norm)
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk. Format is implementation-specific."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, metric: str, dim: int) -> "VectorIndex":
        """Reload a previously saved index from disk."""

    @property
    @abstractmethod
    def index_type(self) -> str:
        """A string tag like 'BRUTE_FORCE' / 'HNSW' / 'IVF_FLAT'."""
```

**Key design points**:

1. **local_id is only valid within a segment**: The Index doesn't know or care what the pk is. The Segment translates local_id back to pk via its own `pks` array. This ensures the index implementation is completely schema-agnostic and pk-type-agnostic.
2. **Distance normalization is done inside the index**: FAISS L2 returns squared L2, IP returns larger-is-more-similar, which doesn't match our `compute_distances` convention. Normalization is done inside `FaissHnswIndex.search`, ensuring the upper layer sees distance semantics identical to brute-force — this is a prerequisite for differential tests to work.
3. **valid_mask is a search parameter, not a build parameter**: Because the mask depends on runtime delta_index and filter_mask. FAISS consumes the mask during the search path via `IDSelectorBitmap`; brute-force takes a subset via `vectors[mask]` before computing distances.
4. **No add / remove interface**: Immutability is part of the contract. Any need to "modify an index" is fulfilled by "discard old segment + build new segment + build new index."

### 3.2 IndexSpec

```python
# milvus_lite/index/spec.py

from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass(frozen=True)
class IndexSpec:
    """Persisted on the Collection (via Manifest) and used by every
    Segment to decide what kind of index to build.

    Mirrors Milvus's IndexParams structure for direct pymilvus mapping.
    """
    field_name: str          # which vector field this index covers
    index_type: str          # "BRUTE_FORCE" | "HNSW" | "IVF_FLAT" | ...
    metric_type: str         # "COSINE" | "L2" | "IP"
    build_params: Dict       # impl-specific: {"M": 16, "efConstruction": 200}
    search_params: Dict = field(default_factory=dict)  # impl-specific defaults: {"ef": 64}

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "IndexSpec": ...
```

**Why frozen**: Consistent with `CompiledExpr` / `FieldSchema` — Manifest persistence + cross-segment sharing + hash safety.

**Why build_params is dict rather than typed**: Different index_types have vastly different parameters (HNSW has M / efConstruction, IVF has nlist); typed would cause `Union[HnswParams, IvfParams, ...]` type bloat. dict + internal validation within each impl is a lighter choice, directly aligned with Milvus proto's `KeyValuePair` representation.

---

## 4. Integration Points with Existing Code

### 4.1 Segment Changes

```python
# milvus_lite/storage/segment.py

class Segment:
    __slots__ = (
        ..., "index",   # new
    )

    def __init__(self, ...):
        ...
        self.index: Optional[VectorIndex] = None

    def attach_index(self, index: "VectorIndex") -> None:
        """Attach a built or loaded index. Idempotent — replaces any
        existing index. Used by build_or_load_index."""
        self.index = index

    def release_index(self) -> None:
        """Drop the index reference. Memory freed when GC collects."""
        self.index = None

    def build_or_load_index(
        self,
        spec: "IndexSpec",
        index_dir: str,
    ) -> None:
        """Try to load index from disk; build + save if not found.

        Called by:
            - Collection.load() for every existing segment
            - flush.execute_flush() for newly created segments (if
              Collection is in 'loaded' state)
            - compaction.run_compaction() for the merged segment
        """
        path = self._index_file_path(index_dir, spec.index_type)
        if os.path.exists(path):
            self.index = build_index_from_factory(...).load(path, spec.metric_type, self.vector_dim)
            return
        # Build from scratch
        self.index = factory.build_index_from_spec(spec, self.vectors)
        self.index.save(path)

    def _index_file_path(self, index_dir: str, index_type: str) -> str:
        """Convention: indexes/<data_filename_stem>.<index_type>.idx"""
        stem = os.path.splitext(os.path.basename(self.file_path))[0]
        return os.path.join(index_dir, f"{stem}.{index_type.lower()}.idx")
```

**Key invariant**: `Segment.file_path` and `Segment.index's on-disk path` have a 1:1 correspondence. Any segment deletion must synchronously delete the corresponding index file, and vice versa.

### 4.2 Collection Changes

```python
# milvus_lite/engine/collection.py

class Collection:
    def __init__(self, ...):
        ...
        self._index_spec: Optional[IndexSpec] = self._manifest.index_spec
        self._load_state: Literal["released", "loading", "loaded"] = "released"

    # ── new public API ───────────────────────────────────────────

    def create_index(self, field_name: str, index_params: dict) -> None:
        """Persist an IndexSpec on the manifest. Does NOT actually build
        any index — that happens at load() time. Mirrors Milvus behavior.

        Raises:
            IndexAlreadyExistsError: if create_index already called
            FilterFieldError: if field_name is not a vector field
        """
        if self._index_spec is not None:
            raise IndexAlreadyExistsError(...)
        spec = IndexSpec(
            field_name=field_name,
            index_type=index_params["index_type"],
            metric_type=index_params["metric_type"],
            build_params=index_params.get("params", {}),
            search_params=index_params.get("search_params", {}),
        )
        self._index_spec = spec
        self._manifest.set_index_spec(spec)
        self._manifest.save()

    def drop_index(self, field_name: str) -> None:
        """Remove the IndexSpec and all on-disk .idx files."""
        if self._index_spec is None:
            return
        # Release in-memory indexes
        for seg in self._segment_cache.values():
            seg.release_index()
        # Delete .idx files
        for seg in self._segment_cache.values():
            path = seg._index_file_path(self._index_dir(seg.partition), self._index_spec.index_type)
            if os.path.exists(path):
                os.remove(path)
        self._index_spec = None
        self._manifest.set_index_spec(None)
        self._manifest.save()
        self._load_state = "released"

    def load(self) -> None:
        """Build or load all segment indexes. Required before search."""
        if self._load_state == "loaded":
            return
        self._load_state = "loading"
        try:
            if self._index_spec is not None:
                for seg in self._segment_cache.values():
                    if seg.index is None:
                        index_dir = self._index_dir(seg.partition)
                        os.makedirs(index_dir, exist_ok=True)
                        seg.build_or_load_index(self._index_spec, index_dir)
            self._load_state = "loaded"
        except Exception:
            self._load_state = "released"
            raise

    def release(self) -> None:
        """Drop all in-memory indexes. Subsequent search() raises."""
        for seg in self._segment_cache.values():
            seg.release_index()
        self._load_state = "released"

    def has_index(self) -> bool:
        return self._index_spec is not None

    def get_index_info(self) -> Optional[dict]:
        return self._index_spec.to_dict() if self._index_spec else None

    # ── search guard ─────────────────────────────────────────────

    def search(self, ...):
        if self._load_state != "loaded":
            raise CollectionNotLoadedError(self.name)
        ...
```

### 4.3 Search Executor Changes

A new `execute_search_with_index` function is added, **coexisting with the original `execute_search`**. Collection.search selects the path based on `_load_state`:

- Always uses the new path `execute_search_with_index`
- Inside the new path, recall is done per-segment; each segment checks whether it has an index: if yes, use the index; if no, use brute-force

```python
# milvus_lite/search/executor_indexed.py  (or extend executor.py)

def execute_search_with_index(
    query_vectors: np.ndarray,
    segments: List["Segment"],
    memtable: "MemTable",
    delta_index: "DeltaIndex",
    top_k: int,
    metric_type: str,
    pk_field: str,
    vector_field: str,
    compiled_filter: Optional["CompiledExpr"] = None,
    partition_names: Optional[List[str]] = None,
) -> List[List[dict]]:
    """Per-segment recall + global merge.

    Algorithm:
        1. For each segment (filtered by partition):
            a. Build per-segment valid_mask via bitmap (dedup intra-segment +
               tombstone + scalar filter mask for THIS segment's rows)
            b. If segment.index is not None: index.search(q, top_k, valid_mask)
               Else: brute-force on segment.vectors[valid_indices]
            c. Translate local_ids back to pks via segment.pks
            d. Collect (pk, distance, segment, row_idx) tuples
        2. Process memtable similarly (always brute-force; no index there)
        3. Global dedup by pk (keep max-seq) — different segments may
           have the same pk due to upsert
        4. Global top-k by distance
        5. Materialize result dicts via segment.row_to_dict / memtable record
    """
    ...
```

**Key complexity points**:
- **Global merge after per-segment top-k**: Each segment recalls `top_k` results, then global top-k is selected from `N_segments * top_k + memtable_topk` candidates. Theoretically, "taking k per segment" may lose some recall (if a segment actually has k+1 nearest neighbors for the query); in practice, the impact is very small when N_segments is small. Whether to "take 2k per segment" is a tunable parameter for Phase 9.5.
- **Cross-segment dedup**: In upsert scenarios, the same pk may appear in multiple segments, requiring dedup by max-seq. This step is handled by `bitmap.build_valid_mask` in the original `execute_search`; the new path needs to redo this during the merge phase.
- **Per-segment slicing of valid_mask**: The original `filter_mask` was globally concatenated; the new path needs to slice it back per-segment (the assembler can provide "per-segment filter mask list" instead of the merged version).

### 4.4 Flush / Compaction Hooks

```python
# milvus_lite/engine/flush.py — Step 8 (new)

def execute_flush(collection):
    ... # Steps 1-7 unchanged

    # Step 8: build indexes for newly created segments if collection is loaded
    if collection._load_state == "loaded" and collection._index_spec is not None:
        for new_segment in newly_added_segments:
            index_dir = collection._index_dir(new_segment.partition)
            os.makedirs(index_dir, exist_ok=True)
            new_segment.build_or_load_index(collection._index_spec, index_dir)
```

```python
# milvus_lite/engine/compaction.py — at end of run_compaction

def run_compaction(collection, ...):
    ...
    # After: new merged segment created, old segments dropped
    # Delete old .idx files for the dropped segments
    for old_seg in dropped_segments:
        old_idx_path = old_seg._index_file_path(...)
        if os.path.exists(old_idx_path):
            os.remove(old_idx_path)
    # Build new index for merged segment if loaded
    if collection._load_state == "loaded" and collection._index_spec is not None:
        index_dir = collection._index_dir(merged_segment.partition)
        merged_segment.build_or_load_index(collection._index_spec, index_dir)
```

### 4.5 Recovery Changes

```python
# milvus_lite/engine/recovery.py

def recover(collection):
    ... # Steps 1-5 unchanged

    # Step 6 (new): default load_state is "released"
    # Even if the manifest has an index_spec, segments are NOT loaded
    # automatically — caller must explicitly call collection.load().
    # This matches Milvus behavior and avoids surprise startup latency.
    collection._load_state = "released"

    # Orphan cleanup also covers .idx files
    _cleanup_orphan_index_files(collection)
```

### 4.6 Manifest Schema Bump

```python
# milvus_lite/storage/manifest.py

# Bump format_version from 1 to 2 to add index_spec support.
# Backward-compat: old manifests without index_spec field load with
# index_spec=None.

@dataclass
class ManifestState:
    ...
    index_spec: Optional[IndexSpec] = None  # NEW
    format_version: int = 2                  # bumped from 1

def to_dict(self) -> dict:
    return {
        ...,
        "index_spec": self.index_spec.to_dict() if self.index_spec else None,
        "format_version": self.format_version,
    }

@classmethod
def from_dict(cls, d: dict) -> "ManifestState":
    fv = d.get("format_version", 1)
    spec_dict = d.get("index_spec")
    spec = IndexSpec.from_dict(spec_dict) if spec_dict else None
    return cls(..., index_spec=spec, format_version=2)
```

**Compatibility strategy**: When loading an old v1 manifest, the `index_spec` field is missing → defaults to None; next save upgrades to v2. No migration tool needed.

---

## 5. Directory Layout

```
data_dir/
└── collections/
    └── <collection_name>/
        ├── schema.json
        ├── manifest.json          # contains index_spec
        ├── manifest.json.prev
        ├── wal/
        │   ├── data_*.arrow
        │   └── delta_*.arrow
        └── partitions/
            └── <partition_name>/
                ├── data/
                │   ├── data_000001_000500.parquet
                │   └── data_000501_001000.parquet
                ├── delta/
                │   └── delta_000501_000503.parquet
                └── indexes/                                  ← Added in Phase 9
                    ├── data_000001_000500.brute_force.idx
                    └── data_000501_001000.hnsw.idx
```

**Naming convention**: `<data_filename_stem>.<index_type_lowercase>.idx`

**Strict invariants**:
1. A segment's .idx filename is uniquely determined by the segment filename + the current IndexSpec's index_type
2. When compaction deletes a segment, the .idx is deleted simultaneously; when writing a new segment, the new .idx is written simultaneously
3. At recovery startup, the indexes/ directory is scanned; orphan .idx files (whose corresponding segment file no longer exists) are unconditionally deleted

---

## 6. FAISS Integration Pitfalls

### 6.1 Metric Sign Alignment (Biggest Pitfall)

| Metric | FAISS Internal Convention | MilvusLite Upper-Layer Convention | Conversion |
|---|---|---|---|
| L2 | squared L2 (smaller is more similar) | raw L2 (smaller is more similar) | `dist = sqrt(faiss_dist)` |
| IP | dot product (larger is more similar) | -dot (smaller is more similar) | `dist = -faiss_dist` |
| COSINE | Equivalent to "IP after L2-normalizing vectors" | `1 - cosine_sim` | `vectors_norm = normalize(vectors); query_norm = normalize(query); dist = 1 - faiss_ip(query_norm, vectors_norm)` |

**Implementation location**: Distance post-processing in `FaissHnswIndex.search`, ensuring the distances returned to the executor are fully consistent with brute-force.

**Testing method**: Differential testing — build two indexes (brute / faiss) on the same data, verify distance error < 1e-3 (for recall-matched pks) across 100 random queries.

### 6.2 IDSelectorBitmap Byte Alignment

```python
# faiss.IDSelectorBitmap accepts bit-packed uint8 arrays
# numpy bool arrays cannot be passed directly; packbits is needed

import faiss
import numpy as np

mask_bool = np.array([True, False, True, ...], dtype=bool)
mask_packed = np.packbits(mask_bool, bitorder='little')  # uint8 array
selector = faiss.IDSelectorBitmap(num_vectors, faiss.swig_ptr(mask_packed))
params = faiss.SearchParametersHNSW(sel=selector)
D, I = index.search(queries, top_k, params=params)
```

Note that `bitorder='little'` must be explicitly specified (FAISS expects LSB-first packing). This detail requires unit test coverage to debug properly.

### 6.3 FAISS HNSW Doesn't Need Training, but IVF Does

```python
# HNSW: just add
index = faiss.IndexHNSWFlat(dim, M)
index.add(vectors)

# IVF: train first, then add
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.train(vectors[:training_subset])
index.add(vectors)
```

Phase 9 MVP focuses on HNSW; IVF_FLAT, IVF_SQ8, and HNSW_SQ have been implemented in subsequent iterations (see §11 for the list of supported indexes).

### 6.4 Persistence

```python
# Save
faiss.write_index(self._index, path)

# Load
loaded = faiss.read_index(path)
```

Note that FAISS's write_index/read_index uses a C++ serialization format, not numpy. One index file = one FAISS object.

### 6.5 macOS arm64 Wheel

The latest `faiss-cpu>=1.7.4` has macOS arm64 wheels on PyPI. However, it is still recommended to:

```toml
# pyproject.toml
[project.optional-dependencies]
faiss = ["faiss-cpu>=1.7.4"]
```

And use the try-import pattern in `milvus_lite/index/factory.py`:

```python
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

def build_index_from_spec(spec: IndexSpec, vectors: np.ndarray) -> VectorIndex:
    if spec.index_type in ("HNSW", "IVF_FLAT", ...):
        if not _FAISS_AVAILABLE:
            raise IndexBackendUnavailableError(
                f"index_type={spec.index_type} requires faiss-cpu; "
                "install with `pip install faiss-cpu`"
            )
        ...
    elif spec.index_type == "BRUTE_FORCE":
        return BruteForceIndex.build(vectors, spec.metric_type, spec.build_params)
```

---

## 7. Lifecycle Sequence Diagrams

### 7.1 Normal Write + Search (Collection Already Loaded)

```
client                Collection           flush.py        Segment      VectorIndex
  │                       │                    │              │              │
  ├─ insert(records) ─────►                    │              │              │
  │                       ├─ memtable.append ──┤              │              │
  │                       │  (memtable full)   │              │              │
  │                       ├─ flush() ──────────►              │              │
  │                       │                    ├─ write data parquet ────────►
  │                       │                    ├─ load Segment ──────────────►
  │                       │                    ├─ if loaded:                 │
  │                       │                    │   build_or_load_index ──────►
  │                       │                    │                             ├─ FaissHnsw.build
  │                       │                    │                             ├─ save .idx
  │                       │                    ◄──────────── attach ─────────┤
  │                       ◄────────────────────┤              │              │
  │                                                                          │
  ├─ search([q], top_k) ──►                                                  │
  │                       ├─ assemble per-segment masks ──────►              │
  │                       │                                  │              │
  │                       ├─ for each segment:                              │
  │                       │   if has index: ──────────────────────────────►
  │                       │                                                ├─ search(q, mask)
  │                       │   else: brute force                            │
  │                       ├─ merge top-k from all segments + memtable      │
  ◄───────────────────────┤                                                │
```

### 7.2 Restart + Explicit Load

```
client                Collection           recovery        Manifest      Segment    VectorIndex
  │                       │                    │              │             │            │
  ├─ open(data_dir) ──────►                    │              │             │            │
  │                       ├─ recover() ────────►              │             │            │
  │                       │                    ├─ load manifest (incl. index_spec)       │
  │                       │                    ├─ replay WAL → memtable                  │
  │                       │                    ├─ load Segments (no index)               │
  │                       │                    ├─ load_state = released                  │
  │                       ◄────────────────────┤              │             │            │
  │                                                                                       │
  ├─ search(...) ─────────►                                                              │
  │                       ├─ raise CollectionNotLoadedError                              │
  ◄───────────────────────┤                                                              │
  │                                                                                       │
  ├─ load() ──────────────►                                                              │
  │                       ├─ load_state = loading                                        │
  │                       ├─ for each segment:                                           │
  │                       │   build_or_load_index ────────────────────────►              │
  │                       │                                                ├─ load .idx (fast) │
  │                       │                                                │   OR build (slow)│
  │                       ├─ load_state = loaded                                         │
  ◄───────────────────────┤                                                              │
```

### 7.3 Compaction Sequence

```
flush.py            compaction.py         Segment(old)    Segment(new)   VectorIndex
  │                       │                    │               │              │
  │                       ├─ pick small files ─►               │              │
  │                       ├─ merge & dedup ───────────────────►               │
  │                       ├─ write new parquet                                │
  │                       ├─ load new Segment ────────────────►               │
  │                       ├─ remove old segments ──────────────►              │
  │                       │   (also rm old .idx files)         │              │
  │                       ├─ if loaded:                                       │
  │                       │   build_or_load_index ─────────────────────────────►
  │                       │                                                   ├─ build + save
  │                       ├─ manifest.swap                                    │
```

---

## 8. Recall Validation Strategy

### 8.1 Differential Test Structure

`tests/index/test_index_differential.py`:

```python
@pytest.mark.parametrize("dim", [4, 32, 128])
@pytest.mark.parametrize("n", [100, 10_000])
@pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
def test_faiss_hnsw_recall_vs_brute_force(dim, n, metric):
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    queries = np.random.randn(20, dim).astype(np.float32)

    brute = BruteForceIndex.build(vectors, metric, {})
    faiss_idx = FaissHnswIndex.build(vectors, metric, {"M": 16, "efConstruction": 200})

    brute_ids, brute_dists = brute.search(queries, top_k=10)
    faiss_ids, faiss_dists = faiss_idx.search(queries, top_k=10, params={"ef": 64})

    # 1) recall@10 ≥ 0.95
    for i in range(20):
        overlap = len(set(brute_ids[i]) & set(faiss_ids[i]))
        assert overlap / 10 >= 0.95, \
            f"recall@10 = {overlap/10} for query {i}, dim={dim}, metric={metric}"

    # 2) distance value parity for hits — within 1e-3 relative error
    for i in range(20):
        faiss_id_to_dist = dict(zip(faiss_ids[i], faiss_dists[i]))
        brute_id_to_dist = dict(zip(brute_ids[i], brute_dists[i]))
        for pid, fdist in faiss_id_to_dist.items():
            if pid in brute_id_to_dist:
                bdist = brute_id_to_dist[pid]
                assert abs(fdist - bdist) < 1e-3 + 1e-3 * abs(bdist), \
                    f"distance mismatch for pk={pid}: faiss={fdist} brute={bdist}"
```

**Why this setup**:
- BruteForce is the mathematical ground truth, FAISS is the object under test — symmetric with Phase 8's differential test structure
- recall@10 ≥ 0.95 is the typical level for HNSW with reasonable parameters
- Distance value parity verifies that metric sign alignment is correct (once the metric conversion is wrong, the second assertion will definitely fail)
- Three metrics × three dims × two scales = 18 cases, fast to run but broad coverage

### 8.2 End-to-End Search Path Differential

`tests/engine/test_search_index_vs_brute.py`:

```python
def test_collection_search_index_path_matches_brute_force(tmp_path):
    """The full Collection.search going through index path returns the
    same top-k as if we forced brute-force everywhere."""
    db = MilvusLite(str(tmp_path))
    col = db.create_collection("test", schema=...)
    col.insert([...])  # 1000 records
    col.flush()

    # Path 1: brute-force
    col.create_index("vec", {"index_type": "BRUTE_FORCE", "metric_type": "COSINE"})
    col.load()
    brute_results = col.search(query_vectors=[[...]], top_k=10)

    # Path 2: HNSW
    col.release()
    col.drop_index("vec")
    col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16}})
    col.load()
    hnsw_results = col.search(query_vectors=[[...]], top_k=10)

    # Top-1 must match (HNSW recall@1 is essentially 1.0 for n=1000)
    assert brute_results[0][0]["id"] == hnsw_results[0][0]["id"]
    # Top-10 set overlap ≥ 9/10
    brute_ids = {r["id"] for r in brute_results[0]}
    hnsw_ids = {r["id"] for r in hnsw_results[0]}
    assert len(brute_ids & hnsw_ids) >= 9
```

---

## 9. Phase 9 Sub-phase Breakdown

| Sub-phase | Content | Completion Criteria | Effort |
|---|---|---|---|
| **9.1** | Fill in prerequisite APIs for pymilvus quickstart: `Collection.create_partition / drop_partition / list_partitions / num_entities / describe` + `search(output_fields=...)` + `MilvusLite.get_collection_stats` | 6 new methods + full test coverage; no index concepts introduced | S |
| **9.2** | `VectorIndex` protocol + `BruteForceIndex` + integrate with `Segment.index` + new `execute_search_with_index` path (still using brute-force implementation) | All old search tests pass on the new path; differential test brute-force-via-index ≡ old execute_search | M |
| **9.3** | `IndexSpec` + `Manifest` v2 upgrade + `Collection.create_index / drop_index / load / release / has_index / get_index_info` + `_load_state` state machine + `CollectionNotLoadedError` | Full chain `col.create_index → col.load → col.search → col.release → col.search raise` passes; manifest v1→v2 compatibility test passes | M |
| **9.4** | Index file persistence (`indexes/<stem>.<type>.idx`) + flush / compaction / recovery hooks + orphan .idx cleanup | Collection restart → load → search is equivalent; no orphan .idx after compaction; crash injection test passes | M |
| **9.5** | `FaissHnswIndex` + factory routing + metric alignment + `IDSelectorBitmap` integration + differential tests + benchmark | 100K vector search QPS ≥50x higher than brute-force; recall@10 ≥ 0.95; macOS arm64 + Linux dual-platform CI passes | L |
| **9.6** | `examples/m9_demo.py` + long-running tests + Phase 9 documentation backfill | m9 demo passes; `@pytest.mark.slow` 100K test passes; `plan/index-design.md` aligned with final code | S |

Total: 2S + 3M + 1L

---

## 10. Key Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| FAISS metric sign alignment implemented incorrectly | High | High (wrong search results) | Differential test is a mandatory gate; metric error will definitely fail |
| FAISS macOS arm64 wheel installation failure | Medium | Medium | Optional extra + BruteForce fallback; CI runs dual matrix |
| `IDSelectorBitmap` packbits order wrong | Medium | High | Unit test coverage for various mask patterns |
| Segment-level top-k merge loses recall | Low | Low | Per-segment takes `2*top_k` candidates; tunable parameter |
| Synchronous index build at end of flush slows writes | High | Medium (UX) | Accept MVP behavior; future: move index build out of flush synchronous path |
| Orphan .idx files after compaction | Medium | Low | recovery startup runs cleanup_orphan_index_files |
| Manifest v1 → v2 compatibility failure | Low | High | Explicit test coverage: old manifest file → new code reads → upgrade and save |
| search called during multi-threaded load | Low | Medium | search throws error when `_load_state == "loading"`; load is not concurrent |

---

## 11. Supported Index Types and Future Extensions

### Supported

| Index Type | Implementation File | Description |
|---|---|---|
| BRUTE_FORCE | `brute_force.py` | NumPy brute-force scan, zero-dependency fallback + differential test baseline + default for small segments |
| HNSW | `faiss_hnsw.py` | FAISS HNSW + IDSelectorBitmap pre-filter |
| IVF_FLAT | `faiss_ivf_flat.py` | FAISS IVF-Flat, requires training step |
| IVF_SQ8 | `faiss_ivf_sq8.py` | FAISS IVF + 8-bit scalar quantization, more memory efficient |
| HNSW_SQ | `faiss_hnsw_sq.py` | FAISS HNSW + scalar quantization variant |
| SPARSE_INVERTED_INDEX | `sparse_inverted.py` | Per-segment BM25 inverted index for full-text search (Phase 11) |

### Not in Current Scope

| Feature | Deferred To |
|---|---|
| IVF-PQ / OPQ and other advanced quantization indexes | Future |
| GPU acceleration (faiss-gpu) | Future |
| Vector quantization (int8, fp16, bf16, binary) | Future |
| Async index build (non-blocking flush) | Future |
| Index warmup / prefetch | Future |
| Multiple vector fields (multiple vector columns per Collection) | Future (Milvus also added this later) |
| Index parameter auto-tuning | Future |

---

## 12. Completion Criteria

- `col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})` can successfully persist to manifest
- After `col.load()`, `col.search([[...]], top_k=10)` goes through the FAISS path, with performance an order of magnitude higher than brute-force
- After `col.release()`, search raises `CollectionNotLoadedError`
- Restart process → `col.load()` completes in seconds (directly loads .idx files, no rebuilding)
- After compaction, old .idx files are automatically cleaned up and new .idx files are automatically built
- Differential test recall@10 ≥ 0.95 all green
- `examples/m9_demo.py` runs successfully
- Running `pytest` with all old tests + new tests all green
