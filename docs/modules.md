# MilvusLite Code Module Design

## 1. Top-Level Package Layout

```
milvus_lite/
├── schema/          # Data model and type system
├── storage/         # Storage layer: persistence + in-memory buffer
├── engine/          # Engine layer: core logic orchestration
├── search/          # Search layer: vector retrieval
├── index/           # Index layer (Phase 9): VectorIndex abstraction + BruteForce / FAISS / IVF / Sparse
├── analyzer/        # Analyzer layer (Phase 11): text analysis (BM25 tokenizer)
├── embedding/       # Embedding layer: vectorization model abstraction + OpenAI provider
├── rerank/          # Rerank layer: re-ranking model abstraction + Cohere provider
├── adapter/         # Adapter layer (Phase 10): gRPC -> engine protocol translation
├── db.py            # DB layer: multi-Collection lifecycle management (MilvusLite)
├── server_manager.py # pymilvus integration entry point (background gRPC server lifecycle)
├── constants.py     # Global constants
├── exceptions.py    # Exception hierarchy
└── __init__.py      # Public API exports
```

## 2. Complete Code Structure

```
lite-v2/
├── MVP.md
├── write-pipeline.md
├── research.md
├── modules.md
├── CLAUDE.md
│
├── milvus_lite/
│   ├── __init__.py                 # Public API exports
│   ├── constants.py                # Global constants
│   ├── exceptions.py               # Exception hierarchy
│   │
│   ├── schema/                     # == Data model and type system ==
│   │   ├── __init__.py             #   Exports: DataType, FieldSchema, CollectionSchema
│   │   ├── types.py                #   DataType enum, FieldSchema, CollectionSchema class definitions
│   │   ├── validation.py           #   validate_schema(), validate_record(), separate_dynamic_fields()
│   │   ├── arrow_builder.py        #   4 Arrow Schema builders (data/delta/wal_data/wal_delta)
│   │   └── persistence.py          #   schema.json read/write (save_schema / load_schema)
│   │
│   ├── storage/                    # == Storage layer ==
│   │   ├── __init__.py             #   Exports: WAL, MemTable, DataFile, DeltaFile, DeltaIndex, Manifest
│   │   ├── wal.py                  #   WAL (Arrow IPC Streaming, dual-file, write/read_operations/close, fsync)
│   │   ├── memtable.py             #   MemTable (RecordBatch list + pk_index + delete_index, seq-aware)
│   │   ├── data_file.py            #   Data Parquet stateless functions (read/write, naming, seq range parsing)
│   │   ├── delta_file.py           #   Delta Parquet stateless functions (read/write)
│   │   ├── delta_index.py          #   DeltaIndex (in-memory pk->max_delete_seq, gc_below)
│   │   └── manifest.py             #   Manifest (JSON, tmp+replace atomic, .prev backup, Partition file lists)
│   │
│   ├── engine/                     # == Engine layer ==
│   │   ├── __init__.py             #   Exports: Collection
│   │   ├── operation.py            #   InsertOp / DeleteOp / Operation Union (write orchestration abstraction layer)
│   │   ├── collection.py           #   Collection core (entry point, _seq allocation, insert/delete/get/search, _apply unified path, Partition CRUD)
│   │   ├── flush.py                #   Flush pipeline (7 steps, synchronous blocking)
│   │   ├── recovery.py             #   Crash recovery (5 steps, replay via WAL.read_operations by seq)
│   │   └── compaction.py           #   Compaction Manager (Size-Tiered + tombstone GC)
│   │
│   ├── search/                     # == Search layer ==
│   │   ├── __init__.py             #   Exports: execute_search
│   │   ├── bitmap.py               #   Bitmap pipeline (dedup + delete filtering + optional filter_mask)
│   │   ├── distance.py             #   Distance computation (cosine / L2 / inner product, NumPy implementation)
│   │   ├── assembler.py            #   Candidate assembly (segments + memtable -> numpy + optional filter_mask)
│   │   ├── executor.py             #   Search executor (collect data + bitmap + vector retrieval + top-k)
│   │   └── filter/                 # == Scalar filter subsystem (Phase 8) ==
│   │       ├── __init__.py         #   Exports: parse_expr, compile_expr, evaluate, FilterError
│   │       ├── exceptions.py       #   FilterError / FilterParseError / FilterFieldError / FilterTypeError
│   │       ├── tokens.py           #   TokenKind enum + Token + tokenize()
│   │       ├── ast.py              #   20 frozen AST nodes + Expr Union
│   │       ├── parser.py           #   Pratt parser (inspired by Milvus Plan.g4)
│   │       ├── semantic.py         #   compile_expr + type inference + field binding + backend selection
│   │       ├── cache.py            #   LRUCache (Phase F2c)
│   │       └── eval/
│   │           ├── __init__.py     #   evaluate() backend dispatcher (arrow / hybrid / python)
│   │           ├── arrow_backend.py #   pyarrow.compute backend (primary)
│   │           ├── hybrid_backend.py#   per-batch JSON preprocessing -> arrow (Phase F3+)
│   │           └── python_backend.py#   row-wise Python backend (differential baseline + hybrid fallback)
│   │
│   ├── index/                      # == Index layer (Phase 9) ==
│   │   ├── __init__.py             #   Exports: VectorIndex, BruteForceIndex, FaissHnswIndex, IndexSpec, build_index_from_spec
│   │   ├── protocol.py             #   VectorIndex ABC: build / search / save / load
│   │   ├── spec.py                 #   IndexSpec frozen dataclass
│   │   ├── brute_force.py          #   BruteForceIndex (NumPy, differential baseline + fallback)
│   │   ├── faiss_hnsw.py           #   FaissHnswIndex (FAISS HNSW + IDSelectorBitmap)
│   │   ├── faiss_ivf_flat.py       #   FaissIvfFlatIndex (FAISS IVF_FLAT)
│   │   ├── faiss_ivf_sq8.py        #   FaissIvfSq8Index (FAISS IVF_SQ8)
│   │   ├── faiss_hnsw_sq.py        #   FaissHnswSqIndex (FAISS HNSW_SQ — HNSW + scalar quantization)
│   │   ├── sparse_inverted.py      #   SparseInvertedIndex (sparse vector inverted index)
│   │   └── factory.py              #   build_index_from_spec / load_index + try-import faiss degradation
│   │
│   ├── analyzer/                   # == Analyzer layer (Phase 11) ==
│   │   ├── __init__.py             #   Exports: Analyzer, StandardAnalyzer, create_analyzer
│   │   ├── protocol.py             #   Analyzer ABC: tokenize
│   │   ├── standard.py             #   StandardAnalyzer (regex tokenizer)
│   │   ├── jieba_analyzer.py       #   JiebaAnalyzer (optional Chinese word segmentation)
│   │   ├── hash.py                 #   term_to_id hash function
│   │   ├── sparse.py               #   sparse vector utilities
│   │   └── factory.py              #   create_analyzer factory
│   │
│   ├── embedding/                  # == Embedding layer ==
│   │   ├── __init__.py
│   │   ├── protocol.py             #   EmbeddingProvider ABC
│   │   ├── openai_provider.py      #   OpenAI embedding provider
│   │   └── factory.py              #   create_embedding_provider factory
│   │
│   ├── rerank/                     # == Rerank layer ==
│   │   ├── __init__.py
│   │   ├── protocol.py             #   RerankProvider ABC
│   │   ├── cohere_provider.py      #   Cohere rerank provider
│   │   ├── decay.py                #   decay function
│   │   └── factory.py              #   create_rerank_provider factory
│   │
│   ├── adapter/                    # == Adapter layer (Phase 10) ==
│   │   └── grpc/                   # -- gRPC -> engine protocol translation --
│   │       ├── __init__.py
│   │       ├── server.py           #   run_server(data_dir, host, port)
│   │       ├── servicer.py         #   MilvusServicer — all RPC implementations
│   │       ├── errors.py           #   MilvusLiteError -> grpc Status mapping
│   │       ├── translators/
│   │       │   ├── schema.py       #   Milvus FieldSchema <-> MilvusLite FieldSchema
│   │       │   ├── records.py      #   FieldData (columnar) <-> list[dict] (row-wise) transposition
│   │       │   ├── search.py       #   SearchRequest parsing
│   │       │   ├── result.py       #   engine results -> SearchResults proto
│   │       │   ├── expr.py         #   Milvus filter pass-through + unsupported function detection
│   │       │   └── index.py        #   IndexParams <-> IndexSpec
│   │       └── proto/              #   Generated stubs (committed to repo)
│   │           ├── milvus_pb2.py
│   │           ├── milvus_pb2_grpc.py
│   │           ├── schema_pb2.py
│   │           ├── common_pb2.py
│   │           └── README.md       #   source commit reference
│   │
│   ├── db.py                       # == DB layer ==
│   │                                #   MilvusLite class (create/get/drop/list_collection, get_collection_stats, close)
│   ├── cmdline.py                   #   unified CLI: milvus-lite server / dump
│   └── server_manager.py           #   pymilvus integration entry point (ServerManager, server_manager_instance)
│
├── tests/
│   ├── conftest.py                 # Shared fixtures: temporary directories, example Schema, random vector generators
│   │
│   ├── schema/
│   │   ├── test_types.py           #   DataType / FieldSchema / CollectionSchema definitions
│   │   ├── test_validation.py      #   Schema validation rules, record validation, dynamic field separation
│   │   ├── test_arrow_builder.py   #   4 Arrow Schema builders, TYPE_MAP mapping
│   │   └── test_persistence.py     #   schema.json serialization/deserialization round-trip
│   │
│   ├── storage/
│   │   ├── test_wal.py             #   WAL write/recover round-trip, dual-file lifecycle, corruption handling, fsync
│   │   ├── test_memtable.py        #   apply_insert/apply_delete/get semantics, upsert, seq-aware out-of-order counterexample, flush Partition split
│   │   ├── test_data_file.py       #   Parquet read/write round-trip, file naming, seq range parsing
│   │   ├── test_delta_file.py      #   delta Parquet read/write round-trip
│   │   ├── test_delta_index.py     #   add_batch / is_deleted / gc_below / rebuild_from
│   │   └── test_manifest.py        #   load/save atomicity, .prev backup and fallback, Partition file list management
│   │
│   ├── engine/
│   │   ├── test_operation.py       #   InsertOp / DeleteOp construction, properties (seq_min/seq_max/num_rows)
│   │   ├── test_collection.py      #   Collection-level E2E: insert/delete/get/search, Partition CRUD, upsert
│   │   ├── test_flush.py           #   Flush pipeline end-to-end, 7 crash points (including fsync), recovery correctness
│   │   ├── test_recovery.py        #   Crash recovery 5 steps, WAL replay by seq, orphan file cleanup
│   │   └── test_compaction.py      #   File bucketing, merge dedup, delete filtering, Manifest update, tombstone GC
│   │
│   ├── search/
│   │   ├── test_bitmap.py          #   Bitmap construction: dedup + delete filtering + filter_mask
│   │   ├── test_distance.py        #   cosine / L2 / IP distance correctness
│   │   ├── test_executor.py        #   Search end-to-end, top-k, Partition Pruning
│   │   └── filter/                 #   -- Filter subsystem unit tests --
│   │       ├── test_tokens.py      #   Various literal lexing + keyword case sensitivity + error positions
│   │       ├── test_parser.py      #   Pratt precedence + parentheses + error recovery
│   │       ├── test_semantic.py    #   Field not found / type mismatch / did-you-mean
│   │       ├── test_arrow_backend.py    #   Each AST node -> correct BooleanArray
│   │       ├── test_python_backend.py   #   Same as above, row-level implementation comparison
│   │       └── test_e2e.py         #   Differential testing: arrow == python; hybrid == python
│   │
│   ├── index/                      #   -- Phase 9 index subsystem unit tests --
│   │   ├── test_brute_force_index.py    #   BruteForceIndex self-correctness
│   │   ├── test_faiss_hnsw.py           #   FaissHnswIndex (skipif faiss not available)
│   │   ├── test_faiss_id_selector.py    #   IDSelectorBitmap packbits order + boundary
│   │   ├── test_index_differential.py   #   recall@10 >= 0.95, distance value parity
│   │   └── test_index_persistence.py    #   .idx file save/load round-trip
│   │
│   ├── adapter/                    #   -- Phase 10 gRPC adapter layer tests --
│   │   ├── test_grpc_server_startup.py  #   server startup / shutdown / pymilvus.connect
│   │   ├── test_grpc_translators_schema.py
│   │   ├── test_grpc_translators_records.py  #   FieldData <-> records bidirectional round-trip
│   │   ├── test_grpc_translators_expr.py
│   │   ├── test_grpc_translators_index.py
│   │   ├── test_grpc_collection_lifecycle.py
│   │   ├── test_grpc_crud.py            #   insert / upsert / delete / query / get
│   │   ├── test_grpc_search.py
│   │   ├── test_grpc_index.py           #   create_index / load / release
│   │   ├── test_grpc_partition.py
│   │   ├── test_grpc_error_mapping.py
│   │   └── test_grpc_quickstart.py      #   L3 smoke test: pymilvus quickstart full flow
│   │
│   ├── test_db.py                  #   Multi-Collection lifecycle, close/cleanup
│   └── test_smoke_e2e.py           #   End-to-end smoke test via public API
│
├── pyproject.toml
└── requirements.txt
```

## Architecture Invariants (Core Constraints)

The following constraints apply across all modules and represent design consensus. They are not repeated in each section interface description. Any module implementation violating any of these is considered a bug.

**Correctness / Data Consistency:**

1. **`_seq` is the total order of operations**. All "override / discard / dedup" decisions must compare `_seq` and must not depend on call order or file physical order. This ensures that recovery out-of-order replay, compaction reordering, and future concurrency introduction will not break correctness.
2. **MemTable cross-clear must be seq-aware**. When put and delete clear each other's buffer entries for the same pk, they **must** first compare `_seq`; only when the current operation's `_seq` is larger does it override; otherwise the current operation is discarded. See section 9.7.
3. **Tombstone GC rule**: A `delta_index` entry `pk -> delete_seq` can be discarded if and only if there is no data file with `seq_min <= delete_seq` that contains that pk. MVP uses the conservative version: all tombstones below the global `min_active_data_seq` can be dropped. See sections 9.9 / 9.16.
4. **Files are immutable**. All disk files (data Parquet, delta Parquet, WAL Arrow, Manifest) can only be deleted as a whole once written; in-place modification is not allowed. This is the foundation of the LSM approach.
5. **Manifest is the single source of truth**, atomically updated (write-tmp + replace), with `manifest.json.prev` as a fallback against one serialization failure. See section 9.10.

**Concurrency Model:**

6. **MVP synchronous flush**. When `Collection.insert/delete` detects MemTable is full, it **blocks to execute** flush and returns only after flush completes. Async/background flush is deferred to the future and not included in MVP.
   - This decision affects the interface shape of MemTable / Collection / Search. Switching to async requires opening a new document to discuss lock/snapshot/RCU boundaries.
7. **Single writer per Collection**. Collection does not do internal locking; multi-threaded concurrent writes to the same Collection within one process is undefined behavior.
8. **Single process per data_dir**. `db.py` acquires `fcntl.flock(data_dir/LOCK)` at startup; if already held, it errors out immediately without waiting.

**Schema / Evolution:**

9. **Schema is immutable**. MVP does not support alter table; schema changes can only be done by creating a new Collection + reindex. This is the key simplification premise: the 4 Arrow Schemas and all historical Parquet files never need to consider compatibility.

**WAL / Durability:**

10. **WAL defaults to `sync_mode="close"`**: performs one `os.fsync` on the sink before `close_and_delete`. This covers the crash scenario of container OOM-kill followed by immediate takeover. See wal-design.md section 8.

**Index / Retrieval (Phase 9):**

11. **Index is bound 1:1 to data files, with strictly aligned lifecycles**. A segment's .idx file name is uniquely determined by the segment file name + IndexSpec.index_type. When compaction deletes a segment, the corresponding .idx must be deleted simultaneously; when writing a new segment, a new .idx is written simultaneously. Recovery runs cleanup_orphan_index_files at startup. See section 10 / index-design.md.
12. **Index is immutable**: follows from data file immutability. The VectorIndex protocol has no add / remove interface; any "modify index" is done through "discard old segment + build new segment + build new index".
13. **Search path prohibits access to unloaded Collections by default**: `Collection.search / get / query` throws `CollectionNotLoadedError` when `_load_state != "loaded"`. `insert / delete` are not affected by this constraint. After restart, Collection defaults to `released` state. See index-design.md section 2.4.
14. **Distance normalization happens inside VectorIndex**: FAISS L2 / IP / cosine internal conventions differ from the upper-layer `compute_distances` convention (FAISS L2 is squared, IP is larger-is-more-similar). Normalization in `FaissHnswIndex.search` ensures the upper layer sees distances identical to `BruteForceIndex`. Differential tests are the mandatory check for this invariant.

**Protocol Layer (Phase 10):**

15. **gRPC adapter layer only translates, never adds capability**. All methods in `adapter/grpc/servicer.py` are thin wrappers around engine API; any implementation that "the engine doesn't support but the servicer pretends to support" is prohibited. Unsupported RPCs return `UNIMPLEMENTED` + friendly message, **never silent fail**. See section 11 / grpc-adapter-design.md.
16. **proto stubs are committed to the repo**, not generated at runtime. `proto/README.md` records the source commit.

---

## 3. Six Package Detailed Descriptions

### 3.1 schema/ — Data Model and Type System

**Responsibility boundary**: Defines what data looks like; does not concern itself with where data is stored or how it flows.

| Submodule | Responsibility | Core Classes/Functions |
|--------|------|------------|
| `types.py` | Type definitions | `DataType` enum, `FieldSchema` dataclass, `CollectionSchema` dataclass, `TYPE_MAP` (DataType->Arrow) |
| `validation.py` | Validation logic | `validate_schema(schema)` — primary key/vector constraints; `validate_record(record, schema)` — field type/non-null/dimension; `separate_dynamic_fields(record, schema)` — schema in/out field separation->$meta |
| `arrow_builder.py` | Arrow Schema construction | `build_data_schema()`, `build_delta_schema()`, `build_wal_data_schema()`, `build_wal_delta_schema()`, `get_primary_field()`, `get_vector_field()` |
| `persistence.py` | Persistence | `save_schema_json(schema, collection_name, path)`, `load_schema_json(path)` |

```python
# schema/__init__.py
from milvus_lite.schema.types import DataType, FieldSchema, CollectionSchema
```

### 3.2 storage/ — Storage Layer

**Responsibility boundary**: Manages the physical storage of data (disk files + in-memory buffers), providing read/write primitives. Does not understand business flows (flush orchestration, recovery orchestration are handled by engine).

| Submodule | Responsibility | Core Classes/Methods |
|--------|------|------------|
| `wal.py` | WAL persistence | `WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number, sync_mode="close")` — `write(op)`, `read_operations(...)→Iterator[Operation]`, `close_and_delete()` (includes fsync) |
| `memtable.py` | In-memory buffer | `MemTable(schema)` — `apply_insert(batch)`, `apply_delete(batch)`, `get(pk)`, `flush()→Dict[partition, (data_table, delta_table)]`, `size()`. **Internal representation: append-only RecordBatch list + pk_index + delete_index; cross-clear must be seq-aware**. |
| `data_file.py` | Data Parquet (stateless) | `write_data_file(table, partition_dir, seq_min, seq_max)→path`, `read_data_file(path)→pa.Table`, `parse_seq_range(filename)→(min, max)`, `get_file_size(path)→int` |
| `delta_file.py` | Delta Parquet (stateless) | `write_delta_file(...)→path`, `read_delta_file(path)→pa.Table` |
| `delta_index.py` | In-memory delete index | `DeltaIndex(pk_name)` — `add_batch(batch)`, `is_deleted(pk, seq)→bool`, `gc_below(min_active_seq)→int`, `rebuild_from(...)` |
| `manifest.py` | Global state | `Manifest(data_dir)` — `load()`, `save()` (atomic+`.prev`), `add/remove_data_file(partition, file)`, `add/remove_delta_file(partition, file)`, `add/remove_partition(name)` |

```python
# storage/__init__.py
from milvus_lite.storage.wal import WAL
from milvus_lite.storage.memtable import MemTable
from milvus_lite.storage.data_file import write_data_file, read_data_file
from milvus_lite.storage.delta_file import write_delta_file, read_delta_file
from milvus_lite.storage.delta_index import DeltaIndex
from milvus_lite.storage.manifest import Manifest
```

### 3.3 engine/ — Engine Layer

**Responsibility boundary**: Orchestrates storage components, implementing business flows (write->flush->persist, crash recovery, compaction). Is the upper-layer caller of storage and search.

| Submodule | Responsibility | Core Content |
|--------|------|---------|
| `operation.py` | Write orchestration abstraction | `InsertOp`, `DeleteOp`, `Operation = Union[InsertOp, DeleteOp]` — frozen dataclass + Arrow batch, pure description with no behavior |
| `collection.py` | Engine core | `Collection(name, data_dir, schema)` — `insert()`, `delete()`, `get()`, `search()`, `_apply(op)` (unified write path), `create/drop/list_partitions()`, `flush()`, `close()`, `_alloc_seq()` |
| `flush.py` | Flush pipeline | `execute_flush(frozen_memtable, frozen_wal, manifest, delta_index, compaction_mgr)` — 7-step flow, **synchronous blocking** |
| `recovery.py` | Crash recovery | `execute_recovery(data_dir, manifest)` — 5-step flow, replayed via `WAL.read_operations()` by seq |
| `compaction.py` | Compaction | `CompactionManager(data_dir, schema)` — `maybe_compact(partition, manifest, delta_index)`, includes tombstone GC |

```python
# engine/__init__.py
from milvus_lite.engine.collection import Collection
```

**Relationship between flush.py and collection.py**:
- `collection.py` calls `flush.execute_flush()` when `insert()`/`delete()` detects MemTable is full
- `flush.py` receives the frozen MemTable/WAL + storage component references and executes the 7-step pipeline
- flush does not hold a Collection reference, only operates on the passed storage components -> **no circular dependency**

**Relationship between recovery.py and collection.py**:
- `collection.py` calls `recovery.execute_recovery()` in `__init__()`
- recovery receives data_dir + storage component references and returns the recovered state
- Likewise no circular dependency

### 3.4 search/ — Search Layer

**Responsibility boundary**: Vector retrieval; input is data + query, output is top-k results. Does not concern itself with where data comes from. **Includes the optional scalar filter subsystem** `search/filter/` (added in Phase 8), see sections 9.19-9.25.

| Submodule | Responsibility | Core Functions |
|--------|------|---------|
| `bitmap.py` | Validity filtering | `build_valid_mask(all_pks, all_seqs, delta_index, filter_mask=None)→np.ndarray[bool]` — dedup + delete filtering + optional scalar filter mask |
| `distance.py` | Distance computation | `cosine_distance(q, candidates)→np.ndarray`, `l2_distance(...)`, `ip_distance(...)`, `compute_distances(q, candidates, metric_type)` — pure math, stateless |
| `assembler.py` | Candidate assembly | `assemble_candidates(segments, memtable, vector_field, partition_names=None, filter_compiled=None)` — assembles data from various sources into unified numpy + optional filter mask |
| `executor.py` | Search orchestration | `execute_search(query_vectors, all_pks, all_seqs, all_vectors, all_records, delta_index, top_k, metric_type, ...)→List[List[dict]]` — bitmap + distance + top-k selection |
| `filter/` | Scalar filter subsystem | `parse_expr(s) → compile_expr(expr, schema) → evaluate(compiled, table) → BooleanArray`, see sections 9.19-9.25 |

```python
# search/__init__.py
from milvus_lite.search.executor import execute_search
from milvus_lite.search.assembler import assemble_candidates
from milvus_lite.search.filter import (
    parse_expr, compile_expr, evaluate,
    FilterError, FilterParseError, FilterFieldError, FilterTypeError,
)
```

**Why search is a separate package**:
- The bitmap pipeline grows in logic after adding scalar filtering (filter_mask)
- distance is a pure math module; replacing with FAISS in the future only requires replacing executor.py
- The filter subpackage is a relatively independent "small DSL"; making it a separate package facilitates testing and future parser implementation swaps
- Decoupled from engine: engine.collection calls `search.execute_search()`, passing in collected data

### 3.5 index/ — Index Layer (Phase 9)

**Responsibility boundary**: Defines the "vector index" abstraction, provides BruteForce and FAISS HNSW implementations.
Bound 1:1 to segments (one .idx file per data parquet), does not hold any business state.
Can be used directly by search/executor. See `plan/index-design.md`.

| Submodule | Responsibility | Core Classes/Functions |
|--------|------|------------|
| `protocol.py` | VectorIndex abstraction | `VectorIndex` ABC — `build(vectors, metric, params) → VectorIndex`, `search(queries, top_k, valid_mask, params) → (local_ids, distances)`, `save(path)`, `load(path, metric, dim) → VectorIndex`, `index_type` property |
| `spec.py` | Index parameters | `IndexSpec` frozen dataclass — `field_name / index_type / metric_type / build_params / search_params`, `to_dict / from_dict` supports manifest serialization |
| `brute_force.py` | NumPy fallback | `BruteForceIndex(VectorIndex)` — directly reuses distance functions from `search/distance.py`; zero external dependencies; retained as differential test baseline + fallback when faiss is unavailable |
| `faiss_hnsw.py` | FAISS HNSW | `FaissHnswIndex(VectorIndex)` — `IndexHNSWFlat`, `IDSelectorBitmap` accepts bitmap pipeline output, metric sign is normalized internally, `faiss.write_index / read_index` for persistence |
| `factory.py` | Factory + routing | `build_index_from_spec(spec, vectors) → VectorIndex`, `load_index(path, spec, dim) → VectorIndex`; `try: import faiss` failure for `index_type ∈ {HNSW, IVF_*}` raises `IndexBackendUnavailableError` |

```python
# index/__init__.py
from milvus_lite.index.protocol import VectorIndex
from milvus_lite.index.spec import IndexSpec
from milvus_lite.index.brute_force import BruteForceIndex
from milvus_lite.index.factory import build_index_from_spec, load_index
try:
    from milvus_lite.index.faiss_hnsw import FaissHnswIndex
    __all__ = ["VectorIndex", "IndexSpec", "BruteForceIndex", "FaissHnswIndex",
               "build_index_from_spec", "load_index"]
except ImportError:
    __all__ = ["VectorIndex", "IndexSpec", "BruteForceIndex",
               "build_index_from_spec", "load_index"]
```

**Relationship with other modules**:
- `storage/segment.py` holds `index: Optional[VectorIndex]`, injected by `Segment.build_or_load_index(spec, dir)`
- `engine/collection.py` manages Collection-level index lifecycle through `_index_spec` + `_load_state` state machine
- `engine/flush.py` / `compaction.py` hooks trigger build_or_load_index immediately after writing new segments (when in loaded state)
- `search/executor.py`'s `execute_search_with_index` path calls `index.search(query, top_k, valid_mask)` when segment.index exists

**Why index is a separate package (not placed under search/)**:
- Its responsibility is symmetric to search rather than being a submodule of search — search is the logic for "running distance + top-k on existing candidates", index is the logic for "using spatial data structures to accelerate nearest neighbor finding"
- Index implementations may depend on external libraries (FAISS / hnswlib / USearch); a separate package facilitates optional extras isolation
- Future extensions may include Sparse / Binary / multi-vector types; a separate package facilitates evolution

### 3.6 adapter/ — Adapter Layer (Phase 10)

**Responsibility boundary**: Translates external protocols (gRPC / HTTP / ...) into engine API calls.
**Only translates, never adds capability**. Each RPC maps to one engine method; unsupported RPCs return UNIMPLEMENTED. See `plan/grpc-adapter-design.md`.

| Submodule | Responsibility | Core Content |
|--------|------|---------|
| `grpc/server.py` | gRPC server lifecycle | `run_server(data_dir, host, port, max_workers)` |
| `grpc/servicer.py` | RPC dispatcher | `MilvusServicer(MilvusServiceServicer)` — implements quickstart subset RPCs, unimplemented ones return UNIMPLEMENTED |
| `grpc/errors.py` | Error code translation | `to_grpc_status(MilvusLiteError) → grpc Status code + reason` |
| `grpc/translators/schema.py` | Schema translation | `milvus_to_milvus_lite_schema(milvus.CollectionSchema) → CollectionSchema`, reverse likewise |
| `grpc/translators/records.py` | Column-row transposition | `fields_data_to_records(fields_data, num_rows) → List[dict]`, reverse `records_to_fields_data(records, schema, output_fields)` |
| `grpc/translators/search.py` | Search request parsing | `parse_search_params(search_params_kv)`, `decode_search_query(request)` |
| `grpc/translators/result.py` | Search result generation | engine `List[List[dict]]` -> milvus `SearchResults` proto |
| `grpc/translators/expr.py` | Filter translation | Mostly pass-through + UNIMPLEMENTED function detection (json_contains, etc.) |
| `grpc/translators/index.py` | Index parameter translation | `IndexParams (KeyValuePair list) ↔ IndexSpec` |
| `grpc/proto/` | Generated stubs | Generated from milvus-io/milvus-proto; committed to repo |
| `../cmdline.py` | Unified CLI entry point | `milvus-lite server --data-dir ./data --port 19530`; also includes `dump` |

```python
# adapter/grpc/__init__.py
from milvus_lite.adapter.grpc.server import run_server
from milvus_lite.adapter.grpc.servicer import MilvusServicer
```

**Relationship with other modules**:
- `servicer.py` holds a `MilvusLite` instance; all RPCs obtain a `Collection` object via `self._db.get_collection(name)` then call engine methods
- Does not directly depend on `storage/` / `search/` / `index/` (these are engine implementation details)
- `errors.py` depends on `milvus_lite.exceptions`

**Why adapter is a top-level package**:
- Conceptually it is the protocol layer, having a vertical relationship with engine / storage / search / index
- Future may introduce HTTP adapter / OpenAPI adapter / Python direct API and other adapters, each in its own subdirectory
- All external dependencies (grpcio / protobuf) are isolated through adapter, keeping the engine core clean

## 4. Dependency Graph

```
                        milvus_lite.__init__
                              │
                           db.py
                              │
                    engine/collection.py
                     /        |        \
                    /         |         \
        engine/flush.py  engine/recovery.py  engine/compaction.py
               \         |         /              |
                \        |        /               |
                 storage/*                  search/executor.py
                 ├── wal.py                    /          \
                 ├── memtable.py      search/bitmap.py  search/distance.py
                 ├── data_file.py          |
                 ├── delta_log.py          |
                 └── manifest.py           |
                        \                  |
                     schema/*              |
                     ├── types.py          |
                     ├── validation.py     |
                     ├── arrow_builder.py  |
                     └── persistence.py    |
                            \             /
                         constants.py + exceptions.py
```

**Dependency direction** (strictly downward, no cycles):

```
Level 0:  constants.py, exceptions.py           <- no internal dependencies
Level 1:  schema/*                              <- depends on L0
Level 2:  storage/*                             <- depends on L0, L1
Level 3:  search/bitmap.py, search/distance.py  <- depends on L0, L2(delta_log)
          search/filter/*                       <- depends on L0, L1
          index/*                               <- depends on L0, L1, search/distance (BruteForce)
Level 4:  search/executor.py, search/assembler.py <- depends on L3
Level 5:  engine/flush.py, recovery.py, compaction.py <- depends on L0-L4 + index (Phase 9)
Level 6:  engine/collection.py                  <- depends on L0-L5
Level 7:  db.py                                 <- depends on L6
Level 8:  adapter/grpc/*                        <- depends on L7 (db.py) — Phase 10
```

**Notes**:
- `index/` is the newly added L3, alongside search/filter, held by storage/segment + called by search/executor
- `adapter/` is the newly added L8, the outermost layer of the project; all external protocol dependencies (grpcio / protobuf) are isolated here

## 5. Public API Exports

```python
# milvus_lite/__init__.py
from milvus_lite.schema import DataType, FieldSchema, CollectionSchema
from milvus_lite.db import MilvusLite
from milvus_lite.exceptions import MilvusLiteError

__all__ = ["MilvusLite", "CollectionSchema", "FieldSchema", "DataType", "MilvusLiteError"]
```

Users only need `from milvus_lite import MilvusLite, CollectionSchema, FieldSchema, DataType`. Collection is obtained through `db.get_collection()` and is not directly exported.

## 6. constants.py Contents

```python
# -- MemTable --
MEMTABLE_SIZE_LIMIT = 10_000

# -- Compaction --
MAX_DATA_FILES = 32
COMPACTION_MIN_FILES_PER_BUCKET = 4
COMPACTION_BUCKET_BOUNDARIES = [1_000_000, 10_000_000, 100_000_000]  # bytes

# -- File naming --
SEQ_FORMAT_WIDTH = 6
DATA_FILE_TEMPLATE = "data_{min:0{w}d}_{max:0{w}d}.parquet"
DELTA_FILE_TEMPLATE = "delta_{min:0{w}d}_{max:0{w}d}.parquet"
WAL_DATA_TEMPLATE = "wal_data_{n:0{w}d}.arrow"
WAL_DELTA_TEMPLATE = "wal_delta_{n:0{w}d}.arrow"

# -- Partition --
DEFAULT_PARTITION = "_default"
ALL_PARTITIONS = "_all"
```

## 7. exceptions.py Contents

```python
class MilvusLiteError(Exception): ...

class SchemaValidationError(MilvusLiteError): ...
class CollectionNotFoundError(MilvusLiteError): ...
class CollectionAlreadyExistsError(MilvusLiteError): ...
class PartitionNotFoundError(MilvusLiteError): ...
class PartitionAlreadyExistsError(MilvusLiteError): ...
class DefaultPartitionError(MilvusLiteError): ...
class WALCorruptedError(MilvusLiteError): ...
```

## 8. Alignment with Implementation Priority

| P# | Implementation Module | Tests |
|----|---------|------|
| P0 | `constants`, `exceptions`, `schema/*` (all) | `tests/schema/*` |
| P1 | `storage/manifest` | `tests/storage/test_manifest` |
| P2 | `storage/memtable`, `storage/wal` | `tests/storage/test_memtable`, `test_wal` |
| P3 | `storage/data_file`, `storage/delta_log`, `engine/flush` | `tests/storage/test_data_file`, `test_delta_log`, `tests/engine/test_flush` |
| P4 | `engine/collection` (partition methods) | `tests/engine/test_collection` (partial) |
| P5 | `engine/collection` (insert/delete/get E2E) | `tests/engine/test_collection` |
| P6 | `search/*` (all) | `tests/search/*` |
| P7 | `engine/compaction` | `tests/engine/test_compaction` |
| P8 | `db.py` | `tests/test_db` |
| P9 | `engine/recovery` | `tests/engine/test_recovery` |
| P10 | `engine/collection` (add_field) | `tests/engine/test_collection` |
| P11 | End-to-end tests | All |

## 9. Module Interface Detailed Descriptions

Designed bottom-up by dependency level; each module only lists external interfaces (public), not internal implementation details.

---

### 9.0 constants.py

```python
# -- MemTable --
MEMTABLE_SIZE_LIMIT: int = 10_000          # Combined threshold for insert_buf + delete_buf

# -- Compaction --
MAX_DATA_FILES: int = 32                    # Max data files per Partition
COMPACTION_MIN_FILES_PER_BUCKET: int = 4    # Trigger threshold for file count within same bucket
COMPACTION_BUCKET_BOUNDARIES: List[int] = [1_000_000, 10_000_000, 100_000_000]  # bytes

# -- File naming --
SEQ_FORMAT_WIDTH: int = 6
DATA_FILE_TEMPLATE: str = "data_{min:0{w}d}_{max:0{w}d}.parquet"
DELTA_FILE_TEMPLATE: str = "delta_{min:0{w}d}_{max:0{w}d}.parquet"
WAL_DATA_TEMPLATE: str = "wal_data_{n:0{w}d}.arrow"
WAL_DELTA_TEMPLATE: str = "wal_delta_{n:0{w}d}.arrow"

# -- Partition --
DEFAULT_PARTITION: str = "_default"
ALL_PARTITIONS: str = "_all"               # Internal marker for cross-Partition delete
```

### 9.1 exceptions.py

```python
class MilvusLiteError(Exception):
    """Base class for all MilvusLite exceptions"""

class SchemaValidationError(MilvusLiteError):
    """Schema definition is invalid or record does not conform to Schema"""

class CollectionNotFoundError(MilvusLiteError):
    """Collection does not exist"""

class CollectionAlreadyExistsError(MilvusLiteError):
    """Collection already exists"""

class PartitionNotFoundError(MilvusLiteError):
    """Partition does not exist"""

class PartitionAlreadyExistsError(MilvusLiteError):
    """Partition already exists"""

class DefaultPartitionError(MilvusLiteError):
    """Attempted to delete the _default Partition"""

class WALCorruptedError(MilvusLiteError):
    """WAL file is corrupted and cannot be recovered"""
```

---

### 9.2 schema/types.py

```python
class DataType(Enum):
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    DOUBLE = "double"
    VARCHAR = "varchar"
    JSON = "json"
    FLOAT_VECTOR = "float_vector"


@dataclass
class FieldSchema:
    name: str
    dtype: DataType
    is_primary: bool = False
    dim: Optional[int] = None           # Only needed for FLOAT_VECTOR
    max_length: Optional[int] = None    # Only needed for VARCHAR
    nullable: bool = False
    default_value: Any = None


@dataclass
class CollectionSchema:
    fields: List[FieldSchema]
    version: int = 1                    # Schema version number, +1 on each change
    enable_dynamic_field: bool = False  # Whether to enable $meta dynamic field


# DataType -> PyArrow type mapping
TYPE_MAP: Dict[DataType, Any] = {
    DataType.BOOL:         pa.bool_(),
    DataType.INT8:         pa.int8(),
    DataType.INT16:        pa.int16(),
    DataType.INT32:        pa.int32(),
    DataType.INT64:        pa.int64(),
    DataType.FLOAT:        pa.float32(),
    DataType.DOUBLE:       pa.float64(),
    DataType.VARCHAR:      pa.string(),
    DataType.JSON:         pa.string(),
    DataType.FLOAT_VECTOR: None,  # Requires dim; generated at runtime via lambda dim: pa.list_(pa.float32(), dim)
}
```

### 9.3 schema/validation.py

```python
def validate_schema(schema: CollectionSchema) -> None:
    """Validate the legality of a CollectionSchema definition.

    Validation rules:
    - Exactly one is_primary=True field, of type VARCHAR or INT64
    - Exactly one FLOAT_VECTOR field (MVP limitation)
    - Vector field must specify dim > 0
    - Primary key field cannot be nullable
    - Field names must be unique
    - Field names must not use reserved names (_seq, _partition, $meta)

    Raises:
        SchemaValidationError
    """


def validate_record(record: dict, schema: CollectionSchema) -> None:
    """Validate that a single record conforms to the Schema.

    Validation rules:
    - Primary key field exists and is not None
    - Vector field exists and dimension == schema.dim
    - Defined fields have matching types
    - Non-nullable fields are not None
    - When enable_dynamic_field=False, fields outside Schema are not allowed

    Raises:
        SchemaValidationError
    """


def separate_dynamic_fields(
    record: dict, schema: CollectionSchema
) -> Tuple[dict, Optional[str]]:
    """Split a record into Schema-defined fields + $meta JSON.

    Args:
        record: Original record provided by user
        schema: CollectionSchema

    Returns:
        (schema_fields, meta_json)
        - schema_fields: Contains only Schema-defined fields (with default values filled in)
        - meta_json: JSON string of dynamic fields serialized, None when there are no dynamic fields

    Raises:
        SchemaValidationError: enable_dynamic_field=False but fields outside Schema exist
    """
```

### 9.4 schema/arrow_builder.py

```python
def get_primary_field(schema: CollectionSchema) -> FieldSchema:
    """Return the primary key FieldSchema."""


def get_vector_field(schema: CollectionSchema) -> FieldSchema:
    """Return the vector FieldSchema."""


def build_data_schema(schema: CollectionSchema) -> pa.Schema:
    """Arrow Schema for data Parquet files.
    Column order: _seq(uint64) + user fields + [$meta(string)]
    Does not include _partition."""


def build_delta_schema(schema: CollectionSchema) -> pa.Schema:
    """Arrow Schema for delta Parquet files.
    Columns: {pk_name}(primary key type) + _seq(uint64)
    Does not include _partition."""


def build_wal_data_schema(schema: CollectionSchema) -> pa.Schema:
    """Arrow Schema for WAL data files.
    Column order: _seq(uint64) + _partition(string) + user fields + [$meta(string)]
    Has an additional _partition column compared to data_schema."""


def build_wal_delta_schema(schema: CollectionSchema) -> pa.Schema:
    """Arrow Schema for WAL delete files.
    Columns: {pk_name}(primary key type) + _seq(uint64) + _partition(string)
    Has an additional _partition column compared to delta_schema."""
```

### 9.5 schema/persistence.py

```python
def save_schema(
    schema: CollectionSchema,
    collection_name: str,
    path: str,
) -> None:
    """Serialize Schema as JSON and write to path.
    JSON structure includes collection_name (self-describing) + version + fields + enable_dynamic_field.
    Uses write-tmp + replace for atomic writing."""


def load_schema(path: str) -> Tuple[str, CollectionSchema]:
    """Load Schema from JSON file.
    Returns: (collection_name, schema)
    Raises: FileNotFoundError, SchemaValidationError (invalid JSON format)"""
```

---

### 9.6 storage/wal.py

```python
class WAL:
    """Write-Ahead Log, Arrow IPC Streaming format, dual-file (data + delta).

    Each write round corresponds to a pair of WAL files (wal_data_{N}.arrow + wal_delta_{N}.arrow),
    which are deleted entirely after successful flush. Writer is lazily initialized (file created on first write).
    """

    def __init__(
        self,
        wal_dir: str,
        wal_data_schema: pa.Schema,
        wal_delta_schema: pa.Schema,
        wal_number: int,
    ) -> None:
        """
        Args:
            wal_dir: Directory where WAL files reside
            wal_data_schema: Arrow Schema used for wal_data files (includes _partition)
            wal_delta_schema: Arrow Schema used for wal_delta files (includes _partition)
            wal_number: WAL number for this round (the N in the filename)
        """

    def write_insert(self, record_batch: pa.RecordBatch) -> None:
        """Append-write to wal_data file. Creates file and writer on first call.
        record_batch schema must match wal_data_schema."""

    def write_delete(self, record_batch: pa.RecordBatch) -> None:
        """Append-write to wal_delta file. Creates file and writer on first call.
        record_batch schema must match wal_delta_schema."""

    def close_and_delete(self) -> None:
        """Close writer, delete both WAL files. Called after successful flush."""

    @staticmethod
    def find_wal_files(wal_dir: str) -> List[int]:
        """Scan wal_dir, return list of existing WAL numbers (used for recovery detection)."""

    @staticmethod
    def recover(
        wal_dir: str,
        wal_number: int,
        wal_data_schema: pa.Schema,
        wal_delta_schema: pa.Schema,
    ) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        """Read WAL files of the specified number, return (data_batches, delta_batches).
        Each RecordBatch contains a _partition column; during recovery, route to MemTable by _partition.
        Raises: WALCorruptedError"""

    @property
    def number(self) -> int:
        """Current WAL number."""

    @property
    def data_path(self) -> Optional[str]:
        """wal_data file path (None when not yet created)."""

    @property
    def delta_path(self) -> Optional[str]:
        """wal_delta file path (None when not yet created)."""
```

### 9.7 storage/memtable.py

**Internal representation**: MemTable does not hold `dict[pk -> record_dict]`. The memory overhead of Python dict + record dict is 10-100x that of Arrow representation, and requires an additional Python -> Arrow conversion at flush time. MemTable internally maintains **append-only RecordBatch list + two lightweight indexes**:

```
_insert_batches: list[pa.RecordBatch]              # append-only, one batch appended per apply_insert
_pk_index:       dict[pk -> (batch_idx, row_idx)]   # points to latest position in _insert_batches
_delete_index:   dict[pk -> delete_seq]             # delete watermark
```

`_pk_index` is lazy -- old versions of the same pk are physically retained in `_insert_batches` until flush time when they are deduped in one pass to extract active rows. This way:

- **Write is zero-copy**: apply_insert directly `append(batch)` + update index
- **Flush is nearly zero-cost**: `pa.Table.from_batches` + take active rows by `_pk_index.values()`
- **Search gets Arrow columns**: no secondary conversion

**Cross-clear must be seq-aware** (architecture invariant section 2). Before put / delete, first compare `_seq`; the current operation takes effect only if its `_seq` is larger; otherwise the current operation is discarded. **This invariant ensures that recovery / future concurrent writes will never trigger data corruption.**

```python
class MemTable:
    """Collection-level shared in-memory buffer.

    Internal representation:
    - _insert_batches: list[pa.RecordBatch], append-only
    - _pk_index: dict[pk -> (batch_idx, row_idx)], latest position index
    - _delete_index: dict[pk -> delete_seq], delete watermark

    cross-clear semantics:
    - apply_insert(seq=S, pk=P): if _delete_index[P] >= S then discard the operation (a newer delete already exists);
      otherwise insert and clear _delete_index[P].
    - apply_delete(seq=S, pk=P): if the latest _seq of P in _insert_batches >= S then discard the operation;
      otherwise update _delete_index[P]=S and clear _pk_index[P].
    """

    def __init__(self, schema: CollectionSchema) -> None:
        """
        Args:
            schema: CollectionSchema, used to determine pk_name and field information
        """

    def apply_insert(self, batch: pa.RecordBatch) -> None:
        """Append a RecordBatch with wal_data schema.
        - Each row in batch must contain _seq, _partition, pk_field, and all user fields
        - Internally updates _pk_index and clears outdated entries in _delete_index per seq-aware rules
        - Batch is physically appended to _insert_batches; old versions are not deleted (deduped at flush)
        """

    def apply_delete(self, batch: pa.RecordBatch) -> None:
        """Process a RecordBatch with wal_delta schema.
        - Batch shares the same _seq; internally updates _delete_index per seq-aware rules
        - Does not hold batch reference (only extracts pk + seq into _delete_index)
        - Handles cross-partition delete when _partition='_all'
        """

    def get(self, pk_value: Any) -> Optional[dict]:
        """Point lookup for a single record.
        Locates latest position via _pk_index, then checks if _delete_index overrides it.
        Returns record dict (without _partition, _seq) or None."""

    def flush(self) -> Dict[str, Tuple[Optional[pa.Table], Optional[pa.Table]]]:
        """Output Arrow Tables split by Partition.

        Implementation:
        1. Concat _insert_batches into one large Table
        2. Take active rows by _pk_index.values() (dedup happens here)
        3. Split by _partition column into per-partition data_tables
        4. Materialize _delete_index into per-partition delta_tables

        Returns: {partition_name: (data_table, delta_table)}
            - data_table: uses data_schema (without _partition), may be None
            - delta_table: uses delta_schema (without _partition), may be None
        Internal state is not cleared after call (caller freezes then discards the entire MemTable)."""

    def size(self) -> int:
        """Returns len(_pk_index) + len(_delete_index).
        Note this is the "active pk count", not the total row count of _insert_batches --
        the flush trigger threshold uses this to avoid memory blowup from repeated upserts of the same pk."""

    def get_active_records(
        self, partition_names: Optional[List[str]] = None
    ) -> List[dict]:
        """Return active records from _pk_index that are not overridden by _delete_index.
        Used for reading MemTable layer data during search/get.
        When partition_names is not None, only returns records matching the specified Partitions.
        Returned dicts do not contain _partition and _seq (clean for external use)."""
```

**Key test points** (must be covered during implementation):

```python
# Verify seq-aware cross-clear: out-of-order apply still yields correct final state
mt = MemTable(schema)
mt.apply_insert(batch_with_pk_X_seq_7)   # First comes seq=7
mt.apply_insert(batch_with_pk_X_seq_5)   # Then comes seq=5 (should be discarded)
mt.apply_delete(batch_with_pk_X_seq_6)   # Then comes delete seq=6 (should be discarded, since seq=7 is newer)
assert mt.get("X")["_seq"] == 7          # <- seq=7 must be retained
```

### 9.8 storage/data_file.py

```python
def write_data_file(
    table: pa.Table,
    partition_dir: str,
    seq_min: int,
    seq_max: int,
) -> str:
    """Write Arrow Table to a data Parquet file.
    File path: {partition_dir}/data/data_{seq_min:06d}_{seq_max:06d}.parquet
    Automatically creates data/ subdirectory (if it doesn't exist).
    Returns: The relative path of the written file (relative to Collection data_dir)."""


def read_data_file(path: str) -> pa.Table:
    """Read a data Parquet file, return complete Arrow Table."""


def parse_seq_range(filename: str) -> Tuple[int, int]:
    """Parse seq range from filename.
    'data_000001_000500.parquet' -> (1, 500)
    'delta_000501_000503.parquet' -> (501, 503)"""


def get_file_size(path: str) -> int:
    """Return file size in bytes, used for Compaction bucketing."""
```

### 9.9 storage/delta_file.py + storage/delta_index.py

**DeltaLog is split into two modules**, symmetric with `data_file.py`:

- `delta_file.py` -- pure IO functions, stateless
- `delta_index.py` -- in-memory `DeltaIndex` class, independently testable

**Why split**: The original `DeltaLog` simultaneously managed Parquet IO + in-memory index + queries, 6 methods with 3 responsibilities, hard to unit test; after splitting, IO functions form a symmetric structure with `data_file.py`, the in-memory index can be unit tested without disk, and future implementation changes (numpy / pyarrow dict array) only touch one place.

#### 9.9.a delta_file.py (stateless IO)

```python
def write_delta_file(
    delta_table: pa.Table,
    partition_dir: str,
    seq_min: int,
    seq_max: int,
) -> str:
    """Write delta Arrow Table to a delta Parquet file.
    File path: {partition_dir}/delta/delta_{seq_min:06d}_{seq_max:06d}.parquet
    Automatically creates delta/ subdirectory.
    Returns: Relative path (relative to Collection data_dir)."""


def read_delta_file(path: str) -> pa.Table:
    """Read a delta Parquet file, return Arrow Table (pk_field + _seq, two columns)."""
```

#### 9.9.b delta_index.py (in-memory index)

```python
class DeltaIndex:
    """In-memory delete watermark index: pk -> max_delete_seq.

    At startup, rebuilt via rebuild_from() from all delta files;
    at runtime, incrementally updated via add_batch();
    via gc_below(), old tombstones are reclaimed after Compaction (architecture invariant section 3).
    """

    def __init__(self, pk_name: str) -> None:
        """
        Args:
            pk_name: Primary key field name (used to extract pk column from batch)
        """

    def add_batch(self, delta_batch: pa.RecordBatch) -> None:
        """Merge (pk, _seq) from delta batch into the in-memory index.
        - For each pk, takes max(existing seq, new seq)
        - Does not write to disk
        - Used for post-flush update + WAL replay
        """

    def is_deleted(self, pk_value: Any, data_seq: int) -> bool:
        """Determine whether a data record has been deleted.
        Rule: _map.get(pk, -1) > data_seq -> deleted."""

    def gc_below(self, min_active_data_seq: int) -> int:
        """Reclaim all tombstones with delete_seq < min_active_data_seq.

        Args:
            min_active_data_seq: Minimum seq_min across all data files in the current manifest
        Returns:
            Number of entries reclaimed

        Correctness: Any tombstone with delete_seq < min_active_data_seq has had all
        corresponding data rows physically consumed by compaction; no residual data rows
        need it for filtering, so it can be safely discarded. See architecture invariant section 3.
        """

    @classmethod
    def rebuild_from(
        cls,
        pk_name: str,
        partition_delta_files: Dict[str, List[str]],
    ) -> "DeltaIndex":
        """One-time rebuild at startup.
        Args:
            partition_delta_files: {partition_name: [list of absolute paths]}
        Returns: Complete DeltaIndex instance
        """

    def __len__(self) -> int:
        """Current number of active tombstone entries (for monitoring/testing)."""

    @property
    def snapshot(self) -> Dict[Any, int]:
        """Read-only snapshot: pk_value -> max_delete_seq (copy, does not hold internal references)."""
```

**Key points**:
- `add_batch` accepts `pa.RecordBatch` rather than `pa.Table`, unified granularity with WAL / MemTable
- `gc_below` is the entry point called by Compaction, encapsulating the GC rule of architecture invariant section 3
- No `remove_files` method -- file management is Manifest's responsibility; DeltaIndex does not hold file paths

### 9.10 storage/manifest.py

```python
class Manifest:
    """Global state snapshot file, updated via atomic replacement (write-tmp + replace).

    Records current _seq, Schema version, each Partition's file list, active WAL.
    Is the system's only source of truth (architecture invariant section 5).

    Persistence layout:
        data_dir/
          ├── manifest.json          # Current version
          └── manifest.json.prev     # Previous version backup (fallback against one serialization failure)
    """

    def __init__(self, data_dir: str) -> None: ...

    # -- Persistence --

    def save(self) -> None:
        """Atomically update manifest.json, version auto-increments by 1.

        Steps:
        1. Serialize to manifest.json.tmp
        2. If manifest.json exists, cp it -> manifest.json.prev (overwriting old .prev)
        3. os.replace(manifest.json.tmp, manifest.json)  <- atomic switch
        4. fsync data_dir directory to ensure replacement is durable

        Failure semantics: Step 1 failure -> orphaned tmp file, no impact;
                          Step 2 failure -> raises exception, disk still has previous successful save's manifest;
                          Step 3 is atomic, either succeeds or fails entirely.
        """

    @classmethod
    def load(cls, data_dir: str) -> "Manifest":
        """Load manifest.json.

        Loading strategy:
        1. Try manifest.json
           - File doesn't exist -> return initial state (version=0, _default Partition), no error
           - Load succeeds -> return
           - Load fails (corrupt JSON, missing fields) -> warning log + proceed to step 2
        2. Try manifest.json.prev
           - Load succeeds -> warning log ("using prev manifest, last save likely corrupted") + return
           - Load fails -> raise ManifestCorruptedError
        """

    # -- File management (per Partition) --

    def add_data_file(self, partition: str, filename: str) -> None:
        """Add a data file to the specified Partition."""

    def add_delta_file(self, partition: str, filename: str) -> None:
        """Add a delta file to the specified Partition."""

    def remove_data_files(self, partition: str, filenames: List[str]) -> None:
        """Remove data files from the specified Partition (after Compaction)."""

    def remove_delta_files(self, partition: str, filenames: List[str]) -> None:
        """Remove delta files from the specified Partition (after Compaction)."""

    def get_data_files(self, partition: str) -> List[str]:
        """Return the data file list for the specified Partition."""

    def get_delta_files(self, partition: str) -> List[str]:
        """Return the delta file list for the specified Partition."""

    def get_all_data_files(self) -> Dict[str, List[str]]:
        """Return data files for all Partitions: {partition: [files]}."""

    def get_all_delta_files(self) -> Dict[str, List[str]]:
        """Return delta files for all Partitions: {partition: [files]}."""

    # -- Partition management --

    def add_partition(self, name: str) -> None:
        """Add a new Partition (initialize empty file lists).
        Raises: PartitionAlreadyExistsError"""

    def remove_partition(self, name: str) -> None:
        """Remove a Partition.
        Raises: DefaultPartitionError (deleting _default not allowed), PartitionNotFoundError"""

    def list_partitions(self) -> List[str]:
        """Return list of all Partition names."""

    def has_partition(self, name: str) -> bool:
        """Check if a Partition exists."""

    # -- Properties --

    @property
    def version(self) -> int:
        """Manifest version number (+1 on each save)."""

    @property
    def current_seq(self) -> int:
        """Current maximum _seq, restores counter at startup."""

    @current_seq.setter
    def current_seq(self, value: int) -> None: ...

    @property
    def schema_version(self) -> int:
        """Current Schema version number."""

    @schema_version.setter
    def schema_version(self, value: int) -> None: ...

    @property
    def active_wal_number(self) -> Optional[int]:
        """Current active WAL number."""

    @active_wal_number.setter
    def active_wal_number(self, value: int) -> None: ...

    # -- Phase 9.3 -- Index spec persistence --

    @property
    def index_spec(self) -> Optional["IndexSpec"]:
        """Current Collection's index specification (None if none)."""

    def set_index_spec(self, spec: Optional["IndexSpec"]) -> None:
        """Set or clear IndexSpec; must call save() afterwards to persist.
        Introduced in Phase 9.3."""

    @property
    def format_version(self) -> int:
        """Manifest schema version number. From Phase 9.3 onward = 2 (v1 old manifest
        defaults index_spec to None on load; next save automatically upgrades)."""
```

**Manifest v1 -> v2 compatibility**:

| Field | v1 | v2 |
|---|---|---|
| `format_version` | 1 (or missing) | 2 |
| `index_spec` | absent | dict or null |

Missing fields during load are handled with default values; save always writes v2. No migration tool needed.

---

### 9.11 search/bitmap.py

```python
def build_valid_mask(
    all_pks: List[Any],
    all_seqs: np.ndarray,
    delta_index: "DeltaIndex",
    filter_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build valid row bitmap (np.ndarray[bool], True=valid).

    Three-step filtering (in order):
    1. Dedup: When the same PK appears multiple times, keep only the row with the largest _seq, mark others as False
    2. Delete filtering: Call delta_index.is_deleted(pk, seq), mark deleted rows as False
    3. Scalar filtering: If filter_mask is not None, bitwise AND it into the final mask

    Args:
        all_pks: Length N pk list
        all_seqs: shape=(N,), _seq of all rows
        delta_index: DeltaIndex instance, provides is_deleted() queries
        filter_mask: Optional, length N bool array, obtained by evaluating the search/filter
            subsystem on pa.Tables from each source in assemble_candidates and concatenating

    Returns:
        np.ndarray[bool] shape=(N,)

    Raises:
        ValueError: filter_mask length does not equal all_pks length
    """
```

### 9.12 search/distance.py

```python
def cosine_distance(
    query: np.ndarray,          # shape=(dim,)
    candidates: np.ndarray,     # shape=(n, dim)
) -> np.ndarray:                # shape=(n,)
    """Cosine distance = 1 - cosine_similarity. Range [0, 2], smaller is more similar."""


def l2_distance(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """L2 (Euclidean) distance. Smaller is more similar."""


def ip_distance(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Inner product distance = -dot(q, c). Negated so smaller is more similar (same direction as cosine/L2)."""


def compute_distances(
    query: np.ndarray,
    candidates: np.ndarray,
    metric_type: str,           # "COSINE" | "L2" | "IP"
) -> np.ndarray:
    """Dispatch to the corresponding distance function based on metric_type.
    Raises: ValueError (unsupported metric_type)"""
```

### 9.13 search/executor.py

```python
def execute_search(
    query_vectors: np.ndarray,      # shape=(nq, dim), nq query vectors
    all_pks: List[Any],             # Length N, PKs of all candidate rows
    all_seqs: np.ndarray,           # shape=(N,), _seq of all candidate rows
    all_vectors: np.ndarray,        # shape=(N, dim), vectors of all candidate rows
    all_records: List[dict],        # Complete records of all candidate rows (for returning entity fields)
    delta_index: "DeltaIndex",
    top_k: int,
    metric_type: str,
    pk_field: str,
    vector_field: str,
    filter_mask: Optional[np.ndarray] = None,    # Phase 8: scalar filter
) -> List[List[dict]]:
    """Execute vector search.

    Flow:
    1. build_valid_mask(filter_mask=filter_mask) -> valid row bitmap (including dedup +
       delete filtering + scalar filtering)
    2. For each query_vector:
       a. Filter candidate vectors using bitmap
       b. compute_distances() -> distance array
       c. argpartition to get top-k smallest distances
       d. Assemble results
    3. Return results

    Returns:
        Outer List = results for each query vector
        Inner List = top-K results, sorted by distance ascending
        Each result: {"id": pk_value, "distance": float, "entity": {field: value}}
    """
```

### 9.13.5 search/assembler.py

```python
def assemble_candidates(
    segments: Iterable["Segment"],
    memtable: "MemTable",
    vector_field: str,
    partition_names: Optional[List[str]] = None,
    filter_compiled: Optional["CompiledExpr"] = None,
) -> Tuple[
    List[Any],         # all_pks
    np.ndarray,        # all_seqs (uint64)
    np.ndarray,        # all_vectors (float32, shape=(N, dim))
    List[dict],        # all_records (entity dicts)
    Optional[np.ndarray],  # filter_mask (bool, length N) or None
]:
    """Assemble segments and MemTable into unified candidate arrays.

    Order: segments first (in iteration order), then MemTable. This order determines
    the concatenation order of filter_mask; the bitmap stage uses the same order.

    If filter_compiled is not None:
        - For each segment entering the candidate set, call evaluator to produce a BooleanArray
        - For MemTable's active data, construct a temporary pa.Table then call evaluator
        - Masks from each source are concatenated in candidate order into a single numpy array
    """
```

`assembler` is the **only module in the search subsystem that simultaneously knows about storage types (Segment, MemTable) and the filter subsystem** -- other search files are storage-agnostic.

---

### 9.14 engine/flush.py

```python
def execute_flush(
    frozen_memtable: "MemTable",
    frozen_wal: "WAL",
    data_dir: str,
    schema: "CollectionSchema",
    manifest: "Manifest",
    delta_index: "DeltaIndex",
    compaction_mgr: "CompactionManager",
    collection: Optional["Collection"] = None,    # Phase 9.4: used for index hook
) -> None:
    """Execute the Flush pipeline (7+1 steps, **synchronous blocking** -- architecture invariant section 6).

    Precondition: Caller has already completed Step 1 (frozen old MemTable/WAL, created new ones).

    Step 2: frozen_memtable.flush() -> {partition: (data_table, delta_table)}
    Step 3: Write Parquet files to each Partition directory (including delta Parquet)
    Step 4: Update delta_index in-memory (add_batch for each delta_table's RecordBatch)
    Step 5: Atomically update Manifest (add new files + update current_seq + switch active_wal + .prev backup)
    Step 6: Delete old WAL (frozen_wal.close_and_delete, includes fsync)
    Step 7: Trigger compaction_mgr.maybe_compact() per Partition (which includes tombstone GC)

    Step 8 (Phase 9.4): For each new Segment created in Step 3:
        - Load Segment into _segment_cache
        - If collection._load_state == "loaded" and collection._index_spec is not None:
            seg.build_or_load_index(spec, index_dir)
        - After flush completes, new data is immediately available for search

    Crash safety:
    - Crash at Step 3 -> Manifest not updated, Parquet becomes orphan files, recovery cleans up
    - Crash before Step 5 -> WAL is intact, replay recovers
    - Crash after Step 5 -> Manifest already updated, WAL replay produces duplicates but _seq dedup ensures correctness
    - Crash at Step 8 -> .idx files may be incomplete, next load() detects failure and rebuilds
    """
```

### 9.15 engine/recovery.py

```python
def execute_recovery(
    data_dir: str,
    schema: "CollectionSchema",
    manifest: "Manifest",
) -> Tuple["MemTable", "DeltaIndex", int]:
    """Execute crash recovery (5+1 steps).

    Precondition: Caller has already loaded Manifest (Step 1).

    Step 2: Scan wal/ directory; if uncleaned WAL exists -> replay Operations in _seq order to a new MemTable
            (Replay via WAL.read_operations() Iterator[Operation], see section 9.x operation)
    Step 3: Verify that files in Manifest actually exist (handle Compaction mid-crash)
    Step 4: Clean up orphan files (files on disk but not in Manifest)
    Step 5: DeltaIndex.rebuild_from(all Partitions' delta_files)
    Step 6 (Phase 9.4): _cleanup_orphan_index_files(data_dir, manifest)
            - Scan each partition's indexes/ directory
            - For each .idx file, parse back the corresponding data file stem from the filename
            - If the corresponding data file is not in manifest, delete the .idx file
            - If a data file in manifest has no corresponding .idx, no action (will be built at load())

    After Phase 9.3: Caller must force Collection._load_state to "released" after receiving these return values.
    Restart does not automatically load index, aligning with Milvus behavior.

    Returns:
        (memtable, delta_index, next_wal_number)
        - memtable: MemTable after WAL replay (empty MemTable if no WAL)
        - delta_index: Rebuilt DeltaIndex
        - next_wal_number: WAL number to use for the next round
    """
```

### 9.16 engine/compaction.py

```python
class CompactionManager:
    """Size-Tiered Compaction Manager, executes independently per Partition."""

    def __init__(self, data_dir: str, schema: CollectionSchema) -> None:
        """
        Args:
            data_dir: Collection data directory
            schema: CollectionSchema (used to determine primary key and schema when reading Parquet)
        """

    def maybe_compact(
        self,
        partition: str,
        manifest: "Manifest",
        delta_index: "DeltaIndex",
        collection: Optional["Collection"] = None,    # Phase 9.4: used for index hook
    ) -> None:
        """Check if the specified Partition needs Compaction; execute if conditions are met.

        Trigger conditions (any one met):
        - File count in the same size bucket >= COMPACTION_MIN_FILES_PER_BUCKET
        - Total file count for the Partition > MAX_DATA_FILES

        Compaction flow:
        1. Bucket files by size
        2. Select files in the target bucket
        3. Read and merge Arrow Tables
        4. Dedup by primary key (keep max _seq)
        5. Filter deleted records using delta_index.is_deleted()
        6. Write new Parquet file
        7. Atomically update Manifest (remove old files + add new files + remove consumed delta files)
        8. Delete old files and consumed delta files
        9. **Tombstone GC**: Call delta_index.gc_below(min_active_data_seq)
           where min_active_data_seq is the global minimum of seq_min across all partitions' data files in Manifest
        10. (Phase 9.4) Delete .idx files corresponding to retired segments
        11. (Phase 9.4) If collection._load_state == "loaded" and _index_spec is not None:
            call build_or_load_index(spec, index_dir) for the new merged segment
        """

    def _global_min_active_data_seq(self, manifest: "Manifest") -> int:
        """Compute the minimum seq_min across all data files in all partitions.

        Used for tombstone GC triggering -- any tombstone with delete_seq less than this value
        cannot possibly filter any data rows (because all data with seq < this value has been
        physically consumed by compaction).

        Returns:
            Global minimum seq_min; if no data files exist, returns sys.maxsize (GC will empty delta_index)
        """
```

**Tombstone GC invariant** (implementation notes for architecture invariant section 3):

```
For any delete tombstone (pk, delete_seq):
  There exist residual data rows that need it for filtering  <=>  There exists some data file containing pk with seq_min <= delete_seq

Conservative GC rule (used in MVP):
  if delete_seq < min(seq_min of all data files):
      drop tombstone(pk, delete_seq)

Correctness proof:
  delete_seq < min_seq_min ==> no data file exists with seq_min <= delete_seq
                           ==> no residual data rows need it for filtering
                           ==> safe to discard
```

### 9.16.5 engine/operation.py (Write Orchestration Abstraction Layer)

**Purpose**: Provide a unified orchestration entry point for the insert / delete pipeline. **Unified orchestration, preserved representation** -- schema, buffer, parquet file types remain separate where they should be; only one abstraction layer at the orchestration level (Collection / WAL / MemTable / recovery).

**Why this layer exists**:

1. Collection's `_apply` is the single write path -- the sole entry point for any write operation; adding new operations in the future (schema migration, bulk import) only requires dispatch in one place
2. WAL.write / MemTable.apply both accept Operation, no need to open a separate method for each operation type
3. The recovery path becomes 5 lines of `for op in WAL.read_operations(): memtable.apply(op)`, eliminating nested row-by-row loops
4. Operation is a frozen dataclass + Arrow batch, does not hold Collection / WAL references -- pure description, no behavior

```python
# engine/operation.py

from dataclasses import dataclass
from typing import Union
import pyarrow as pa


@dataclass(frozen=True)
class InsertOp:
    """Transaction description for one insert call.

    batch's schema = wal_data_schema (contains _seq, _partition, user fields, $meta),
    each row has been assigned an independent _seq.
    """
    partition: str          # Single partition name
    batch: pa.RecordBatch   # Contains _seq, _partition, user fields, $meta

    @property
    def seq_min(self) -> int:
        """Minimum _seq in the batch."""

    @property
    def seq_max(self) -> int:
        """Maximum _seq in the batch."""

    @property
    def num_rows(self) -> int:
        """Batch row count."""


@dataclass(frozen=True)
class DeleteOp:
    """Transaction description for one delete call.

    batch's schema = wal_delta_schema (contains pk, _seq, _partition),
    the entire batch shares a single _seq. partition can be '_all' (delete across all partitions).
    """
    partition: str          # Can be ALL_PARTITIONS = "_all"
    batch: pa.RecordBatch   # Contains pk, _seq, _partition

    @property
    def seq(self) -> int:
        """The shared _seq of the batch."""

    @property
    def num_rows(self) -> int:
        """Batch row count (number of PKs being deleted)."""


Operation = Union[InsertOp, DeleteOp]
```

**What the Collection entry point looks like** (section 9.17 will rewrite insert/delete; this is illustrative):

```python
def insert(self, records, partition_name="_default"):
    self._validate_records(records)
    seq_start = self._alloc_seq(len(records))
    batch = self._build_wal_data_batch(records, partition_name, seq_start)
    op = InsertOp(partition=partition_name, batch=batch)
    self._apply(op)
    return [r[self.pk_field] for r in records]

def delete(self, pks, partition_name=None):
    seq = self._alloc_seq(1)
    partition = partition_name or ALL_PARTITIONS
    batch = self._build_wal_delta_batch(pks, partition, seq)
    op = DeleteOp(partition=partition, batch=batch)
    self._apply(op)
    return len(pks)

def _apply(self, op: Operation) -> None:
    """Single write path -- dispatch to raw batch interfaces.

    The Operation abstraction only lives in the engine layer. WAL / MemTable
    do not know about it, so dispatch must be done explicitly here.
    """
    if isinstance(op, InsertOp):
        self.wal.write_insert(op.batch)
        self.memtable.apply_insert(op.batch)
    else:  # DeleteOp
        self.wal.write_delete(op.batch)
        self.memtable.apply_delete(op.batch)
    if self.memtable.size() >= MEMTABLE_SIZE_LIMIT:
        self._trigger_flush()
```

**WAL / MemTable do not know about Operation** -- they still only accept raw `pa.RecordBatch`; dispatch is done in Collection.\_apply.

**Why this design**: Dependency hierarchy. `storage/` is at Level 2, `engine/` is at Level 5/6; having storage reverse-import engine's `Operation` type would break the layering. Operation is a transaction orchestration concept that belongs to the engine layer.

```python
class WAL:
    # Still raw batch interfaces (landed in Phase 1)
    def write_insert(self, record_batch: pa.RecordBatch) -> None: ...
    def write_delete(self, record_batch: pa.RecordBatch) -> None: ...

    # Recovery path is wrapped in engine/recovery.py:
    # raw WAL.recover() -> assembled into Iterator[Operation]
    @staticmethod
    def recover(wal_dir, wal_number) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        ...


class MemTable:
    # Still raw batch interfaces
    def apply_insert(self, batch: pa.RecordBatch) -> None: ...
    def apply_delete(self, batch: pa.RecordBatch) -> None: ...
```

**The unified replay of Operations is implemented in engine/recovery.py** (landed in Phase 3):

```python
# engine/recovery.py
def replay_wal_operations(
    wal_dir: str, wal_number: int, pk_field: str,
) -> Iterator[Operation]:
    """Read WAL files, yield Operations in _seq order.

    Implementation:
    1. WAL.recover(wal_dir, wal_number) -> (data_batches, delta_batches)
    2. Wrap each batch into InsertOp(partition=..., batch=b) / DeleteOp(...)
    3. Merge-sort by starting _seq then yield

    Replaying in _seq order ensures that after recovery, MemTable's max observed seq
    naturally equals the last yielded op.seq -- this is not required for correctness
    (MemTable is already seq-aware), but makes next_seq derivation cleaner.
    """
```

**Recovery benefits**:

```python
def execute_recovery(...):
    memtable = MemTable(schema)
    max_seq = manifest.current_seq
    for n in WAL.find_wal_files(wal_dir):
        for op in WAL.read_operations(wal_dir, n, pk_field):
            memtable.apply(op)
            if isinstance(op, InsertOp):
                max_seq = max(max_seq, op.seq_max)
            else:
                max_seq = max(max_seq, op.seq)
    return memtable, delta_index, max_seq + 1
```

**Parts that stay unchanged** (don't be tempted by "unification"):

- `insert_buf` and `delete_buf` internal representations **are not merged** (different semantics: override vs max accumulation)
- `wal_data_schema` and `wal_delta_schema` **are not merged** (schema is a data contract, not an orchestration concept)
- `InsertOp` and `DeleteOp` **do not inherit from a common base class** -- use `Union` + `isinstance` dispatch
- **Do not** add an `execute(collection)` method to Operation -- it would become a god object
- **Do not** let the storage layer import Operation -- dispatch stays in Collection.\_apply

### 9.17 engine/collection.py

```python
class Collection:
    """Collection layer: engine core, manages WAL / MemTable / Manifest / Compaction.

    Holds all storage component instances, orchestrates read/write paths.
    Automatically executes recovery during __init__.
    """

    def __init__(self, name: str, data_dir: str, schema: CollectionSchema) -> None:
        """Initialize Collection, load Manifest, execute recovery.
        Args:
            name: Collection name
            data_dir: Collection data directory (contains manifest.json, schema.json, wal/, partitions/)
            schema: CollectionSchema
        """

    # --- Write operations (partition_name: singular str) ---

    def insert(
        self,
        records: List[dict],
        partition_name: str = "_default",
    ) -> List:
        """Batch write, with natural upsert semantics (overrides if PK exists).

        Flow: validate -> allocate _seq -> WAL -> MemTable -> (trigger flush if full)
        Each record is assigned an independent _seq.

        Args:
            records: List[dict], each dict contains user fields
            partition_name: Target Partition

        Returns:
            List of written PKs

        Raises:
            SchemaValidationError: Record does not conform to Schema
            PartitionNotFoundError: Specified Partition does not exist
        """

    def delete(
        self,
        pks: List,
        partition_name: Optional[str] = None,
    ) -> int:
        """Batch delete, does not check whether records exist.

        Flow: allocate shared _seq -> WAL -> MemTable -> (trigger flush if full)
        Multiple PKs share a single _seq.

        Args:
            pks: List of PK values
            partition_name: Target Partition, None means delete across all Partitions

        Returns:
            Number of PKs processed

        Raises:
            PartitionNotFoundError: Specified Partition does not exist
        """

    # --- Read operations (partition_names: plural List[str]) ---

    def get(
        self,
        pks: List,
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,                  # Phase 8
    ) -> List[dict]:
        """Batch lookup by PK.

        Flow: MemTable lookup -> segment lookup (take max-seq version) -> delta_index filtering ->
              (if expr is provided) call filter.evaluate on matched rows for additional filtering.

        Args:
            pks: List of PK values
            partition_names: Search scope, None means search all Partitions
            expr: Optional Milvus-style filter expression (see sections 9.19-9.25). Matched
                pks must additionally satisfy this expression to appear in results.

        Returns:
            List[dict], each dict is one record (absence from return list = pk doesn't exist
            or was filtered out)
        """

    def search(
        self,
        query_vectors: List[list],
        top_k: int = 10,
        metric_type: str = "COSINE",
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,                  # Phase 8
        output_fields: Optional[List[str]] = None,   # Phase 9.1
    ) -> List[List[dict]]:
        """Vector retrieval.

        Flow: (if expr is provided) parse_expr -> compile_expr ->
              assemble_candidates(filter_compiled=...) ->
              execute_search_with_index(filter_mask=...)

        From Phase 9 onward:
        - Collection must be in loaded state before calling, otherwise raises CollectionNotLoadedError
        - Each segment uses index.search if .index exists, otherwise falls back to brute force
        - MemTable always uses brute force (latest writes have no index built)

        Args:
            query_vectors: List of query vectors, each element is list[float]
            top_k: Number of nearest neighbors to return
            metric_type: "COSINE" | "L2" | "IP"
            partition_names: Search scope, None means search all Partitions
            expr: Optional Milvus-style filter expression (see sections 9.19-9.25)
            output_fields: List of fields to retain in the returned entity; None means all

        Returns:
            Outer List = per query vector
            Inner List = top-K results
            Each entry: {"id": pk, "distance": float, "entity": {field: value}}
        """

    def query(                                       # Phase 8 new method
        self,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Pure scalar query (no vectors, no distance).

        Flow: parse_expr -> compile_expr -> assemble_candidates(no query) ->
              build_valid_mask(filter_mask=...) -> take all True rows ->
              project output_fields -> truncate to limit.

        Args:
            expr: Required, filter expression (see sections 9.19-9.25)
            output_fields: List of fields to return, None returns all fields (excluding _seq, _partition)
            partition_names: Search scope
            limit: Maximum number of rows to return (None = unlimited)

        Returns:
            List[dict], each dict is one matching record
        """

    # --- Partition management ---

    def create_partition(self, partition_name: str) -> None:
        """Create a new Partition. Creates directory + updates Manifest.
        Raises: PartitionAlreadyExistsError"""

    def drop_partition(self, partition_name: str) -> None:
        """Delete a Partition. Deletes directory + updates Manifest.
        Raises: DefaultPartitionError, PartitionNotFoundError"""

    def list_partitions(self) -> List[str]:
        """Return list of all Partition names."""

    def has_partition(self, partition_name: str) -> bool:
        """Phase 9.1: Check if a Partition exists."""

    # --- Statistics and description (Phase 9.1) ---

    @property
    def num_entities(self) -> int:
        """Phase 9.1: Total row count of Collection (memtable active rows + all segment rows - deleted)."""

    def describe(self) -> dict:
        """Phase 9.1: Return dict containing name / schema / num_entities /
        load_state / index_spec / partitions."""

    # --- Index lifecycle (Phase 9.3) ---

    def create_index(self, field_name: str, index_params: dict) -> None:
        """Phase 9.3: Persist IndexSpec to manifest. **Does not build immediately** --
        actual building happens at load() time.

        Args:
            field_name: Must be a vector field in the schema
            index_params: dict containing "index_type" / "metric_type" / "params" /
                "search_params"

        Raises:
            IndexAlreadyExistsError: create_index has already been called
            FilterFieldError: field_name is not a vector field
        """

    def drop_index(self, field_name: str) -> None:
        """Phase 9.3: Release in-memory indexes + delete all .idx files on disk +
        clear IndexSpec in manifest. After drop, _load_state is forced to released."""

    def has_index(self) -> bool:
        """Phase 9.3: Whether create_index has been called on the Collection."""

    def get_index_info(self) -> Optional[dict]:
        """Phase 9.3: Return IndexSpec.to_dict() or None."""

    def load(self) -> None:
        """Phase 9.3: State machine released -> loading -> loaded.

        For each segment, calls build_or_load_index: first tries to load .idx file,
        falls back to build + save on failure. Collections without IndexSpec are also
        allowed to load (directly enters loaded state, search uses brute force).

        Rolls back to released state on any exception.
        """

    def release(self) -> None:
        """Phase 9.3: Release all segments' in-memory indexes; state machine -> released."""

    @property
    def load_state(self) -> str:
        """Phase 9.3: "released" | "loading" | "loaded"."""

    # --- Schema changes ---

    def add_field(self, field: FieldSchema) -> None:
        """Add a new field to Schema. schema_version +1, persists the update.
        Leverages Parquet's natural schema evolution: old files missing the column automatically fill with null.
        Does not support deleting fields or modifying field types."""

    # --- Lifecycle ---

    def flush(self) -> None:
        """Manually trigger flush (without waiting for MemTable to be full).
        No-op when MemTable is empty."""

    def close(self) -> None:
        """Close Collection: flush residual data -> close WAL writer."""
```

### 9.18 db.py

```python
class MilvusLite:
    """DB layer: manages the lifecycle of multiple Collections.

    Corresponds to a root directory on disk; each Collection is a subdirectory under the root.
    """

    def __init__(self, root_dir: str) -> None:
        """Initialize DB, scan root_dir to discover existing Collections (lazy loading).
        Args:
            root_dir: Database root directory, auto-created if it doesn't exist
        """

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
    ) -> "Collection":
        """Create a new Collection.
        Creates subdirectory + writes schema.json + initializes Manifest + creates _default Partition.
        Returns: Collection instance
        Raises: CollectionAlreadyExistsError"""

    def get_collection(self, collection_name: str) -> "Collection":
        """Get an existing Collection (loads and executes recovery on first access).
        Raises: CollectionNotFoundError"""

    def drop_collection(self, collection_name: str) -> None:
        """Delete a Collection. Close -> delete entire subdirectory.
        Raises: CollectionNotFoundError"""

    def list_collections(self) -> List[str]:
        """Return list of all Collection names."""

    def has_collection(self, collection_name: str) -> bool:
        """Phase 9.1: Check if a Collection exists."""

    def get_collection_stats(self, collection_name: str) -> dict:
        """Phase 9.1: Return dict, containing at least a row_count field;
        directly consumed by gRPC adapter layer GetCollectionStatistics."""

    def close(self) -> None:
        """Close all loaded Collections."""
```

---

## Phase 8: search/filter Subsystem Interface Detailed Descriptions

**Goal**: Allow `Collection.search` / `get` / `query` to accept Milvus-style scalar filter expressions.

**Architecture**: Three-stage compilation + dual backend dispatcher.

```
source string  --parse_expr()-->  Expr (raw AST, schema-independent)
                                      |
                                      | compile_expr(expr, schema)
                                      v
                              CompiledExpr (field binding + type checking + backend tag)
                                      |
                                      | evaluate(compiled, table)
                                      v
                              pa.BooleanArray (length == table.num_rows)
```

**Architecture invariant supplement (written into the top-level "Architecture Invariants" section)**:

11. **Filter parser and evaluator are decoupled via AST** -- AST is the stable interface; future parser implementation swaps (e.g., ANTLR) do not affect type checker / backends.
12. **Filter backend is statically determined at compile time** -- no dispatch on the evaluate hot path; F1 always selects arrow; future F2b when encountering refs containing dynamic fields will upgrade to python.

### 9.19 search/filter/exceptions.py

```python
from milvus_lite.exceptions import MilvusLiteError

class FilterError(MilvusLiteError):
    """Base class for filter expression errors."""

class FilterParseError(FilterError):
    """Lexing or parsing failed.

    Carries (source, pos) for caret-style rendering:

        FilterParseError: unexpected token '>' at column 5
          age >> 18
              ^
    """
    def __init__(self, message: str, source: str, pos: int) -> None: ...

class FilterFieldError(FilterError):
    """Reference to a field that does not exist in the schema.

    Includes did-you-mean suggestion via difflib:

        FilterFieldError: unknown field 'agg' at column 1
          agg > 18
          ^^^
        did you mean 'age'?
    """

class FilterTypeError(FilterError):
    """Type mismatch in expression operands.

        FilterTypeError: type mismatch at column 7
          age > 'eighteen'
                ^^^^^^^^^
        left side is int (field 'age'), right side is string
    """
```

### 9.20 search/filter/tokens.py

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, List

class TokenKind(Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOL = "BOOL"
    IDENT = "IDENT"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    COMMA = ","
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "AND"     # and / AND / &&
    OR = "OR"       # or / OR / ||
    NOT = "NOT"     # not / NOT / !
    IN = "IN"       # in / IN
    SUB = "-"       # Unary negation (not recognized as part of INT literal)
    EOF = "EOF"

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str       # original source slice
    pos: int        # column in source
    value: Any      # parsed literal value (None for non-literals)

def tokenize(source: str) -> List[Token]:
    """Single-pass lexer.

    Behaviour (matches Milvus Plan.g4 where applicable):
        - Whitespace ' \\t \\r \\n' skipped (no comments)
        - Identifiers case-sensitive: [a-zA-Z_][a-zA-Z_0-9]*
        - Keywords case-insensitive: 'and'/'AND'/'&&', 'or'/'OR'/'||',
          'not'/'NOT'/'!', 'in'/'IN'
        - Booleans: 'true'/'True'/'TRUE', 'false'/'False'/'FALSE'
          (only these 6 forms — 'tRuE' rejected with did-you-mean)
        - Strings: "..." or '...' with escapes \\" \\' \\\\ \\n \\t \\r
        - Numbers: decimal int + decimal float + scientific notation
          (1, 3.14, 1e3, 1.5e-2). Negative sign is unary, not part of literal.
        - '==' is the equality operator; '=' alone raises FilterParseError
          with hint "did you mean '=='?".

    Raises:
        FilterParseError: on lex errors. Always carries source + pos.
    """
```

### 9.21 search/filter/ast.py

```python
from dataclasses import dataclass
from typing import Tuple, Union

# -- Literals --------------------------------------------------------

@dataclass(frozen=True)
class IntLit:
    value: int
    pos: int

@dataclass(frozen=True)
class FloatLit:
    value: float
    pos: int

@dataclass(frozen=True)
class StringLit:
    value: str
    pos: int

@dataclass(frozen=True)
class BoolLit:
    value: bool
    pos: int

@dataclass(frozen=True)
class ListLit:
    """Homogeneous literal list, used inside `in [...]`."""
    elements: Tuple["Literal", ...]
    pos: int

# -- Reference -------------------------------------------------------

@dataclass(frozen=True)
class FieldRef:
    name: str
    pos: int

# -- Operations ------------------------------------------------------

@dataclass(frozen=True)
class CmpOp:
    op: str          # "==", "!=", "<", "<=", ">", ">="
    left: "Expr"
    right: "Expr"
    pos: int

@dataclass(frozen=True)
class InOp:
    field: FieldRef
    values: ListLit
    negate: bool     # True for "not in"
    pos: int

@dataclass(frozen=True)
class And:
    operands: Tuple["Expr", ...]
    pos: int

@dataclass(frozen=True)
class Or:
    operands: Tuple["Expr", ...]
    pos: int

@dataclass(frozen=True)
class Not:
    operand: "Expr"
    pos: int

# -- Type aliases ----------------------------------------------------

Literal = Union[IntLit, FloatLit, StringLit, BoolLit]

Expr = Union[
    Literal, ListLit, FieldRef,
    CmpOp, InOp, And, Or, Not,
]
```

**Key design decisions**:
- 11 frozen dataclasses, all value semantics, hashable, automatic `__eq__`
- Uses `tuple` not `list` (frozen-friendly)
- No common base class -- uses `Union` + `isinstance` dispatch (consistent with Operation)
- No methods -- behavior is in the backends
- Each node carries `pos` for error message source tracing
- Node naming is simplified compared to Milvus: a single `CmpOp` (with op field) replaces Milvus's `Equality`/`Relational`

### 9.22 search/filter/parser.py

```python
class Parser:
    def __init__(self, tokens: List[Token], source: str) -> None: ...

    def parse(self) -> Expr:
        """Parse one expression and verify EOF."""

    # Pratt-style descent (low -> high precedence)
    def parse_or(self) -> Expr: ...      # prec 1: a or b or c
    def parse_and(self) -> Expr: ...     # prec 2: a and b and c
    def parse_not(self) -> Expr: ...     # prec 3: not a
    def parse_cmp(self) -> Expr: ...     # prec 4: a == b, a in [...]
    def parse_primary(self) -> Expr: ... # literal | ident | ( expr )

def parse_expr(source: str) -> Expr:
    """Public entry. Lex + parse."""
```

**Precedence table** (aligned with Milvus Plan.g4):

| Prec | Operator | Associativity |
|---|---|---|
| 1 | `or`, `OR`, `\|\|` | left |
| 2 | `and`, `AND`, `&&` | left |
| 3 | `not`, `NOT`, `!` (prefix) | right |
| 4 | `==, !=, <, <=, >, >=` | left (chained comparison is parse-accepted, semantic-rejected) |
| 4 | `in [...]`, `not in [...]` | non-assoc |
| 5 | unary `-` (prefix) | right |
| 6 | literal / ident / `(...)` | — |

**Restrictions tightened compared to Milvus**:
- RHS of `in` must be a literal array (Milvus accepts arbitrary expr, but in practice only literal arrays are used)
- Array literal elements must be literals (Milvus accepts expr, F1 only accepts literal)
- F1 does not support arithmetic / `like` / `is null` / `exists` / `$meta` / function calls -- these tokens are rejected at the lex or parse stage with a "Phase F2/F3 will support" hint

### 9.23 search/filter/semantic.py

```python
@dataclass(frozen=True)
class FieldInfo:
    name: str
    dtype: DataType
    nullable: bool

@dataclass(frozen=True)
class CompiledExpr:
    """Type-checked, schema-bound expression ready for evaluation."""
    ast: Expr
    fields: Dict[str, FieldInfo]   # all field names referenced
    backend: str                    # "arrow" | "hybrid" | "python"

def compile_expr(expr: Expr, schema: CollectionSchema) -> CompiledExpr:
    """Bind field references, check types, choose backend.

    Steps:
        1. Walk AST, collect all FieldRef
        2. For each: lookup in schema; reject reserved (_seq, _partition,
           $meta) or vector fields; produce did-you-mean on miss
        3. Walk again, infer + check types (cmp operands compat,
           list elements homogeneous, bool combinators bool operands)
        4. Choose backend:
           - "arrow"  for pure schema field expressions (fast path)
           - "hybrid" for $meta expressions (F3+: per-batch JSON
             preprocessing then arrow path)
           - "python" reserved for future UDF / truly dynamic things;
             not selected automatically in F3+ — only used as differential
             baseline and as hybrid_backend's runtime fallback target
        5. Wrap in CompiledExpr

    Raises:
        FilterFieldError: unknown field reference
        FilterTypeError:  operand type mismatch
    """
```

**Type inference rules**:

| Node | Inferred type | Validation |
|---|---|---|
| `IntLit` | `int` | — |
| `FloatLit` | `float` | — |
| `StringLit` | `string` | — |
| `BoolLit` | `bool` | — |
| `ListLit` | `list[T]` | Elements are mutually compatible |
| `FieldRef` | Schema-declared type | Must exist, not reserved, not vector |
| `CmpOp` | `bool` | Left and right types are compatible |
| `InOp` | `bool` | Field type is compatible with list element type |
| `And/Or/Not` | `bool` | Operand is bool |

**Compatible types**:
- `int ~ int` yes
- `int ~ float` yes (promotion)
- `string ~ string` yes
- `bool ~ bool` yes
- All others: no

### 9.24 search/filter/eval/arrow_backend.py

```python
import functools
import pyarrow as pa
import pyarrow.compute as pc

_CMP_KERNELS = {
    "==": pc.equal, "!=": pc.not_equal,
    "<":  pc.less,  "<=": pc.less_equal,
    ">":  pc.greater, ">=": pc.greater_equal,
}

def evaluate_arrow(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Walk the AST, translating each node into pyarrow.compute calls.

    NULL handling: top-level result is fill_null(False) so any null
    in operand chain becomes "no row matches" rather than three-valued
    result. AND/OR use the Kleene variants (and_kleene, or_kleene).
    """
```

**Dispatch table** (each AST node -> pyarrow.compute call):

| AST | pyarrow operation |
|---|---|
| `IntLit / FloatLit / StringLit / BoolLit` | `pa.scalar(value)` |
| `FieldRef` | `table.column(name)` |
| `CmpOp(op, l, r)` | `_CMP_KERNELS[op](_eval(l), _eval(r))` |
| `InOp(field, values, negate)` | `pc.is_in(col, value_set=values)`, `negate` -> `pc.invert` |
| `And(operands)` | `functools.reduce(pc.and_kleene, masks)` |
| `Or(operands)` | `functools.reduce(pc.or_kleene, masks)` |
| `Not(operand)` | `pc.invert(_eval(operand))` |

### 9.25 search/filter/eval/python_backend.py

```python
def evaluate_python(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Row-wise interpreter. Slow but flexible.

    Use cases:
        - Differential test baseline (arrow_backend / hybrid_backend output must match this)
        - hybrid_backend's runtime fallback for heterogeneous JSON types / arrow kernel incompatibility
        - Future F3 UDF / truly dynamic semantics ultimate fallback

    After F3+, no longer automatically selected by dispatcher ($meta expressions default to hybrid_backend).

    NULL semantics: Uses Kleene three-valued logic for AND/OR/NOT, final result None -> False.
    """
```

**Dispatch**: Mirrors arrow_backend, but each node accepts a row dict and returns a Python value:

| AST | Python operation |
|---|---|
| `IntLit` etc. | `node.value` |
| `FieldRef` | `row.get(node.name)` |
| `CmpOp(op, l, r)` | `_CMP_OPS[op](_eval(l, row), _eval(r, row))`, None propagation |
| `InOp` | Set lookup + optional negation |
| `And/Or` | Kleene three-valued short-circuit |
| `Not` | None propagation |

### 9.25a search/filter/eval/hybrid_backend.py (F3+)

```python
def evaluate_hybrid(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Per-batch JSON preprocessing + arrow_backend delegation.

    Strategy:
        1. collect_meta_keys(ast) — gather all $meta["key"] references
        2. _augment_table(data, keys) — pull $meta column once, json.loads
           each row, materialize one Arrow column per key under synthetic
           name __meta__<key>
        3. _rewrite_meta_access(ast, keys) — replace MetaAccess(key) with
           FieldRef("__meta__<key>") in a new AST tree
        4. dataclasses.replace the CompiledExpr with arrow backend tag
        5. Delegate to evaluate_arrow on the augmented table

    Performance: 100K rows + simple expression goes from ~500ms (python_backend
    row-wise) to ~50–100ms. Bottleneck shifts from per-row Python to JSON parse.

    Fallback: the entire augment + arrow eval is wrapped in try/except. Any
    failure (heterogeneous JSON types, null-typed synthetic column with no
    arrow kernel match, type promotion failures) falls back to evaluate_python
    for this single call. Per-evaluate fallback, not per-row.
    """
```

### 9.26 search/filter/eval/__init__.py

```python
def evaluate(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Backend dispatcher.

    Backend decision has already been made at compile_expr time and fixed in
    the CompiledExpr.backend field; evaluate hot path does not repeat the decision.
    """
    if compiled.backend == "arrow":
        return evaluate_arrow(compiled, data)
    if compiled.backend == "hybrid":
        return evaluate_hybrid(compiled, data)
    if compiled.backend == "python":
        return evaluate_python(compiled, data)
    raise ValueError(f"unknown filter backend: {compiled.backend!r}")
```

### 9.27 Phase 8 Implementation Phases

| Phase | Target grammar | Backend |
|---|---|---|
| **F1** done | Tier 1: `==/!=/<.../in/and/or/not` + literals + field references + parentheses | arrow_backend only (python_backend used as differential test baseline) |
| **F2a** done | + `like` + arithmetic (`+ - * / %`) + `is null` | Still arrow_backend |
| **F2b** done | + `$meta["key"]` dynamic fields | Introduces python_backend dispatch ($meta expressions auto-downgrade) |
| **F2c** done | + filter LRU cache + `query()` public method | Backend-independent |
| **F3+** done | Performance optimization: per-batch JSON preprocessing -> arrow path; hybrid replaces python as default $meta dispatch | Introduces hybrid_backend |
| **F3** | + `json_contains` / `array_contains` / UDF / strict Milvus compatibility | Extends python_backend; optional ANTLR parser swap |

### 9.28 Phase 8 Design References

- **Milvus Plan.g4**: `internal/parser/planparserv2/Plan.g4` (master branch)
- **Operator precedence, keyword case sensitivity, literal syntax all aligned with Milvus**
- **AST node shape inspired by Milvus PlanNode concepts**, but simplified (`CmpOp` replaces `Equality`/`Relational`)
- **F1 does not pursue binary compatibility** -- document our subset, consider in future F3
- **F1 uses hand-written Pratt parser** rather than ANTLR: F1 grammar is small, error messages are friendlier, zero dependencies. AST is the stable interface; switching to ANTLR in F3 does not affect type checker / backends

---

## 10. Phase 9 Index Subsystem Interface Detailed Descriptions

**Goal**: Upgrade `Collection.search` retrieval path from NumPy brute-force scan to ANN, defaulting to FAISS HNSW. See `plan/index-design.md` for in-depth design.

### 10.1 index/protocol.py

```python
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

class VectorIndex(ABC):
    """Abstract per-segment vector index.

    Implementations: BruteForceIndex, FaissHnswIndex, future FaissIvf*Index.
    Lifetime: build -> save -> load -> search -> close. Indexes are immutable --
    no add/remove after build.
    """

    metric: str       # "COSINE" | "L2" | "IP"
    num_vectors: int
    dim: int

    @classmethod
    @abstractmethod
    def build(cls, vectors: np.ndarray, metric: str, params: dict) -> "VectorIndex": ...

    @abstractmethod
    def search(
        self,
        queries: np.ndarray,
        top_k: int,
        valid_mask: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (local_ids, distances), each (nq, top_k).
        Distance convention: smaller = more similar (regardless of metric).
        valid_mask is the bitmap pipeline output AFTER dedup + tombstone +
        scalar filter, used as IDSelectorBitmap (or numpy slice for brute-force)."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, metric: str, dim: int) -> "VectorIndex": ...

    @property
    @abstractmethod
    def index_type(self) -> str:
        """'BRUTE_FORCE' | 'HNSW' | 'IVF_FLAT' | ..."""
```

### 10.2 index/spec.py

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class IndexSpec:
    field_name: str          # which vector field this index covers
    index_type: str          # "BRUTE_FORCE" | "HNSW" | ...
    metric_type: str         # "COSINE" | "L2" | "IP"
    build_params: Dict       # {"M": 16, "efConstruction": 200}
    search_params: Dict = field(default_factory=dict)  # {"ef": 64}

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> "IndexSpec": ...
```

### 10.3 index/brute_force.py

```python
class BruteForceIndex(VectorIndex):
    """NumPy implementation, zero external dependencies. Differential test baseline + fallback when faiss is unavailable.

    Internally stores full vectors; search directly calls search/distance.compute_distances.
    valid_mask is implemented via vectors[mask].
    """

    @classmethod
    def build(cls, vectors, metric, params): ...
    def search(self, queries, top_k, valid_mask=None, params=None): ...
    def save(self, path): ...    # numpy .npy
    @classmethod
    def load(cls, path, metric, dim): ...

    index_type = "BRUTE_FORCE"
```

### 10.4 index/faiss_hnsw.py

```python
import faiss
import numpy as np

class FaissHnswIndex(VectorIndex):
    """FAISS HNSW implementation.

    Key details:
    - Metric sign is normalized internally (FAISS L2 is squared, IP is larger-is-more-similar ->
      upper layer unified to "smaller is more similar")
    - cosine: both vectors and query are L2 normalized, then use IndexFlatIP
    - IDSelectorBitmap accepts numpy bool mask (packbits little-endian)
    - faiss.write_index / read_index for persistence
    """

    @classmethod
    def build(cls, vectors, metric, params): ...
    def search(self, queries, top_k, valid_mask=None, params=None): ...
    def save(self, path): ...    # faiss.write_index
    @classmethod
    def load(cls, path, metric, dim): ...

    index_type = "HNSW"
```

### 10.5 index/factory.py

```python
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

def build_index_from_spec(spec: IndexSpec, vectors: np.ndarray) -> VectorIndex:
    """Select implementation based on IndexSpec.index_type.

    Raises:
        IndexBackendUnavailableError: HNSW/IVF requested but faiss is unavailable
        ValueError: Unrecognized index_type
    """

def load_index(path: str, spec: IndexSpec, dim: int) -> VectorIndex:
    """Deserialize disk file based on IndexSpec.index_type."""
```

### 10.6 storage/segment.py Phase 9 Extension

```python
class Segment:
    __slots__ = (..., "index")    # New addition

    index: Optional[VectorIndex]

    def attach_index(self, index: VectorIndex) -> None:
        """Inject index, idempotent."""

    def release_index(self) -> None:
        """Release index reference."""

    def build_or_load_index(self, spec: IndexSpec, index_dir: str) -> None:
        """First try to load .idx file, falls back to build + save on failure.
        Path convention: indexes/<data_filename_stem>.<index_type_lowercase>.idx"""
```

### 10.7 Index File Naming and Path Conventions

```
data_dir/collections/<col>/partitions/<partition>/
├── data/
│   └── data_000001_000500.parquet
└── indexes/
    └── data_000001_000500.hnsw.idx        # stem + index_type.lower()
```

**Strict invariant** (architecture invariant section 11): A segment's .idx file name is uniquely determined by segment file name + IndexSpec.index_type; deleting a segment must simultaneously delete its .idx.

### 10.8 Phase 9 Implementation Phases

| Phase | Content | Effort |
|---|---|---|
| **9.1** | Fill in pymilvus quickstart prerequisite APIs (5 new Collection methods + db.get_collection_stats) | S |
| **9.2** | VectorIndex protocol + BruteForceIndex + Segment.index + execute_search_with_index path | M |
| **9.3** | IndexSpec + Manifest v2 + Collection.create_index/drop_index/load/release/has_index/get_index_info + _load_state state machine | M |
| **9.4** | Index file persistence + flush/compaction/recovery hooks + orphan cleanup | M |
| **9.5** | FaissHnswIndex + factory + metric alignment + differential tests | L |
| **9.6** | m9 demo + long-run tests | S |

---

## 11. Phase 10 gRPC Adapter Layer Interface Detailed Descriptions

**Goal**: Allow pymilvus clients to connect to MilvusLite without any code changes. See `plan/grpc-adapter-design.md` for in-depth design.

### 11.1 adapter/grpc/server.py

```python
def run_server(
    data_dir: str,
    host: str = "0.0.0.0",
    port: int = 19530,
    max_workers: int = 10,
) -> None:
    """Start gRPC server, blocks until KeyboardInterrupt."""
```

### 11.2 adapter/grpc/servicer.py — RPC Mapping Table

| RPC | engine API | Notes |
|---|---|---|
| `CreateCollection` | `db.create_collection(name, schema)` | translator: schema.py |
| `DropCollection` | `db.drop_collection(name)` | |
| `HasCollection` | `db.has_collection(name)` | |
| `DescribeCollection` | `db.get_collection(name).describe()` + schema serialization | |
| `ShowCollections` | `db.list_collections()` | |
| `GetCollectionStatistics` | `col.num_entities` | row_count field |
| `CreatePartition` | `col.create_partition(name)` | |
| `DropPartition` | `col.drop_partition(name)` | |
| `HasPartition` | `col.has_partition(name)` | |
| `ShowPartitions` | `col.list_partitions()` | |
| `Insert` / `Upsert` | `col.insert(records, partition)` | translator: records.py — FieldData column-row transposition |
| `Delete(ids=)` | `col.delete(pks, partition)` | |
| `Delete(filter=)` | `col.query(filter) -> extract pk -> col.delete` | |
| `Query` | `col.query(expr, output_fields, partition_names, limit)` or `col.get(pks, ...)` | id expression goes to get |
| `Search` | `col.search(query_vectors, top_k, metric_type, partition_names, expr, output_fields)` | translator: search.py + result.py |
| `CreateIndex` | `col.create_index(field, params)` | translator: index.py |
| `DropIndex` | `col.drop_index(field)` | |
| `DescribeIndex` | `col.get_index_info()` | |
| `LoadCollection` | `col.load()` | |
| `ReleaseCollection` | `col.release()` | |
| `GetLoadState` | `col.load_state` enum mapping | |
| `Flush` | `col.flush()` | |
| `ListDatabases` | stub returns `["default"]` | |
| Aliases / RBAC / Backup / Replica / ResourceGroup / etc | — | UNIMPLEMENTED + friendly message |

### 11.3 adapter/grpc/translators/records.py

```python
def fields_data_to_records(
    fields_data: List["FieldData"],
    num_rows: int,
) -> List[Dict[str, Any]]:
    """Milvus columnar -> engine row-wise.

    Supported types: INT64 / INT32 / VARCHAR / BOOL / FLOAT / DOUBLE / FLOAT_VECTOR / JSON
    Unsupported: BinaryVector / SparseFloatVector / Float16/BFloat16 -> UnsupportedFieldTypeError
    """

def records_to_fields_data(
    records: List[Dict[str, Any]],
    schema: "CollectionSchema",
    output_fields: Optional[List[str]] = None,
) -> List["FieldData"]:
    """engine row-wise -> Milvus columnar (for Query/Get/Search return)."""
```

### 11.4 adapter/grpc/errors.py

```python
_EXCEPTION_TO_CODE = {
    CollectionNotFoundError:       (4,   "CollectionNotExists"),
    CollectionAlreadyExistsError:  (1,   "CollectionAlreadyExists"),
    PartitionNotFoundError:        (200, "PartitionNotExists"),
    SchemaValidationError:         (6,   "IllegalArgument"),
    FilterParseError:              (6,   "IllegalArgument"),
    FilterTypeError:               (6,   "IllegalArgument"),
    FilterFieldError:              (6,   "IllegalArgument"),
    CollectionNotLoadedError:      (101, "CollectionNotLoaded"),
    IndexAlreadyExistsError:       (35,  "IndexAlreadyExists"),
    IndexNotFoundError:            (11,  "IndexNotExist"),
    IndexBackendUnavailableError:  (26,  "IndexBuildFailed"),
}

def to_grpc_status(exc: MilvusLiteError) -> dict:
    """Exception -> grpc Status code/reason, aligned with Milvus 2.3 numeric codes."""
```

### 11.5 Phase 10 Implementation Phases

| Phase | Content | Effort |
|---|---|---|
| **10.1** | proto pull + stub generation + empty servicer + run_server + CLI | M |
| **10.2** | Collection lifecycle RPC + translators/schema.py | M |
| **10.3** | insert/get/delete/query RPC + translators/records.py bidirectional transposition | L |
| **10.4** | search + create_index + load + release RPC + translators/{search,result,expr,index}.py | L |
| **10.5** | Partition + flush + stats + m10 demo + pymilvus quickstart smoke test | M |
| **10.6** | Error code mapping + UNIMPLEMENTED friendly messages | S |

### 11.6 Phase 10 Completion Criteria

- pymilvus quickstart full flow passes (create -> insert -> create_index -> load -> search -> query -> delete -> release -> drop)
- recall parity: grpc search and engine direct search top-k are identical
- Unsupported RPCs return `UNIMPLEMENTED` + friendly message (no silent fail)
- m10 demo passes
