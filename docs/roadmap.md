# MilvusLite Development Roadmap

This document is the vertical-slice development plan from the current state (only `wal.py` landed) to MVP. Each Phase is an end-to-end runnable state, with verification methods running throughout.

## Core Principles

1. **Design freeze first**: P1 design gaps identified in modules.md must be resolved at the document level before writing code.
2. **Vertical slice > horizontal layering**: Every milestone must run end-to-end; no stacking "finish the storage layer first".
3. **One demo per Phase**: `examples/m{N}_demo.py` serves as living documentation + smoke test.
4. **One git tag per Phase**: enables bisection if something breaks.
5. **Crash injection testing is the soul of LSM**: run routinely from Phase 3 onward.

---

## Phase 0 — Design Freeze (no code)

Commit prior discussion conclusions into `modules.md` / `wal-design.md`.

| Decision Item | Affected Module |
|---|---|
| MemTable cross-clear must be seq-aware (avoid recovery order-sensitive bugs) | MemTable |
| Tombstone GC rule: can discard when `delete_seq < min_active_data_seq` | DeltaIndex + Compaction |
| Concurrency: MVP **synchronous flush** (async deferred to future) | Collection |
| MemTable internal representation: append-only RecordBatch list + pk_index + delete_index | MemTable |
| Manifest keeps `.prev` backup | Manifest |
| Operation abstraction: `InsertOp` / `DeleteOp` unified orchestration layer | engine/operation.py |
| DeltaLog split into `delta_file.py` (IO) + `delta_index.py` (in-memory index) | Storage layer |
| WAL adds `sync_mode="close"` default fsync | WAL |
| LOCK file prevents multi-process access | db.py |
| Schema is immutable (MVP does not support alter) | Top-level invariant |

**Completion criteria**: modules.md diff is meaningful, unambiguous, no remaining "discuss later" items.

---

## Phase 1 — WAL Fixes + Basic Utilities

| Task | File | Verification |
|---|---|---|
| `_read_wal_file` uses `with` + tighten except | `storage/wal.py` | Unit test: truncated file does not leak fd |
| `close_and_delete` exception-safe (each writer has its own try/finally) | `storage/wal.py` | Unit test: mock close raises exception, verify `_closed=True` and second writer is also attempted to close |
| Remove dead parameter schema from `recover()` | `storage/wal.py` | Compile/test pass |
| `sync_mode="close"` + fsync on close | `storage/wal.py` | Unit test: fsync mock called once |
| `schema/validation.py` | new | Unit tests cover each DataType |
| `schema/persistence.py` | new | dump → load round-trip |
| `exceptions.py` fill in remaining | — | Naturally consumed by subsequent modules |

**Completion criteria**: `pytest tests/storage/test_wal.py tests/schema/` all green.

---

## Phase 2 — First Vertical Slice: insert/get working in memory

**Goal**: Can `insert()` a record and `get()` it back, only touching WAL (no flush).

**New modules**:

```
storage/manifest.py      # Minimal subset: load/save/add_partition/has_partition
engine/operation.py      # InsertOp / DeleteOp / Operation types
storage/memtable.py      # apply_insert / apply_delete / get / size
                         # Internal: RecordBatch list + pk_index + delete_index
engine/collection.py     # insert + get + _seq + WAL/MemTable orchestration
                         # Not implemented: flush, search, delete, compaction
```

**Demo**: `examples/m2_demo.py` — 5-line insert + get.

**Verification matrix**:

| Type | Content |
|---|---|
| Unit test `test_operation.py` | InsertOp / DeleteOp construction, properties |
| Unit test `test_manifest.py` | load non-existent file → initial state; save→load round-trip; `.prev` takes effect; version monotonically +1 |
| Unit test `test_memtable.py` | upsert override; delete + put order-sensitive counterexample (verify seq-aware); size accuracy |
| **Key unit test** | Out-of-order apply: first seq=7 insert, then seq=5 insert, then seq=6 delete → get(X) must return seq=7 data |
| Integration test | M2 demo passes; duplicate pk goes through upsert |

**Completion criteria**: M2 demo passes; `kill -9` then restart get returns nothing (expected, solved in next phase).

---

## Phase 3 — Persistence: flush + recovery

**Goal**: Data survives process crash. LSM core value proof point.

**New modules**:

```
storage/data_file.py     # write_data_file / read_data_file / parse_seq_range
storage/delta_file.py    # write_delta_file / read_delta_file
storage/delta_index.py   # add_batch / is_deleted / rebuild_from
storage/manifest.py      # Complete add/remove_data_file, current_seq
engine/flush.py          # Synchronous 7-step pipeline
engine/recovery.py       # 5-step recovery
```

**Collection upgrade**: Triggers synchronous flush; `__init__` calls recovery; delete still not implemented.

**Demo**: `examples/m3_demo.py` — write process `os._exit(0)`, read process recovers and can get the data.

**Verification matrix**:

| Type | Content |
|---|---|
| Unit test `test_data_file.py` | Read-write round-trip; parse_seq_range edge cases |
| Unit test `test_delta_index.py` | rebuild_from multiple files; is_deleted edge cases; GC rules |
| Unit test `test_flush.py` | 7-step manifest + disk state matches expectations at each step |
| Unit test `test_recovery.py` | wal-design.md §7.1 five scenarios A-E, one case each |
| **Crash injection** `test_crash_recovery.py` | `os._exit(0)` between each flush step, verify recovery is correct |
| **Property test** | hypothesis generates random insert sequences + flush + crash + recover, state is consistent |

**Key crash injection template**:

```python
@pytest.mark.parametrize("crash_after_step", range(1, 8))
def test_crash_during_flush(tmp_path, crash_after_step):
    # monkeypatch makes flush raise SystemExit after step N
    col = Collection("t", str(tmp_path), schema)
    col.insert([...])  # fill memtable
    with pytest.raises(SystemExit):
        col.insert([trigger])

    col2 = Collection("t", str(tmp_path), schema)  # recovery
    # Invariant: all "successfully inserted" records can be retrieved via get
    # Invariant: orphan files are cleaned up by recovery
```

**Completion criteria**: M3 demo passes; 7 crash injection cases all green; property test runs 1000 random cases with no failures.

---

## Phase 4 — Search

**Goal**: Brute-force KNN search works (covering both MemTable and Disk).

**New modules**:

```
search/distance.py       # cosine / l2 / ip
search/bitmap.py         # build_valid_mask
search/assembler.py      # segment cache + memtable → numpy assembly
search/executor.py       # execute_search
storage/segment.py       # data Parquet in-memory cache (pks/seqs/vectors numpy)
```

**Collection upgrade**: `search()` implemented; maintains `_segment_cache`; registers new segment after flush.

**Demo**: `examples/m4_demo.py` — 1000 vectors + top-10 query.

**Verification matrix**:

| Type | Content |
|---|---|
| Unit test `test_distance.py` | Three distance types compared with hand calculations |
| Unit test `test_bitmap.py` | Same pk multi-seq dedup; empty input; all deleted |
| Unit test `test_segment.py` | Load parquet → numpy shape is correct |
| Unit test `test_executor.py` | top-k correctness; nq > 1 |
| Integration `test_search_e2e.py` | 1000 random vectors, search results match numpy brute-force comparison |
| **Flush boundary** | Half in MemTable, half in Parquet, results correct |

**Completion criteria**: M4 demo top-10 matches numpy brute-force calculation.

---

## Phase 5 — Delete + Delta + Complete recovery

**Goal**: delete works end-to-end; delete recovery path works.

**New/upgraded**:

```
engine/collection.py     # Add delete()
engine/flush.py          # flush also writes delta parquet
engine/recovery.py       # Replay wal_delta + rebuild delta_index
search/bitmap.py         # Enable is_deleted filtering
```

**Demo**: `examples/m5_demo.py` — insert → delete → search does not return → restart still does not return.

**Verification matrix**:

| Type | Content |
|---|---|
| Unit test `test_collection_delete.py` | Single/multi pk delete; non-existent pk; cross-partition (`partition_name=None`) |
| Unit test `test_delta_index.py` | is_deleted correct after rebuild |
| **Key integration** `test_insert_delete_insert.py` | insert(X) → delete(X) → insert(X) → search returns new X (end-to-end verification of bug fix #1) |
| **Crash injection** | Crash during flush after delete, recovery shows X still deleted |
| **Property** | Random insert/delete mix + crash + recover state is consistent |

**Completion criteria**: M5 demo passes; 50 random interleaving orders of insert/delete all correct.

---

## Phase 6 — Compaction + Tombstone GC

**Goal**: Under long-running use, file count and deleted_map do not grow unbounded.

**New**:

```
engine/compaction.py     # CompactionManager, maybe_compact, size-tiered
                         # GC: calls delta_index.gc_below(min_active_seq)
```

**Collection upgrade**: flush calls `compaction_mgr.maybe_compact()` at the end.

**Demo**: `examples/m6_demo.py` — 1 million inserts + periodic deletes, verify file count and memory are bounded.

**Verification matrix**:

| Type | Content |
|---|---|
| Unit test `test_compaction.py` | Bucketing; trigger conditions; merge dedup correctness |
| Unit test `test_tombstone_gc.py` | Construct delete_seq < min_data_seq scenario, after GC entry is gone and query is still correct |
| Regression | All previous tests pass after triggering compaction |
| Crash injection | Crash while compaction writes new file → old files still present → query correct after recovery |
| **Long-run** `@pytest.mark.slow` | After M6 demo completes: assert data files ≤ 32; delete_index size ≤ 5% × total inserts |

**Completion criteria**: Long-run test passes; after 1 million inserts, file count and memory both have upper bounds.

---

## Phase 7 — DB Layer + Wrap-up

**Goal**: Multi-collection lifecycle; cross-collection isolation; clean public API.

**New/upgraded**:

```
db.py                    # MilvusLite.create_collection / drop_collection / list_collections
                         # LOCK file (fcntl.flock)
__init__.py              # Public API
```

**Demo**: `examples/m7_demo.py` — Create multiple collections, write to each, drop one, verify with list.

**Verification matrix**:

| Type | Content |
|---|---|
| Unit test `test_db.py` | create / drop / list; duplicate name create raises error; disk cleanup after drop |
| Multi-process isolation | Second process opening same data_dir is rejected by LOCK |
| Smoke `test_smoke_e2e.py` | Run all M2-M7 demos (CI entry point) |

**Completion criteria**: Can write a README quickstart that runs successfully.

---

## Phase 8 — Scalar Filter Expression

**Goal**: Enable `Collection.search` / `get` / `query` to accept Milvus-style filter expressions
(e.g., `"age > 18 and category == 'tech'"`), enabling "vector recall + scalar filter" hybrid queries.

**Architecture**: Three-stage compilation + dual backend dispatcher.

```
parse_expr(s)            → Expr (raw AST, schema-independent)
compile_expr(expr, schema) → CompiledExpr (type checking + backend selection)
evaluate(compiled, table) → pa.BooleanArray
```

**New modules** (see modules.md §9.19-9.28 for details):

```
milvus_lite/search/filter/
├── __init__.py        # parse_expr / compile_expr / evaluate
├── exceptions.py      # FilterError family
├── tokens.py          # Tokenizer
├── ast.py             # 11 frozen AST nodes
├── parser.py          # Pratt parser (inspired by Milvus Plan.g4)
├── semantic.py        # compile_expr + type inference
└── eval/
    ├── __init__.py    # backend dispatcher
    ├── arrow_backend.py  # pyarrow.compute (primary)
    └── python_backend.py # row-wise (fallback + differential baseline)
```

**Collection upgrade**:
- `search(query_vectors, ..., expr=None)`
- `get(pks, ..., expr=None)`
- **New method** `query(expr, output_fields=None, partition_names=None, limit=None) → List[dict]`
  (pure scalar query, no query vector)

### Sub-phase Breakdown

| Sub-phase | Grammar Increment | Backend | Status |
|---|---|---|---|
| **F1** | Tier 1: `==/!=/<.../in/and/or/not` + literals + field references + parentheses | arrow_backend only; python_backend only as differential test baseline | ✅ done |
| **F2a** | + `like` + arithmetic (`+ - * / %`) + `is null` | Still arrow_backend | ✅ done |
| **F2b** | + `$meta["key"]` dynamic fields | Introduce python_backend dispatch | ✅ done |
| **F2c** | filter LRU cache + `query()` public method | Backend-independent | ✅ done |
| **F3+** | Performance optimization: per-batch JSON preprocessing → arrow path; hybrid replaces python as default $meta dispatch | Introduce hybrid_backend; python_backend remains as fallback and differential baseline | ✅ done |
| **F3** | + `json_contains` / `array_contains` / UDF / strict Milvus compatibility | Extend python_backend; optional ANTLR parser swap | — |

### Phase F1 Task List

| # | Task | File |
|---|---|---|
| F1.1 | exceptions.py + rendering logic | `search/filter/exceptions.py` |
| F1.2 | tokens.py (TokenKind + Token + tokenize) | `search/filter/tokens.py` + `tests/.../test_tokens.py` |
| F1.3 | ast.py (11 frozen dataclasses) | `search/filter/ast.py` |
| F1.4 | parser.py (Pratt parser) | `search/filter/parser.py` + `test_parser.py` |
| F1.5 | semantic.py (compile_expr + type inference) | `search/filter/semantic.py` + `test_semantic.py` |
| F1.6 | eval/arrow_backend.py | + `test_arrow_backend.py` |
| F1.7 | eval/python_backend.py | + `test_python_backend.py` |
| F1.8 | eval/__init__.py (dispatcher) + differential tests | + `test_e2e.py` |
| F1.9 | filter/__init__.py public API | — |
| F1.10 | bitmap.py add filter_mask parameter | + test updates |
| F1.11 | assembler.py calls evaluator + returns mask | + `test_assembler_filter.py` |
| F1.12 | executor.py accepts filter_mask | + test updates |
| F1.13 | Collection.search/get add expr parameter | + `test_collection_filter.py` partial |
| F1.14 | Collection.query new method | + complete integration tests |
| F1.15 | __init__.py expose query / FilterError | + smoke supplement |
| F1.16 | examples/m8_demo.py | — |
| F1.17 | Run full pytest suite | — |

**M8 demo**: `examples/m8_demo.py` — 100 records with `age + category + score` fields
+ search/get/query three usage patterns + various expr expressions.

### Verification Strategy: Differential Testing

In `test_e2e.py` every case **runs both backends simultaneously**, asserting results are equal:

```python
@pytest.mark.parametrize("expr_str", [
    "age > 18",
    "category == 'tech'",
    "age in [10, 20, 30]",
    "age >= 18 and category == 'tech' or score > 0.5",
    "not (status == 'draft')",
    # ... 50+ cases
])
def test_arrow_python_equivalence(expr_str, sample_table, sample_schema):
    expr = parse_expr(expr_str)
    compiled = compile_expr(expr, sample_schema)

    arrow_result = evaluate_arrow(compiled, sample_table)

    py_compiled = CompiledExpr(ast=compiled.ast, fields=compiled.fields, backend="python")
    py_result = evaluate_python(py_compiled, sample_table)

    assert arrow_result.equals(py_result)
```

Differential testing is the safety net for F1: two implementations cross-validate each other, and a bug in either side is exposed by the other.
NULL three-valued logic, type promotion, and edge values — areas prone to bugs — are caught by symmetry.

### Not in Phase F1 Scope (explicitly deferred)

- ❌ `like` operator → F2a
- ❌ Arithmetic (`+, -, *, /, %`) → F2a
- ❌ NULL operators (`is null` / `is not null`) → F2a
- ❌ `$meta` dynamic fields → F2b
- ❌ JSON / array functions → F3
- ❌ UDF → F3
- ❌ Expression cache → F2c
- ❌ ANTLR-based parser → F3+ optional swap
- ❌ DuckDB backend → F3+ opt-in extra

### Completion Criteria

- F1 done: `col.search([[...]], expr="age > 18 and category in ['tech', 'news']")` works;
  differential tests 50+ cases all green; m8 demo passes.
- F2 done: `col.search(expr="title like 'AI%' and $meta['priority'] > 5")` works.
- F3+ done: hybrid_backend replaces python_backend as default $meta dispatch; differential tests hybrid vs python consistent.
- F3 done: passes pymilvus expression test suite subset.

---

## Phase 9 — Vector Index (FAISS HNSW + segment-level)

**Goal**: Upgrade the `Collection.search` retrieval path from NumPy brute-force scan to FAISS HNSW, reducing search latency by 1-2 orders of magnitude for 100K+ vectors.

**Architecture decisions** (see `plan/index-design.md` for details):
- **Index bound at segment-level**: Each data parquet file corresponds to one .idx file, 1:1 binding, naturally aligned with the LSM immutable architecture
- **Index library is FAISS-cpu**: `IDSelectorBitmap` is isomorphic with the Phase 8 bitmap pipeline; index family aligns with Milvus
- **BruteForceIndex kept long-term**: Differential test baseline + fallback when faiss is not installed
- **load/release state machine introduced in Phase 9.3**: Aligns with Milvus behavior

**New modules**:

```
milvus_lite/index/
├── __init__.py
├── protocol.py            # VectorIndex ABC
├── spec.py                # IndexSpec dataclass
├── brute_force.py         # BruteForceIndex
├── faiss_hnsw.py          # FaissHnswIndex (introduced in Phase 9.5)
└── factory.py             # build_index_from_spec / load_index
```

### Sub-phase Breakdown

| Sub-phase | Content | Changed Files | Effort |
|---|---|---|---|
| **9.1** | Fill in pymilvus quickstart prerequisite APIs: `Collection.create_partition / drop_partition / list_partitions / num_entities / describe` + `search(output_fields=...)` + `MilvusLite.get_collection_stats` | `engine/collection.py`, `db.py`, `search/executor.py` | S |
| **9.2** | `VectorIndex` protocol + `BruteForceIndex` + `Segment.index` + new `execute_search_with_index` path | new `index/`, `storage/segment.py`, `search/executor.py` | M |
| **9.3** | `IndexSpec` + Manifest v2 upgrade + `Collection.create_index / drop_index / load / release` + `_load_state` state machine + `CollectionNotLoadedError` | `engine/collection.py`, `storage/manifest.py`, `exceptions.py` | M |
| **9.4** | Index file persistence (`indexes/<stem>.<type>.idx`) + flush / compaction / recovery hooks + orphan cleanup | `engine/flush.py`, `engine/compaction.py`, `engine/recovery.py`, `storage/segment.py` | M |
| **9.5** | `FaissHnswIndex` + factory routing + metric alignment + IDSelectorBitmap + differential tests | new `index/faiss_hnsw.py`, `pyproject.toml`, `tests/index/test_index_differential.py` | L |
| **9.6** | `examples/m9_demo.py` + long-run tests + documentation backfill | new `examples/m9_demo.py` | S |

### Phase 9 Task List (key file details)

| # | Task | File |
|---|---|---|
| 9.1.1 | `Collection.create_partition / drop_partition / list_partitions / has_partition` | `engine/collection.py` + `tests/engine/test_collection_partition.py` |
| 9.1.2 | `Collection.num_entities` + `Collection.describe()` | `engine/collection.py` + `tests/engine/test_collection_describe.py` |
| 9.1.3 | `Collection.search(output_fields=...)` parameter support | `engine/collection.py`, `search/executor.py` + test updates |
| 9.1.4 | `MilvusLite.get_collection_stats(name)` | `db.py` + `tests/test_db.py` |
| 9.2.1 | `VectorIndex` protocol + `BruteForceIndex` implementation | new `index/protocol.py`, `index/brute_force.py` + `tests/index/test_brute_force_index.py` |
| 9.2.2 | `Segment.index / attach_index / release_index / build_or_load_index` | `storage/segment.py` + `tests/storage/test_segment_index.py` |
| 9.2.3 | `execute_search_with_index` new path + Collection.search switching | `search/executor.py` (new `executor_indexed.py` or extend) + `tests/search/test_executor_with_index.py` |
| 9.3.1 | `IndexSpec` dataclass + Manifest v2 schema | new `index/spec.py`, `storage/manifest.py` + `tests/storage/test_manifest_v2_compat.py` |
| 9.3.2 | `Collection.create_index / drop_index / has_index / get_index_info` | `engine/collection.py` + `tests/engine/test_collection_create_index.py` |
| 9.3.3 | `_load_state` state machine + `Collection.load / release` + `CollectionNotLoadedError` | `engine/collection.py`, `exceptions.py` + `tests/engine/test_collection_load_release.py` |
| 9.4.1 | Index file naming convention + `Segment.build_or_load_index` persistence path | `storage/segment.py` + `tests/index/test_index_persistence.py` |
| 9.4.2 | flush post-step build index hook (when in loaded state) | `engine/flush.py` + `tests/engine/test_flush_with_index.py` |
| 9.4.3 | compaction old .idx cleanup + new .idx build | `engine/compaction.py` + `tests/engine/test_compaction_with_index.py` |
| 9.4.4 | recovery startup `_cleanup_orphan_index_files` | `engine/recovery.py` + `tests/engine/test_recovery_orphan_idx.py` |
| 9.5.1 | `FaissHnswIndex` build / search / save / load + metric alignment | new `index/faiss_hnsw.py` + `tests/index/test_faiss_hnsw.py` |
| 9.5.2 | `IDSelectorBitmap` integration (packbits order, selector params) | `index/faiss_hnsw.py` + `tests/index/test_faiss_id_selector.py` |
| 9.5.3 | factory routing + try-import degradation | `index/factory.py`, `pyproject.toml` |
| 9.5.4 | Differential test recall@10 ≥ 0.95 + distance parity | `tests/index/test_index_differential.py` |
| 9.5.5 | Benchmark script (QPS comparison) | `examples/m9_benchmark.py` (optional) |
| 9.6.1 | `examples/m9_demo.py` | new |
| 9.6.2 | Long-run `@pytest.mark.slow` 100K + periodic compaction | `tests/test_smoke_index_longrun.py` |
| 9.6.3 | Run full pytest suite | — |

### Verification Strategy: Differential Testing + Recall

```python
# tests/index/test_index_differential.py
@pytest.mark.parametrize("dim,n,metric", [
    (4, 100, "COSINE"), (32, 10000, "L2"), (128, 10000, "IP"),
])
def test_faiss_hnsw_recall_vs_brute_force(dim, n, metric):
    vectors = np.random.randn(n, dim).astype(np.float32)
    queries = np.random.randn(20, dim).astype(np.float32)
    brute = BruteForceIndex.build(vectors, metric, {})
    faiss_idx = FaissHnswIndex.build(vectors, metric, {"M": 16, "efConstruction": 200})
    brute_ids, _ = brute.search(queries, 10)
    faiss_ids, _ = faiss_idx.search(queries, 10, params={"ef": 64})
    for i in range(20):
        recall = len(set(brute_ids[i]) & set(faiss_ids[i])) / 10
        assert recall >= 0.95
```

### Not in Phase 9 Scope

- IVF / IVF-PQ / OPQ and other quantization indexes → Phase 9.5+ onward
- GPU index → Future
- Async index build → Future
- Multi-vector fields → Future
- Sparse / Binary vector indexes → Future

### Completion Criteria

- `col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16}})` persists to manifest
- After `col.load()`, `col.search` uses FAISS path, 100K vector search QPS ≥ brute-force × 50
- After `col.release()`, search raises `CollectionNotLoadedError`
- Restart → load completes in seconds (directly reads .idx files)
- After compaction, .idx files are 1:1 synchronized
- recall@10 ≥ 0.95 all green
- m9 demo passes

---

## Phase 10 — gRPC Adapter Layer (pymilvus compatibility)

**Goal**: Build a gRPC service layer on top of the engine, allowing pymilvus clients to connect to MilvusLite without code modifications. The final mile for "local version of Milvus" protocol compatibility.

**Prerequisites**: Phase 9 must be completed first (CreateIndex / LoadCollection / Search RPCs directly map to Phase 9 APIs).

**Architecture decisions** (see `plan/grpc-adapter-design.md` for details):
- **Proto source**: Directly copy official milvus proto, but only implement the quickstart subset; others return `UNIMPLEMENTED` (Approach C — balances pymilvus compatibility + manageable implementation effort)
- **Error codes align with Milvus 2.3 numeric codes**
- **grpcio dependency is included in the default dependencies**

**New modules**:

```
milvus_lite/adapter/grpc/
├── __init__.py
├── server.py                  # run_server(data_dir, host, port)
├── servicer.py                # MilvusServicer
├── errors.py                  # MilvusLiteError → grpc Status
├── translators/
│   ├── schema.py
│   ├── records.py             # FieldData ↔ list[dict] column-row transposition
│   ├── search.py
│   ├── result.py
│   ├── expr.py
│   └── index.py
└── proto/                     # Generated stubs (committed to repo)
    ├── milvus_pb2.py
    ├── milvus_pb2_grpc.py
    ├── schema_pb2.py
    └── common_pb2.py
```

### Sub-phase Breakdown

| Sub-phase | Content | Effort |
|---|---|---|
| **10.1** | Proto fetch + stub generation + empty servicer + `run_server` + CLI | M |
| **10.2** | Collection lifecycle RPCs (create / drop / has / describe / list) + `translators/schema.py` | M |
| **10.3** | insert/get/delete/query RPCs + `translators/records.py` (FieldData ↔ records bidirectional transposition) | L |
| **10.4** | search + create_index + load + release RPCs + `translators/{search,result,expr,index}.py` | L |
| **10.5** | Partition RPCs + flush + stats + `examples/m10_demo.py` + pymilvus quickstart smoke test | M |
| **10.6** | Error code mapping + UNIMPLEMENTED friendly messages | S |

### Phase 10 Task List

| # | Task | File |
|---|---|---|
| 10.1.1 | Fetch milvus proto → `proto/`, record source commit | `adapter/grpc/proto/README.md` |
| 10.1.2 | Generate _pb2 / _pb2_grpc with grpcio-tools, commit | `adapter/grpc/proto/*_pb2.py` |
| 10.1.3 | `MilvusServicer` empty framework (inherit + all UNIMPLEMENTED) | `adapter/grpc/servicer.py` |
| 10.1.4 | `run_server` + unified CLI | `adapter/grpc/server.py`, `cmdline.py`, `pyproject.toml` |
| 10.1.5 | Server startup test (pymilvus.connect succeeds) | `tests/adapter/test_grpc_server_startup.py` |
| 10.2.1 | `translators/schema.py`: Milvus FieldSchema ↔ MilvusLite bidirectional | + `tests/adapter/test_grpc_translators_schema.py` |
| 10.2.2 | `CreateCollection / DropCollection / HasCollection / DescribeCollection / ShowCollections` | `servicer.py` + `tests/adapter/test_grpc_collection_lifecycle.py` |
| 10.3.1 | `translators/records.py`: FieldData ↔ records, covering all supported types | + `tests/adapter/test_grpc_translators_records.py` |
| 10.3.2 | `Insert / Upsert / Delete / Query / Get` RPCs | `servicer.py` + `tests/adapter/test_grpc_crud.py` |
| 10.4.1 | `translators/expr.py`: filter passthrough + unsupported function detection | + `tests/adapter/test_grpc_translators_expr.py` |
| 10.4.2 | `translators/index.py`: IndexParams ↔ IndexSpec | + `tests/adapter/test_grpc_translators_index.py` |
| 10.4.3 | `translators/search.py + result.py`: SearchRequest parsing + SearchResults generation | + `tests/adapter/test_grpc_translators_search.py` |
| 10.4.4 | `Search / CreateIndex / DropIndex / DescribeIndex / LoadCollection / ReleaseCollection / GetLoadState` | `servicer.py` + `tests/adapter/test_grpc_search.py`, `test_grpc_index.py` |
| 10.5.1 | `CreatePartition / DropPartition / ShowPartitions / HasPartition / Flush / GetCollectionStatistics / ListDatabases` | `servicer.py` + `tests/adapter/test_grpc_partition.py` |
| 10.5.2 | `examples/m10_demo.py` | new |
| 10.5.3 | pymilvus quickstart L3 smoke test | `tests/adapter/test_grpc_quickstart.py` |
| 10.6.1 | `errors.py`: MilvusLiteError → ErrorCode mapping | + `tests/adapter/test_grpc_error_mapping.py` |
| 10.6.2 | servicer exception wrapping middleware | `servicer.py` |
| 10.6.3 | UNIMPLEMENTED stub friendly messages (aliases / RBAC stubs ...) | `servicer.py` |

### Verification Strategy: pymilvus quickstart smoke test

```python
# tests/adapter/test_grpc_quickstart.py
def test_pymilvus_quickstart(grpc_server):
    client = MilvusClient(uri=f"http://localhost:{grpc_server.port}")
    client.create_collection("demo", dimension=4)
    client.insert("demo", data=[{"id": i, "vector": [float(i)]*4} for i in range(100)])
    client.flush("demo")
    client.create_index("demo", index_params={
        "field_name": "vector", "index_type": "HNSW",
        "metric_type": "COSINE", "params": {"M": 16},
    })
    client.load_collection("demo")
    res = client.search("demo", data=[[0.1, 0.2, 0.3, 0.4]], limit=10)
    assert len(res[0]) == 10
    client.query("demo", filter="id >= 50", limit=20)
    client.delete("demo", ids=[1, 2, 3])
    client.release_collection("demo")
    client.drop_collection("demo")
```

### Not in Phase 10 Scope

- TLS / mTLS encryption → Future
- Token / Username-Password authentication → Future
- RBAC / Multi-tenancy → Future (unnecessary for embedded)
- Backup / Restore RPCs → Future
- Bulk insert / Import → Future
- Replica / Resource Group → Future
- Aliases → Future
- Search iterator / pagination → Future
- Database multi-instance → Future (always default)
- Binary vector types → Future

### Completion Criteria

- `milvus-lite server --data-dir ./data --port 19530` starts server
- pymilvus quickstart full flow passes
- Recall parity test: grpc search and engine direct search top-k are identical
- All unsupported RPCs return `UNIMPLEMENTED` + friendly message (no silent fail)
- m10 demo passes

---

## Phase 11 — Full Text Search

**Goal**: Support Milvus-compatible BM25 full text search + text_match filtering; pymilvus users can directly use `Function(type=BM25)` for text search.

**Prerequisites**: Phase 10 complete. gRPC adapter layer provides FunctionSchema / SparseFloatArray proto infrastructure.

**Detailed design document**: `plan/fts-design.md`

### Core Architecture

```
Insert: text → Analyzer tokenization → BM25 Function → {term_hash: TF} → sparse vector column
Load:   sparse vectors → build inverted index + per-segment statistics (docCount, avgdl, df)
Search: query text → tokenization → inverted index lookup → BM25 scoring → top-k
```

### Task Breakdown

| ID | Task | Deliverable |
|---|---|---|
| 11.1 | Schema extension ✅ | DataType.SPARSE_FLOAT_VECTOR, Function/FunctionType, FieldSchema new attributes, validation, persistence |
| 11.2 | Analyzer tokenization subsystem | `analyzer/` package: StandardAnalyzer (regex) + JiebaAnalyzer (optional) + factory + hash |
| 11.3 | Sparse vector storage | sparse_to_bytes/bytes_to_sparse codec; WAL/Parquet sparse vector column support |
| 11.4 | BM25 Function engine | Auto-tokenize on insert → generate TF sparse vectors; engine search supports anns_field |
| 11.5 | Sparse inverted index + BM25 search | SparseInvertedIndex: build/search/save/load; integrated into segment state machine |
| 11.6 | text_match filter | Filter subsystem adds text_match function; three backend implementations |
| 11.7 | gRPC adapter layer extension | FunctionSchema translation; SparseFloatArray codec; BM25 search request handling |
| 11.8 | Integration tests | pymilvus end-to-end BM25 search + text_match + hybrid scenario tests |

### Key Design Decisions

- **Per-segment inverted index**: Consistent with Phase 9's VectorIndex 1:1 binding; immutable segment → immutable index
- **BM25 computed at query time**: Insert stores TF; search computes IDF + BM25 score in real-time based on per-segment statistics
- **term ID = hash**: MurmurHash3 maps term → uint32, no global vocabulary needed
- **Distance convention**: distance = -bm25_score (negated, consistent with VectorIndex protocol)
- **anns_field parameter**: search API supports specifying which vector field to search, breaking the single-vector limitation

### Not in Phase 11 Scope

- Multi-Analyzer (multi-language dynamic selection) → Future
- phrase_match (phrase matching + slop) → Future
- LexicalHighlighter (result highlighting) → Future
- TextEmbedding Function → Future
### Completion Criteria

- pymilvus BM25 full text search end-to-end passes
- text_match filter works in combination with vector search
- BM25 index correctly rebuilt after Flush / Compaction
- Load / Release state machine covers BM25 index
- All existing tests do not regress

---

## Phase 12 — Hybrid Search (Multi-route Vector Fusion Retrieval)

**Goal**: Support pymilvus `hybrid_search()` API, allowing simultaneous execution of multiple ANN searches (e.g., dense COSINE + BM25 sparse) with results merged via Reranker.

**Prerequisites**: Phase 11 complete. BM25 search + dense vector search + `anns_field` routing are ready.

### Core Architecture

```
pymilvus.hybrid_search(reqs=[dense_req, bm25_req], ranker=WeightedRanker(0.6, 0.4))
  ↓
HybridSearchRequest proto (multiple SearchRequests + rank_params)
  ↓
servicer.HybridSearch:
  1. Parse each sub-SearchRequest → (query_vectors, anns_field, metric, filter, limit)
  2. Call Collection.search() for each route separately → List[List[dict]]
  3. Apply Reranker to merge results → unified top-k
  4. Build SearchResults response
```

### Task Breakdown

| ID | Task | Deliverable |
|---|---|---|
| 12.1 | Reranker implementation | `adapter/grpc/reranker.py`: WeightedRanker (weighted score normalization merge) + RRFRanker (Reciprocal Rank Fusion) |
| 12.2 | HybridSearch RPC implementation | `servicer.py`: Parse HybridSearchRequest → multi-route search → rerank → SearchResults |
| 12.3 | Integration tests | pymilvus `hybrid_search()` end-to-end: dense+BM25, multi-dense, filter, output_fields |

### Key Design Decisions

**Reranker strategies**:

| Strategy | Formula | Description |
|---|---|---|
| **WeightedRanker** | `final_score = Σ(weight_i × normalize(score_i))` | Normalize each route's scores to [0,1] then weighted sum |
| **RRFRanker** | `final_score = Σ 1/(k + rank_i)` | Rank-based fusion, k defaults to 60, independent of score scale |

**Score normalization** (required by WeightedRanker):
- Different routes have different distance scales (dense COSINE ∈ [0,2], BM25 score ∈ (-∞,0])
- Normalization method: per-query min-max → [0,1], then `1 - normalized` (unified to higher = better)

**HybridSearchRequest parsing**:
- `requests`: repeated SearchRequest, each sub-request has independent placeholder_group, anns_field, search_params, dsl (filter)
- `rank_params`: KeyValuePair list, containing strategy ("weighted"/"rrf"), params (weights/k), limit, offset
- `output_fields`: global output fields, shared across all sub-searches

**Result merging**:
- Each route returns `List[List[dict]]` (nq × top_k_per_route)
- Reranker deduplicates by pk + merges scores → global top-k
- When the same pk appears in multiple routes' results, the merge strategy is determined by the Reranker

### Not in Phase 12 Scope

- FunctionScore reranker (external function re-ranking) → Future
- group_by grouped re-ranking → Future
- Async parallel execution of multi-route searches → Future (MVP serial is sufficient)

### Completion Criteria

- pymilvus `hybrid_search(reqs=[dense, bm25], ranker=WeightedRanker(...))` end-to-end passes
- pymilvus `hybrid_search(reqs=[...], ranker=RRFRanker(...))` end-to-end passes
- Each sub-search can have independent filter expressions
- Results are correctly sorted by reranker score
- All existing tests do not regress

---

## Phase 13 — Group By Search (Search Result Grouping & Deduplication)

**Goal**: Support pymilvus `search(group_by_field=...)` and `hybrid_search(group_by_field=...)`, returning search results grouped by a scalar field.

**Prerequisites**: Phase 12 complete.

### Core Mechanism

```
Search → top-N candidates → group by group_by_field → top group_size per group → return first limit groups
```

### Task Breakdown

| ID | Task | Deliverable |
|---|---|---|
| 13.1 | Engine layer group_by post-processing | Collection.search() adds group_by_field/group_size/strict_group_size parameters |
| 13.2 | gRPC adaptation | search_params parses group_by parameters; SearchResultData adds group_by_field_value |
| 13.3 | Tests | pymilvus end-to-end group search + hybrid group_by |

### Completion Criteria

- pymilvus `search(group_by_field="category", group_size=3)` end-to-end passes
- strict_group_size=True/False behavior is correct
- group_by works in combination with hybrid_search
- Supports INT64/VARCHAR/BOOL grouping fields

---

## Phase 14 — Range Search (Distance Range Filtered Search)

**Goal**: Support pymilvus `search(search_params={"params": {"radius": ..., "range_filter": ...}})` distance range filtering.

### Parameter Semantics

- `radius`: distance lower bound (exclusive), `range_filter`: distance upper bound (inclusive)
- Result range: `radius < distance <= range_filter`
- Both parameters are optional; when both present, requires `radius < range_filter`

### Task Breakdown

| ID | Task | Deliverable |
|---|---|---|
| 14.1 | Engine layer range filtering | search() adds radius/range_filter, filters by distance after search |
| 14.2 | gRPC adaptation + tests | search_params parsing + pymilvus end-to-end tests |

### Completion Criteria

- pymilvus `search(params={"radius": ..., "range_filter": ...})` end-to-end passes
- Range filtering correct for each metric: L2/COSINE/IP/BM25
- Correctly handles cases with only radius or only range_filter

---

## Phase 15 — Auto ID (Auto-increment Primary Key)

**Completed.** FieldSchema.auto_id=True, INT64 primary keys auto-increment generated.

---

## Phase 16 — Iterator (query_iterator / search_iterator)

**Completed.** query(expr=None) returns all records, supports pymilvus client-side pk cursor pagination and distance range pagination.

---

## Phase 17 — Offset Pagination

**Completed.** search(offset=N) and query(offset=N) skip the first N results.

---

## Phase 18 — Multi-Vector Independent Indexing

**Completed.** Manifest/Segment/Collection refactored from single IndexSpec to Dict[str, IndexSpec], each vector field indexed independently.

---

## Performance Optimizations (Completed)

1. **Batch .to_pylist()** — BM25 search segment data batch conversion replaces row-by-row .as_py()
2. **Lazy materialization** — assembler/executor/memtable full pipeline only materializes records for top-k winners
3. **BM25 segment-level index caching** — immutable segment inverted indexes built once, reused permanently

---

## CRUD Alignment Fixes (Completed)

- delete(filter=...) no longer requires load_collection
- query(output_fields=["count(*)"]) count aggregation
- get(ids, output_fields=[...]) field filtering
- search(round_decimal=N) distance rounding
- output_fields=["*"] wildcard expansion
- JSON field field["key"] path filter syntax
- JSON dict value Arrow serialization
- Nullable FLOAT_VECTOR end-to-end support

---

## CI/CD + Packaging (Completed)

- GitHub Actions: Python 3.10-3.13 × ubuntu + macos matrix testing
- PyPI packaging: Apache-2.0 license, complete metadata
- 1529 tests, 0 skip

---

## TODO

### Code TODOs

| Location | Description |
|------|------|
| `engine/collection.py:552` | BM25 per-segment IDF should be changed to global statistics (cross-segment aggregation of doc_count/avgdl/df) |

### Unimplemented RPCs (UNIMPLEMENTED stubs)

| RPC | Reason |
|-----|------|
| AlterCollection | Schema is immutable |
| LoadPartitions / ReleasePartitions | Only collection-level load/release supported |

### Feature Comparison with Milvus (updated 2026-04-17)

#### Aligned

| Feature | Description |
|------|------|
| Collection CRUD | create/drop/has/describe/list/rename |
| Insert / Upsert / Delete | upsert semantics + partial update (field merge) |
| Get / Query / Search | Including scalar filter expressions |
| Vector Search (dense) | COSINE / L2 / IP |
| HNSW / HNSW_SQ Index | FAISS HNSW + scalar quantization |
| IVF_FLAT / IVF_SQ8 Index | FAISS IVF family |
| AUTOINDEX | Automatic index type selection |
| BruteForce Index | Small segment fallback + differential baseline |
| SPARSE_INVERTED_INDEX | BM25 inverted index |
| Load / Release state machine | Aligned with Milvus behavior |
| Partition management | create/drop/list/has |
| Partition Key | Automatic hash bucket routing |
| Array field type | DataType.ARRAY + array_contains/all/any + array_length + subscript access |
| Scalar Filter | `==, !=, <, >, in, and, or, not, like, is null, $meta["key"]` |
| gRPC protocol compatibility | pymilvus direct connection |
| BM25 full text search | Function(type=BM25) + text_match |
| TEXT_EMBEDDING Function | Automatic text-to-vector conversion |
| RERANK Function | Cohere + Decay (gauss/exp/linear) |
| Hybrid Search | WeightedRanker + RRFRanker |
| Group By Search | group_by_field + group_size |
| Range Search | radius + range_filter |
| Auto ID | INT64 auto-increment |
| Nullable fields + Default values | Including is null / is not null |
| Dynamic fields | enable_dynamic_field + $meta["key"] |
| Iterator | Client-side pk cursor / distance range pagination |
| Offset pagination | search/query(offset=N) |
| Multi-vector independent indexing | Each vector field has independent IndexSpec |
| Compaction | Size-Tiered + tombstone GC |
| count(\*) aggregation | query(output_fields=["count(*)"]) |
| output_fields=["\*"] | Wildcard expansion |
| JSON field + path filtering | field["key"] syntax |
| Sparse Vector (BM25) | SPARSE_FLOAT_VECTOR |
| round_decimal | Search distance rounding |
| Analyzer | Standard + Jieba Chinese/English tokenization |
| Crash Recovery | WAL + crash injection tests |
| delete(filter=...) without load | Fixed |

#### Feature Gaps

##### P0 — Directly related to basic CRUD, frequently used by users

| Feature | Corresponding Milvus API | Description | Effort |
|------|-----------------|------|--------|
| Alias management | create/drop/alter/describe/list_aliases | **Completed** — collection aliases persisted in `aliases.json` and accepted by read/write APIs | Small |
| Truncate Collection | truncate_collection | **Completed** — clears collection data while preserving schema | Small |
| list_indexes | list_indexes | **Completed** — returns field-name index names via DescribeIndex for pymilvus iterator compatibility | Small |
| get_partition_stats | get_partition_stats | **Completed** — partition-level `row_count` | Small |
| Search Iterator | search_iterator | Large result set paginated traversal (server-side cursor) | Medium |
| Query Iterator | query_iterator | Query result paginated traversal (distinct from offset/limit) | Medium |

##### P1 — Commonly used advanced features, affecting user migration experience

| Feature | Corresponding Milvus API | Description | Effort |
|------|-----------------|------|--------|
| LoadPartitions / ReleasePartitions | load_partitions / release_partitions | Partition-level load and release | Medium |
| Schema changes | add_collection_field / alter_collection_field | Dynamically add fields, modify field properties | Large |
| Collection properties | alter/drop_collection_properties | Collection-level configuration (TTL, etc.) | Medium |
| Index properties | alter/drop_index_properties | Modify index parameters | Small |
| Collection Functions | add/alter/drop_collection_function | Dynamically manage BM25 and other functions | Medium |
| query order_by | query(... order_by=...) | Scalar sorting (Milvus 2.5+) | Medium |
| Scalar Index | INVERTED / BITMAP index on scalar fields | Scalar filter performance for large datasets | Large |
| FLOAT16 / BFLOAT16 vectors | FLOAT16_VECTOR / BFLOAT16_VECTOR | Memory-efficient vector formats | Medium |

##### P2 — Advanced features, needed for specific scenarios

| Feature | Corresponding Milvus API | Description | Effort |
|------|-----------------|------|--------|
| Database management | create/drop/list/use_database | Multi-database isolation | Medium |
| DiskANN Index | DiskANN index type | Disk-based index for large-scale data | Large |
| IVF_PQ / OPQ / SCANN | Quantization index family | Quantization compression, memory efficiency | Large |
| Binary Vector | BIN_FLAT / BIN_IVF_FLAT | Binary vector indexes | Medium |
| Bulk Insert | import / get_import_state | Batch file import | Large |
| phrase_match | text_match enhancement | Ordered phrase matching + slop | Medium |
| run_analyzer | run_analyzer | Analyzer debugging interface | Small |
| SPARSE_WAND | Sparse vector acceleration | Accelerated sparse vector retrieval | Medium |
| JSON Path Index | JSON field indexing | JSON field index acceleration | Medium |
| Boost Ranker | Weighted search ranking | Multi-field weighted search | Medium |
| Text Highlighter | Search result highlighting | FTS result snippet highlighting | Medium |
| Snapshot | create/restore_snapshot | Collection snapshot backup and restore | Large |
| RESTful API | HTTP interface | REST alternative to gRPC | Large |

##### P3 — Low Priority

| Feature | Description |
|------|------|
| Geometry types | WKT format + spatial queries |
| Struct/Array nesting | Structured array fields |
| MinHash vectors | MinHash vector type |
| Clustering Key | Clustered compaction |
| Warmup | Collection pre-warming |
| MMap | Memory-mapped storage |
| Nullable vector Parquet persistence | Currently null vectors stored as zero vectors, null semantics lost after restart |

##### Explicitly Not Implementing (not needed for embedded scenarios)

| Feature | Reason |
|------|------|
| User/Role/Permission (RBAC) | Embedded single-user |
| GPU Index | Not needed for embedded scenarios |
| TLS / mTLS | Local embedded |
| Multi-replica / Resource Group | Single-process architecture |
| Streaming CDC | No distributed consumption needs |
| Row-Level Security | Enterprise security feature |
| Privilege Groups | Not needed for embedded access control |
| Metrics / GetComponentStates | Distributed monitoring |
| Consistency Levels | Single-process synchronous architecture is naturally Strong Consistency; no need for multiple levels |

**Coverage**: Measured against the Milvus pymilvus test suite (55 test files), MilvusLite P0 covers approximately ~85% of core features, P0+P1 covers approximately ~90%. Remaining gaps are concentrated in advanced index types, extended data types, and enterprise-grade operational capabilities.

---

## Milvus Feature Integration Backlog

This section tracks Milvus features that are worth bringing into MilvusLite after the current compatibility baseline. The selection principle is:

1. Prefer features that improve local development, notebooks, tests, and small-scale RAG applications.
2. Prefer features that fit the existing LSM + immutable segment architecture.
3. Prefer pymilvus compatibility gaps that users are likely to hit when moving code between MilvusLite and Milvus.
4. Avoid distributed, enterprise, and operational-control features that add surface area without improving the embedded use case.

### P0 - High-Value Compatibility and Performance

| Feature | Milvus Surface | Why It Matters | MilvusLite Design Direction | Acceptance Criteria |
|---|---|---|---|---|
| Scalar Index | `INVERTED`, `BITMAP`, `NGRAM` scalar indexes | Current scalar filters are functional but can become scan-heavy as local datasets grow. Scalar indexes are the highest-impact performance upgrade because search/query/delete all depend on filters. | Add segment-level immutable `.sidx` files. Filter compilation should first produce an optional indexed bitmap, then merge it with the existing bitmap pipeline. Start with scalar equality/range/in-list indexes, then add NGRAM for `LIKE`/prefix-style VARCHAR predicates. | Indexed and non-indexed filter results are identical under differential tests; large filtered search/query benchmarks show lower scanned-row count. |
| JSON and Dynamic Field Indexes | JSON path index, dynamic field key index | MilvusLite already supports JSON path and `$meta["key"]` filtering, but those paths likely remain scan-bound. This is a common local RAG metadata pattern. | Reuse the scalar index framework, with extracted path columns stored per segment. Index declared paths first; later consider adaptive path materialization. | Filters such as `meta["source"] == "x"` and `$meta["tenant"] in [...]` can use index bitmaps and preserve current expression semantics. |
| Compact Vector Types | `FLOAT16_VECTOR`, `BFLOAT16_VECTOR`, `INT8_VECTOR`, `BINARY_VECTOR` | README currently lists binary/float16/bfloat16 vectors as unsupported. These types are important for compatibility and reduce local memory/disk pressure. | Extend schema, Arrow builders, validation, gRPC translators, and distance kernels. Store compact formats natively; convert to float32 for unsupported index/search paths. Start with brute-force parity, then FAISS paths where available. | pymilvus create/insert/search smoke tests pass for each type; distance parity is documented for any conversion path. |
| IVF_PQ | `IVF_PQ` index | README currently lists Product Quantization as unsupported. FAISS is already a core dependency, so this is a natural index-family extension. | Add `FaissIvfPqIndex` under the existing `VectorIndex` protocol and segment-bound `.idx` lifecycle. Keep `AUTOINDEX` unchanged unless PQ is explicitly requested. | Build/load/search round-trip works; recall benchmark is compared against `BRUTE_FORCE` and existing IVF indexes. |

### P1 - Retrieval Quality and RAG Ergonomics

| Feature | Milvus Surface | Why It Matters | MilvusLite Design Direction | Acceptance Criteria |
|---|---|---|---|---|
| Global BM25 Statistics | BM25 sparse search | README notes BM25 IDF is currently segment-local. Cross-segment ranking quality can drift after flushes and compaction. | Maintain collection-level term document-frequency and total-document stats. Update stats during flush/compaction and keep a conservative recovery path. | BM25 results are stable across flush boundaries; differential tests verify segment-local fallback does not change correctness when global stats are unavailable. |
| Phrase Match | `phrase_match` / text phrase predicates | Phrase search is useful for local document QA and is now part of Milvus text retrieval surface. | Extend the sparse/text inverted index to store term positions. Implement ordered phrase matching with optional slop, then integrate it into filter parsing/evaluation. | `phrase_match(text, "...")` works in query/search filters and matches Milvus-style phrase semantics for supported analyzer output. |
| Analyzer Debugging | `run_analyzer` | Users building FTS pipelines need to inspect tokenizer/analyzer behavior locally. This is cheap and useful for notebooks. | Expose analyzer execution through the adapter and internal helper API. Reuse existing `Analyzer` implementations. | pymilvus-compatible analyzer debug calls return deterministic token streams for Standard and Jieba analyzers. |
| Bulk Import | Milvus import / BulkWriter JSON/Parquet inputs | MilvusLite already has dump/export. Import closes the loop for local reproduction of Milvus/Zilliz datasets. | Support Milvus BulkWriter JSON first, then Parquet. Prefer routing through validation + flush to preserve WAL/manifest invariants; direct segment build can be a later optimization. | Export from MilvusLite and import into a fresh MilvusLite DB produces equivalent query/search-visible rows. |
| Text Highlighting | FTS result highlighting | Useful for search demos and local RAG inspection, but not core storage/search correctness. | Implement as post-processing over output text fields using analyzer token offsets when available. | Search output can include highlight snippets without changing ranking. |

### P2 - Domain-Specific Compatibility

| Feature | Milvus Surface | Why It Matters | MilvusLite Design Direction | Acceptance Criteria |
|---|---|---|---|---|
| `TIMESTAMPTZ` follow-ups | Timezone-aware timestamp scalar type | Basic TIMESTAMPTZ storage, UTC normalization, gRPC FieldData, database/collection/request-level timezone parsing, `time_fields`, and ISO/INTERVAL filters are implemented. Remaining work is index acceleration. | Later integrate with scalar index or `STL_SORT` equivalent. | Indexed timestamp range filters return the same rows as scan mode. |
| Geometry | `GEOMETRY`, `ST_*` spatial predicates | Useful for geo-constrained vector search, but domain-specific and optional. | Start with optional `shapely`-backed brute-force predicates. Add segment-level R-tree only if benchmarks justify it. | Geometry fields can be inserted, persisted, and filtered with a small supported subset such as contains/within/intersects. |
| Binary Vector Indexes | `BIN_FLAT`, `BIN_IVF_FLAT` | Complements `BINARY_VECTOR`; useful for compatibility but less common than float/sparse vectors in RAG. | Add brute-force Hamming/Jaccard first. Add FAISS binary indexes only after storage and metric semantics are stable. | Binary search results match brute-force reference for supported metrics. |
| SPARSE_WAND | Sparse vector accelerated retrieval | Useful when BM25/sparse collections grow, but depends on sparse index maturity. | Add WAND-style upper-bound pruning inside `SparseInvertedIndex` without changing public API. | Sparse search returns the same top-k as exhaustive sparse scoring on test corpora. |
| MMap / Warmup | Memory mapped storage, collection pre-warming | Useful for larger local databases, but secondary to indexes and compact vectors. | Introduce only behind explicit collection/index properties. Avoid making mmap a correctness dependency. | Load/search memory profile improves on large read-only datasets without changing results. |

### Reconsider Later

These features are not rejected forever, but should not preempt the P0/P1 work.

| Feature | Reason to Defer |
|---|---|
| Full schema alter | Current design assumes immutable schemas. Safe support probably means only additive nullable/default scalar fields first. |
| Multi-database isolation | Current single default namespace is enough for most local workflows. Full DB isolation adds adapter/storage surface area. |
| REST API | gRPC/pymilvus compatibility is the primary contract. REST is useful only if there is a concrete local tooling requirement. |
| Snapshots | Valuable for backup/restore, but direct filesystem copy is often sufficient for embedded usage until concurrent readers/writers become more complex. |
| DiskANN | Heavy dependency and architecture cost. Revisit only if MilvusLite explicitly targets larger-than-memory local datasets. |
| SCANN / OPQ / HNSW_PQ | Useful index variants, but IVF_PQ should land first as the quantization baseline. |

### Explicitly Out of Scope for MilvusLite Core

| Feature | Reason |
|---|---|
| RBAC, users, roles, privilege groups | Embedded single-user process; security should be handled by the host application. |
| Resource groups, replicas, shard balancing | Distributed serving controls do not map to a single-process local engine. |
| Full consistency-level matrix | Single-process synchronous writes naturally provide strong consistency for the embedded use case. Snapshot sequence support should stay internal for iterators/MVCC-style reads. |
| GPU indexes | Dependency footprint and platform variance are too high for the default local package. |
| TLS / mTLS | Local embedded deployments should terminate transport security outside MilvusLite if needed. |
| Distributed metrics / component states | MilvusLite has no QueryNode/DataNode/Coord component topology to report. |

---

## Verification System

| Level | Tool | Trigger | Value |
|---|---|---|---|
| L1 Unit tests | pytest | Every commit | Module-level correctness |
| L2 Integration | pytest | Every commit | Module combination correctness |
| L3 Demo scripts | python `examples/m*.py` | End of milestone | "It actually works" confidence |
| L4 Property | hypothesis | Every commit | Invariants hold under random input |
| L5 Crash injection | pytest + monkeypatch + `os._exit` | Every commit | Soul of LSM — crash safety |
| L6 Long-run | `@pytest.mark.slow` | nightly / manual | Resource boundedness |

**Key**: L4 + L5 run routinely from Phase 3 onward; every change to flush/recovery goes through them. This is the only way to prevent "silent regressions".

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Phase 3 flush+recovery harder than expected | First version: prefer synchronous + simple; no premature optimization |
| Phase 4 segment cache memory explosion | M4 starts with full cache (Segment never evicted); LRU deferred to P3 optimization |
| Phase 6 compaction concurrent with search | MVP synchronous path has no concurrency; shelved |
| Schema change requirement emerges mid-way | Phase 0 hard-codes "MVP does not support alter"; new requirements go into a separate document |
| Phase 9 FAISS macOS arm64 wheel installation failure | faiss-cpu default dependency + BruteForceIndex fallback; CI runs dual matrix |
| Phase 9 FAISS metric sign alignment bugs | Differential test distance value parity is a mandatory checkpoint |
| Phase 10 milvus proto cross-version drift | proto/README.md records source commit; future upgrades use git diff |
| Phase 10 pymilvus client version compatibility | Test matrix covers at least pymilvus 2.3.x / 2.4.x / 2.5.x |

---

## Usage Instructions

1. **Phase 0 must be done first** — prerequisite for all subsequent phases
2. **Phases 1-10 strictly sequential** (Phase 8 contains F1/F2/F3 sub-phases, Phase 9 contains 9.1-9.6, Phase 10 contains 10.1-10.6); tasks within a phase can be done in any order
3. **Each phase completion = git tag** (`m1-fixes`, `m2-write`, ...)
4. **Each phase has one `examples/m{N}_demo.py`** kept permanently
5. **PR descriptions reference the corresponding section of this document**, leaving a decision trail
6. **In-depth designs for Phase 9 / Phase 10 are in separate documents**: `plan/index-design.md` and `plan/grpc-adapter-design.md`
