# Sparse IP Support Implementation Plan

Status: implemented. All sparse acceptance tests pass; the broader regression
limitations are recorded below.

Design: [Sparse Vector IP and BM25 Design](../../sparse-vector-design.md).

## Objective and boundaries

Add general sparse-vector retrieval through a dedicated `SparseIpIndex`.
Keep `SparseInvertedIndex` as the BM25 implementation and reuse small posting,
validation, and result-assembly helpers where useful. The Engine selects the
implementation using the target field and its validated metric.

The public field type remains `SPARSE_FLOAT_VECTOR`; both implementations report
the public index family `SPARSE_INVERTED_INDEX`. Ordinary sparse fields accept
vectors and use IP. BM25 Function outputs accept text queries and use BM25.

Retain packed sparse storage, Manifest IndexSpecs, and lazy in-memory segment
indexes. This plan does not add sparse disk sidecars, global BM25 statistics,
WAND/MaxScore/SINDI acceleration, a new DataType, or a BM25 class rename.

## Starting point

- [x] Investigate the 3.2.1 scoring and input paths.
- [x] Define the separate IP/BM25 field and index contracts.
- [x] Add index, Engine, and adapter contract tests, including alternating IP/BM25
  searches through lifecycle transitions.
- [x] Migrate existing BM25 integration queries from precomputed TF dictionaries
  to raw text; keep dictionary inputs in low-level BM25 formula tests.

Before implementation, the index/Engine run had 28 passing and 101 failing cases, including
22 passing BM25-related cases. IP unit tests failed because `index/sparse_ip.py`
did not exist. Engine failures also exposed input-validation, analyzer-selection,
cache, and visibility problems; adding the IP scorer alone will not fix them.
The adapter contract tests were also red. These are historical baselines; the
completed validation results follow.

The existing contracts are a starting point. Add the focused cases identified
below when implementing behavior that is not yet covered. Keep expected-behavior
tests enabled; do not turn missing functionality into skips or xfails.

## Delivery order

| Step | Deliverable | Depends on | Completion gate |
|---|---|---|---|
| 1 | Standalone sparse IP index | Existing contracts | IP and BM25 index suites pass |
| 2 | Field and input validation | Step 1 validation primitives where shared | Input/analyzer/write contract tests pass |
| 3 | Engine metric resolution and dispatch | Steps 1–2 | Direct Engine IP/BM25 search and index configuration tests pass |
| 4 | Cache, visibility, recovery, and compatibility | Step 3 | Entire Engine sparse contract suite passes |
| 5 | gRPC and pymilvus integration | Steps 3–4 | Adapter sparse contracts and BM25/hybrid regressions pass |
| 6 | Remaining coverage and release readiness | Steps 1–5 | All sparse contracts pass; no new broader regressions |

Each step should form a reviewable change. Complete its focused checks before
continuing; later-step contracts may remain red until their owning step lands.

## Step 1 — Implement the independent IP index

**Files:** new `milvus_lite/index/sparse_ip.py`; existing
`milvus_lite/index/sparse_inverted.py` only for justified helper extraction;
`tests/index/test_sparse_ip.py`; `tests/index/test_sparse_inverted.py`.

- [x] Implement `SparseIpIndex()` with `build`, `search`, `save`, `load`, and
  `index_type`. Do not introduce `k1/b` or a metric-switching constructor.
- [x] Build postings from original weights, retain original row IDs, and apply
  build/search masks before top-k. Rebuild replaces prior data and mask state.
- [x] Implement exact posting-list dot-product accumulation and negative internal
  distances. Return only positive-score hits, with `-1/+inf` padding and the
  existing `(nq, top_k)` result shape; ties need no stable ID order.
- [x] Validate finite non-negative weights and masks. Ignore explicit zeros and
  handle empty rows, empty query batches, and zero low-level top-k as specified.
- [x] Persist IP-tagged JSON containing raw postings and sufficient row/mask
  information. Reject absent/non-IP metric tags on IP load. Keep old BM25 files
  and positional constructors readable by the BM25 implementation.
- [x] Extract shared helpers only where they remove actual duplication. Keep
  BM25 statistics and scoring in the BM25 class; do not couple IP to its state.
- [x] Pass all current IP unit contracts and all existing BM25 tests.

```bash
.venv/bin/python -m pytest -q \
  tests/index/test_sparse_ip.py tests/index/test_sparse_inverted.py
```

**Gate:** exact scores agree with the independent NumPy oracle; normalization,
corpus-statistics, masking, ranking, and persistence regressions are excluded.
The public Engine can still lack IP dispatch at this checkpoint.

## Step 2 — Enforce field ownership and input contracts

**Files:** `milvus_lite/schema/validation.py`,
`milvus_lite/engine/collection.py`, sparse validation/codec helpers as needed;
`tests/engine/test_sparse_field_contract.py`, relevant schema tests.

- [x] Classify sparse fields from schema Function output bindings. Reject an
  orphan `is_function_output` flag; do not classify by query shape or field name.
- [x] Make `_prepare_sparse_queries(query_vectors, vector_field)` field-aware.
  Ordinary sparse fields accept vectors; BM25 outputs accept text and use the
  analyzer bound to their own input/output pair. Validate the complete batch.
- [x] Share value rules between inserts and queries: valid uint32-range IDs,
  numeric finite non-negative float32-compatible weights, and ignored zeros.
  Wrap low-level validation errors as field-specific `SchemaValidationError`.
- [x] Distinguish empty user records from empty queries/internal rows. Validate
  nullable/default behavior and float32 overflow/underflow before packing.
- [x] Reject caller-supplied BM25 output fields before Function execution and
  before any WAL writes for the batch.
- [x] For partial upserts, validate the original patch, decode inherited ordinary
  sparse bytes, remove inherited BM25 outputs from the merged copy, and regenerate
  them from effective source text. Do not add a public validation-bypass option.
- [x] Add focused nullable/default, conversion-boundary, and failure-atomicity
  cases where current tests do not cover these rules.

```bash
.venv/bin/python -m pytest -q tests/engine/test_sparse_field_contract.py \
  -k 'query_input or analyzer or caller_cannot or partial_upsert or public_sparse_values or empty_stored or empty_query or unproduced'
```

**Gate:** invalid input fails before ingestion/WAL mutation; valid partial updates
after flush succeed; multiple BM25 Functions select the correct analyzer. Keep
low-level BM25 dictionary tests passing.

## Step 3 — Resolve metrics and dispatch Engine searches

**Files:** `milvus_lite/engine/collection.py`; optionally a small sparse-specific
index factory; index specification validation as needed;
`tests/engine/test_sparse_field_contract.py` and focused dispatch tests.

- [x] Add the Engine-owned `resolve_search_metric(anns_field=None,
  metric_type=None)` and preserve omission with `Collection.search(...,
  metric_type=None, ...)`.
- [x] Resolve the target field before choosing a metric: field IndexSpec first;
  otherwise ordinary sparse defaults to IP and BM25 output to BM25. Preserve
  dense defaults and existing `anns_field` selection behavior.
- [x] Reject explicit field/metric conflicts and unsupported sparse index types.
  Normalize valid metric strings; check every new IndexSpec before persistence.
- [x] Reject BM25 parameters for public IP indexes, unsupported acceleration
  selectors, and nonzero drop-ratio requests. Add tests for these parameter
  checks and lowercase metric normalization where coverage is missing.
- [x] Select `SparseIpIndex` or the existing BM25 class for both Segment and
  MemTable searches. If a factory is introduced, keep sparse dictionary dispatch
  separate from dense NumPy/FAISS dispatch and test it directly.
- [x] Use one effective metric through search and postprocessing. Convert IP
  distance to public score once; preserve current BM25 score conventions.
- [x] Test ordinary and BM25 sparse fields with/without indexes alongside a dense
  index; neither dense settings nor another sparse field may supply the default.

```bash
.venv/bin/python -m pytest -q tests/engine/test_sparse_field_contract.py \
  -k 'search_metric or conflicting or invalid_index_request'
```

**Gate:** direct Engine IP results are true dot products and BM25 uses the
existing formula; index creation failures leave no invalid persisted spec.
Cache replacement and historical configuration handling are completed next.

## Step 4 — Complete cache and storage lifecycle correctness

**Files:** `milvus_lite/storage/segment.py`,
`milvus_lite/engine/collection.py`; recovery/flush/compaction code only where a
failing contract identifies a necessary change;
`tests/engine/test_sparse_field_contract.py`, segment lifecycle tests.

- [x] Introduce `Segment.sparse_indexes`, keyed by actual field names. Replace
  synthetic `_sparse_<field>` cache entries and validate the concrete index kind
  and relevant configuration when reusing a cache.
- [x] Invalidate caches on IndexSpec creation, drop/recreation, release, and
  parameter changes. Publish replacement indexes without mutating objects held
  by existing search snapshots.
- [x] Preserve lazy index construction on flush, compaction, and reopen. Maintain
  correct original row IDs and avoid introducing sparse sidecar persistence.
- [x] Ensure partition/filter masks, tombstones, and `_seq` visibility are applied
  before local top-k. Investigate the current deletion/old-version failures rather
  than assuming a scoring change will fix them.
- [x] Validate legacy field/metric combinations on load/search while leaving
  opening and management operations available for repair. Error messages name
  the field, invalid/allowed metric, and release/drop/recreate action.
- [x] Pass score-invariance tests across MemTable, flush, compaction, reload, and
  reopen; pass alternating IP/BM25 searches with both indexes present.
- [x] Add targeted cache tests for configuration replacement and per-field
  invalidation if current lifecycle tests leave a branch unverified.

```bash
.venv/bin/python -m pytest -q tests/engine/test_sparse_field_contract.py
```

**Gate:** the entire Engine contract suite passes. IP matches the reference
regardless of layout; BM25 and IP caches cannot contaminate each other; legacy
invalid configurations can be inspected and repaired.

## Step 5 — Integrate the protocol and score consumers

**Files:** `milvus_lite/adapter/grpc/servicer.py`, search/index translators,
Function Chain score consumers where necessary;
`tests/adapter/test_sparse_field_contract.py` and existing regression suites.

- [x] Preserve an omitted wire metric as `None` in search decoding; retain text
  versus sparse-vector placeholder types for Engine validation.
- [x] Reuse Engine resolution for ordinary Search and every HybridSearch route.
  Eliminate fallback to the first/default dense index for a sparse target.
- [x] Feed the effective metric to range, reranking, and fusion consumers. IP
  public scores must not be negated twice; leave BM25 sign handling consistent.
- [x] Validate all hybrid routes before returning fused results. Invalid routes
  produce a protocol validation error, not a silent fallback or partial success.
- [x] Round-trip IndexSpec metrics through CreateIndex/DescribeIndex. Add focused
  protocol validation cases if SDK preprocessing would otherwise hide bad input
  from the server under test.
- [x] Run against a real loopback gRPC server, including pure-sparse and mixed
  collections. Resolve local socket sandbox restrictions through the normal
  execution approval mechanism when necessary.

```bash
.venv/bin/python -m pytest -q \
  tests/adapter/test_sparse_field_contract.py \
  tests/adapter/test_grpc_fts.py tests/adapter/test_milvus_fts_compat.py \
  tests/adapter/test_hybrid_search.py tests/adapter/test_multi_index.py
```

**Gate:** pymilvus receives exact IP scores with explicit or omitted metrics;
hybrid routes keep the right ranking and score direction; BM25 regressions pass.

## Step 6 — Close coverage gaps and prepare release documentation

**Files:** sparse/schema/Engine/adapter tests as needed; `docs/modules.md`,
`docs/sparse-vector-design.md`, `docs/fts-design.md`, `docs/index-design.md`,
README examples and release notes in the repository's established location.

- [x] Audit the design's acceptance matrix against implemented tests. Add missing
  boundary cases rather than weakening existing assertions to obtain green runs.
- [x] Run all sparse contracts together, then the affected schema, dense default
  metric, projection, lifecycle, and Function Chain regressions.
- [x] Run the regular non-slow/non-soak suite. Record unrelated existing failures
  separately and preserve unrelated worktree changes; all required sparse
  contracts must pass before declaring this feature complete.
- [x] Document both public paths with working examples and no new DataType.
- [x] Document changed scores for old IP users, rejected direct BM25-vector
  queries/writes, and how to repair legacy invalid IndexSpecs.
- [x] Mark design/module API descriptions implemented only after the checks pass.
  Inspect the final diff for unintended storage-format or BM25 scoring changes.

```bash
.venv/bin/python -m pytest -q \
  tests/index/test_sparse_ip.py tests/index/test_sparse_inverted.py \
  tests/engine/test_sparse_field_contract.py tests/adapter/test_sparse_field_contract.py

.venv/bin/python -m pytest -q -m "not slow and not soak" --ignore=tests/benchmark

git diff --check
```

## Completion criteria

The feature is complete when ordinary sparse embeddings use exact IP through
both direct Engine and pymilvus APIs, BM25 text search retains its intended
behavior, both indexes coexist across lifecycle transitions, invalid inputs and
configurations fail predictably, and all required contract tests pass. A working
standalone scorer or a lower failure count alone is not completion.


## Completed validation

- Final combined sparse index, factory, BM25, Engine, and pymilvus contract run:
  **161 passed**. The real loopback gRPC server was used.
- Final schema, partition, sparse Engine, and index checks: **302 passed**,
  including the cross-partition upsert regression added during final review.
- The existing BM25/hybrid/multi-index adapter regressions and sparse adapter
  contracts passed together: **61 passed**.
- The regular suite was run in two batches: **3149 passed**, **29 skipped**,
  **6 xfailed**, and **4 failures**, all in the pre-existing issue #369 HNSW
  reproductions. The three Engine failures were independently reproduced using
  an isolated copy of the unmodified HEAD package; the adapter failure exercises
  the same missing HNSW row. Those unrelated files were left unchanged.
- `tests/search/test_hnsw_exact_fallback.py` could not be collected because it
  imports the pre-existing missing `HNSW_EXACT_SEARCH_THRESHOLD` constant. It was
  excluded from the regular run.
- `tests/compatibility/test_database_gaps.py` stalled in pymilvus retries during
  the non-ASCII database-name case. That run was interrupted after four expected
  failures; the remaining files were run separately without repeating completed
  tests. The database-gap file was left unchanged.
- Whitespace, Python syntax, and local documentation links were checked.

The final review also found that sparse version resolution pruned partitions too
soon: after moving a primary key to another partition and flushing, the old
partition could reveal the stale version. Sparse search now resolves the latest
collection-wide `_seq` before partition pruning; both before/after-flush cases
pass, along with the focused schema and partition regressions above.
