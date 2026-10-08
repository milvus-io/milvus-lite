# Sparse Vector IP and BM25 Design

Status: implemented. Ordinary sparse fields use the independent IP index;
BM25 Function outputs retain their text-query scoring path.

## 1. Goal and current behavior

Support exact inner-product retrieval for user-supplied sparse embeddings while
keeping Function-driven BM25 text retrieval as a separate public API path. Both
paths share `SPARSE_FLOAT_VECTOR` storage and reusable posting-list operations;
their input contracts and scoring implementations differ. Use a dedicated
`SparseIpIndex` for ordinary embeddings and retain `SparseInvertedIndex` as the
existing BM25 implementation. No new vector data type is required.

In the investigated 3.2.1 implementation:

- Schema validation and gRPC record decoding accept user-supplied sparse vectors
  without a BM25 Function.
- `_prepare_sparse_queries()` accepts dictionaries directly. Only text queries
  require an analyzer, currently selected from the first BM25 Function.
- `_search_sparse()` constructs `SparseInvertedIndex` for every sparse field.
  The index always computes BM25 with source-local statistics.
- Selecting IP only negates the internal distance at the result boundary. It
  does not change candidate scores or ranking to inner product.
- Ordinary gRPC Search resolves an omitted metric from the default dense field,
  rather than the requested `anns_field`.
- Sparse indexes are built lazily in memory during search. The Engine does not
  persist their standalone JSON representation as segment index sidecars.

The implementation adds a general sparse-vector retrieval path with its own IP
index, field-aware routing, and validation, reusing the existing sparse storage.

## 2. Public field contracts

Classify sparse fields using `CollectionSchema.functions`, keyed by each
Function's output field name. Do not infer their purpose from the query type,
the field name, or the presence of some other BM25 Function in the collection.

| Property | User-supplied sparse field | BM25 Function output field |
|---|---|---|
| Field type | `SPARSE_FLOAT_VECTOR` | `SPARSE_FLOAT_VECTOR` |
| Source | User-generated embeddings or client-generated weights | The target field's BM25 Function |
| Write input | Sparse vector | Source text; output vector is generated internally |
| Search input | Sparse vector | Raw text |
| Metric | `IP` | `BM25` |
| Analyzer | None | Analyzer attached to that Function's input field |
| Stored values | Original embedding weights | Raw term frequencies |

These are Engine and public gRPC contracts. The low-level index always accepts
lists of sparse dictionaries, including dictionaries prepared internally for
BM25. A dictionary accepted by an index unit test is not permission to submit a
precomputed query to a public BM25 Function field.

### 2.1 Writes and query validation

- Reject a text query for an ordinary sparse field, even if another field has a
  BM25 Function. IP does not implicitly tokenize or embed text.
- Reject a vector query for a BM25 Function output field. Convert text using the
  analyzer associated with that exact output field.
- Reject user attempts to write BM25 output values. Check the incoming records
  before the ingestion chain generates those fields.
- BM25 ingestion copies each record dictionary before replacing top-level fields.
  Read-only nested vectors, arrays, and JSON values are reused during ingestion;
  mutable schema defaults retain their separate deep copies. Serialization owns
  the stored data before insert returns, so later caller mutations cannot change it.
- For partial upserts, check the original user patch before merging stored
  records. Existing generated output values in the merged internal record must
  not be mistaken for user input. Remove generated BM25 outputs from the merged
  copy before passing it to `insert()`, then regenerate them from the effective
  source text. Do not introduce a public validation-bypass flag.
  Decode inherited ordinary sparse fields from their stored packed-byte form
  before record validation; otherwise a partial upsert after flush rejects a
  valid stored embedding as a non-dictionary input.
- Validate the entire query batch before executing any search; mixed text/vector
  batches are invalid for either field kind.
- A sparse field marked `is_function_output` without a corresponding supported
  producer is invalid; the flag alone must not bypass validation or select BM25.

For user-supplied values, dimension IDs are integers (not booleans) in
`[0, 2**32 - 2]`. Weights are numeric, finite, non-negative, and representable as
finite float32 values, matching the storage encoding. Validate before packing;
use the same rules for inserted vectors and queries. Explicit zero entries do
not create postings or query terms. No normalization or IDF transformation is
applied to user weights.

New non-null user-supplied records must contain at least one nonzero weight after
float32 conversion. An explicit empty/all-zero query returns no hits. Low-level
indexes accept empty rows, including historical stored rows, nullable rows, and
BM25 output from empty analyzed text; these rows produce no IP hits. Existing
nullable/default rules still apply to omitted or null fields. Validate supplied
sparse defaults against the same value rules.

Invalid public inputs raise `SchemaValidationError` with the target field and
the expected input/metric. Low-level index argument validation uses `ValueError`.

## 3. Index creation and metric resolution

### 3.1 Creating an index

Keep the public index type `SPARSE_INVERTED_INDEX`. The initial implementation
supports this explicit sparse index type; it does not introduce sparse HNSW,
`SPARSE_WAND`, or an implicit sparse `AUTOINDEX` mapping.

`create_index()` must validate the field/index/metric combination before updating
the Manifest or caches:

| Target field | Accepted metric | Rejected combinations |
|---|---|---|
| Ordinary sparse | `IP` | `BM25`, `COSINE`, `L2`, dense index types |
| BM25 Function output | `BM25` | `IP`, `COSINE`, `L2`, dense index types |

Keep the existing requirement to specify `metric_type` when creating a vector
index. Normalize valid string metric names to uppercase at the Engine boundary.
Persist the result in the existing `IndexSpec.metric_type`; no Manifest schema
change or new field-purpose flag is needed.

BM25-specific build parameters belong to BM25 indexes. Reject explicitly supplied
`bm25_k1`/`bm25_b` for a public IP index, rather than implying that they affect IP.
`SparseIpIndex` also has no `k1` or `b` constructor parameters. These parameters
belong exclusively to the BM25 implementation; there is no previous IP constructor
interface to preserve.

### 3.2 Resolving a search metric

The Engine search default is `metric_type: Optional[str] = None` so that
omission is distinguishable from an explicit `COSINE`. Preserve the existing
rules for selecting `anns_field` and dense search defaults.

Resolve the target field first, then:

1. If an index exists for that field, its validated metric is authoritative.
   An explicit conflicting request is an error.
2. Without an index, an ordinary sparse field uses IP and a BM25 Function output
   uses BM25. An explicit incompatible metric is an error.
3. Dense fields retain their current supported metrics; omitted metric uses the
   target field's index metric or the existing COSINE fallback.

No-index sparse search remains supported under the Engine's existing load-state
rules. Creating a persisted IndexSpec is not required to choose the correct
scoring path. Another field's index must never supply the default metric.

Provide one Engine-owned resolver:

```python
Collection.resolve_search_metric(
    anns_field: Optional[str] = None,
    metric_type: Optional[str] = None,
) -> str
```

The Engine uses this resolver for direct calls. The adapter uses the same resolver
when it needs an effective metric for result processing or HybridSearch fusion;
it must not duplicate field-purpose rules. Resolve and validate once per request
route and use that value consistently for scoring, range filtering, and reranking.

## 4. Sparse index contract

Use two concrete indexes with compatible sparse-dictionary build/search results.
The new IP implementation does not inherit BM25 scoring or document statistics:

```python
# milvus_lite/index/sparse_ip.py (new)
class SparseIpIndex:
    def __init__(self) -> None: ...

    def build(self, sparse_vectors, valid_mask=None) -> None: ...
    def search(self, query_sparse_vectors, top_k, valid_mask=None): ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str): ...

# milvus_lite/index/sparse_inverted.py (existing BM25 implementation)
class SparseInvertedIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None: ...

    def build(self, sparse_vectors, valid_mask=None) -> None: ...
    def search(self, query_sparse_vectors, top_k, valid_mask=None): ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str, k1: float = 1.5, b: float = 0.75): ...
```

Both classes expose `index_type == "SPARSE_INVERTED_INDEX"`, which is the public
index family rather than a unique Python class identifier. Neither constructor
selects between IP and BM25. Preserve the BM25 class name and existing positional
`k1, b` calls; renaming it is not a prerequisite for IP support.

After validating field purpose and resolving the effective metric, the Engine
selects `SparseIpIndex()` for IP or `SparseInvertedIndex(k1, b)` for BM25. Reuse
this selection for Segment and MemTable indexes. If extracted into a factory,
keep it separate from the dense NumPy/FAISS factory and test the sparse dispatch
there. Invalid metrics are rejected by the Engine/dispatch layer, not by turning
the BM25 class into a multi-metric constructor.

### 4.1 Shared build representation

Both implementations build postings of `dimension -> [(local_row_id, original_weight)]`.
They never replace weights with precomputed BM25 scores. For IP, a dimension is
an embedding coordinate; for BM25, it is a hashed term ID and its weight is TF.

Extract small reusable helpers for postings, row/mask validation, and top-k
assembly where that avoids duplication. Reuse through composition or functions;
do not make IP a subclass of a BM25 scorer. No new public base class is required.

IP needs the original row count, visible postings, and original weights. BM25
separately owns `doc_count`, `doc_lengths`, `avgdl`, and `df`. Shared helpers must
not require BM25 statistics. Each field/index instance owns its data; code reuse
does not mean sharing mutable posting lists across fields.

Tests assert scores, visibility, and lifecycle behavior rather than private
helper names or whether a particular intermediate array was allocated. The
existing corpus-size and document-length counterexamples establish that IP is
independent of BM25 statistics without supplying BM25 parameters to the IP class.

Build masks remove rows from the indexed population while retaining their original
local IDs. Search masks further restrict that population; they cannot restore a
build-excluded row or mutate the index. Both masks must match the original row
count. Rebuilding an index replaces all previous postings and mask state.

### 4.2 Exact IP scoring

`Collection._prepare_sparse_queries` validates and canonicalizes the whole query
batch once. Segment and MemTable searches reuse those dictionaries through the
internal prepared-query dispatch. `SparseIpIndex.search` remains the validated
standalone entry point and calls the same scoring kernel after preparing its own
queries. The kernel does not modify shared queries and still checks mask shape
and top-k for every source.

```text
score(q, d) = sum(q[i] * d[i] for shared nonzero dimensions i)
internal_distance(q, d) = -score(q, d)
```

Use full posting-list accumulation without dropping query dimensions or approximate
pruning. Reject explicit nonzero drop-ratio requests until those semantics are
implemented; this version promises exact scoring over the stored float32 weights.
It must not silently claim support for WAND/MaxScore/SINDI algorithm selectors.

Only positive-score rows are hits. Do not fill unused slots with disjoint or
zero-score rows. This is sparse retrieval semantics, not dense exhaustive top-k
over all rows. Negative weights are rejected rather than introducing a separate
zero-versus-negative ranking policy.

The index returns signed integer IDs and floating distances, shaped `(nq, top_k)`.
Distances are ascending, with `(-1, +inf)` padding. Ties have no required ID order.
An empty query batch has shape `(0, top_k)`; low-level `top_k=0` has shape `(nq, 0)`.
Public search limit validation remains separate from the low-level shape contract.

Accumulate products without intermediate float32 rounding; emit distances using
the existing float32 result representation. Compare scores with numerical
tolerances, not bit equality. Unit-norm self-matches score approximately 1;
arbitrary self-matches score the squared norm, not cosine similarity.

### 4.3 BM25 behavior

Preserve the current weighted-query BM25 formula, configurable `k1` and `b`, and
source-local statistics. Fixing global BM25 statistics is independent work.
Field-aware text preparation feeds TF dictionaries to this existing formula.

Keep score-sign conversion separate from this change: the index returns negative
scores for both modes; the Engine currently exposes positive IP and negative BM25
distances. Existing BM25 fusion helpers account for that sign. Do not change BM25
public score signs incidentally or negate IP a second time in HybridSearch.

## 5. Storage and lifecycle

### 5.1 Durable data and standalone serialization

No WAL, Arrow, Parquet, or schema.json format change is needed. Sparse values
remain sorted packed `(uint32, float32)` pairs. The Manifest remains authoritative
for each field's IndexSpec and metric.

The Engine continues to build sparse indexes lazily on search: one cached index
per immutable segment and field, plus a temporary MemTable index per search.
`create_index()` saves validated configuration; `load()` validates it and enables
search under the existing state machine. Neither action adds a new sparse sidecar
pipeline in this change.

Standalone persistence belongs to each implementation:

- `SparseIpIndex.save/load` stores a JSON payload with `metric_type: "IP"`, row
  identity/mask information, and raw postings. Its loader returns an IP index and
  rejects a missing or non-IP metric instead of interpreting a BM25 file as IP.
- `SparseInvertedIndex.save/load` retains the existing BM25 JSON format and
  `k1/b` handling. Old files without metric metadata continue to load through the
  BM25 loader. It is not a generic loader for the new IP implementation.

Both preserve row IDs and build-mask effects on reload. These files are not
Manifest entries or Engine-managed sidecars. Common posting representation does
not require byte-identical index files or redundant BM25 statistics in IP files.

### 5.2 Segment cache ownership

Use a dedicated `Segment.sparse_indexes` map keyed by the actual field name.
This replaces the current synthetic `_sparse_<field>` entries in the general
index map and avoids ambiguity with real field names and dense indexes.

A cached entry is usable only when its concrete index implementation and relevant
parameters match the effective configuration. IP selects `SparseIpIndex`; BM25
selects `SparseInvertedIndex` and additionally checks `k1` and `b`. Keep any cache
configuration metadata with the cache entry rather than requiring a mutable
metric selector on both scorers. Publish an immutable replacement without changing
an index still referenced by a search snapshot.

- `release_index(field_name)` also removes that field's sparse cache.
- `release_index()` clears all sparse caches.
- Creating/replacing an IndexSpec invalidates any prior no-index sparse cache.
- Dropping an index invalidates its field's cache and configuration.
- Flush creates new segments whose caches are built on first search.
- Compaction creates new segments and caches; old snapshots keep their existing
  index references until released.
- Reopen reads the persisted spec and rebuilds caches. No score transformation
  or vector rewrite is required.

### 5.3 Filtering and cross-source merge

Keep the Engine's bitmap and `_seq` rules. Exclude tombstoned rows, stale versions,
unselected partitions, and scalar-filter failures before selecting local top-k.
Use the same resolved metric for every segment and the MemTable.

For plain IP search, local top-k followed by global merge is exact once visibility
is resolved: a row excluded from its source's top-k cannot outrank the global
top-k. Equal-score ID selection is unspecified. This guarantee does not extend
to arbitrary later rerankers/grouping stages that use bounded candidate sets.

The IP score for a given stored row/query pair is independent of corpus size,
filter selectivity, partition layout, flush boundaries, or compaction. Apply this
as an invariant in Engine tests; BM25 does not currently have this invariant.

## 6. Adapter behavior

`translators/search.py` must preserve an omitted metric as `None`, decode
SparseFloatVector placeholders into dictionaries, and decode text placeholders
into strings. It does not choose a scoring algorithm based on placeholder type.

Ordinary Search and each HybridSearch route resolve the target field and metric
through the Engine. Pass the effective metric to all downstream score conversion,
Function Chain, and fusion code. A bad route fails the request with a validation
error; it must not silently become a BM25 route or yield partial fused results.

CreateIndex and DescribeIndex round-trip the validated metric. Keep business
validation in the Engine so direct Python and gRPC clients see the same behavior.

## 7. Compatibility and migration

| Existing usage | Behavior after implementation |
|---|---|
| Ordinary sparse field with IP index | True IP scores and ranking; no vector rewrite |
| Ordinary sparse field without index | IP default; formerly implicit BM25 results change |
| BM25 output field queried with text | BM25 preserved; correct target-field analyzer selected |
| BM25 output queried with a dictionary | Validation error; submit source text instead |
| User writes a BM25 output value | Validation error; write source text instead |
| Ordinary sparse field with BM25 index | Invalid configuration; release/drop/recreate as IP |
| BM25 output field with IP index | Invalid configuration; release/drop/recreate as BM25 |
| Standalone old index JSON without metric | Load through the BM25 implementation; the IP loader rejects it |

Do not silently rewrite legacy IndexSpecs. Validate them when loading/searching
the affected configuration; preserve collection opening and management operations
so users can inspect, release, drop, and recreate an invalid index. Error messages
must name the field, persisted metric, allowed metric, and recovery action.

Historical sparse rows that violate the new weight constraints fail explicitly
when used to build an index, rather than being clipped or reinterpreted as TF.
Users must correct/reinsert those records. Empty historical rows remain readable
and nonmatching as described above.

Record these behavior changes in release notes. Low-level BM25 backward
compatibility does not imply preserving previously accepted but invalid public
query/field combinations.

## 8. Implementation map and acceptance tests

Follow the ordered [Sparse IP Support Implementation Plan](superpowers/plans/2026-10-08-sparse-ip-support.md)
for dependencies, file ownership, focused test commands, and completion gates.

| Step | Modules | Acceptance |
|---|---|---|
| 1. Standalone IP index | New `index/sparse_ip.py`, reusable helpers, existing BM25 implementation | Exact IP and independent serialization; BM25 constructor/formula/file compatibility |
| 2. Field and input contracts | `schema/validation.py`, Collection ingestion/query preparation | Text/vector separation, correct analyzer, value validation, safe insert/partial upsert |
| 3. Engine metric resolution and dispatch | `engine/collection.py`, optional sparse factory | Valid IndexSpecs, target-field defaults, correct Segment/MemTable scoring |
| 4. Cache, visibility, and compatibility | `storage/segment.py`, Collection lifecycle/snapshot handling | Lifecycle invariance, `_seq` visibility, cache isolation, legacy configuration repair |
| 5. Protocol integration | gRPC servicer and search/index translators | Explicit/omitted metrics, ordinary/hybrid search, score direction and useful errors |
| 6. Integration and documentation | Tests, docs and release notes | Complete acceptance matrix, migration examples and no new broader regressions |

`tests/index/test_sparse_ip.py` targets `SparseIpIndex` directly using an
independent NumPy oracle. `tests/index/test_sparse_factory.py` checks dispatch,
instance isolation, and cache matching. These tests exercise real scorers without
substituting mocks or marking expected behavior as xfail.

`tests/index/test_sparse_inverted.py` owns BM25 formula, positional constructor,
and legacy-file compatibility tests. Its dictionary queries remain valid because
that class is below the public text-input boundary. IP tests do not pass `k1/b`,
and BM25 tests do not require a new `metric_type` constructor argument. Engine
tests cover invalid public metric and parameter combinations.

`tests/engine/test_sparse_field_contract.py` and
`tests/adapter/test_sparse_field_contract.py` exercise the public field boundary,
metric resolution, and lifecycle through the real Engine and pymilvus. Existing
BM25 integration fixtures now submit text; low-level index tests retain TF dicts.

`tests/adapter/test_milvus_sparse_compat.py` adds upstream-derived ordinary sparse
scenarios and an independent score oracle. See the
[migration mapping](sparse-upstream-test-migration.md) for source methods,
parameter adaptations, strict output-projection checks, and the remaining
error-protocol gap. Passing the local contracts does not imply every upstream
assertion passes.

Add Engine and adapter tests covering:

- The full field/input/metric matrix, with and without a persisted index, and
  explicit versus omitted search metrics.
- Two BM25 Functions with different analyzers, proving selection by output field.
- A collection containing dense, ordinary sparse, and BM25 output fields.
- Alternating IP and BM25 searches in one collection with both indexes present,
  including after flush, release/load, and reopen. Check each score against its
  own reference to catch cache collisions or shared mutable scoring state.
- Rejection of caller-supplied BM25 outputs before ingestion; valid partial
  upserts that carry generated values internally; failures produce no WAL writes.
- Finite float32 values, dimension limits, empty/null/default behavior, and
  consistent write/query validation through both Engine and gRPC.
- IP scores and top-k against an independent brute-force baseline before/after
  flush, compaction, release/load, and reopen; use distinct scores for ID-order
  comparisons and tolerance-based score checks.
- Delete/upsert visibility by `_seq`, partition restrictions, scalar filters,
  per-field cache invalidation, and configuration changes after no-index search.
- Legacy invalid IndexSpecs with actionable errors and successful repair.
- HybridSearch score direction and no double negation of IP results.

Existing Engine/gRPC BM25 tests that submit dictionaries to a Function output must
be migrated to text queries. Keep dictionary-based BM25 formula tests at the index
layer. Do not weaken the new boundary solely to retain those old API tests.

Global BM25 statistics, new sparse disk sidecars, sparse ANN/pruning algorithms,
and additional sparse embedding Function types are outside this implementation.

## 9. Compatibility references

- [Milvus sparse vectors](https://milvus.io/docs/sparse_vector.md): user-provided
  sparse vectors and IP retrieval.
- [Milvus full text search](https://milvus.io/docs/full-text-search.md): text input
  and BM25 Function-generated sparse fields.
- `milvus/pkg/util/typeutil/schema.go`, `ValidateSparseFloatRows`: finite,
  non-negative weights and the reserved maximum uint32 dimension ID.
- `milvus/tests/python_client/milvus_client/test_milvus_client_sparse_inverted_index.py`:
  sparse IP empty-query and non-overlap tests.

The last two references were inspected in the adjacent Milvus checkout; they
establish compatibility targets, not new MilvusLite dependencies.
