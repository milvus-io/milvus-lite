# Upstream Sparse IP Test Migration

The migrated suite is `tests/adapter/test_milvus_sparse_compat.py`. It exercises
ordinary `SPARSE_FLOAT_VECTOR` fields through pymilvus and a real local gRPC
server, without a BM25 Function or upstream test-framework dependency.

## Source and scope

Source checkout: `milvus/tests/python_client/milvus_client`, commit
`a11a375e05356420fdeed0e2cc39994312433b69`.

The following upstream files supply the scenarios:

- `test_milvus_client_sparse_search.py`
- `test_milvus_client_sparse_inverted_index.py`
- `test_milvus_client_insert.py`
- `test_milvus_client_upsert.py`
- `test_milvus_client_range_search.py`

This is a selected behavior migration, not a claim of full upstream sparse index
compatibility. Generic cases extracted from SINDI fixtures use Lite's exact
backend and do not validate SINDI itself. No external Milvus server is required.

## Coverage mapping

Names below omit the common `test_milvus_client_` filename prefix.

| Upstream source method | Local test | Coverage retained or strengthened |
|---|---|---|
| `sparse_search.py::test_sparse_search_default` | Same method name | Query count, limit, requested vector output, descending scores, exact IP oracle |
| `sparse_search.py::test_sparse_search_with_filter` | Same method name | PK filter before top-k and exact scores of eligible rows |
| `sparse_search.py::test_sparse_search_output_field` | Same method name | Exact requested field set plus actual scalar/vector values |
| `sparse_search.py::test_sparse_search_different_nq` | Same method name | nq=1 and nq=100; independent oracle for every query |
| `sparse_search.py::test_sparse_search_invalid_metric_type` | Same name plus `test_sparse_invalid_metric_upstream_error_contract` | L2/COSINE rejection; Lite error assertions and separate unmodified upstream error assertions |
| `sparse_search.py::test_sparse_index_dim` | Same method name | High coordinates at 32768 and the maximum legal uint32 coordinate; actual query overlap at that coordinate |
| `sparse_search.py::test_sparse_search_after_delete` | Same method name | Deleted PKs excluded before and after flushing tombstones |
| `sparse_search.py::test_sparse_search_iterator` | Same method name | Batch sizes 10/100/500, limit 500, termination, no duplicate IDs, complete ordered IP results |
| `sparse_inverted_index.py::test_ip_sindi_search_topk_and_score_order` | `test_ip_topk_and_score_order` | Limits 1/10/100 and the original monotonic weight pattern; scores checked numerically |
| `sparse_inverted_index.py::test_ip_sindi_search_empty_query` / `test_ip_sindi_search_no_overlap_query` | `test_sparse_query_without_matches` | Empty results for empty or disjoint queries |
| `sparse_inverted_index.py::test_ip_default_index_params_search_success` / `test_ordinary_sparse_algorithm_search_after_reload` | `test_sparse_index_search_after_reload` | Default and TAAT_NAIVE configuration, release/load, identical ranking, index type and metric metadata |
| `sparse_inverted_index.py::test_ip_sindi_search_with_high_filter_ratio` | `test_sparse_search_high_filter_ratio` | Remove 40%/90% of candidates, retain exact top-k among eligible rows |
| `sparse_inverted_index.py::test_sparse_index_rejects_unsupported_metric` | Same name plus `test_sparse_invalid_index_metric_upstream_error_contract` | Reject L2/COSINE during CreateIndex and leave no persisted index |
| `range_search.py::test_range_search_sparse` | `test_sparse_range_search` | IP lower-exclusive/upper-inclusive bounds, nonempty deterministic expected result |
| `upsert.py::test_milvus_client_upsert_sparse_data` | `test_sparse_upsert_preserves_count_and_updates_scores` | Five upsert/flush cycles; stats/count stability; changed weights must change scores |
| `insert.py::test_milvus_client_insert_sparse_vector_scipy` | `test_sparse_scipy_insert_and_search` | Native csr_matrix and csr_array inputs on insert and search |
| `insert.py::test_milvus_client_insert_sparse_vector_scipy_to_csr` | Same local method, additional parameters | CSC/COO/DOK/LIL/coo_array converted to CSR as in the upstream test |

## Explicit adaptations

1. Replace `TestMilvusClientV2Base`, `ResponseChecker`, and `common_func` with the
   repository's `milvus_client` fixture and direct assertions. Preserve the
   upstream primary/scalar/vector field names. Add finite RPC timeouts and a
   bounded iterator loop so a failing test cannot loop indefinitely.
2. Use deterministic data: 640 rows for common/reload/iterator scenarios, 100 for
   high coordinates, and 128 for CSR/upsert. Keep the iterator's original limit
   of 500 and batch-size matrix. Use common nonzero dimensions 0/1 in random
   vectors, including CSR data, to guarantee enough positive-score candidates.
3. Use `SPARSE_INVERTED_INDEX` with default parameters or `TAAT_NAIVE`. Upstream's
   shared `default_sparse_search_params` includes `drop_ratio_search="0.2"`;
   migrated generic behavior cases use no dropping. Pruning/recall semantics are
   outside this migration.
4. Validate exact IP using an independent document scan with `math.fsum`, float32
   wire weights, and float32 expected scores. Require all strictly better rows
   while allowing arbitrary IDs/order among exact cutoff ties. No dense vector
   allocation is made for large dimension IDs.
5. Preserve the exact-output-fields assertions. Additional primary-key fields in
   `entity` are a compatibility failure, not permitted extra output.
6. Separate the known error-protocol gap: behavioral tests check Lite's current
   IllegalArgument code 6, while strict `xfail` cases retain upstream code 1100
   and message assertions. An XPASS fails the suite so the annotation must be
   revisited when compatibility improves. These xfails are not counted as passes.
7. Release/load cases do not port the V3 fixture's explicit compaction step or
   its multi-algorithm index matrix. Lite's existing Engine tests separately
   cover automatic compaction and reopen.

SINDI, SPARSE_WAND, DAAT_WAND/DAAT_MAXSCORE, block codecs, quantization, mmap, and
nonzero drop-ratio scenarios are not migrated as successful capability tests.

## Running the suite

SciPy is a development dependency for the CSR cases, not a runtime dependency of
milvus-lite. Install development dependencies before running; CSR tests must not
silently disappear because SciPy is missing.

```bash
pip install -e ".[dev]"
.venv/bin/python -m pytest -q tests/adapter/test_milvus_sparse_compat.py --tb=short
```

The shared adapter fixture additionally requires pymilvus and grpcio, and binds
a loopback port. Run with local socket access in sandboxed environments.

## Compatibility status

- Explicit sparse output projection now returns exactly the requested fields.
  Search always retains the primary key in the top-level hit ID, and includes
  it in `entity` only when selected explicitly or by `*`. The two upstream
  projection assertions pass without being relaxed or marked xfail. Query/Get
  still include the primary key under their existing projection rules.
- Unsupported search/index metrics use Lite's error code/message rather than
  upstream code 1100 and wording. Four parameterized strict xfails retain those
  upstream contracts alongside the behavioral rejection tests.

The remaining gap concerns error-protocol compatibility. The numerical oracle separately
checks inner-product scoring so a correct sort order cannot hide BM25 scoring.
