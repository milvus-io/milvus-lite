# Function Chain L2 Upstream Test Migration

**Status:** Proposed

**Date:** 2026-07-17

**Scope:** Migrate the ten Milvus Python client L2 Function Chain tests into a standalone Milvus Lite compatibility test module

## 1. Summary

Milvus Lite will migrate the ten `L2_RERANK` scenarios from Milvus's
`tests/python_client/testcases/test_function_chain.py` into a standalone
compatibility module:

```text
tests/compatibility/test_function_chain_l2.py
```

The compatibility module will preserve the upstream test names, datasets,
Function Chain DSL shapes, and behavioral assertions while replacing the
Milvus integration-test framework with local Milvus Lite gRPC fixtures.

The migration does not add public `L0_RERANK`, XGBoost, FileResource, MinIO,
or Hybrid Search Function Chain support. The only implementation change
required by the ten L2 scenarios is rejecting `order_by_fields` when an
ordinary Search request also contains a public Function Chain.

## 2. Source Test Set

The source is:

```text
/Users/junjie.jiang/Workspace/dev/milvus/tests/python_client/testcases/test_function_chain.py
```

The following ten L2 tests are migrated with their names unchanged:

1. `test_search_with_l2_function_chain_sdk_reranks_by_scalar_field`
2. `test_search_with_l2_function_chain_sdk_uses_hidden_input_field`
3. `test_search_with_l2_function_chain_sdk_temp_column_not_returned`
4. `test_search_with_l2_function_chain_sdk_limit_op`
5. `test_search_rejects_l2_function_chain_write_readonly_system_column`
6. `test_search_rejects_l2_function_chain_reserved_temp_output`
7. `test_search_rejects_l2_function_chain_read_internal_system_input`
8. `test_search_rejects_l2_function_chain_read_unknown_system_input`
9. `test_search_rejects_l2_function_chain_with_function_score`
10. `test_search_rejects_l2_function_chain_with_order_by`

## 3. Goals

- Maintain a visible one-to-one parity suite for the upstream Milvus L2
  Function Chain scenarios.
- Exercise the public PyMilvus DSL and the real ordinary Search gRPC path.
- Preserve upstream test names so additions and behavior changes can be
  compared mechanically in future updates.
- Verify hidden input planning and final result projection through the public
  client API.
- Verify system-column validation and request-level API conflicts through the
  public client API.
- Keep the compatibility suite runnable with the PyMilvus checkout that
  defines Function Chain protobuf messages.
- Avoid changing the existing `Collection.search()` Python API.

## 4. Non-Goals

- Migrating any `L0_RERANK` test.
- Migrating XGBoost or FileResource tests.
- Implementing `order_by` result sorting.
- Changing Function Chain error codes to match distributed Milvus code 1100.
- Replacing the more detailed unit and adapter tests already present under
  `tests/function/` and `tests/adapter/`.
- Removing existing tests that overlap with the upstream parity scenarios.

## 5. Test Organization

The selected approach is a standalone compatibility module rather than adding
the cases to `tests/adapter/test_function_chain.py`.

```text
tests/compatibility/
└── test_function_chain_l2.py
```

Intentional overlap with lower-level tests is accepted. The new module answers
the specific compatibility question "Do the current upstream L2 scenarios work
through Milvus Lite?" Existing adapter and function tests continue to provide
more focused diagnostics and edge-case coverage.

The module must not import Milvus's `TestMilvusClientV2Base`, `CheckTasks`,
`CaseLabel`, `common_func`, or other integration-test helpers.

## 6. Fixtures and Test Data

The module owns lightweight fixtures following existing compatibility-test
patterns:

- A module-scoped temporary Milvus Lite server.
- A per-test `MilvusClient` connected to that server.
- Per-test cleanup of all created collections.
- Module teardown that stops the server, closes the database, and removes the
  temporary data directory.

The collection schema mirrors the upstream test:

```text
id      INT64 primary key
ts      INT64
vector  FLOAT_VECTOR dim=2
```

Rows are preserved exactly:

```python
[
    {"id": 1, "ts": 10, "vector": [0.0, 0.0]},
    {"id": 2, "ts": 20, "vector": [0.01, 0.0]},
    {"id": 3, "ts": 30, "vector": [0.02, 0.0]},
]
```

The collection uses an L2-compatible flat or brute-force index, is loaded
before Search, and does not rely on explicit `flush()` for correctness.

## 7. PyMilvus Version Handling

The compatibility module uses the public PyMilvus imports:

```python
from pymilvus import (
    DataType,
    Function,
    FunctionChain,
    FunctionChainStage,
    FunctionScore,
    FunctionType,
    MilvusClient,
)
from pymilvus.function_chain import col, fn
```

When the installed PyMilvus package does not expose the Function Chain DSL or
protobuf fields, the module skips at collection time with a clear reason. With
the target PyMilvus checkout, all ten tests must collect and execute.

The canonical command is:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  pytest tests/compatibility/test_function_chain_l2.py -v
```

## 8. Helper API

The compatibility module provides local helpers equivalent to the upstream
test class helpers:

```python
def create_function_chain_collection(client: MilvusClient) -> str: ...

def score_plus_ts_chain() -> FunctionChain: ...

def hit_field(hit, field_name: str): ...

def assert_search_error(
    client: MilvusClient,
    collection_name: str,
    chain: FunctionChain,
    message: str,
    **kwargs,
) -> None: ...
```

`assert_search_error` calls `MilvusClient.search()` and checks the resulting
client exception. It asserts the stable semantic error substring rather than
the distributed Milvus error code 1100. Milvus Lite continues to use its
existing IllegalArgument mapping.

## 9. Scenario Mapping

### 9.1 Scalar Rerank

The Chain maps `$score` from `num_combine($score, ts, mode="sum")`, then sorts
by `$score` descending with `$id` as the explicit tie-break. The expected ID and
`ts` order is `[3, 2, 1]` and `[30, 20, 10]`.

### 9.2 Hidden Input Field

Search requests only the primary key while the Chain reads `ts`. The internal
search projection must fetch `ts`, the Chain must rerank successfully, and the
final entity projection must not expose `ts`.

### 9.3 Temporary Column Projection

The Chain writes `tmp_score`, reads it in a second map, writes `$score`, and
sorts. `tmp_score` must not appear in any returned hit entity.

### 9.4 Limit Operator

Search requests three results and the Chain applies `limit(2)`. Exactly two
hits must be returned.

### 9.5 Read-Only System Output

A map writing `$id` must fail with an error containing that the system output
is not writable.

### 9.6 Reserved Temporary Output

A map writing `$tmp_score` must fail because user outputs may not use the
reserved `$` system namespace.

### 9.7 Internal System Input

A map reading `$seg_offset` must fail because ordinary public L2 Chains expose
only supported public system inputs.

### 9.8 Unknown System Input

A map reading an undefined `$tmp_score` must fail before ANN Search.

### 9.9 FunctionScore Conflict

Search must reject a request that contains both a public Function Chain and a
legacy `FunctionScore` ranker. The compatibility test accepts Milvus Lite's
stable conflict message.

### 9.10 Order-By Conflict

Search must reject a request that contains both a public Function Chain and
`order_by_fields`. This is conflict validation only; Milvus Lite does not need
to implement order-by sorting for this migration.

## 10. Order-By Conflict Implementation

PyMilvus serializes `order_by_fields` into Search request parameters. The
ordinary Search translator must expose the decoded value:

```python
order_by_fields = raw_params.get("order_by_fields")
```

`parse_search_request()` returns it without interpreting sort semantics.

`prepare_search_function_chain()` receives the value and rejects a non-empty
order-by specification before compiling the Chain:

```text
order_by and function rerank cannot be used together
```

The validation runs before `Collection.search()`, so conflicting requests do
not execute ANN Search.

No `Collection.search()` argument, engine operator, or result ordering behavior
is added for `order_by_fields`.

## 11. Error Assertion Policy

The migrated tests preserve the upstream behavioral conditions but use Milvus
Lite error conventions:

- Public client exceptions are asserted rather than Milvus's `CheckTasks`.
- Stable message fragments are asserted.
- Tests do not require distributed Milvus error code 1100.
- Existing raw gRPC tests remain responsible for exact status-code checks.

This keeps the parity suite focused on user-visible API behavior without
duplicating lower-level error-mapping tests.

## 12. Existing Coverage Relationship

Several scenarios already have equivalent or stronger coverage in:

```text
tests/adapter/test_function_chain.py
tests/function/test_function_chain_validator.py
tests/function/test_public_function_chain.py
```

The compatibility module does not replace them. It deliberately exercises the
same upstream scenarios as an integrated group using public SDK calls and
upstream test names.

## 13. Files Changed During Implementation

Expected implementation files:

```text
milvus_lite/adapter/grpc/translators/search.py
milvus_lite/adapter/grpc/function_chain.py
milvus_lite/adapter/grpc/servicer.py
tests/compatibility/test_function_chain_l2.py
tests/adapter/test_function_chain.py
docs/modules.md
```

`tests/adapter/test_function_chain.py` receives a focused raw gRPC regression
proving that the order-by conflict is rejected before ANN Search. The ten
upstream parity tests remain isolated in the compatibility module.

## 14. Acceptance Criteria

- The ten upstream L2 test names exist in
  `tests/compatibility/test_function_chain_l2.py`.
- All ten execute through `MilvusClient.search()` using the target PyMilvus
  checkout.
- All ten pass against Milvus Lite.
- The order-by conflict is rejected before ANN Search.
- Older PyMilvus packages without Function Chain support skip the module rather
  than failing test collection.
- Existing Function Chain adapter, validator, compiler, runtime, Boost Ranker,
  ordinary Search, and Hybrid Search tests remain green.
- No L0, XGBoost, FileResource, MinIO, or Hybrid Function Chain behavior is
  added.
- `Collection.search()` retains its existing signature.
- Documentation identifies the compatibility module and the order-by conflict
  rule.
