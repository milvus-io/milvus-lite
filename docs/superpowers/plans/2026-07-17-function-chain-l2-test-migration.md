# Function Chain L2 Upstream Test Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the ten upstream Milvus ordinary-search `L2_RERANK` Function Chain tests into a standalone Milvus Lite compatibility module and add the missing rejection of `order_by_fields` combined with a public Function Chain.

**Architecture:** Keep the engine API unchanged and implement the missing conflict check entirely in the gRPC adapter. `parse_search_request()` exposes the serialized `order_by_fields` value, `prepare_search_function_chain()` rejects it before ANN Search, and the servicer passes the parsed value into planning. The compatibility suite uses public `MilvusClient.search()` calls against a temporary Milvus Lite gRPC server and preserves the ten upstream test names.

**Tech Stack:** Python 3.14, pytest, PyMilvus Function Chain DSL, protobuf/gRPC, Milvus Lite adapter and engine test fixtures.

---

## File Map

- Modify `milvus_lite/adapter/grpc/translators/search.py` to expose `order_by_fields` from serialized Search parameters.
- Modify `milvus_lite/adapter/grpc/function_chain.py` to reject non-empty `order_by_fields` whenever public Function Chains are present.
- Modify `milvus_lite/adapter/grpc/servicer.py` to pass the parsed order-by value into Function Chain planning before ANN Search.
- Modify `tests/adapter/test_function_chain.py` with a raw gRPC regression proving the conflict is rejected as `IllegalArgument` before `Collection.search()` runs.
- Create `tests/compatibility/test_function_chain_l2.py` with standalone server/client fixtures, PyMilvus feature detection, shared helpers, and the ten upstream L2 test names.
- Modify `docs/modules.md` to document the ordinary Search conflict rule and confirm that no engine search API changes are involved.

## Execution Constraints

- Work in `/Users/junjie.jiang/Workspace/dev/milvus-lite/.worktrees/function-chain-api` on branch `feature/function-chain-api`.
- Use `/opt/homebrew/opt/python@3.14/bin/python3.14` for all test commands.
- Set `PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus` so tests use the Function Chain-enabled PyMilvus checkout.
- gRPC tests bind loopback ports and may require sandbox escalation.
- Do not add `L0_RERANK`, XGBoost, FileResource, MinIO, Hybrid Search Function Chains, or actual `order_by` result sorting.
- Do not change `milvus_lite.engine.collection.Collection.search()` or its callers outside the gRPC adapter.

### Task 1: Reject Order-By Before ANN Search

**Files:**
- Modify: `tests/adapter/test_function_chain.py:1`
- Modify: `tests/adapter/test_function_chain.py:540`
- Modify: `milvus_lite/adapter/grpc/translators/search.py:79`
- Modify: `milvus_lite/adapter/grpc/translators/search.py:139`
- Modify: `milvus_lite/adapter/grpc/function_chain.py:34`
- Modify: `milvus_lite/adapter/grpc/servicer.py:535`

- [ ] **Step 1: Add the raw gRPC regression test**

Add `json` to the imports at the top of `tests/adapter/test_function_chain.py`:

```python
import json
import uuid
from types import SimpleNamespace
```

Add this test immediately after `test_raw_search_rejects_function_score_function_chain_conflict`:

```python
def test_raw_search_rejects_order_by_function_chain_conflict_before_ann(
    milvus_client,
    grpc_server,
    monkeypatch,
):
    from milvus_lite.engine.collection import Collection

    name = _collection_name("function_chain_order_by_conflict")
    _create_collection(milvus_client, name)

    def fail_if_ann_runs(self, *args, **kwargs):
        raise AssertionError("ANN search must not run")

    monkeypatch.setattr(Collection, "search", fail_if_ann_runs)
    request = _search_request(name)
    request.function_chains.append(_score_plus_popularity_chain().to_proto())
    request.search_params.add(
        key="order_by_fields",
        value=json.dumps("popularity:asc"),
    )
    channel, stub = _raw_stub(grpc_server)

    try:
        response = stub.Search(request)
    finally:
        channel.close()

    assert response.status.error_code == 6
    assert response.status.code == 6
    assert response.status.reason == (
        "order_by and function rerank cannot be used together"
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/adapter/test_function_chain.py::test_raw_search_rejects_order_by_function_chain_conflict_before_ann -v
```

Expected: FAIL because the request reaches the monkeypatched `Collection.search()` and the response reason contains `ANN search must not run` instead of the order-by conflict message.

- [ ] **Step 3: Parse `order_by_fields` in the Search translator**

In `milvus_lite/adapter/grpc/translators/search.py`, read the raw value after the group-by parameters are decoded:

```python
    order_by_fields = raw_params.get("order_by_fields")
```

Expose it in the returned dictionary:

```python
        "group_by_field": group_by_field,
        "group_size": group_size,
        "group_size_strict": strict_group_size,
        "order_by_fields": order_by_fields,
        "radius": radius,
```

Do not parse the order-by expression further. Milvus Lite only needs to distinguish absence/empty values from a requested order-by when a Function Chain is present.

- [ ] **Step 4: Add the planner conflict parameter and validation**

Change `prepare_search_function_chain()` in `milvus_lite/adapter/grpc/function_chain.py` to accept the parsed value and reject it after confirming a chain is present:

```python
def prepare_search_function_chain(
    *,
    function_chains,
    has_function_score: bool,
    schema,
    num_queries: int,
    requested_output_fields: Optional[list[str]],
    order_by_fields=None,
) -> SearchFunctionChainPlan | None:
    chains = list(function_chains or ())
    if not chains:
        return None
    if has_function_score:
        raise SchemaValidationError(
            "function_score and function_chains cannot be used together"
        )
    if order_by_fields:
        raise SchemaValidationError(
            "order_by and function rerank cannot be used together"
        )
    if len(chains) != 1:
        raise SchemaValidationError(
            "ordinary search supports exactly one function chain"
        )

    representation = function_chain_to_repr(chains[0])
    validated = validate_function_chain(representation, schema, num_queries)
    return SearchFunctionChainPlan(
        chain=compile_function_chain(validated),
        required_fields=validated.required_schema_fields,
        requested_output_fields=(
            tuple(requested_output_fields)
            if requested_output_fields is not None
            else None
        ),
    )
```

Keep `has_function_score` validation first so the existing ranker conflict remains deterministic when both conflicts are present.

- [ ] **Step 5: Pass the parsed value from the servicer**

Update the call in `milvus_lite/adapter/grpc/servicer.py`:

```python
            public_chain_plan = prepare_search_function_chain(
                function_chains=parsed["function_chains"],
                has_function_score=parsed["has_function_score"],
                order_by_fields=parsed.get("order_by_fields"),
                schema=col.schema,
                num_queries=len(parsed["query_vectors"]),
                requested_output_fields=requested_output_fields,
            )
```

- [ ] **Step 6: Run the focused test and verify GREEN**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/adapter/test_function_chain.py::test_raw_search_rejects_order_by_function_chain_conflict_before_ann -v
```

Expected: PASS. The response has status code `6`, the exact semantic reason, and the monkeypatched ANN method is not called.

- [ ] **Step 7: Run adjacent adapter regressions**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/adapter/test_function_chain.py::test_raw_search_rejects_function_score_function_chain_conflict \
  tests/adapter/test_function_chain.py::test_raw_search_rejects_unsupported_rerank_param_before_ann \
  tests/adapter/test_function_chain.py::test_raw_search_rejects_order_by_function_chain_conflict_before_ann -v
```

Expected: `3 passed`.

- [ ] **Step 8: Commit the adapter change**

```bash
git add \
  milvus_lite/adapter/grpc/translators/search.py \
  milvus_lite/adapter/grpc/function_chain.py \
  milvus_lite/adapter/grpc/servicer.py \
  tests/adapter/test_function_chain.py
git commit -m "fix: reject order by with function chains"
```

### Task 2: Scaffold Standalone Compatibility Fixtures

**Files:**
- Create: `tests/compatibility/test_function_chain_l2.py`

- [ ] **Step 1: Create module imports and Function Chain feature detection**

Create `tests/compatibility/test_function_chain_l2.py` with this module header and import block:

```python
"""Upstream L2 Function Chain compatibility tests through MilvusClient."""

from __future__ import annotations

import shutil
import tempfile
import uuid

import pytest

try:
    from pymilvus import (
        DataType,
        Function,
        FunctionChain,
        FunctionChainStage,
        FunctionScore,
        FunctionType,
        MilvusClient,
    )
    from pymilvus.exceptions import MilvusException
    from pymilvus.function_chain import col, fn
    from pymilvus.grpc_gen import milvus_pb2
except (ImportError, AttributeError) as exc:
    pytest.skip(
        f"PyMilvus Function Chain API is unavailable: {exc}",
        allow_module_level=True,
    )

if "function_chains" not in milvus_pb2.SearchRequest.DESCRIPTOR.fields_by_name:
    pytest.skip(
        "PyMilvus SearchRequest has no function_chains field",
        allow_module_level=True,
    )

from milvus_lite.adapter.grpc.server import start_server_in_thread


DIM = 2
VECTOR_FIELD = "vector"
SCALAR_FIELD = "ts"
```

The module-level skip must occur before fixtures are defined so older PyMilvus installations skip collection cleanly instead of failing import or request construction.

- [ ] **Step 2: Add isolated server and client fixtures**

Append:

```python
@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="function_chain_l2_")
    grpc_server, database, port = start_server_in_thread(data_dir)
    yield port
    grpc_server.stop(grace=2)
    database.close()
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def client(server):
    milvus_client = MilvusClient(uri=f"http://127.0.0.1:{server}")
    yield milvus_client
    for name in milvus_client.list_collections():
        milvus_client.drop_collection(name)
    milvus_client.close()
```

- [ ] **Step 3: Add collection and assertion helpers**

Append:

```python
def _collection_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _create_function_chain_collection(client: MilvusClient) -> str:
    name = _collection_name("function_chain_l2")
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field(SCALAR_FIELD, DataType.INT64)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="BRUTE_FORCE",
        metric_type="L2",
        params={},
    )
    client.create_collection(
        name,
        schema=schema,
        index_params=index_params,
    )
    client.insert(
        name,
        [
            {"id": 1, SCALAR_FIELD: 10, VECTOR_FIELD: [0.0, 0.0]},
            {"id": 2, SCALAR_FIELD: 20, VECTOR_FIELD: [0.01, 0.0]},
            {"id": 3, SCALAR_FIELD: 30, VECTOR_FIELD: [0.02, 0.0]},
        ],
    )
    client.load_collection(name)
    return name


def _score_plus_ts_chain():
    return (
        FunctionChain(FunctionChainStage.L2_RERANK, name="score_plus_ts")
        .map(
            "$score",
            fn.num_combine(col("$score"), col(SCALAR_FIELD), mode="sum"),
        )
        .sort(col("$score"), desc=True, tie_break_col=col("$id"))
    )


def _hit_field(hit: dict, field: str):
    if field in hit:
        return hit[field]
    return hit.get("entity", {}).get(field)


def _assert_search_error(
    client: MilvusClient,
    collection_name: str,
    function_chains,
    message: str,
    **kwargs,
) -> None:
    with pytest.raises(MilvusException) as exc_info:
        client.search(
            collection_name=collection_name,
            data=[[0.0, 0.0]],
            anns_field=VECTOR_FIELD,
            search_params={"metric_type": "L2", "params": {}},
            limit=3,
            function_chains=function_chains,
            **kwargs,
        )

    assert message in str(exc_info.value)
```

- [ ] **Step 4: Verify the scaffold imports successfully**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -c \
  'import runpy; runpy.run_path("tests/compatibility/test_function_chain_l2.py")'
```

Expected: exit code `0` with no import or protobuf feature-detection error under the target PyMilvus checkout.

- [ ] **Step 5: Commit the fixture scaffold**

```bash
git add tests/compatibility/test_function_chain_l2.py
git commit -m "test: scaffold L2 function chain compatibility"
```

### Task 3: Migrate Four Positive L2 Scenarios

**Files:**
- Modify: `tests/compatibility/test_function_chain_l2.py`

- [ ] **Step 1: Add scalar reranking and hidden-input tests**

Append the first two upstream-named tests:

```python
def test_search_with_l2_function_chain_sdk_reranks_by_scalar_field(client):
    collection_name = _create_function_chain_collection(client)

    result = client.search(
        collection_name=collection_name,
        data=[[0.0, 0.0]],
        anns_field=VECTOR_FIELD,
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=[SCALAR_FIELD],
        function_chains=_score_plus_ts_chain(),
    )

    assert [hit["id"] for hit in result[0]] == [3, 2, 1]
    assert [_hit_field(hit, SCALAR_FIELD) for hit in result[0]] == [30, 20, 10]


def test_search_with_l2_function_chain_sdk_uses_hidden_input_field(client):
    collection_name = _create_function_chain_collection(client)

    result = client.search(
        collection_name=collection_name,
        data=[[0.0, 0.0]],
        anns_field=VECTOR_FIELD,
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=["id"],
        function_chains=_score_plus_ts_chain(),
    )

    assert [hit["id"] for hit in result[0]] == [3, 2, 1]
    assert all(_hit_field(hit, SCALAR_FIELD) is None for hit in result[0])
```

- [ ] **Step 2: Run the two characterization tests**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/compatibility/test_function_chain_l2.py::test_search_with_l2_function_chain_sdk_reranks_by_scalar_field \
  tests/compatibility/test_function_chain_l2.py::test_search_with_l2_function_chain_sdk_uses_hidden_input_field -v
```

Expected: `2 passed`. These are parity characterization tests for already implemented hidden-field planning and projection, so they are expected to be GREEN on first execution.

- [ ] **Step 3: Add temporary-column and limit tests**

Append:

```python
def test_search_with_l2_function_chain_sdk_temp_column_not_returned(client):
    collection_name = _create_function_chain_collection(client)
    chain = (
        FunctionChain(FunctionChainStage.L2_RERANK, name="l2_temp_score")
        .map(
            "tmp_score",
            fn.num_combine(col("$score"), col(SCALAR_FIELD), mode="sum"),
        )
        .map(
            "$score",
            fn.num_combine(col("tmp_score"), col("$score"), mode="sum"),
        )
        .sort(col("$score"), desc=True, tie_break_col=col("$id"))
    )

    result = client.search(
        collection_name=collection_name,
        data=[[0.0, 0.0]],
        anns_field=VECTOR_FIELD,
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        output_fields=[SCALAR_FIELD],
        function_chains=chain,
    )

    assert [hit["id"] for hit in result[0]] == [3, 2, 1]
    assert [_hit_field(hit, SCALAR_FIELD) for hit in result[0]] == [30, 20, 10]
    assert all(_hit_field(hit, "tmp_score") is None for hit in result[0])


def test_search_with_l2_function_chain_sdk_limit_op(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="l2_limit",
    ).limit(2)

    result = client.search(
        collection_name=collection_name,
        data=[[0.0, 0.0]],
        anns_field=VECTOR_FIELD,
        search_params={"metric_type": "L2", "params": {}},
        limit=3,
        function_chains=chain,
    )

    assert len(result[0]) == 2
```

- [ ] **Step 4: Run all four positive tests**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/compatibility/test_function_chain_l2.py -k \
  'reranks_by_scalar_field or uses_hidden_input_field or temp_column_not_returned or limit_op' -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the positive parity tests**

```bash
git add tests/compatibility/test_function_chain_l2.py
git commit -m "test: migrate positive L2 function chain cases"
```

### Task 4: Migrate Four System-Column Validation Scenarios

**Files:**
- Modify: `tests/compatibility/test_function_chain_l2.py`

- [ ] **Step 1: Add the read-only and reserved-output tests**

Append:

```python
def test_search_rejects_l2_function_chain_write_readonly_system_column(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="bad_l2_write_id",
    ).map(
        "$id",
        fn.num_combine(col("$score"), col(SCALAR_FIELD), mode="sum"),
    )

    _assert_search_error(
        client,
        collection_name,
        chain,
        'system output "$id" is not writable',
    )


def test_search_rejects_l2_function_chain_reserved_temp_output(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="bad_l2_reserved_temp_output",
    ).map(
        "$tmp_score",
        fn.num_combine(col("$score"), col(SCALAR_FIELD), mode="sum"),
    )

    _assert_search_error(
        client,
        collection_name,
        chain,
        'system output "$tmp_score" is not writable',
    )
```

- [ ] **Step 2: Add internal and unknown system-input tests**

Append:

```python
def test_search_rejects_l2_function_chain_read_internal_system_input(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="bad_l2_seg_offset_input",
    ).map(
        "$score",
        fn.num_combine(col("$seg_offset"), col("$score"), mode="sum"),
    )

    _assert_search_error(
        client,
        collection_name,
        chain,
        'system input "$seg_offset" is not supported',
    )


def test_search_rejects_l2_function_chain_read_unknown_system_input(client):
    collection_name = _create_function_chain_collection(client)
    chain = FunctionChain(
        FunctionChainStage.L2_RERANK,
        name="bad_l2_unknown_system_input",
    ).map(
        "$score",
        fn.num_combine(col("$tmp_score"), col("$score"), mode="sum"),
    )

    _assert_search_error(
        client,
        collection_name,
        chain,
        'system input "$tmp_score" is not supported',
    )
```

- [ ] **Step 3: Run the four system-column tests**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/compatibility/test_function_chain_l2.py -k \
  'write_readonly_system_column or reserved_temp_output or read_internal_system_input or read_unknown_system_input' -v
```

Expected: `4 passed`. Each assertion checks a stable semantic fragment rather than distributed Milvus error code `1100`.

- [ ] **Step 4: Commit the system-column parity tests**

```bash
git add tests/compatibility/test_function_chain_l2.py
git commit -m "test: migrate L2 system column validation cases"
```

### Task 5: Migrate FunctionScore and Order-By Conflict Scenarios

**Files:**
- Modify: `tests/compatibility/test_function_chain_l2.py`

- [ ] **Step 1: Add the public FunctionScore conflict test**

Append:

```python
def test_search_rejects_l2_function_chain_with_function_score(client):
    collection_name = _create_function_chain_collection(client)
    function = Function(
        name="boost_ts",
        function_type=FunctionType.RERANK,
        input_field_names=[],
        output_field_names=[],
        params={"reranker": "boost", "weight": "1.5"},
    )
    function_score = FunctionScore(functions=[function])

    _assert_search_error(
        client,
        collection_name,
        _score_plus_ts_chain(),
        "function_chains and ranker cannot be used together",
        ranker=function_score,
    )
```

- [ ] **Step 2: Add the public order-by conflict test**

Append:

```python
def test_search_rejects_l2_function_chain_with_order_by(client):
    collection_name = _create_function_chain_collection(client)

    _assert_search_error(
        client,
        collection_name,
        _score_plus_ts_chain(),
        "order_by and function rerank cannot be used together",
        order_by_fields=[{"field": SCALAR_FIELD, "order": "asc"}],
    )
```

- [ ] **Step 3: Run the two public conflict tests**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/compatibility/test_function_chain_l2.py::test_search_rejects_l2_function_chain_with_function_score \
  tests/compatibility/test_function_chain_l2.py::test_search_rejects_l2_function_chain_with_order_by -v
```

Expected: `2 passed`. This verifies both conflicts through the public `MilvusClient.search()` API rather than only through raw protobuf requests.

- [ ] **Step 4: Verify exactly the ten upstream test names are present**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/compatibility/test_function_chain_l2.py --collect-only -q
```

Expected: exactly these ten tests are collected:

```text
test_search_with_l2_function_chain_sdk_reranks_by_scalar_field
test_search_with_l2_function_chain_sdk_uses_hidden_input_field
test_search_with_l2_function_chain_sdk_temp_column_not_returned
test_search_with_l2_function_chain_sdk_limit_op
test_search_rejects_l2_function_chain_write_readonly_system_column
test_search_rejects_l2_function_chain_reserved_temp_output
test_search_rejects_l2_function_chain_read_internal_system_input
test_search_rejects_l2_function_chain_read_unknown_system_input
test_search_rejects_l2_function_chain_with_function_score
test_search_rejects_l2_function_chain_with_order_by
```

- [ ] **Step 5: Run the complete standalone compatibility module**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/compatibility/test_function_chain_l2.py -v
```

Expected: `10 passed` with the target PyMilvus checkout. With a PyMilvus installation lacking Function Chain API/protobuf support, collection should report one clear module-level skip instead of an import error.

- [ ] **Step 6: Commit the conflict parity tests**

```bash
git add tests/compatibility/test_function_chain_l2.py
git commit -m "test: migrate L2 function chain conflict cases"
```

### Task 6: Document and Verify the Complete Migration

**Files:**
- Modify: `docs/modules.md:2615`
- Test: `tests/compatibility/test_function_chain_l2.py`
- Test: `tests/adapter/test_function_chain.py`
- Test: `tests/function/test_function_chain_validator.py`
- Test: `tests/function/test_public_function_chain.py`

- [ ] **Step 1: Document the Search adapter conflict rule**

Replace the `Search` RPC row in `docs/modules.md` with:

```markdown
| `Search` | `col.search(query_vectors, top_k, metric_type, partition_names, expr, output_fields)` | translator: search.py + result.py; supports exactly one public L2 Function Chain; public Function Chains reject FunctionScore/ranker and `order_by_fields` before ANN Search |
```

This documents adapter behavior without implying that `Collection.search()` accepts or implements order-by sorting.

- [ ] **Step 2: Run the complete compatibility module**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/compatibility/test_function_chain_l2.py -v
```

Expected: `10 passed`.

- [ ] **Step 3: Run the focused adapter module**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/adapter/test_function_chain.py -v
```

Expected: all tests pass, including the new pre-ANN order-by conflict regression.

- [ ] **Step 4: Run related Function Chain regressions**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m pytest \
  tests/compatibility/test_function_chain_l2.py \
  tests/adapter/test_function_chain.py \
  tests/function/test_function_chain_validator.py \
  tests/function/test_public_function_chain.py -v
```

Expected: all selected tests pass with no unexpected skips under the target PyMilvus checkout.

- [ ] **Step 5: Verify the engine API was not changed**

Run:

```bash
git diff -- milvus_lite/engine/collection.py
```

Expected: no output.

- [ ] **Step 6: Review scope and test-name parity**

Run:

```bash
git diff --name-only 6012a4f..HEAD
rg -n '^def test_' tests/compatibility/test_function_chain_l2.py
```

Expected changed implementation/test/document files are limited to:

```text
milvus_lite/adapter/grpc/function_chain.py
milvus_lite/adapter/grpc/servicer.py
milvus_lite/adapter/grpc/translators/search.py
tests/adapter/test_function_chain.py
tests/compatibility/test_function_chain_l2.py
docs/modules.md
docs/superpowers/plans/2026-07-17-function-chain-l2-test-migration.md
```

The compatibility module must list exactly the ten upstream L2 test names and must not contain L0, XGBoost, FileResource, MinIO, or Hybrid Search scenarios.

- [ ] **Step 7: Commit documentation and final verification state**

```bash
git add docs/modules.md
git commit -m "docs: document function chain order by conflict"
```

- [ ] **Step 8: Inspect the final commit series**

Run:

```bash
git status --short
git log --oneline -6
```

Expected: the worktree is clean and the recent commits show the focused adapter fix, fixture scaffold, three compatibility-test batches, and documentation update.
