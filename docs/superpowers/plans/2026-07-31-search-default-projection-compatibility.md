# Search Default Projection Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Milvus-compatible default Search output so omitted `output_fields` retains all non-primary schema fields, including vectors, while preserving dynamic fields.

**Architecture:** Fix the behavior at the shared projection-plan source rather than adding Function Chain or gRPC special cases. The gRPC result translator will continue deriving protocol-visible fields from that plan, so ordinary Search and Function Chain Search share identical projection semantics.

**Tech Stack:** Python 3.10+, pytest, PyMilvus gRPC compatibility layer.

---

### Task 1: Lock the compatible default projection in unit tests

**Files:**
- Modify: `tests/engine/test_projection.py`
- Modify: `tests/engine/test_collection_search_output_fields.py`
- Modify: `tests/adapter/test_grpc_translators_result.py`

- [ ] **Step 1: Update the projection expectation to include vectors**

Change the Search assertion in `test_default_projection_includes_dynamic_fields` to:

```python
assert search_plan.response_schema_fields == ("text", "vector")
```

- [ ] **Step 2: Update the gRPC result expectation and input entity**

Include a vector value in the result entity and require it in the protocol field list:

```python
"entity": {
    "vector": [1.0, 0.0],
    "popularity": 2.0,
    "dynamic_tag": "first",
},
```

```python
assert list(result.output_fields) == ["vector", "popularity", "$meta"]
```

- [ ] **Step 3: Run the unit tests to verify RED**

Run:

```bash
python3 -m pytest -q \
  tests/engine/test_projection.py::test_default_projection_includes_dynamic_fields \
  tests/adapter/test_grpc_translators_result.py::test_default_dynamic_search_declares_meta_output_field
```

Expected: both tests fail because the current Search projection removes vector fields.

- [ ] **Step 4: Update the legacy Collection.search expectation**

Rename the default-output test to describe the compatible behavior and require
the vector alongside scalar fields:

```python
def test_default_output_fields_none_returns_all_non_primary_fields(col):
    res = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=1)
    hit = res[0][0]
    assert set(hit["entity"].keys()) == {
        "vec", "title", "score", "active"
    }
```

### Task 2: Restore the default Search schema projection

**Files:**
- Modify: `milvus_lite/engine/projection.py:43-51`
- Test: `tests/engine/test_projection.py`
- Test: `tests/adapter/test_grpc_translators_result.py`

- [ ] **Step 1: Implement the minimal source fix**

Replace the special scalar-only Search branch with a non-primary-field projection:

```python
if output_fields is None:
    if api_kind == "search":
        response_schema = tuple(
            field.name for field in schema.fields if not field.is_primary
        )
    else:
        response_schema = schema_names
```

Do not alter explicit projection, Query, Get, or dynamic-field handling.

- [ ] **Step 2: Run the unit tests to verify GREEN**

Run:

```bash
python3 -m pytest -q \
  tests/engine/test_projection.py \
  tests/adapter/test_grpc_translators_result.py
```

Expected: all selected tests pass.

### Task 3: Verify Function Chain and broader compatibility

**Files:**
- Test: `tests/adapter/test_function_chain.py`
- Test: `tests/adapter/test_dynamic_field_coverage.py`

- [ ] **Step 1: Run the previously failing Function Chain test**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
python3 -m pytest -q \
  tests/adapter/test_function_chain.py::test_search_function_chain_limit_changes_each_query_chunk
```

Expected: PASS, with `vector`, scalar fields, and dynamic projection intact.

- [ ] **Step 2: Run the Function Chain and dynamic-field suites**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
python3 -m pytest -q \
  tests/adapter/test_function_chain.py \
  tests/adapter/test_dynamic_field_coverage.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the regular test suite**

Run:

```bash
PYTHONPATH=/Users/junjie.jiang/Workspace/dev/pymilvus \
python3 -m pytest --tb=short -q \
  -m "not slow and not soak" --ignore=tests/benchmark
```

Expected: no failures.

- [ ] **Step 4: Check the final diff**

Run:

```bash
git diff --check
git diff --name-only --diff-filter=U
```

Expected: both commands produce no errors or unresolved files.

- [ ] **Step 5: Commit the implementation**

```bash
git add \
  milvus_lite/engine/projection.py \
  tests/engine/test_projection.py \
  tests/adapter/test_grpc_translators_result.py \
  docs/superpowers/plans/2026-07-31-search-default-projection-compatibility.md
git commit -m "fix: preserve vectors in default search projection"
```
