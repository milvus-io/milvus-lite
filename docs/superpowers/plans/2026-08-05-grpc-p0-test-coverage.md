# gRPC P0 Test Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add public-API gRPC integration coverage for implemented collection-property, compaction, truncate, and statistics RPCs.

**Architecture:** Tests call `pymilvus.MilvusClient`, use the existing shared adapter server where restart is unnecessary, and use the established standalone restart harness for persistence cases. Production changes are allowed only when a new compatibility test exposes a real defect.

**Tech Stack:** Python 3.10+, pytest, pymilvus, grpcio, Milvus Lite gRPC adapter.

---

## File Structure

- Create `tests/adapter/test_grpc_collection_properties.py`: property lifecycle, alias routing, database isolation, and failure atomicity.
- Create `tests/adapter/test_grpc_compaction.py`: manual-compaction compatibility response and logical-data preservation.
- Modify `tests/adapter/test_p0_features.py`: truncate state transitions plus collection and partition statistics.
- Modify `tests/adapter/test_server_restart.py`: property and truncate persistence across a full server restart.
- Modify production files only if a focused test demonstrates a defect; likely candidates are `milvus_lite/db.py` and `milvus_lite/adapter/grpc/servicer.py`.

### Task 1: Collection property lifecycle

**Files:**
- Create: `tests/adapter/test_grpc_collection_properties.py`
- Reference: `tests/adapter/test_grpc_timestamptz_milvus_compat.py`
- Reference: `tests/adapter/test_grpc_database.py`

- [ ] **Step 1: Add shared schema and property helpers**

```python
import pytest
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException


def _schema():
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    return schema


def _create(client, name):
    client.create_collection(name, schema=_schema())


def _properties(client, name):
    return client.describe_collection(name).get("properties", {})
```

- [ ] **Step 2: Add set, replace, and delete property tests**

```python
def test_collection_properties_set_replace_and_delete(milvus_client):
    _create(milvus_client, "props")

    milvus_client.alter_collection_properties(
        "props", {"custom.mode": "first", "custom.keep": "yes"}
    )
    assert _properties(milvus_client, "props")["custom.mode"] == "first"

    milvus_client.alter_collection_properties("props", {"custom.mode": "second"})
    props = _properties(milvus_client, "props")
    assert props["custom.mode"] == "second"
    assert props["custom.keep"] == "yes"

    milvus_client.drop_collection_properties("props", ["custom.mode"])
    props = _properties(milvus_client, "props")
    assert "custom.mode" not in props
    assert props["custom.keep"] == "yes"
```

- [ ] **Step 3: Add missing-resource atomicity test**

```python
def test_alter_missing_collection_has_no_side_effect(milvus_client):
    _create(milvus_client, "sentinel")
    milvus_client.alter_collection_properties("sentinel", {"custom.keep": "yes"})

    with pytest.raises(MilvusException):
        milvus_client.alter_collection_properties("missing", {"custom.keep": "no"})

    assert _properties(milvus_client, "sentinel")["custom.keep"] == "yes"
    assert milvus_client.has_collection("sentinel")
```

- [ ] **Step 4: Add alias routing and database isolation tests**

```python
def test_collection_properties_via_alias(milvus_client):
    _create(milvus_client, "target")
    milvus_client.create_alias("target", "target_alias")

    milvus_client.alter_collection_properties(
        "target_alias", {"custom.route": "alias"}
    )

    assert _properties(milvus_client, "target")["custom.route"] == "alias"
    assert milvus_client.describe_alias("target_alias")["collection_name"] == "target"


def test_collection_properties_are_database_isolated(milvus_client):
    milvus_client.create_database("tenant_a")
    milvus_client.create_database("tenant_b")
    for database, value in (("tenant_a", "a"), ("tenant_b", "b")):
        milvus_client.using_database(database)
        _create(milvus_client, "shared")
        milvus_client.alter_collection_properties("shared", {"custom.owner": value})

    milvus_client.using_database("tenant_a")
    assert _properties(milvus_client, "shared")["custom.owner"] == "a"
    milvus_client.using_database("tenant_b")
    assert _properties(milvus_client, "shared")["custom.owner"] == "b"
```

- [ ] **Step 5: Run the focused property tests**

Run: `pytest tests/adapter/test_grpc_collection_properties.py -v`

Expected: all tests pass. If `describe_collection()` omits generic properties or alias routing fails, retain the failing test and fix only the demonstrated adapter/engine defect.

- [ ] **Step 6: Commit the property coverage**

```bash
git add tests/adapter/test_grpc_collection_properties.py milvus_lite/db.py milvus_lite/adapter/grpc/servicer.py
git commit -m "test: cover grpc collection properties"
```

Only add production files that were actually changed.

### Task 2: Manual compaction compatibility

**Files:**
- Create: `tests/adapter/test_grpc_compaction.py`
- Reference: `milvus_lite/adapter/grpc/servicer.py`

- [ ] **Step 1: Add the compaction helpers and empty-collection case**

```python
from pymilvus import DataType, MilvusClient


def _create(client, name):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("value", DataType.INT64)
    client.create_collection(name, schema=schema)


def test_compact_empty_collection_reports_completed(milvus_client):
    _create(milvus_client, "compact_empty")

    job_id = milvus_client.compact("compact_empty")

    assert isinstance(job_id, int)
    assert milvus_client.get_compaction_state(job_id).lower() == "completed"
```

- [ ] **Step 2: Add logical-data preservation after mutation history**

```python
def test_compact_preserves_logical_rows_and_search(milvus_client):
    _create(milvus_client, "compact_data")
    milvus_client.insert("compact_data", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "value": 10},
        {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "value": 20},
    ])
    milvus_client.upsert("compact_data", [
        {"id": 2, "vec": [0.0, 0.0, 1.0, 0.0], "value": 200},
        {"id": 3, "vec": [0.0, 0.0, 0.0, 1.0], "value": 30},
    ])
    milvus_client.delete("compact_data", ids=[1])
    milvus_client.flush("compact_data")
    before = milvus_client.query(
        "compact_data", filter="id >= 0", output_fields=["id", "value"], limit=10
    )

    job_id = milvus_client.compact("compact_data")

    assert milvus_client.get_compaction_state(job_id).lower() == "completed"
    after = milvus_client.query(
        "compact_data", filter="id >= 0", output_fields=["id", "value"], limit=10
    )
    assert sorted(before, key=lambda row: row["id"]) == sorted(
        after, key=lambda row: row["id"]
    )
    hits = milvus_client.search(
        "compact_data", data=[[0.0, 0.0, 1.0, 0.0]], limit=1,
        output_fields=["value"],
    )
    assert hits[0][0]["id"] == 2
    assert hits[0][0]["entity"]["value"] == 200
```

- [ ] **Step 3: Run the focused compaction tests**

Run: `pytest tests/adapter/test_grpc_compaction.py -v`

Expected: both tests pass and freeze the documented compatibility-shim behavior. Do not add segment-count or asynchronous-progress assertions.

- [ ] **Step 4: Commit the compaction coverage**

```bash
git add tests/adapter/test_grpc_compaction.py
git commit -m "test: cover grpc compaction compatibility"
```

### Task 3: Truncate state transitions

**Files:**
- Modify: `tests/adapter/test_p0_features.py`

- [ ] **Step 1: Add index, alias, mutation-history, and post-truncate read coverage**

```python
def test_truncate_preserves_metadata_and_clears_mutation_history(milvus_client):
    _create_basic_collection(milvus_client, "truncate_history")
    milvus_client.create_alias("truncate_history", "truncate_alias")
    idx = milvus_client.prepare_index_params()
    idx.add_index(
        field_name="vec", index_type="BRUTE_FORCE", metric_type="COSINE", params={}
    )
    milvus_client.create_index("truncate_history", idx)
    milvus_client.alter_collection_properties(
        "truncate_history", {"custom.keep": "yes"}
    )
    milvus_client.insert("truncate_history", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "delete"},
        {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "upsert"},
    ])
    milvus_client.upsert("truncate_history", [
        {"id": 2, "vec": [0.0, 0.0, 1.0, 0.0], "tag": "updated"},
        {"id": 3, "vec": [0.0, 0.0, 0.0, 1.0], "tag": "new"},
    ])
    milvus_client.delete("truncate_history", ids=[1])

    milvus_client.truncate_collection("truncate_alias")

    assert int(milvus_client.get_collection_stats("truncate_history")["row_count"]) == 0
    assert milvus_client.query(
        "truncate_history", filter="id >= 0", output_fields=["id"]
    ) == []
    assert milvus_client.list_indexes("truncate_history") == ["vec"]
    assert milvus_client.describe_alias("truncate_alias")["collection_name"] == "truncate_history"
    assert milvus_client.describe_collection("truncate_history")["properties"]["custom.keep"] == "yes"
```

- [ ] **Step 2: Add released, flush, and database-isolation cases**

```python
def test_truncate_released_collection_then_flush(milvus_client):
    _create_basic_collection(milvus_client, "truncate_released")
    milvus_client.insert("truncate_released", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "old"}
    ])
    milvus_client.release_collection("truncate_released")

    milvus_client.truncate_collection("truncate_released")
    milvus_client.flush("truncate_released")

    assert int(milvus_client.get_collection_stats("truncate_released")["row_count"]) == 0


def test_truncate_is_database_isolated(milvus_client):
    for database, row_id in (("truncate_db_a", 1), ("truncate_db_b", 2)):
        milvus_client.create_database(database)
        milvus_client.using_database(database)
        _create_basic_collection(milvus_client, "shared")
        milvus_client.insert("shared", [
            {"id": row_id, "vec": [1.0, 0.0, 0.0, 0.0], "tag": database}
        ])

    milvus_client.using_database("truncate_db_a")
    milvus_client.truncate_collection("shared")
    assert int(milvus_client.get_collection_stats("shared")["row_count"]) == 0
    milvus_client.using_database("truncate_db_b")
    assert int(milvus_client.get_collection_stats("shared")["row_count"]) == 1
```

- [ ] **Step 3: Run truncate-focused tests**

Run: `pytest tests/adapter/test_p0_features.py -k truncate -v`

Expected: all truncate tests pass. If metadata disappears, fix `MilvusLite.truncate_collection()` to preserve the persisted metadata named by the failing assertion.

- [ ] **Step 4: Commit truncate coverage and any minimal fix**

```bash
git add tests/adapter/test_p0_features.py milvus_lite/db.py
git commit -m "test: cover grpc truncate state transitions"
```

Only add `milvus_lite/db.py` if changed.

### Task 4: Collection and partition statistics

**Files:**
- Modify: `tests/adapter/test_p0_features.py`

- [ ] **Step 1: Add collection-statistics mutation tests**

```python
def test_collection_stats_follow_upsert_delete_flush_and_truncate(milvus_client):
    _create_basic_collection(milvus_client, "collection_stats")
    assert int(milvus_client.get_collection_stats("collection_stats")["row_count"]) == 0
    milvus_client.insert("collection_stats", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "one"},
        {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "two"},
    ])
    milvus_client.upsert("collection_stats", [
        {"id": 2, "vec": [0.0, 0.0, 1.0, 0.0], "tag": "updated"},
        {"id": 3, "vec": [0.0, 0.0, 0.0, 1.0], "tag": "three"},
    ])
    assert int(milvus_client.get_collection_stats("collection_stats")["row_count"]) == 3
    milvus_client.delete("collection_stats", ids=[1, 999])
    assert int(milvus_client.get_collection_stats("collection_stats")["row_count"]) == 2
    milvus_client.flush("collection_stats")
    assert int(milvus_client.get_collection_stats("collection_stats")["row_count"]) == 2
    milvus_client.truncate_collection("collection_stats")
    assert int(milvus_client.get_collection_stats("collection_stats")["row_count"]) == 0
```

- [ ] **Step 2: Add partition-statistics mutation and negative tests**

```python
def test_partition_stats_follow_upsert_delete_and_flush(milvus_client):
    _create_basic_collection(milvus_client, "partition_stats")
    milvus_client.create_partition("partition_stats", "archive")
    milvus_client.insert("partition_stats", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "tag": "default"}
    ])
    milvus_client.insert("partition_stats", [
        {"id": 10, "vec": [0.0, 1.0, 0.0, 0.0], "tag": "old"},
        {"id": 11, "vec": [0.0, 0.0, 1.0, 0.0], "tag": "keep"},
    ], partition_name="archive")
    milvus_client.upsert("partition_stats", [
        {"id": 10, "vec": [0.0, 0.0, 0.0, 1.0], "tag": "updated"},
        {"id": 12, "vec": [1.0, 1.0, 0.0, 0.0], "tag": "new"},
    ], partition_name="archive")
    milvus_client.delete("partition_stats", ids=[11], partition_name="archive")
    assert int(milvus_client.get_partition_stats("partition_stats", "_default")["row_count"]) == 1
    assert int(milvus_client.get_partition_stats("partition_stats", "archive")["row_count"]) == 2
    milvus_client.flush("partition_stats")
    assert int(milvus_client.get_partition_stats("partition_stats", "archive")["row_count"]) == 2
```

```python
def test_statistics_missing_resources_raise(milvus_client):
    import pytest
    from pymilvus.exceptions import MilvusException

    with pytest.raises(MilvusException):
        milvus_client.get_collection_stats("missing")

    _create_basic_collection(milvus_client, "missing_partition")
    with pytest.raises(MilvusException):
        milvus_client.get_partition_stats("missing_partition", "missing")
    milvus_client.create_partition("missing_partition", "gone")
    milvus_client.drop_partition("missing_partition", "gone")
    with pytest.raises(MilvusException):
        milvus_client.get_partition_stats("missing_partition", "gone")
```

- [ ] **Step 3: Run focused statistics tests**

Run: `pytest tests/adapter/test_p0_features.py -k stats -v`

Expected: all statistics tests pass and `row_count` reflects logical visible primary keys, not physical versions.

- [ ] **Step 4: Commit statistics coverage and any minimal fix**

```bash
git add tests/adapter/test_p0_features.py milvus_lite/db.py
git commit -m "test: cover grpc collection and partition stats"
```

Only add `milvus_lite/db.py` if changed.

### Task 5: Restart persistence

**Files:**
- Modify: `tests/adapter/test_server_restart.py`
- Reference: `tests/adapter/test_server_restart.py:33`

- [ ] **Step 1: Add collection-property persistence using the existing restart pattern**

Add a test that starts a server on one temporary directory, creates a
collection, sets `custom.persist=yes`, stops both server and DB, restarts from
the same directory, and asserts:

```python
assert client2.describe_collection("restart_props")["properties"]["custom.persist"] == "yes"
```

Use the existing `_stop(server, db)` helper and the same `try/finally` cleanup
structure as `test_insert_survives_restart()`.

- [ ] **Step 2: Add truncate persistence using the same restart pattern**

Before stopping the first server:

```python
client.insert("restart_truncate", [{
    "id": 1,
    "vec": [1.0, 0.0, 0.0, 0.0],
    "text": "old",
}])
client.flush("restart_truncate")
client.truncate_collection("restart_truncate")
assert int(client.get_collection_stats("restart_truncate")["row_count"]) == 0
```

After restarting:

```python
assert int(client2.get_collection_stats("restart_truncate")["row_count"]) == 0
assert client2.query(
    "restart_truncate", filter="id >= 0", output_fields=["id"]
) == []
```

- [ ] **Step 3: Run restart tests**

Run: `pytest tests/adapter/test_server_restart.py -k 'properties or truncate' -v`

Expected: both new tests pass across a real server stop and restart.

- [ ] **Step 4: Commit restart coverage**

```bash
git add tests/adapter/test_server_restart.py milvus_lite/db.py
git commit -m "test: cover grpc p0 restart persistence"
```

Only add `milvus_lite/db.py` if changed.

### Task 6: Full verification and cleanup

**Files:**
- Verify: `tests/adapter/test_grpc_collection_properties.py`
- Verify: `tests/adapter/test_grpc_compaction.py`
- Verify: `tests/adapter/test_p0_features.py`
- Verify: `tests/adapter/test_server_restart.py`

- [ ] **Step 1: Run the four affected files together**

Run:

```bash
pytest \
  tests/adapter/test_grpc_collection_properties.py \
  tests/adapter/test_grpc_compaction.py \
  tests/adapter/test_p0_features.py \
  tests/adapter/test_server_restart.py -v
```

Expected: all selected tests pass with no skips caused by test logic.

- [ ] **Step 2: Run the complete adapter suite**

Run: `pytest tests/adapter -q`

Expected: all adapter tests pass; environment-based optional dependency skips are acceptable if already present before this work.

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors, only intended P0 test files and demonstrated minimal fixes are modified, and the pre-existing untracked `docs/superpowers/plans/2026-07-30-wal-recovery-validation.md` remains untouched.

- [ ] **Step 4: Commit any final test-only cleanup**

```bash
git add \
  tests/adapter/test_grpc_collection_properties.py \
  tests/adapter/test_grpc_compaction.py \
  tests/adapter/test_p0_features.py \
  tests/adapter/test_server_restart.py
git commit -m "test: complete grpc p0 coverage"
```

Skip this commit if the worktree is already clean for the task.
