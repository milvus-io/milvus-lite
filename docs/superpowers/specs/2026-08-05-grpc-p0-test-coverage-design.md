# gRPC P0 Test Coverage Design

## Goal

Complete end-to-end `pymilvus.MilvusClient` coverage for the implemented gRPC
P0 surfaces that are currently uncovered or only covered by a single happy
path: collection properties, compaction compatibility, truncate, collection
statistics, and partition statistics.

The work is test-first and compatibility-focused. It does not add coverage for
RPCs that intentionally return `UNIMPLEMENTED`, and it does not expand the
Milvus Lite feature set merely to make an upstream Milvus test pass.

## Reference and Selection Policy

Use the upstream tests under
`/Users/junjie.jiang/Workspace/dev/milvus/tests/python_client/milvus_client`
as behavioral references. Port the intent of a test rather than its fixtures or
distributed-cluster assumptions.

A candidate test is included only when all of the following are true:

1. The underlying RPC is implemented by `MilvusServicer`.
2. The behavior is meaningful for a local, single-process database.
3. The scenario is not already asserted by an equivalent adapter test.
4. The expected result follows Milvus Lite's documented compatibility model.

## Test Organization

Keep the existing shared `milvus_client` fixture and exercise the public
`MilvusClient` API so every test covers request translation, the gRPC method,
the engine call, response translation, and pymilvus decoding.

Add two focused files and extend two existing files:

- Create `tests/adapter/test_grpc_collection_properties.py` for
  `AlterCollection` property lifecycle and database isolation.
- Create `tests/adapter/test_grpc_compaction.py` for `ManualCompaction` and
  `GetCompactionState` compatibility behavior.
- Extend `tests/adapter/test_grpc_snapshot.py` with truncate state-transition
  and restart cases.
- Extend `tests/adapter/test_p0_features.py` with collection and partition
  statistics cases, retaining its existing basic truncate/statistics tests.

Tests use unique collection, partition, alias, and database names when a case
creates more than one namespace. Restart tests use the existing direct-server
or restart fixture pattern instead of introducing a second server harness.

## Collection Property Coverage

Cover the `AlterCollection` RPC through
`alter_collection_properties()` and `drop_collection_properties()`:

1. Set a property and observe it through `describe_collection()`.
2. Replace an existing property value.
3. Delete one property while preserving unrelated properties.
4. Alter a missing collection and verify the request fails without changing a
   sentinel collection.
5. Create same-named collections in two databases and verify properties are
   isolated.
6. Alter properties through an alias and verify the underlying collection is
   updated while the alias remains usable.
7. Restart the server and verify collection properties persist.

The tests will use neutral custom property keys unless a property has explicit
engine behavior. Timezone behavior is already covered by
`test_grpc_timestamptz_milvus_compat.py` and will not be duplicated.

## Compaction Compatibility Coverage

Milvus Lite automatically compacts after flush. Its implemented manual
compaction RPC is a compatibility shim: `compact()` succeeds and
`get_compaction_state()` reports `Completed`. Tests must freeze that contract
without claiming that the RPC synchronously triggers a new compaction.

Cover:

1. Compact an empty collection successfully.
2. Insert, upsert, delete, and flush data, then compact successfully.
3. Validate the returned compaction identifier is accepted by
   `get_compaction_state()`.
4. Validate the reported state represents completion.
5. Query and search before and after compact and verify logical data is
   unchanged.
6. Record the current missing-collection behavior with an explicit test after
   checking the pymilvus call path. If the shim does not receive a collection
   identity, test the protocol-level behavior instead of inventing validation.

## Truncate Coverage

Extend the existing schema-preservation case to cover state transitions:

1. Preserve schema, collection properties, aliases, and index metadata.
2. Clear live rows after mixed insert, upsert, and delete history.
3. Allow new insert, query, and search operations after truncate.
4. Truncate a released collection and preserve its released state unless the
   current engine contract explicitly reloads it.
5. Truncate through an alias and observe the underlying collection as empty.
6. Flush after truncate and verify the collection remains empty.
7. Restart after truncate and verify removed rows do not recover from WAL or
   immutable files.
8. Verify truncating a collection in one database does not affect a same-named
   collection in another database.

Each stateful case asserts both `row_count` and a read operation where
applicable. This avoids accepting a statistics-only implementation bug.

## Statistics Coverage

### Collection statistics

Cover `get_collection_stats()` for:

1. Empty collection: `row_count == 0`.
2. Inserted rows.
3. Upsert of existing and new primary keys without double counting.
4. Delete of existing and missing primary keys.
5. Stable results before and after flush.
6. Zero rows after truncate.
7. Missing collection error.

### Partition statistics

Cover `get_partition_stats()` for:

1. Independent counts for `_default` and a named partition.
2. Upsert and delete effects scoped to the target partition.
3. Stable results before and after flush.
4. Missing collection error.
5. Missing or dropped partition error.

Statistics assertions normalize `row_count` with `int(...)` because pymilvus
and Milvus versions may expose the value as either an integer or a decimal
string.

## Error and Side-Effect Rules

For every negative case, assert more than an exception type:

- The error message identifies the missing or invalid resource when stable.
- Existing collections, aliases, properties, and rows remain unchanged.
- The gRPC channel remains usable by issuing a small successful operation
  after the failure when the scenario could expose server instability.

Do not require exact server error text or internal numeric error codes in E2E
tests. Those belong in `test_grpc_error_mapping.py`.

## Verification

Run focused files first:

```bash
pytest tests/adapter/test_grpc_collection_properties.py -v
pytest tests/adapter/test_grpc_compaction.py -v
pytest tests/adapter/test_grpc_snapshot.py -v
pytest tests/adapter/test_p0_features.py -v
```

Then run the adapter suite:

```bash
pytest tests/adapter -q
```

If a new test exposes a production defect, keep the reproducing test and make
the minimal implementation fix in a separately described task. Do not weaken
the assertion solely to preserve current behavior unless the behavior is an
intentional compatibility shim documented above.

## Out of Scope

- `LoadPartitions`, `ReleasePartitions`, or any inherited unimplemented RPC.
- RBAC, replicas, shards, resource groups, and distributed consistency.
- Server-side asynchronous compaction progress or segment-count assertions.
- Force merge, external tables, snapshot service, and schema add/drop field.
- Duplicating broad JSON, array, FTS, filter, and vector-index test suites.
