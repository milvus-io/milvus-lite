# Deep Dive Design: gRPC Adapter Layer (Phase 10)

## 1. Overview

MilvusLite Phase 10 builds a gRPC service layer on top of the internal engine, enabling **pymilvus clients to connect to MilvusLite without any code changes**. This is the final mile of protocol compatibility for the project as a "local version of Milvus."

**Core positioning**: Protocol translation only — no additional engine capabilities. The set of working RPCs is exactly equal to the set of methods already implemented in the engine. The work in this layer is concentrated on:
1. Field mapping + data structure conversion from protocol to engine API (the bulk of the work)
2. Transposition between Milvus FieldData columnar format and engine records row-wise list[dict]
3. Error code translation
4. RPC stubs for implemented RPCs + friendly UNIMPLEMENTED messages for the rest

**Prerequisites**: Phase 9 must be completed first — Phase 10's `CreateIndex / LoadCollection / Search` RPCs map directly to Phase 9's `Collection.create_index / load / search`. Reversing the order of these two phases would leave Phase 10's servicer without a stable lower-layer API to interface with.

---

## 2. Service Definition Source Decision

### 2.1 Three Candidate Approaches

| Approach | Pros | Cons |
|---|---|---|
| **A. Copy official Milvus proto directly** | pymilvus connects out of the box; strongest behavioral consistency; can pull new proto when future Milvus versions upgrade | Proto volume is large (~100 RPCs, hundreds of KB); large stub implementation workload; some RPCs' server-side behavior depends on Milvus internal state machines |
| **B. Hand-write a minimal proto subset** | Simple proto, small implementation effort | pymilvus won't recognize it — loses the biggest selling point |
| **C. Copy Milvus proto but only implement a quickstart subset; return UNIMPLEMENTED for everything else** | Combines pymilvus compatibility from A + manageable workload from B | Users hitting unimplemented RPCs need friendly error messages |

### 2.2 Decision: Approach C

**Rationale**:
1. **pymilvus must be able to connect** is a core project goal — Approach B is vetoed outright
2. Approach A implementing all RPCs is unnecessary waste — Milvus's backup / RBAC / replica / resource group and many other RPCs have no meaning for a local embedded database
3. Approach C equals "protocol surface = Milvus, functional surface = MilvusLite" — pymilvus can connect + most quickstart calls work, while a few RPCs return clear "MilvusLite does not support X" messages

**Prior art**: Zilliz's official `milvus-lite` project (SQLite backend) takes exactly this approach. We are essentially "replacing milvus-lite's SQLite with LSM Parquet + FAISS."

### 2.3 Proto File Source

Pull the following files from [milvus-io/milvus-proto](https://github.com/milvus-io/milvus-proto):
- `proto/milvus.proto` — main RPC definitions
- `proto/schema.proto` — schema / FieldData types
- `proto/common.proto` — common types (Status, KeyValuePair, MsgBase, etc.)

Place them in `milvus_lite/adapter/grpc/proto/`, generate `_pb2.py` / `_pb2_grpc.py` with `grpcio-tools`, and **commit the generated output to the repo** (no runtime generation, to avoid build-time dependency on grpcio-tools).

`proto/README.md` records "which milvus-proto commit was used for generation" for traceability and upgrades.

---

## 3. Module Structure

```
milvus_lite/
└── adapter/
    └── grpc/
        ├── __init__.py
        ├── server.py                   # run_server(data_dir, host, port)
        ├── servicer.py                 # MilvusServicer — all RPC implementations
        ├── translators/
        │   ├── __init__.py
        │   ├── schema.py               # Milvus FieldSchema ↔ milvus_lite FieldSchema
        │   ├── records.py              # FieldData (columnar) ↔ list[dict] (row-wise)
        │   ├── search.py               # SearchRequest parsing
        │   ├── result.py               # engine results → SearchResults proto
        │   ├── expr.py                 # Milvus filter expr ↔ MilvusLite filter
        │   └── index.py                # IndexParams ↔ IndexSpec
        ├── proto/                      # generated stubs
        │   ├── __init__.py
        │   ├── milvus_pb2.py
        │   ├── milvus_pb2_grpc.py
        │   ├── schema_pb2.py
        │   ├── common_pb2.py
        │   └── README.md               # source commit reference
        ├── errors.py                   # MilvusLiteError → grpc Status mapping
        └── server.py                    # server lifecycle used by milvus-lite server
```

**Dependencies**:
```toml
# pyproject.toml
[project.optional-dependencies]
grpc = ["grpcio>=1.50", "protobuf>=4.21"]
```

`grpcio-tools` is only needed during dev / build (for generating stubs) — it is not a runtime dependency.

---

## 4. Full Mapping Table: pymilvus → engine API

### 4.1 Collection Lifecycle

| pymilvus | Milvus RPC | engine API | Notes |
|---|---|---|---|
| `MilvusClient(uri="...")` | (TCP connect) | `MilvusLite(data_dir)` held at server startup | Server mode serves a single data_dir |
| `create_collection(name, dim, ...)` quick mode | `CreateCollection` | `db.create_collection(name, schema)` | Translator generates a default schema from quickstart params (id INT64 + vector FLOAT_VECTOR) |
| `create_collection(name, schema)` full schema | `CreateCollection` | Same as above | Translator parses the `CollectionSchema` proto, converting each field to `FieldSchema` |
| `drop_collection(name)` | `DropCollection` | `db.drop_collection(name)` | Direct mapping |
| `has_collection(name)` | `HasCollection` | `db.has_collection(name)` | bool wrapped in `BoolResponse` |
| `describe_collection(name)` | `DescribeCollection` | `db.get_collection(name).describe()` + schema serialization | Translator converts MilvusLite schema back to Milvus proto schema |
| `list_collections()` | `ShowCollections` | `db.list_collections()` | Direct mapping |
| `get_collection_stats(name)` | `GetCollectionStatistics` | `col.num_entities` | Wrapped as `KeyValuePair[("row_count", str(n))]` |
| `rename_collection(old, new)` | `RenameCollection` | `db.rename_collection(old, new)` | Direct mapping |
| Aliases (`create/drop/alter/describe/list_aliases`) | `CreateAlias` / `DropAlias` / `AlterAlias` / `DescribeAlias` / `ListAliases` | `db.*_alias()` | Persisted in `aliases.json`; aliases are resolved before opening a Collection |
| `truncate_collection(name)` | `TruncateCollection` | `db.truncate_collection(name)` | Clears data while preserving schema and aliases |
| `alter_collection_properties` | `AlterCollection` | Schema is immutable | UNIMPLEMENTED |

### 4.2 Partition

| pymilvus | Milvus RPC | engine API | Notes |
|---|---|---|---|
| `create_partition(collection, partition)` | `CreatePartition` | `col.create_partition(name)` | API added in Phase 9.1 |
| `drop_partition(collection, partition)` | `DropPartition` | `col.drop_partition(name)` | API added in Phase 9.1 |
| `has_partition(collection, partition)` | `HasPartition` | `partition in col.list_partitions()` | bool wrapper |
| `list_partitions(collection)` | `ShowPartitions` | `col.list_partitions()` | Direct mapping |
| `get_partition_stats` | `GetPartitionStatistics` | `col.partition_num_entities(name)` | Returns partition-level `row_count` |
| `load_partitions` / `release_partitions` | `LoadPartitions` / `ReleasePartitions` | Engine load/release is at Collection level | UNIMPLEMENTED or mapped to collection load/release |

### 4.3 Data CRUD

| pymilvus | Milvus RPC | engine API | Notes |
|---|---|---|---|
| `insert(collection, data, partition_name=None)` | `Insert` | `col.insert(records, partition_name)` | **Most complex conversion** — InsertRequest.fields_data is a columnar FieldData structure that needs to be transposed into a records list |
| `upsert(collection, data, partition_name=None)` | `Upsert` | `col.insert(records, partition_name)` | Engine insert already has upsert semantics; both RPCs share the same servicer method |
| `delete(collection, ids=, partition_name=None)` | `Delete` | `col.delete(pks, partition_name)` | When DeleteRequest's expr field is in `id in [...]` form, take the pk path |
| `delete(collection, filter=, partition_name=None)` | `Delete` | `col.query(filter) → extract pk → col.delete(pks)` | Expression-based delete: query to find pks first, then delete |
| `get(collection, ids=, partition_names=, output_fields=)` | `Query`(`id in [...]` expr) | `col.get(pks, partition_names, expr)` | pymilvus get actually goes through the Query RPC |
| `query(collection, filter, output_fields, limit, partition_names)` | `Query` | `col.query(expr, output_fields, partition_names, limit)` | Direct mapping |

### 4.4 Search

| pymilvus | Milvus RPC | engine API | Notes |
|---|---|---|---|
| `search(collection, data, anns_field, limit, filter, output_fields, search_params, partition_names)` | `Search` | `col.search(query_vectors, top_k, metric_type, partition_names, expr, output_fields)` | Translator parses SearchParams to extract metric / topk / search_params; return value structure conversion is the most complex |
| `hybrid_search(collection, reqs, ...)` | `HybridSearch` | Multiple `col.search()` + `reranker.rerank()` | Phase 12: Parse each sub-SearchRequest for independent search, merge results via WeightedRanker / RRFRanker |
| `search_iterator(...)` | (client-side wrapper) | Engine adds offset support | Optional; MVP UNIMPLEMENTED |

### 4.5 Index

| pymilvus | Milvus RPC | engine API | Notes |
|---|---|---|---|
| `create_index(collection, index_params)` | `CreateIndex` | `col.create_index(field, params)` | Translator converts IndexParams' KeyValuePair list to IndexSpec |
| `drop_index(collection, field_name, index_name)` | `DropIndex` | `col.drop_index(field)` | index_name is ignored (engine only supports one index per field) |
| `describe_index(collection, field_name)` | `DescribeIndex` | `col.get_index_info()` | IndexSpec → IndexDescription proto |
| `list_indexes(collection)` | `DescribeIndex` (pymilvus wrapper) | `col.list_indexes()` / `DescribeIndex` | Returns field-name index names for pymilvus iterator compatibility |
| `get_index_state` / `get_index_build_progress` | `GetIndexState` / `GetIndexBuildProgress` | Engine build is synchronous, always returns `Finished` / `100%` | Trivial implementation |

### 4.6 Load / Release

| pymilvus | Milvus RPC | engine API | Notes |
|---|---|---|---|
| `load_collection(collection, replica_number=1)` | `LoadCollection` | `col.load()` | replica_number is ignored |
| `release_collection(collection)` | `ReleaseCollection` | `col.release()` | Direct mapping |
| `get_load_state(collection)` | `GetLoadState` | `col._load_state` | Enum mapping: released → NotLoad, loading → Loading, loaded → Loaded |
| `get_loading_progress(collection)` | `GetLoadingProgress` | Engine load is synchronous, always 100% | Trivial |

### 4.7 Others

| pymilvus | Milvus RPC | engine API | Notes |
|---|---|---|---|
| `flush(collection)` | `Flush` | `col.flush()` | Direct mapping |
| `compact(collection)` | `ManualCompaction` | `col.compact()` | Engine adds a manual compact trigger method |
| `list_databases()` | `ListDatabases` | Stub returns `["default"]` | MilvusLite has no multi-database instances |
| `using_database(name)` | `UseDatabase` | Only accepts "default", returns error otherwise | Trivial |
| User / Role / Privilege | `CreateCredential`, etc. | Embedded mode has no RBAC | Always returns OK with empty results (so pymilvus won't crash) |
| Backup / Restore | Various | Not supported | UNIMPLEMENTED |
| Resource Group | `CreateResourceGroup`, etc. | Not supported | UNIMPLEMENTED |
| Replica | `GetReplicas`, etc. | Single process | UNIMPLEMENTED |
| QueryNode / DataNode and other internal RPCs | Various | — | These are Milvus internal inter-component communication; pymilvus doesn't call them, so they are not implemented |

### 4.8 Handling Strategy for Unsupported RPCs

**Three-tier strategy**:

1. **Explicit UNIMPLEMENTED** (recommended default): Return `grpc.StatusCode.UNIMPLEMENTED` + friendly message
   - Applied to: RPCs with missing functionality such as bulk_insert
   - Error message format: `"MilvusLite does not support X. Reason: <one-line reason>. See https://..."`

2. **Silent success** (rare cases): Return `Success` + empty result
   - Applied to: RBAC series (embedded mode has no user concept; pymilvus calls should not crash)
   - Applied to: Multi-database (always "default")
   - **Decision principle**: Only use this when "pretending to succeed" is completely harmless to user experience; switch back to UNIMPLEMENTED immediately if there is any risk of misleading

3. **Ignore optional parameters** (forward compatibility): Process known fields, ignore unknown fields
   - Applied to: SearchRequest's consistency_level, travel_timestamp, etc.
   - **This is not "pretending to support"** — it is reasonable forward compatibility

---

## 5. Key Conversion: FieldData ↔ records

### 5.1 Milvus FieldData Structure

Milvus InsertRequest uses a per-field columnar structure:

```protobuf
message InsertRequest {
  string collection_name = 1;
  string partition_name = 2;
  repeated FieldData fields_data = 3;
  repeated uint32 hash_keys = 4;
  uint32 num_rows = 5;
}

message FieldData {
  schema.DataType type = 1;
  string field_name = 2;
  oneof field {
    schema.ScalarField scalars = 3;
    schema.VectorField vectors = 4;
  }
  int64 field_id = 5;
}

message ScalarField {
  oneof data {
    BoolArray bool_data = 1;
    IntArray int_data = 2;
    LongArray long_data = 3;
    FloatArray float_data = 4;
    DoubleArray double_data = 5;
    StringArray string_data = 6;
    BytesArray bytes_data = 7;
    ArrayArray array_data = 8;
    JSONArray json_data = 9;
  }
}

message VectorField {
  int64 dim = 1;
  oneof data {
    FloatArray float_vector = 2;
    bytes binary_vector = 3;
    bytes float16_vector = 4;
    bytes bfloat16_vector = 5;
    SparseFloatArray sparse_float_vector = 6;
  }
}
```

**Key differences**:
- Milvus is columnar — one array per field, all fields have the same length = `num_rows`
- MilvusLite engine is row-wise — `List[Dict[field_name, value]]`
- Vector fields in Milvus are flat float arrays (length = `num_rows * dim`), requiring slicing by dim
- Many types — each oneof branch corresponds to a set of conversion logic

### 5.2 Transposition Algorithm

```python
# milvus_lite/adapter/grpc/translators/records.py

def fields_data_to_records(
    fields_data: List["FieldData"],
    num_rows: int,
) -> List[Dict[str, Any]]:
    """Transpose Milvus columnar fields_data into engine row-wise records.

    Args:
        fields_data: list of FieldData proto messages
        num_rows: declared row count from InsertRequest.num_rows

    Returns:
        records: list of dicts, length num_rows. Each dict has all field
                 names from fields_data.

    Raises:
        ValueError: if any FieldData length mismatches num_rows
        UnsupportedFieldTypeError: if a FieldData uses a type MilvusLite
                                    doesn't support (e.g. SparseFloat)
    """
    records: List[Dict[str, Any]] = [{} for _ in range(num_rows)]

    for fd in fields_data:
        column = _extract_column(fd, num_rows)
        for i in range(num_rows):
            records[i][fd.field_name] = column[i]

    return records


def _extract_column(fd: "FieldData", num_rows: int) -> List[Any]:
    """Pull a single FieldData out as a length-num_rows Python list."""
    if fd.HasField("scalars"):
        scalars = fd.scalars
        if scalars.HasField("long_data"):
            data = list(scalars.long_data.data)
        elif scalars.HasField("int_data"):
            data = list(scalars.int_data.data)
        elif scalars.HasField("float_data"):
            data = list(scalars.float_data.data)
        elif scalars.HasField("double_data"):
            data = list(scalars.double_data.data)
        elif scalars.HasField("string_data"):
            data = list(scalars.string_data.data)
        elif scalars.HasField("bool_data"):
            data = list(scalars.bool_data.data)
        elif scalars.HasField("json_data"):
            # JSON values are stored as bytes in proto; decode + parse
            data = [json.loads(b.decode("utf-8")) for b in scalars.json_data.data]
        else:
            raise UnsupportedFieldTypeError(
                f"unsupported scalar field type for {fd.field_name}"
            )
    elif fd.HasField("vectors"):
        vectors = fd.vectors
        dim = vectors.dim
        if vectors.HasField("float_vector"):
            flat = vectors.float_vector.data
            data = [list(flat[i*dim:(i+1)*dim]) for i in range(num_rows)]
        elif vectors.HasField("binary_vector"):
            raise UnsupportedFieldTypeError("binary vectors not supported in MVP")
        elif vectors.HasField("sparse_float_vector"):
            raise UnsupportedFieldTypeError("sparse vectors not supported in MVP")
        else:
            raise UnsupportedFieldTypeError(
                f"unsupported vector field type for {fd.field_name}"
            )
    else:
        raise ValueError(f"FieldData {fd.field_name} has no scalars or vectors")

    if len(data) != num_rows:
        raise ValueError(
            f"FieldData {fd.field_name} has {len(data)} rows, expected {num_rows}"
        )
    return data
```

### 5.3 Reverse Conversion: records → FieldData

Only needed in the return values of `Query` / `Get` / `Search` RPCs — converting engine-returned list[dict] back to Milvus FieldData columnar structure. The algorithm is symmetric:

```python
def records_to_fields_data(
    records: List[Dict[str, Any]],
    schema: "CollectionSchema",
    output_fields: Optional[List[str]] = None,
) -> List["FieldData"]:
    """Build columnar FieldData list from row-wise records, based on
    the collection schema (which knows each field's type).

    output_fields: optional whitelist; only these fields are emitted.
    """
    if not records:
        return []

    field_names = output_fields or [f.name for f in schema.fields]
    fields_data = []

    for fname in field_names:
        fschema = schema.get_field(fname)
        column = [r.get(fname) for r in records]
        fd = _build_field_data(fname, fschema, column)
        fields_data.append(fd)

    return fields_data
```

### 5.4 Test Coverage

`tests/adapter/test_grpc_translators_records.py`:
- One round-trip test per type: build FieldData → fields_data_to_records → records_to_fields_data → should be equivalent
- num_rows mismatch → ValueError
- Unsupported type → UnsupportedFieldTypeError
- Vector dim slicing correctness
- Empty fields_data → empty records list
- Partially nullable fields → handling of None values

---

## 6. Filter Expression Translation

### 6.1 Most Cases: Pass-through

MilvusLite Phase 8's filter grammar is modeled after Milvus, so **the vast majority of expressions are passed through directly**:

```
"age > 18 and category in ['tech', 'news']"   # Fully compatible, passed as-is to col.search(expr=...)
```

### 6.2 Few Scenarios Requiring Rewrite

| Milvus Syntax | MilvusLite Support | Handling Strategy |
|---|---|---|
| `field == value` and other comparisons | Supported | Pass-through |
| `field in [...]` | Supported | Pass-through |
| `field like "pattern"` | Supported (Phase F2a) | Pass-through |
| `$meta["key"] == value` | Supported (Phase F2b) | Pass-through |
| Arithmetic + - * / | Supported (Phase F2a) | Pass-through |
| `is null` / `is not null` | Supported (Phase F2a) | Pass-through |
| `json_contains(json_field, value)` | Supported | Pass-through (parser natively supports it) |
| `array_contains(array_field, value)` | Supported | Pass-through (parser natively supports array_contains / array_contains_all / array_contains_any) |
| `text_match(text_field, query)` | Supported (Phase 11) | Pass-through (engine has built-in BM25 full-text index + analyzer) |
| `phrase_match` | Not supported | UNIMPLEMENTED |

### 6.3 Implementation Approach

**Current state**: After Phase 11, the MilvusLite parser natively supports `text_match`, `json_contains`, `array_contains`, and other functions. All expressions are passed through directly to the engine parser for processing, with no rewriting or interception needed at the adapter layer. Only `phrase_match` remains unsupported (the parser throws `FilterParseError` on unknown functions). A separate `translators/expr.py` file was not created — expressions are passed through directly.

---

## 7. Servicer Implementation Skeleton

```python
# milvus_lite/adapter/grpc/servicer.py

import grpc
from .proto import milvus_pb2, milvus_pb2_grpc, common_pb2
from .errors import to_grpc_status
from .translators.records import fields_data_to_records, records_to_fields_data
from .translators.schema import milvus_to_milvus_lite_schema, milvus_lite_to_milvus_schema
from .translators.expr import translate_filter_expr
from milvus_lite import MilvusLite
from milvus_lite.exceptions import MilvusLiteError

class MilvusServicer(milvus_pb2_grpc.MilvusServiceServicer):
    def __init__(self, db: MilvusLite):
        self._db = db

    # ── Collection lifecycle ─────────────────────────────────────

    def CreateCollection(self, request, context):
        try:
            schema = milvus_to_milvus_lite_schema(request.schema)
            self._db.create_collection(request.collection_name, schema)
            return common_pb2.Status(code=0, reason="")
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    def DropCollection(self, request, context):
        try:
            self._db.drop_collection(request.collection_name)
            return common_pb2.Status(code=0)
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    def HasCollection(self, request, context):
        try:
            exists = self._db.has_collection(request.collection_name)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(code=0),
                value=exists,
            )
        except MilvusLiteError as e:
            return milvus_pb2.BoolResponse(status=common_pb2.Status(**to_grpc_status(e)))

    # ── Data CRUD ────────────────────────────────────────────────

    def Insert(self, request, context):
        try:
            records = fields_data_to_records(request.fields_data, request.num_rows)
            col = self._db.get_collection(request.collection_name)
            partition = request.partition_name or "_default"
            inserted_pks = col.insert(records, partition_name=partition)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(code=0),
                IDs=_pks_to_ids_proto(inserted_pks, schema=col.schema),
                insert_cnt=len(inserted_pks),
                succ_index=list(range(len(inserted_pks))),
            )
        except MilvusLiteError as e:
            return milvus_pb2.MutationResult(status=common_pb2.Status(**to_grpc_status(e)))

    def Search(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            queries = _decode_search_query(request)
            top_k, metric_type, _ = _parse_search_params(request.search_params)
            expr = translate_filter_expr(request.dsl) if request.dsl else None
            results = col.search(
                query_vectors=queries,
                top_k=top_k,
                metric_type=metric_type,
                partition_names=list(request.partition_names) or None,
                expr=expr,
                output_fields=list(request.output_fields) or None,
            )
            return _build_search_results(results, col.schema, request.output_fields)
        except MilvusLiteError as e:
            return milvus_pb2.SearchResults(status=common_pb2.Status(**to_grpc_status(e)))

    # ── Index lifecycle ──────────────────────────────────────────

    def CreateIndex(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            params = _kv_pairs_to_dict(request.extra_params)
            index_params = {
                "index_type": params.get("index_type", "HNSW"),
                "metric_type": params.get("metric_type", "COSINE"),
                "params": json.loads(params.get("params", "{}")) if params.get("params") else {},
            }
            col.create_index(request.field_name, index_params)
            return common_pb2.Status(code=0)
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    def LoadCollection(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.load()
            return common_pb2.Status(code=0)
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    def ReleaseCollection(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.release()
            return common_pb2.Status(code=0)
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    # ── Catch-all UNIMPLEMENTED ──────────────────────────────────

    def _unimplemented(self, context, rpc_name: str, reason: str = ""):
        msg = f"MilvusLite does not support {rpc_name}"
        if reason:
            msg += f": {reason}"
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(msg)
        return common_pb2.Status(code=2, reason=msg)

    def HybridSearch(self, request, context):
        # Phase 12: Multi-route search + WeightedRanker/RRFRanker fusion
        # Parse each sub-SearchRequest → col.search() → reranker.rerank()
        # Implementation in servicer.py + reranker.py
        ...

    # ... (other UNIMPLEMENTED stubs)
```

---

## 8. Error Code Mapping

```python
# milvus_lite/adapter/grpc/errors.py

from milvus_lite.exceptions import (
    MilvusLiteError,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    PartitionNotFoundError,
    SchemaError,
    FilterParseError,
    FilterTypeError,
    FilterFieldError,
    CollectionNotLoadedError,
    IndexAlreadyExistsError,
    IndexNotFoundError,
    IndexBackendUnavailableError,
)

# Milvus standard ErrorCode
# 0  Success
# 1  UnexpectedError
# 4  CollectionNotExists
# 6  IllegalArgument
# 11 IndexNotExist
# 26 IndexBuildFailed
# 101 CollectionNotLoaded
# 1100 (newer) BadRequest
# ... (full list in milvus common.proto)

_EXCEPTION_TO_CODE = {
    CollectionNotFoundError:       (4,   "CollectionNotExists"),
    CollectionAlreadyExistsError:  (1,   "CollectionAlreadyExists"),
    PartitionNotFoundError:        (200, "PartitionNotExists"),
    SchemaError:                   (6,   "IllegalArgument"),
    FilterParseError:              (6,   "IllegalArgument"),
    FilterTypeError:               (6,   "IllegalArgument"),
    FilterFieldError:              (6,   "IllegalArgument"),
    CollectionNotLoadedError:      (101, "CollectionNotLoaded"),
    IndexAlreadyExistsError:       (35,  "IndexAlreadyExists"),
    IndexNotFoundError:            (11,  "IndexNotExist"),
    IndexBackendUnavailableError:  (26,  "IndexBuildFailed"),
}

def to_grpc_status(exc: MilvusLiteError) -> dict:
    code, _ = _EXCEPTION_TO_CODE.get(type(exc), (1, "UnexpectedError"))
    return {
        "code": code,
        "reason": str(exc),
    }
```

**Note**: Milvus's ErrorCode underwent significant changes between 2.3 and 2.4 (from numeric codes to string-based errors). Phase 10 MVP aligns with the 2.3-style numeric codes, which pymilvus clients of both versions can recognize.

---

## 9. server.py + CLI

```python
# milvus_lite/adapter/grpc/server.py

import grpc
from concurrent import futures
from .servicer import MilvusServicer
from .proto import milvus_pb2_grpc
from milvus_lite import MilvusLite

def run_server(
    data_dir: str,
    host: str = "0.0.0.0",
    port: int = 19530,
    max_workers: int = 10,
):
    db = MilvusLite(data_dir)
    servicer = MilvusServicer(db)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    milvus_pb2_grpc.add_MilvusServiceServicer_to_server(servicer, server)
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    print(f"MilvusLite gRPC server listening on {addr} (data_dir={data_dir})")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop(grace=5)
        db.close()
```

How to start:
```bash
milvus-lite server --data-dir ./data --port 19530
```

---

## 10. Phase 10 Sub-phase Breakdown

| Sub-phase | Content | Completion Criteria | Effort |
|---|---|---|---|
| **10.1** | Proto fetching + stub generation + empty servicer + `run_server` + CLI | `milvus-lite server --data-dir /tmp/x --port 19530` starts the server, pymilvus.connect() does not error; all RPCs return UNIMPLEMENTED | M |
| **10.2** | Collection lifecycle RPCs + `translators/schema.py` (minimal type set: INT64 / VARCHAR / FLOAT_VECTOR / BOOL / FLOAT / DOUBLE) | pymilvus runs `create / list / has / describe / drop` all passing; unsupported types raise UnsupportedFieldTypeError | M |
| **10.3** | insert/get/delete/query RPCs + `translators/records.py` bidirectional transposition + unit tests covering every supported type | pymilvus inserts 100 records → query returns equivalent data; delete by id works; delete by filter works | L |
| **10.4** | search + create_index + load + release RPCs + `translators/search.py` + `translators/result.py` + `translators/expr.py` + `translators/index.py` | pymilvus quickstart full flow passes: create_collection → insert → create_index(HNSW) → load → search(filter) → release → drop | L |
| **10.5** | Partition RPCs + flush + stats RPCs + `examples/m10_demo.py` + `tests/adapter/test_grpc_quickstart.py` as L3 smoke test | m10 demo passes; smoke test passes in CI | M |
| **10.6** | Error code mapping + exception wrapping middleware + UNIMPLEMENTED friendly messages | Every MilvusLiteError has a corresponding grpc status code test | S |

Total: 1S + 3M + 2L

---

## 11. Validation Strategy

### 11.1 Unit Tests

| Test File | Coverage |
|---|---|
| `tests/adapter/test_grpc_server_startup.py` | Server startup, shutdown, port binding |
| `tests/adapter/test_grpc_translators_schema.py` | Milvus FieldSchema ↔ MilvusLite FieldSchema bidirectional |
| `tests/adapter/test_grpc_translators_records.py` | FieldData ↔ records round-trip for every type |
| `tests/adapter/test_grpc_translators_expr.py` | Pass-through + UNIMPLEMENTED function detection |
| `tests/adapter/test_grpc_translators_index.py` | IndexParams ↔ IndexSpec |
| `tests/adapter/test_grpc_collection_lifecycle.py` | create / list / has / describe / drop |
| `tests/adapter/test_grpc_crud.py` | insert / upsert / delete / query / get |
| `tests/adapter/test_grpc_search.py` | search with all parameters (filter / top_k / output_fields / partition_names) |
| `tests/adapter/test_grpc_index.py` | create_index / load / release / drop_index |
| `tests/adapter/test_grpc_error_mapping.py` | Every exception → grpc status code |

### 11.2 Integration Test — pymilvus Smoke Test

`tests/adapter/test_grpc_quickstart.py`:

```python
import pytest
from pymilvus import MilvusClient

@pytest.fixture
def grpc_server(tmp_path):
    """Start MilvusLite gRPC server in a thread, yield port, stop after test."""
    from milvus_lite.adapter.grpc.server import run_server_in_thread
    port = _find_free_port()
    server, db = run_server_in_thread(str(tmp_path), port=port)
    yield port
    server.stop(grace=2)
    db.close()


def test_pymilvus_quickstart(grpc_server):
    client = MilvusClient(uri=f"http://localhost:{grpc_server}")

    # 1. Create
    client.create_collection("demo", dimension=4)
    assert client.has_collection("demo")
    assert "demo" in client.list_collections()

    # 2. Insert
    data = [{"id": i, "vector": [float(i)]*4} for i in range(100)]
    res = client.insert("demo", data=data)
    assert res["insert_count"] == 100

    # 3. Flush + Index
    client.flush("demo")
    client.create_index(
        "demo",
        index_params={
            "field_name": "vector",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )

    # 4. Load
    client.load_collection("demo")

    # 5. Search
    results = client.search(
        "demo",
        data=[[0.1, 0.2, 0.3, 0.4]],
        limit=10,
    )
    assert len(results[0]) == 10

    # 6. Query
    rows = client.query("demo", filter="id >= 50", limit=20)
    assert len(rows) == 20

    # 7. Delete
    client.delete("demo", ids=[1, 2, 3])

    # 8. Release + Drop
    client.release_collection("demo")
    client.drop_collection("demo")
    assert not client.has_collection("demo")
```

This is Phase 10's **completion criteria test** — it must pass.

### 11.3 Recall Consistency Test

`tests/adapter/test_grpc_search_parity.py`:

```python
def test_grpc_search_returns_same_topk_as_engine_directly(grpc_server, tmp_path):
    """Search via gRPC and via engine directly should return the same top-k."""
    db = MilvusLite(str(tmp_path))
    col = db.create_collection("test", schema=...)
    col.insert([...])
    col.create_index("vec", {"index_type": "HNSW", ...})
    col.load()

    # Direct engine
    direct_results = col.search([[...]], top_k=10)

    # Via gRPC
    client = MilvusClient(uri=f"http://localhost:{grpc_server}")
    grpc_results = client.search("test", data=[[...]], limit=10)

    # IDs should match exactly (HNSW recall@1 = 1.0 for small datasets)
    direct_ids = [r["id"] for r in direct_results[0]]
    grpc_ids = [r["id"] for r in grpc_results[0]]
    assert direct_ids == grpc_ids
```

---

## 12. Dependencies and Build

```toml
# pyproject.toml additions

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "grpcio-tools>=1.50",   # for regenerating proto stubs
]
faiss = ["faiss-cpu>=1.7.4"]
grpc = ["grpcio>=1.50", "protobuf>=4.21"]
all = ["milvus_lite[faiss,grpc]"]

[project.scripts]
milvus-lite = "milvus_lite.cmdline:main"
```

`pymilvus` is intentionally not a dependency of `milvus-lite`; pymilvus
installs Milvus Lite as its local backend. Compatibility tests import
`pymilvus` when it is available and skip otherwise.

`pip install -e ".[dev,faiss,grpc]"` installs the complete development environment.

---

## 13. Out of Phase 10 Scope

| Feature | Deferred To |
|---|---|
| TLS / mTLS encryption | Future |
| Token / Username-Password authentication | Future |
| RBAC / Multi-tenancy | Future (unnecessary for embedded mode) |
| Backup / Restore RPCs | Future |
| Bulk insert / Import | Future |
| Replica / Resource Group | Future (not needed for single process) |
| Hybrid search (multi-vector) | Future |
| Search iterator / pagination | Future (just need to add offset parameter to engine) |
| Database concept (multiple db instances) | Future |
| Async stream RPC (e.g., grpc client streaming) | Future |
| Sparse / Binary vector types | Future |

---

## 14. Completion Criteria

- `milvus-lite server --data-dir ./data --port 19530` can start the server
- pymilvus quickstart (the script in Section 11.2) runs through the entire flow from `connect → create → insert → create_index → load → search → query → delete → release → drop`
- Recall parity test passes: gRPC search and direct engine search return exactly the same top-k
- Error code mapping tests cover all MilvusLiteError subclasses
- `examples/m10_demo.py` is a 1:1 counterpart to the README quickstart
- Unsupported RPCs return `UNIMPLEMENTED` + friendly messages (no silent fail, no pretending to succeed)
- Running `pytest tests/adapter/` is all green
