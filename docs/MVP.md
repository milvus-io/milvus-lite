# MilvusLite - Local Vector Database MVP Design

## 1. Overview

A local vector database based on the LSM-Tree concept, using PyArrow for the in-memory layer and Parquet format for the persistence layer. Adopts the **DB → Collection → Partition** three-level data organization (aligned with Milvus), supports the Collection Schema model (typed fields, Schema versioning, dynamic fields) and Partition-level data isolation. Insert data and delete records are stored separately, supporting real-time CRUD and similarity search for vectors.

## 2. Core Design Principles

### On-Disk Files Are Strictly Immutable

This is the most fundamental constraint of the entire system:

- **Data Files (Parquet)**: Never modified after writing, only deleted entirely during Compaction
- **Delta Log (Parquet)**: Never modified after writing, deleted entirely after Compaction consumes them
- **WAL Files**: Append-only writing, deleted entirely after successful flush
- **Manifest**: The only state file updated through atomic replacement (write-tmp + replace)
- **No mutable auxiliary files** (no bitmaps, no sidecar files)

### Insert Data and Delete Records Are Separated

Following Milvus's design, Insert and Delete follow **two independent data flows**, landing in different files:

```
Insert("doc_1", vec)  →  Data File (Parquet)    # Contains vectors and metadata
Delete("doc_1")       →  Delta Log (Parquet)   # Only records (_id, _seq)
```

Benefits:
- Data files contain no deletion markers, allowing direct vector index construction on files in the future
- Delta Log is small in size (only id + seq), can be fully loaded into memory
- During search, the in-memory deleted set is used for filtering, no disk file modifications needed

### Flat File Organization, No Levels

The traditional LSM-Tree Level structure is designed to optimize point lookups (keys within L1+ layers don't overlap, allowing skipping of irrelevant files). However, the core operation of a vector database is **full scan**, where Levels provide no extra benefit but add complexity. Therefore:

- All data files reside in a single **flat directory**
- Point lookups scan from new to old using the seq range in filenames, combined with `deleted_map` for validity checking
- Compaction uses a **Size-Tiered** strategy (merging small files of similar sizes), not Leveled

### Trade-offs

- **Read amplification**: Search needs to read all data files, relying on Compaction to control file count
- **Space amplification**: Old versions of deleted/updated records still occupy space until Compaction
- **Write amplification**: Compaction rewrites data, but results in fewer files and better read performance
- **Benefits gained**: Simple implementation, crash-safe, concurrency-friendly, naturally supports future vector indexes

## 3. Overall Architecture

### 3.1 Data Hierarchy

```
DB ("my_app")                        ← Namespace, corresponds to a root directory
  ├── Collection ("documents")       ← Schema defined at this level, shared WAL / MemTable / _seq
  │     ├── Partition ("2024_Q1")    ← Data files isolated by Partition
  │     ├── Partition ("2024_Q2")
  │     └── Partition ("_default")   ← Default Partition, cannot be deleted
  └── Collection ("images")
        └── Partition ("_default")
```

- **DB**: Pure namespace, corresponds to a root directory on disk, no storage logic
- **Collection**: Owner of the Schema, has independent WAL, MemTable, Manifest, and `_seq` counter
- **Partition**: Data shard within a Collection, shares Schema, has independent data file and Delta Log directories

### 3.2 Component Architecture

```
                    ┌──────────────────────────┐
                    │       Client API          │
                    │  insert / delete / update │
                    │  search / get             │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │        DB Engine          │
                    │  (Manages multi DB /      │
                    │   Collection)             │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │    Collection Engine      │
                    │  (Global _seq allocation, │
                    │   scheduling)             │
                    └──┬──────────────────────┬┘
                       │                      │
            ┌──────────▼──────────┐  ┌────────▼─────────┐
            │    Write Path       │  │    Read Path      │
            │                     │  │                    │
            │ WAL → MemTable      │  │ MemTable + Data   │
            │ → Flush(by Partition)│  │ Files - Delta Set │
            └──────────┬──────────┘  └──────────────────┘
                       │
         ┌─────────────▼──────────────────────┐
         │     Storage Layer (per Collection)  │
         │                                     │
         │  ┌─────────┐  ┌──────────────────┐ │
         │  │  WAL    │  │  Partition "Q1"  │ │
         │  │(Arrow   │  │  ┌─────┐┌─────┐ │ │
         │  │  IPC)   │  │  │Data ││Delta│ │ │
         │  │(shared) │  │  │Files││Logs │ │ │
         │  └─────────┘  │  └─────┘└─────┘ │ │
         │               ├──────────────────┤ │
         │  ┌──────────┐ │  Partition "Q2"  │ │
         │  │ Manifest │ │  ┌─────┐┌─────┐ │ │
         │  │(global   │ │  │Data ││Delta│ │ │
         │  │ state)   │ │  │Files││Logs │ │ │
         │  └──────────┘ │  └─────┘└─────┘ │ │
         │               └──────────────────┘ │
         │  ┌──────────┐                      │
         │  │Compaction│                      │
         │  │ Manager  │                      │
         │  └──────────┘                      │
         └─────────────────────────────────────┘
```

## 4. Data Model

### 4.1 Collection Schema (Aligned with Milvus)

Following Milvus's Schema design, uses the **Collection Schema + Field Schema** model. Users define the Schema when creating a Collection, including primary key fields, vector fields, scalar fields, etc.

```python
class DataType(Enum):
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    DOUBLE = "double"
    VARCHAR = "varchar"
    JSON = "json"
    ARRAY = "array"               # Requires specifying element_type
    FLOAT_VECTOR = "float_vector"
    SPARSE_FLOAT_VECTOR = "sparse_float_vector"

class FieldSchema:
    name: str               # Field name
    dtype: DataType         # Field type
    is_primary: bool        # Whether it is the primary key (exactly one required)
    dim: Optional[int]      # Vector dimension (only required for vector fields)
    max_length: Optional[int]  # VARCHAR max length
    nullable: bool          # Whether null is allowed
    default_value: Any      # Default value
    element_type: Optional[DataType]  # ARRAY element type (only required for ARRAY fields)
    max_capacity: Optional[int]       # ARRAY max capacity

class CollectionSchema:
    fields: List[FieldSchema]
    version: int            # Schema version number (incremented on each change)
    enable_dynamic_field: bool  # Whether to enable dynamic fields ($meta JSON column)
    functions: List[Function]   # Schema-level functions (e.g., BM25)

class FunctionType(IntEnum):
    BM25 = 1               # Full-text search BM25 function

class Function:
    """Schema-level function definition for automatic field derivation (e.g., BM25 sparse vectors)."""
    name: str
    function_type: FunctionType
    input_field_names: List[str]    # Input fields (e.g., VARCHAR text column)
    output_field_names: List[str]   # Output fields (e.g., SPARSE_FLOAT_VECTOR column)
```

#### Schema Constraints

- Must have exactly one `is_primary=True` field, of type `VARCHAR` or `INT64`
- Must have at least one `FLOAT_VECTOR` field; `SPARSE_FLOAT_VECTOR` fields can be automatically derived by `Function` (e.g., BM25)
- Primary key field cannot be null
- Schema is persisted as `data_dir/schema.json`

#### DataType → Arrow Type Mapping

```python
TYPE_MAP = {
    DataType.BOOL:         pa.bool_(),
    DataType.INT8:         pa.int8(),
    DataType.INT16:        pa.int16(),
    DataType.INT32:        pa.int32(),
    DataType.INT64:        pa.int64(),
    DataType.FLOAT:        pa.float32(),
    DataType.DOUBLE:       pa.float64(),
    DataType.VARCHAR:      pa.string(),
    DataType.JSON:         pa.string(),       # JSON serialized as string
    DataType.ARRAY:        None,              # Resolved at runtime based on element_type
    DataType.FLOAT_VECTOR: lambda dim: pa.list_(pa.float32(), list_size=dim),
    DataType.SPARSE_FLOAT_VECTOR: pa.binary(),  # packed uint32+float32 pairs
}
```

### 4.2 Schema System (Automatically Generated from Collection Schema)

The system internally has **four Arrow Schema variants**, all automatically derived from the Collection Schema:

```python
def _build_user_fields(collection_schema: CollectionSchema) -> list:
    """Extract Arrow types for user-defined fields"""
    fields = []
    for f in collection_schema.fields:
        arrow_type = TYPE_MAP[f.dtype]
        if callable(arrow_type):
            arrow_type = arrow_type(f.dim)
        fields.append((f.name, arrow_type))
    if collection_schema.enable_dynamic_field:
        fields.append(("$meta", pa.string()))
    return fields

def build_data_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """Schema for data Parquet files (without _partition)"""
    fields = [("_seq", pa.uint64())]
    fields += _build_user_fields(collection_schema)
    return pa.schema(fields)

def build_delta_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """Schema for Delta Parquet files (without _partition)"""
    pk = get_primary_field(collection_schema)
    return pa.schema([(pk.name, TYPE_MAP[pk.dtype]), ("_seq", pa.uint64())])

def build_wal_data_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """Schema for WAL data files (data_schema plus a _partition column)"""
    fields = [("_seq", pa.uint64()), ("_partition", pa.string())]
    fields += _build_user_fields(collection_schema)
    return pa.schema(fields)

def build_wal_delta_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """Schema for WAL delete files (delta_schema plus a _partition column)"""
    pk = get_primary_field(collection_schema)
    return pa.schema([(pk.name, TYPE_MAP[pk.dtype]), ("_seq", pa.uint64()), ("_partition", pa.string())])
```

#### Relationship Between the Four Schemas

| Schema | Purpose | Contains `_partition` | File Format |
|--------|---------|:-:|------|
| `wal_data_schema` | WAL data file | **Yes** | Arrow IPC |
| `wal_delta_schema` | WAL delete file | **Yes** | Arrow IPC |
| `data_schema` | Data Parquet file | No | Parquet |
| `delta_schema` | Delta Parquet file | No | Parquet |

**Why does WAL need `_partition` while Parquet does not?**
- WAL is a single Collection-level shared file; during recovery, each record's Partition needs to be known
- Parquet files are already isolated by Partition directories; the file path itself indicates the belonging, no redundant column needed
```

Example (using the documents Collection):

```python
schema = CollectionSchema(
    fields=[
        FieldSchema("doc_id", DataType.VARCHAR, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128),
        FieldSchema("source", DataType.VARCHAR, nullable=True),
        FieldSchema("score", DataType.FLOAT, nullable=True),
    ],
    enable_dynamic_field=True,
)

# data_schema (Data Parquet)               wal_data_schema (WAL Data)
# ─────────────────────────                ─────────────────────────
# _seq:      uint64                        _seq:       uint64
#                                          _partition: string        ← WAL only
# doc_id:    string                        doc_id:     string
# embedding: list<f32, 128>                embedding:  list<f32, 128>
# source:    string                        source:     string
# score:     float32                       score:      float32
# $meta:     string                        $meta:      string

# delta_schema (Delta Parquet)             wal_delta_schema (WAL Delete)
# ─────────────────────────                ─────────────────────────
# doc_id:    string                        doc_id:     string
# _seq:      uint64                        _seq:       uint64
#                                          _partition: string        ← WAL only
```

Notes:
- `_seq` is an internal system field, not exposed in the user Schema
- `_partition` only exists in WAL (Parquet uses directory isolation for Partitions)
- The primary key field is user-defined (replacing the former hard-coded `_id`)
- Scalar fields have explicit types, enabling Parquet predicate pushdown

### 4.3 Schema Versioning

Schema changes (e.g., adding fields) leverage **Parquet's native schema evolution**:

- Adding fields: New files are written with the new Schema; when reading old files, missing columns are automatically filled with null
- Deleting fields: Not supported (additions only)
- Modifying field types: Not supported

```json
// data_dir/schema.json (name comes from the create_collection parameter, persisted to schema.json for self-description)
{
    "collection_name": "documents",
    "version": 2,
    "fields": [
        {"name": "doc_id", "dtype": "varchar", "is_primary": true},
        {"name": "embedding", "dtype": "float_vector", "dim": 128},
        {"name": "source", "dtype": "varchar", "nullable": true},
        {"name": "score", "dtype": "float", "nullable": true},
        {"name": "category", "dtype": "varchar", "nullable": true}  // ← Added in v2
    ],
    "enable_dynamic_field": true
}
```

### 4.4 Dynamic Fields

When `enable_dynamic_field=True`, data files include an additional `$meta` column (JSON string) to store fields not defined in the Schema:

```python
# During insertion, fields outside the Schema are automatically placed into $meta
db.insert(doc_id="doc_1", embedding=[...], source="wiki",
          category="science", tags=["ml", "ai"])
# → source goes into the source column (within Schema)
# → category, tags go into $meta: '{"category": "science", "tags": ["ml", "ai"]}'
```

### 4.5 Delta Log Schema (Stores Delete Records, Unchanged)

| Field       | Type     | Description                         |
| ---------- | -------- | ---------------------------- |
| `{pk_name}`| PK type | Primary key of the deleted record              |
| `_seq`     | `uint64` | Sequence number of the delete operation, used for comparison with data versions |

```python
def build_delta_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """Generate the Arrow Schema for Delta Log from the Collection Schema"""
    pk = get_primary_field(collection_schema)
    pk_type = TYPE_MAP[pk.dtype]
    return pa.schema([
        (pk.name, pk_type),
        ("_seq", pa.uint64()),
    ])
```

### 4.6 Version Resolution Rules (Unchanged)

For the same primary key, both data records and delete records may exist simultaneously. Resolution logic:

```
data_seq  = Maximum _seq for that primary key in data files
delta_seq = Maximum _seq for that primary key in delta log

if delta_seq > data_seq  → The record has been deleted
if data_seq > delta_seq  → The record is valid (re-inserted after deletion)
if only data_seq         → The record is valid
if only delta_seq        → The record has been deleted (can be ignored)
```

## 5. Core Component Design

### 5.1 WAL (Write-Ahead Log)

Uses the **Arrow IPC Streaming** format, writing binary directly to disk, avoiding the 3x write amplification of JSONL text encoding for vector data.

Inserts and deletes use **two independent WAL files**, corresponding one-to-one with MemTable's dual buffers:

- **`wal_data_{N}.arrow`**: Records insert/update operations, uses `wal_data_schema` (`data_schema` plus a `_partition` column)
- **`wal_delta_{N}.arrow`**: Records delete operations, uses `wal_delta_schema` (`delta_schema` plus a `_partition` column)

```python
class WAL:
    def __init__(self, wal_dir, wal_data_schema, wal_delta_schema):
        self.data_writer = None   # pa.ipc.RecordBatchStreamWriter
        self.delta_writer = None  # pa.ipc.RecordBatchStreamWriter

    def write_insert(self, record_batch: pa.RecordBatch):
        """Append to wal_data file (RecordBatch contains _partition column)"""
        if self.data_writer is None:
            self.data_writer = pa.ipc.new_stream(data_path, self.wal_data_schema)
        self.data_writer.write_batch(record_batch)

    def write_delete(self, record_batch: pa.RecordBatch):
        """Append to wal_delta file (RecordBatch contains _partition column)"""
        if self.delta_writer is None:
            self.delta_writer = pa.ipc.new_stream(delta_path, self.wal_delta_schema)
        self.delta_writer.write_batch(record_batch)

    def close_and_delete(self):
        """Close and delete both WAL files after successful flush"""

    def recover(self) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        """Read uncleaned WAL files at startup, return (data_batches, delta_batches)
        Each RecordBatch contains _partition column, routed to MemTable by _partition during recovery"""
```

- **Lifecycle**: After MemTable successfully flushes, the corresponding two WAL files are deleted entirely
- **Recovery**: At startup, reads uncleaned WAL files to rebuild MemTable's two buffers

```
data_dir/
  wal/
    wal_data_000001.arrow
    wal_delta_000001.arrow
```

### 5.2 MemTable

MemTable is **Collection-level shared** (consistent with Milvus: WAL/MemTable are at Collection level, not split by Partition), internally maintaining two independent buffers with field structure driven by the Collection Schema. Each record carries a `_partition` marker, and during flush, output is split by Partition.

**Upsert semantics**: `insert_buf` uses dict[pk → record], where the same PK directly overwrites — if the PK exists, it updates; if not, it inserts. The internal engine only provides a single `insert()` write method, naturally providing upsert semantics. PK uniqueness is guaranteed at the Collection level.

```python
class MemTable:
    def __init__(self, schema: CollectionSchema):
        self.schema = schema
        self.pk_name = get_primary_field(schema).name
        self.insert_buf = {}   # pk_value -> {_partition: str, field_name: value, ...}
        self.delete_buf = {}   # pk_value -> (_seq, _partition)
        self.lock = threading.Lock()

    def put(self, _seq: int, _partition: str, **fields):
        """Write an insert record to insert_buf"""
        pk = fields[self.pk_name]
        record = {"_seq": _seq, "_partition": _partition, **fields}
        self.insert_buf[pk] = record
        self.delete_buf.pop(pk, None)

    def delete(self, pk_value, _seq: int, _partition: str):
        """Write a delete record to delete_buf"""
        self.delete_buf[pk_value] = (_seq, _partition)
        self.insert_buf.pop(pk_value, None)

    def get(self, pk_value) -> Optional[dict]:
        """Point lookup: check delete_buf first, then insert_buf"""
        if pk_value in self.delete_buf:
            del_seq, _ = self.delete_buf[pk_value]
            if pk_value in self.insert_buf and self.insert_buf[pk_value]["_seq"] > del_seq:
                return self.insert_buf[pk_value]
            return None
        return self.insert_buf.get(pk_value)

    def flush(self) -> Dict[str, Tuple[Optional[pa.Table], Optional[pa.Table]]]:
        """Split output by Partition {partition_name: (data_table, delta_table)}"""
        result = {}
        # Group insert_buf by _partition
        for pk, record in self.insert_buf.items():
            part = record["_partition"]
            result.setdefault(part, ([], []))
            result[part][0].append(record)
        # Group delete_buf by _partition
        for pk, (seq, part) in self.delete_buf.items():
            result.setdefault(part, ([], []))
            result[part][1].append({self.pk_name: pk, "_seq": seq})
        # Convert to Arrow Table
        return {
            part: (
                build_arrow_table(inserts, self.schema) if inserts else None,
                build_delta_table(deletes, self.schema) if deletes else None,
            )
            for part, (inserts, deletes) in result.items()
        }

    def size(self) -> int:
        return len(self.insert_buf) + len(self.delete_buf)
```

- **Size limit**: Default `MEMTABLE_SIZE_LIMIT = 10000` records (insert + delete combined, across Partitions)
- **Flush trigger**: When the threshold is reached, the current MemTable is frozen and a new MemTable is created to accept writes
- **Flush output**: Split by Partition, each Partition gets independent data Parquet + delta Parquet
- **Dynamic field handling**: During `put()`, fields outside the Schema are serialized as JSON into `$meta`
- **`_partition` is not written to Parquet**: `_partition` is only used for routing during flush; the Parquet file itself does not contain this column (Partition is indicated by directory isolation)

### 5.3 Data Files (Parquet)

Stores inserted/updated vector records, **contains no deletion markers**.

- **File organization**: Flat directory, all data files at the same level
- **File naming**: `data_{seq_min}_{seq_max}.parquet`
- **Sorting**: Within each file, sorted by `_id` for easy merging
- **Immutable**: Never modified after writing, the entire file is deleted during Compaction
- **Self-describing**: Leverages Parquet's built-in column statistics (min/max) to obtain id range and seq range

```
data_dir/
  data/
    data_000001_000500.parquet
    data_000501_001000.parquet
    data_000001_002000.parquet    ← Compaction merge output
```

### 5.4 Delta Log (Parquet)

Stores delete operation records, completely separated from data files.

- **File organization**: Flat directory, a new file is generated with each flush
- **File naming**: `delta_{seq_min}_{seq_max}.parquet`
- **Content**: Only two columns `(_id, _seq)`, much smaller than data files
- **Immutable**: Never modified after writing, the entire file is deleted after Compaction consumes it
- **In-memory cache**: At startup, all delta logs are fully loaded into memory, building `deleted_map: dict[str, int]` (_id → max delete _seq)

```
data_dir/
  deltas/
    delta_000501_000503.parquet
    delta_001001_001002.parquet
```

```python
class DeltaLog:
    def __init__(self, data_dir, pk_name: str):
        self.data_dir = data_dir
        self.pk_name = pk_name
        self.deleted_map = {}  # pk_value -> max delete _seq (memory-resident)

    def load_all(self):
        """Rebuild deleted_map from all delta parquet files at startup"""
        for f in glob(deltas/*.parquet):
            table = pq.read_table(f)
            for pk, _seq in zip(table[self.pk_name], table["_seq"]):
                cur = self.deleted_map.get(pk.as_py(), 0)
                self.deleted_map[pk.as_py()] = max(cur, _seq.as_py())

    def add(self, delta_table: pa.Table):
        """Write a new delta file during flush and update memory"""
        pq.write_table(delta_table, path)
        # Also update deleted_map

    def is_deleted(self, pk_value, data_seq: int) -> bool:
        """Determine whether a data record has been deleted"""
        del_seq = self.deleted_map.get(pk_value, 0)
        return del_seq > data_seq

    def remove_files(self, files: List[str]):
        """Delete consumed delta files after Compaction, and clean up corresponding entries in deleted_map"""
```

### 5.5 Manifest (Inspired by LanceDB)

The Manifest is the database's **global state snapshot file**, recording which files currently exist, the current _seq, and other key state. Inspired by LanceDB's Manifest design, it solves three problems:

1. **Fast startup**: No need to scan directories, directly obtain file lists and _seq from manifest
2. **Atomic state changes**: After Flush / Compaction, consistency is guaranteed through atomic manifest replacement
3. **Snapshot foundation**: A snapshot = a copy of the manifest at a point in time

#### Manifest Content

The Manifest organizes file lists by Partition; WAL and `_seq` are shared at the Collection level:

```json
// data_dir/manifest.json
{
    "version": 42,
    "current_seq": 15023,
    "schema_version": 2,
    "partitions": {
        "_default": {
            "data_files": ["_default/data_000001_005000.parquet"],
            "delta_files": ["_default/delta_005001_005003.parquet"]
        },
        "2024_Q1": {
            "data_files": [
                "2024_Q1/data_000001_003000.parquet",
                "2024_Q1/data_003001_008000.parquet"
            ],
            "delta_files": ["2024_Q1/delta_008001_008002.parquet"]
        },
        "2024_Q2": {
            "data_files": ["2024_Q2/data_005001_010000.parquet"],
            "delta_files": []
        }
    },
    "active_wal": {
        "data": "wal_data_000003.arrow",
        "delta": "wal_delta_000003.arrow"
    }
}
```

| Field | Description |
|------|------|
| `version` | Manifest version number, incremented on each update |
| `current_seq` | Current maximum _seq, restores counter at startup (Collection level) |
| `schema_version` | Current Collection Schema version number |
| `partitions` | File lists organized by Partition name, each Partition has independent `data_files` and `delta_files` |
| `active_wal` | Path to the currently active WAL files (Collection-level shared) |

#### Atomic Update

Atomicity is guaranteed through **write-tmp + replace**:

```python
class Manifest:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "manifest.json")
        self.version = 0
        self.current_seq = 0
        self.schema_version = 1
        self.partitions = {"_default": {"data_files": [], "delta_files": []}}
        self.active_wal = {"data": None, "delta": None}

    def save(self):
        """Atomic update: write temporary file → replace"""
        self.version += 1
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)  # Atomic overwrite

    @classmethod
    def load(cls, data_dir: str) -> "Manifest":
        """Load manifest at startup"""
        path = os.path.join(data_dir, "manifest.json")
        if os.path.exists(path):
            return cls.from_dict(json.load(open(path)))
        return cls(data_dir)  # First-time creation
```

`os.replace` atomically overwrites the destination on supported local file systems, including when the destination already exists on Windows.

#### Update Timing

| Event | Manifest Changes |
|------|--------------|
| **Flush** | New files added to the corresponding Partition's `data_files` / `delta_files`, `current_seq` updated, `active_wal` switched |
| **Compaction** | Old files removed + new files added in the corresponding Partition's `data_files`, consumed files removed from `delta_files` |
| **Create Partition** | New key added to `partitions`, initialized with empty file lists |
| **Drop Partition** | Key removed from `partitions` (files subsequently deleted asynchronously) |
| **Schema Change** | `schema_version` +1 |
| **WAL Rotation** | `active_wal` updated |

#### Crash Recovery

Recovery logic at startup:

```
1. Read manifest.json
   ├─ Exists → Obtain file lists and current_seq
   └─ Does not exist → Fallback to directory scanning (first startup or manifest corruption)
2. Check if wal/ directory has uncleaned WAL files
   ├─ None → Normal startup
   └─ Has → Replay WAL to MemTable (state in manifest is the consistent snapshot before WAL)
3. Verify that files in manifest actually exist (guard against mid-compaction crash)
4. Rebuild deleted_map from delta_files
5. Recovery complete, save new manifest
```

Key point: The manifest is always updated **before** the WAL is deleted. If flush completes (data written + manifest updated) but the WAL is not deleted, replaying the WAL will produce duplicate data, but due to `_seq` deduplication, correctness is not affected.

### 5.6 Compaction Manager

Uses the **Size-Tiered Compaction** strategy: no Levels, files are grouped and merged by size. **Compaction is executed independently per Partition** — files in different Partitions do not interfere with each other.

#### Trigger Conditions

| Condition | Threshold | Description |
| ---- | ---- | ---- |
| Too many small files | >= 4 files of the same order of magnitude | Merge into one larger file |
| Too many total files | Total file count > `MAX_DATA_FILES` (default 32) | Select the smallest files to merge |

#### File Size Grouping

Files are bucketed by size (logarithmic scale):
- Bucket 0: < 1MB
- Bucket 1: 1MB ~ 10MB
- Bucket 2: 10MB ~ 100MB
- ...

When the number of files in a bucket reaches >= 4, a merge is triggered.

#### Compaction Process

1. For a given Partition, select multiple files from the same size bucket
2. Read and merge Arrow Tables
3. Deduplicate by primary key (keep the version with the largest `_seq`)
4. Filter deleted records using `deleted_map`
5. Write a new Parquet file (to that Partition's data/ directory)
6. **Atomically update Manifest** (within that Partition: remove old files + add new files + remove consumed delta files)
7. Delete old data files and consumed delta log files
8. In the MVP phase, executed synchronously on the main thread, checking each Partition sequentially

Note the order of steps 6-7: Manifest is updated before old files are deleted. If a crash occurs before old files are deleted, the Manifest at startup already points to new files, and orphan old files can be safely cleaned up.

```python
class CompactionManager:
    def __init__(self, data_dir, delta_log): ...

    def maybe_compact(self):
        # 1. Bucket data files by size
        # 2. Find buckets with >= 4 files
        # 3. Merge files in that bucket
        #    a. Read all Arrow Tables
        #    b. Deduplicate by _id (keep max _seq)
        #    c. Filter using delta_log.is_deleted()
        #    d. Write new file, delete old files
        # 4. Clean up reclaimable delta log files

    def merge_tables(self, tables: List[pa.Table]) -> pa.Table: ...
```

### 5.7 Vector Search

Search uses a **build bitmap first, then execute vector retrieval** pipeline. The bitmap uniformly handles delete filtering (MVP) and scalar filtering (future), and vector retrieval only looks at valid rows in the bitmap.

#### Search Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1. Collect    │ ──→ │ 2. Build     │ ──→ │ 3. Vector    │
│ data         │     │ bitmap       │     │ retrieval    │
│ MemTable +   │     │ Dedup +      │     │ Only compute │
│ all data     │     │ delete       │     │ valid rows   │
│ files        │     │ (+ scalar    │     │ Return top_k │
│              │     │   filter)    │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

#### Bitmap Construction Rules

```python
# valid_mask[i] = True means row i participates in vector retrieval
valid_mask = np.ones(n, dtype=bool)

# 1. Dedup: for the same primary key, keep only the row with max _seq, mark others as False
for duplicate_rows in group_by_pk(records):
    keep only max _seq, mark others False

# 2. Delete filtering: check deleted_map
for i, (pk, _seq) in enumerate(records):
    if delta_log.is_deleted(pk, _seq):
        valid_mask[i] = False

# 3. Scalar filtering (future):
# if filter_expr:
#     scalar_mask = evaluate_filter(typed_columns, filter_expr)  # Parquet predicate pushdown
#     valid_mask &= scalar_mask
```

#### MVP Implementation

The MVP phase uses NumPy brute-force scan + bitmap mask:

```python
def search(self, vectors: List[list], top_k: int = 10,
           metric_type: str = "COSINE",
           partition_names: List[str] = None) -> List[List[dict]]:
    # 1. Collect data: MemTable active records + data Parquet files of target Partitions
    all_pks, all_seqs, all_vectors = collect_all(self.schema, partition_names)

    # 2. Build bitmap
    valid_mask = build_valid_mask(all_pks, all_seqs, self.delta_log)

    # 3. Vector retrieval (only compute distances for valid rows)
    results = []
    for query_vector in vectors:
        valid_vectors = all_vectors[valid_mask]
        distances = compute_distances(query_vector, valid_vectors, metric_type)
        top_indices = np.argpartition(distances, top_k)[:top_k]
        results.append(build_results(top_indices, ...))
    return results  # List[List[{"id": pk, "distance": float, "entity": {...}}]]
```

#### Phase 9: FAISS HNSW Integrated (Per-Segment Index)

Phase 9 upgrades the retrieval path from "NumPy brute-force scan" to "FAISS HNSW per-segment". The same bitmap pipeline is reused — `valid_mask` is directly fed to FAISS's `IDSelectorBitmap` for pre-filtering:

```python
# Phase 9 actual implementation (milvus_lite/index/faiss_hnsw.py)
import faiss, numpy as np
mask_packed = np.packbits(valid_mask, bitorder='little')
sel = faiss.IDSelectorBitmap(num_vectors, faiss.swig_ptr(mask_packed))
params = faiss.SearchParametersHNSW(sel=sel, efSearch=ef)
distances, ids = index.search(query, top_k, params=params)
```

**Architecture decisions** (see `plan/index-design.md` for details):
- Index is bound at the **segment level** (one .idx file per data parquet, 1:1 binding)
- Index library choice is **FAISS-cpu** (IDSelectorBitmap is isomorphic with the bitmap pipeline; index family aligns with Milvus)
- **BruteForceIndex is permanently retained** as a differential baseline + fallback when faiss is not installed
- **load/release state machine** aligned with Milvus behavior — defaults to released after restart, explicit load required

No changes to the storage layer — this is exactly the benefit that the LSM immutable architecture + bitmap pipeline reserved since Phase 0.

- Supported distance metrics: `cosine` (default), `l2`, `ip` (inner product)

## 6. Read/Write Paths

### 6.1 Write Path (Insert)

The internal engine has only one write method `insert()`, which naturally provides upsert semantics (overwrites if PK exists). Input is always `List[dict]`, and parameter normalization is handled by the upper layer (gRPC adapter layer).

```
Collection.insert(records=[{"doc_id": "doc_1", ...}], partition_name="_default")
  │
  ├─ 1. Resolve target Partition (if partition_name not specified, use "_default")
  ├─ 2. Schema validation (field types, primary key non-null, vector dimensions)
  ├─ 3. Separate Schema fields vs dynamic fields (→ $meta JSON)
  ├─ 4. Allocate globally incrementing _seq (Collection level, independent _seq per record)
  ├─ 5. Write to WAL (wal_data): Arrow IPC RecordBatch (with _partition marker)
  ├─ 6. Write to MemTable.insert_buf (with _partition, same PK directly overwrites → upsert semantics)
  ├─ 7. Check MemTable size
  │     ├─ Not full → Return
  │     └─ Full → Freeze current MemTable
  │               ├─ Create new MemTable + new WAL
  │               ├─ Flush split by Partition:
  │               │   ├─ Partition A: insert_buf → data Parquet, delete_buf → delta Parquet
  │               │   └─ Partition B: insert_buf → data Parquet, delete_buf → delta Parquet
  │               ├─ Update in-memory deleted_map
  │               ├─ Atomically update Manifest (new files per Partition + update current_seq + switch active_wal)
  │               ├─ Delete old WAL (wal_data + wal_delta)
  │               └─ Trigger Compaction check (independent per Partition)
  └─ Return list of written PKs
```

- **Upsert semantics**: When the same primary key is inserted again, a higher `_seq` is assigned, directly overwriting the old version in MemTable

### 6.2 Delete Path (Delete)

Input is always `List[pk]`, where multiple PKs share the same `_seq`. Parameter normalization (single value → list) is handled by the upper layer.

```
Collection.delete(pks=["doc_1"])                                   # partition_name=None → Delete across all Partitions
Collection.delete(pks=["doc_1", "doc_2"], partition_name="2024_Q1") # Specified partition → Delete only in that Partition
  │
  ├─ 1. Resolve target Partition:
  │     ├─ partition_name specified → Use that Partition
  │     └─ Not specified (None) → Not bound to Partition (delete across all Partitions)
  ├─ 2. Allocate globally incrementing _seq (the entire batch shares one _seq)
  ├─ 3. Write to WAL (wal_delta): Arrow IPC RecordBatch (with _partition marker)
  ├─ 4. Write to MemTable.delete_buf (with _partition)
  ├─ 5. Same flush logic (flushed together when threshold is reached, split by Partition)
  └─ Return count of processed PKs
```

Note: Delete does not need to confirm whether the record exists. Just write it; if the primary key does not exist, the entry in the delta log will be naturally cleaned up during compaction.

### 6.3 Read Path (Get by PK)

Input is always `List[pk]`.

```
Collection.get(pks=["doc_1"], partition_names=None)
Collection.get(pks=["doc_1", "doc_2"], output_fields=["doc_id", "source"], partition_names=["2024_Q1"])
  │
  ├─ 1. For each PK:
  │     ├─ Check MemTable → insert/delete conflicts already handled internally, add to results if hit
  │     ├─ Check frozen MemTable (if any)
  │     ├─ Determine search scope:
  │     │     ├─ partition_names specified → Only scan files of specified Partitions
  │     │     └─ Not specified → Scan files of all Partitions
  │     ├─ Scan data files (get file list from Manifest, from newest to oldest by seq)
  │     │     After finding data record → Check deleted_map:
  │     │       delta_seq > data_seq → Deleted, skip
  │     │       Otherwise → Add to results (trim returned fields by output_fields)
  │     └─ Not found → That PK does not appear in results
  └─ Return List[dict] (each dict is one record, unfound PKs are not in the list)
```

Since deleted_map is memory-resident, checking whether a record is deleted is just one dict lookup, O(1). Specifying partition_names reduces the file scan scope.

### 6.4 Vector Search Path (Search)

```
Collection.search(vectors=[[0.1, 0.2, ...]], top_k=10, metric_type="COSINE", partition_names=None)
  │
  │  ── Phase 0: Partition Pruning ──
  ├─ 0. Determine search scope:
  │     ├─ partition_names specified → Only search files of specified Partitions (Partition Pruning)
  │     └─ Not specified → Search all Partitions
  │
  │  ── Phase 1: Collect Data ──
  ├─ 1. Collect active records from MemTable (filtered by target Partitions)
  ├─ 2. Get file list of target Partitions from Manifest, read data Parquet files
  │
  │  ── Phase 2: Build Bitmap ──
  ├─ 3. Deduplicate by primary key (keep max _seq), mark duplicate rows as invalid
  ├─ 4. Mark deleted rows as invalid using deleted_map
  ├─ 5. (Future) Scalar filtering (extension point reserved in bitmap pipeline)
  │
  │  ── Phase 3: Vector Retrieval ──
  ├─ 6. Only compute distances for valid rows in the bitmap
  └─ 7. Return List[List[dict]] (outer = each query vector, inner = top-K results)
       Each result: {"id": pk_value, "distance": float, "entity": {field: value, ...}}
```

## 7. Disk File Structure

```
root_dir/                                   # DB root directory
  my_app/                                   # DB name
    documents/                              # Collection name
      manifest.json                         # Global state snapshot (updated by atomic replacement)
      schema.json                           # Collection Schema definition (with version number)
      wal/                                  # WAL (Collection-level shared)
        wal_data_000001.arrow               #   Data WAL (Arrow IPC), deleted entirely after flush
        wal_delta_000001.arrow              #   Delete WAL (Arrow IPC), deleted entirely after flush
      _default/                             # Partition: _default (default, cannot be deleted)
        data/
          data_000001_000500.parquet        #   Data file, written once and immutable
          data_000501_001000.parquet
        deltas/
          delta_000501_000503.parquet       #   Delta Log, written once and immutable
      2024_Q1/                              # Partition: 2024_Q1
        data/
          data_000001_003000.parquet
          data_003001_008000.parquet
        deltas/
          delta_008001_008002.parquet
      2024_Q2/                              # Partition: 2024_Q2
        data/
          data_005001_010000.parquet
        deltas/
          (empty)
    images/                                 # Another Collection
      manifest.json
      schema.json
      wal/
      _default/
        data/
        deltas/
```

**Hierarchy rules**:
- **DB** → directory, pure namespace
- **Collection** → directory, contains `manifest.json` + `schema.json` + `wal/` + Partition subdirectories
- **Partition** → directory, contains `data/` + `deltas/`, Drop Partition = delete entire subdirectory + update Manifest
- **WAL** → Collection-level shared, not split by Partition

Data files and Delta Logs have a lifecycle of: **create → immutable → delete entirely**, no exceptions.
`manifest.json` and `schema.json` are updated through **write-tmp + replace** atomic replacement.

## 8. Code Directory Structure

```
lite-v2/
├── MVP.md
├── milvus_lite/
│   ├── __init__.py
│   ├── db.py               # DB layer: manages lifecycle of multiple Collections
│   ├── collection.py       # Collection layer: _seq allocation, WAL/MemTable scheduling, Partition management
│   ├── schema.py           # CollectionSchema / FieldSchema / DataType
│   │                       # Schema validation, Arrow Schema generation, schema.json read/write
│   ├── manifest.py         # Manifest management (loading, atomic save, Partition-level file lists)
│   ├── memtable.py         # MemTable (insert_buf + delete_buf, with _partition routing)
│   ├── wal.py              # WAL implementation (Arrow IPC Streaming, dual files, Collection-level shared)
│   ├── sstable.py          # Data file (Parquet) read/write
│   ├── delta_log.py        # Delta Log management (read/write + in-memory deleted_map)
│   ├── compaction.py       # Compaction Manager (Size-Tiered, executed independently per Partition)
│   └── search.py           # Vector search (brute-force scan + bitmap pipeline + Partition Pruning)
├── tests/
│   ├── test_schema.py      # Schema definition, validation, Arrow conversion
│   ├── test_manifest.py    # Manifest loading, saving, Partition-level file management
│   ├── test_memtable.py
│   ├── test_wal.py
│   ├── test_sstable.py
│   ├── test_delta_log.py
│   ├── test_compaction.py
│   ├── test_search.py
│   ├── test_collection.py  # Collection-level end-to-end tests
│   └── test_db.py          # Multi-DB / multi-Collection tests
├── pyproject.toml
└── requirements.txt
```

## 9. Internal Engine API

The internal engine API is implementation-oriented, with inputs already normalized (always List). A gRPC adapter layer will be added on top of the engine in the future, responsible for Milvus protocol compatibility (parameter normalization, expression parsing, return value wrapping), allowing pymilvus to connect directly.

### 9.1 API Overview

```python
class MilvusLite:
    """DB layer: manages lifecycle of multiple Collections"""

    def __init__(self, root_dir: str): ...
    def create_collection(self, collection_name: str, schema: CollectionSchema) -> Collection: ...
    def get_collection(self, collection_name: str) -> Collection: ...
    def drop_collection(self, collection_name: str): ...
    def list_collections(self) -> List[str]: ...
    def close(self): ...


class Collection:
    """Collection layer: engine core, manages WAL / MemTable / Manifest / Compaction"""

    # ─── Write operations (partition_name: singular str, writes to one Partition) ───
    def insert(self, records: List[dict], partition_name: str = "_default") -> List:
        """Batch write. Overwrites if PK already exists (upsert semantics). Returns list of written PKs."""

    def delete(self, pks: List, partition_name: str = None) -> int:
        """Batch delete. partition_name=None means across all Partitions. Returns count of processed PKs.
        Multiple PKs share the same _seq."""

    # ─── Read operations (partition_names: plural List[str], can span multiple Partitions) ───
    def get(self, pks: List, output_fields: List[str] = None,
            partition_names: List[str] = None) -> List[dict]:
        """Batch query by PK. Unfound PKs are not in the returned list."""

    def search(self, vectors: List[list], top_k: int = 10,
               metric_type: str = "COSINE",
               partition_names: List[str] = None) -> List[List[dict]]:
        """Vector search. Returns outer = each query vector, inner = top-K results.
        Each result: {"id": pk, "distance": float, "entity": {field: value}}"""

    # ─── Index lifecycle (Phase 9) ───
    def create_index(self, field_name: str, index_params: dict) -> None: ...
    def drop_index(self, field_name: str) -> None: ...
    def has_index(self) -> bool: ...
    def get_index_info(self) -> Optional[dict]: ...

    # ─── Load / Release state machine (Phase 9) ───
    # released → loading → loaded; search/get/query require loaded state
    # Defaults to released after restart, must explicitly load()
    def load(self) -> None: ...
    def release(self) -> None: ...
    @property
    def load_state(self) -> str: ...  # "released" | "loading" | "loaded"

    # ─── Partition management ───
    def create_partition(self, partition_name: str): ...
    def drop_partition(self, partition_name: str): ...
    def list_partitions(self) -> List[str]: ...

    # ─── Schema changes ───
    def add_field(self, field: FieldSchema): ...
```

### 9.2 Usage Examples

```python
from milvus_lite import MilvusLite, CollectionSchema, FieldSchema, DataType

# ═══ DB ═══
db = MilvusLite(root_dir="./my_data")

# ═══ Collection ═══
schema = CollectionSchema(
    fields=[
        FieldSchema("doc_id", DataType.VARCHAR, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128),
        FieldSchema("source", DataType.VARCHAR, nullable=True),
        FieldSchema("score", DataType.FLOAT, nullable=True),
    ],
    enable_dynamic_field=True,
)
col = db.create_collection(collection_name="documents", schema=schema)

# ═══ Partition ═══
col.create_partition("2024_Q1")

# ═══ Insert (input is always List[dict]) ═══
col.insert(
    records=[{"doc_id": "doc_1", "embedding": [0.1, 0.2, ...], "source": "wiki"}],
)
col.insert(
    records=[
        {"doc_id": "doc_3", "embedding": [...], "source": "arxiv", "score": 0.95},
        {"doc_id": "doc_4", "embedding": [...], "source": "web"},
    ],
    partition_name="2024_Q1",
)
# Dynamic fields (fields outside the Schema are automatically stored in $meta)
col.insert(records=[{"doc_id": "doc_5", "embedding": [...], "source": "wiki",
                     "category": "science", "tags": ["ml", "ai"]}])
# Upsert: inserting the same PK again overwrites it
col.insert(records=[{"doc_id": "doc_1", "embedding": [0.3, 0.4, ...], "source": "updated"}])

# ═══ Delete (input is always List[pk]) ═══
col.delete(pks=["doc_1"])                                    # partition=None → Across all Partitions
col.delete(pks=["doc_2", "doc_3"], partition_name="2024_Q1") # Specified Partition

# ═══ Get ═══
records = col.get(pks=["doc_1"])
records = col.get(pks=["doc_1", "doc_2"], output_fields=["doc_id", "source"],
                  partition_names=["2024_Q1"])

# ═══ Search ═══
results = col.search(vectors=[[0.1, 0.2, ...]], top_k=5, metric_type="COSINE")
results = col.search(vectors=[[0.1, 0.2, ...]], top_k=5, metric_type="L2",
                     partition_names=["2024_Q1"])

# ═══ Close ═══
db.close()
```

### 9.3 Write Operations vs Read Operations Partition Parameter Convention

| Operation Type | Parameter Name | Type | Semantics |
|---------|--------|------|------|
| **Write operations** (insert/delete) | `partition_name` | `Optional[str]` | Target Partition (singular, writes to one Partition) |
| **Read operations** (get/search) | `partition_names` | `Optional[List[str]]` | Search scope (plural, can span multiple Partitions) |

### 9.4 gRPC Adapter Layer (Phase 10 Landed)

Phase 10 builds a gRPC service layer on top of the internal engine, allowing pymilvus clients to connect without code changes:

```
pymilvus ──gRPC──→ [ milvus_lite/adapter/grpc/ ] ──→ [ Internal Engine ]

Adapter layer responsibilities (see plan/grpc-adapter-design.md for details):
├─ Milvus Insert/Upsert RPC  →  engine.insert(records, partition_name)
├─ Milvus Delete(ids=) RPC   →  engine.delete(pks, partition_name)
├─ Milvus Delete(filter=) RPC→  query(filter) → extract PKs → engine.delete(pks)
├─ Milvus Get RPC            →  engine.get(pks, ...)
├─ Milvus Query RPC          →  engine.query(filter, output_fields, limit)
├─ Milvus Search RPC         →  parse search_params → engine.search(vectors, top_k, expr, output_fields)
├─ Milvus CreateIndex RPC    →  engine.create_index(field, params)
├─ Milvus LoadCollection RPC →  engine.load()
├─ Milvus ReleaseCollection  →  engine.release()
├─ FieldData ↔ records column-row transposition (translators/records.py)
├─ Error code translation (MilvusLiteError → grpc Status code)
└─ Unsupported RPCs return UNIMPLEMENTED + friendly message (never silent fail)

Startup:
$ milvus-lite server --data-dir ./data --port 19530
```

**Architecture principle**: The adapter layer only does protocol translation, never adds engine capability. pymilvus compatibility boundary is described in section 10.

## 10. MVP Boundary and Limitations

**Included:**
- **DB → Collection → Partition** three-level data organization
- **Internal engine API**: `insert(records)` / `delete(pks)` / `get(pks)` / `search(vectors)`, inputs already normalized (always List)
- **Insert naturally has upsert semantics**: Overwrites if PK exists (Collection-level PK uniqueness)
- **Delete supports global deletion**: When `partition_name=None`, deletes across all Partitions
- Collection Schema model (typed fields, primary key constraints, dynamic fields, name not in Schema)
- Partition support (create / drop / list, `_default` cannot be deleted)
- Partition Pruning (skip irrelevant Partition files during search)
- Schema versioning and persistence
- Arrow IPC Streaming WAL (dual files: wal_data + wal_delta, Collection-level shared)
- Manifest global state management (Partition-level file lists, atomic updates, crash recovery)
- Single-process, single-thread safe reads and writes (`threading.Lock`)
- WAL + Manifest ensure crash recovery
- Insert data and delete records separated (data files + Delta Log)
- Flat file organization + Size-Tiered Compaction (executed independently per Partition)
- Bitmap pipeline (dedup + delete filtering → vector retrieval)
- Cosine / L2 / Inner-Product distance
- **Scalar filtering** (Phase 8) — Milvus-style expressions (comparison / IN / AND / OR / NOT / LIKE / arithmetic / IS NULL / `$meta` dynamic fields) + filter LRU cache + `query()` public method + hybrid backend optimization
- **FAISS HNSW vector index** (Phase 9) — per-segment index + IDSelectorBitmap pre-filter + load/release state machine + index persistence; fallback to BruteForceIndex
- **gRPC adapter layer** (Phase 10) — pymilvus clients can connect to MilvusLite without code changes

**Not included (future iterations):**
- Auto ID (automatic primary key generation)
- Expression-based deletion (pymilvus `delete(filter=)` takes the query → delete indirect path in MVP)
- Partition Key (automatic hash partitioning; currently only manual Partition specification is supported)
- IVF-PQ / OPQ and other advanced quantization vector indexes (HNSW / IVF_FLAT / IVF_SQ8 / HNSW_SQ are already supported)
- Binary / Float16 / BFloat16 vector types (SPARSE_FLOAT_VECTOR is supported via BM25 Function)
- Multiple vector fields (multiple vector columns in one Collection)
- Hybrid Search (multi-vector recall)
- Snapshot (see iteration plan below)
- Bloom Filter to accelerate point lookup positioning
- Multi-threaded background Compaction
- Async index build (Phase 9 is synchronous inline during flush)
- Multi-process concurrent access
- Distributed support
- Authentication / RBAC (embedded mode defaults to none)
- Backup / Restore RPC

### Storage Layer Foundations Reserved for Future Features

The current storage design already provides underlying support for the following features, without needing to change the file format:

| Feature | Storage Layer Foundation | Status |
|---------|----------------|---|
| FAISS Index | Data files contain no deletion markers, can directly build indexes on files; segment-level index naturally matches the immutable architecture | Phase 9 |
| Scalar Filtering | Typed fields + bitmap pipeline filter_mask extension point, Parquet supports predicate pushdown | Phase 8 |
| gRPC Adapter Layer | Engine API inputs are already normalized (List[dict] / List[pk]), translation layer only does protocol wrapping | Phase 10 |
| Snapshot | `_seq` provides a time anchor, Manifest provides state snapshot, immutable files can be shared across multiple snapshots | TODO |
| Schema Changes | Parquet schema evolution, old files automatically fill null for missing columns | TODO |
| Partition Key | Manifest already organizes files by Partition, only needs hash routing logic added | TODO |
| IVF_FLAT / IVF_SQ8 / HNSW_SQ | VectorIndex protocol + factory routing | Already supported |
| IVF-PQ / OPQ and other advanced quantization indexes | VectorIndex protocol has reserved interfaces for multiple FAISS index_type | TODO |
| Multiple vector fields | Schema modification + multiple .idx file naming convention | TODO |

### pymilvus Compatibility Boundary (Phase 10)

| pymilvus Call | Status | Notes |
|---|---|---|
| `connect / disconnect` | Supported | gRPC server mode |
| `create_collection / drop_collection / has_collection / describe_collection / list_collections` | Supported | Direct mapping |
| `create_partition / drop_partition / has_partition / list_partitions` | Supported | Direct mapping |
| `insert / upsert` | Supported | Engine insert already has upsert semantics, both RPCs share the same implementation |
| `delete(ids=...)` | Supported | Direct mapping |
| `delete(filter=...)` | Partial | Implemented indirectly via query → delete inside the servicer |
| `get / query` | Supported | Direct mapping |
| `search(filter, output_fields, top_k)` | Supported | output_fields fully supported; filter passed through to Phase 8 |
| `create_index / drop_index / describe_index` | Supported | HNSW / IVF_FLAT / IVF_SQ8 / HNSW_SQ / BruteForce / SPARSE_INVERTED_INDEX |
| `load_collection / release_collection / get_load_state` | Supported | Full state machine |
| `flush / compact` | Supported | Direct mapping |
| `get_collection_stats` | Supported | row_count |
| `hybrid_search` (multi-vector) | UNIMPLEMENTED | Engine does not support multiple vector fields |
| `search_iterator / pagination` | UNIMPLEMENTED | Engine does not yet support offset |
| Aliases (`create_alias` etc.) | Supported | Persisted in `aliases.json`; aliases resolve to collections for CRUD/search |
| Backup / Restore | UNIMPLEMENTED | |
| RBAC / User / Role / Privilege | Stub OK | Embedded mode defaults to single user, stubs return success to avoid pymilvus crash |
| Resource Group / Replica | UNIMPLEMENTED | Not needed for single process |
| `list_databases / using_database` | Stub | Always `default` |
| Sparse / Binary / Float16 / BFloat16 vector | Not supported | Engine only supports FLOAT_VECTOR |
| `json_contains / array_contains / text_match` | FilterUnsupportedError | Phase F3 todo |

## 11. Future Iteration Plan: Snapshot

### Overview

A Snapshot is a consistent read-only view of the database at a specific `_seq` point in time. Based on the Manifest design, a Snapshot **does not need to copy data**; it is essentially **saving a copy of the Manifest at that time**.

### Implementation Approach

With the Manifest, implementing Snapshots becomes very natural:

**Creating a snapshot** = Copy the current Manifest and mark the `_seq` upper bound:

```python
# Persisted as data_dir/snapshots/{snap_name}.manifest.json
{
    "name": "snap_001",
    "seq": 1500,                    # _seq upper bound at snapshot time
    "manifest_version": 42,          # Manifest version at creation time
    "partitions": {                  # Copied from the Manifest at that time (organized by Partition)
        "_default": {
            "data_files": ["_default/data_000001_000500.parquet"],
            "delta_files": ["_default/delta_000501_000503.parquet"]
        },
        "2024_Q1": {
            "data_files": ["2024_Q1/data_000001_003000.parquet"],
            "delta_files": []
        }
    }
}
```

**File reference counting**:

- When a snapshot is created: reference count +1 for referenced files
- When a snapshot is released: reference count -1
- Before Compaction deletes old files: check reference count, skip if > 0

**Snapshot read path**:

```
snapshot.search(query_vector, top_k)
  │
  ├─ 1. Only read the data_files recorded in this snapshot (from the snapshot Manifest)
  ├─ 2. Filter each file by _seq <= snapshot_seq
  ├─ 3. Build a temporary deleted_map from delta_files by _seq <= snapshot_seq
  ├─ 4. Build bitmap → vector retrieval (reuses the same pipeline)
  └─ 5. Return results
```

Does not use the global `deleted_map` (which only reflects the latest state), but rebuilds from delta files using a time window.

**Compaction adaptation**:

- When active snapshots exist, files referenced by snapshots cannot be deleted
- After Compaction produces new files, old files are only physically deleted when reference count reaches zero

### Requirements on MVP Storage Layer (Already Met)

| Requirement | Current Status |
|------|---------|
| `_seq` globally incrementing, can serve as time anchor | Already available |
| Manifest records complete file lists, can serve as snapshot foundation | Already available |
| Files are immutable, can be shared across multiple snapshots | Already available |
| Delta Log is separated from data files, can be independently filtered by _seq | Already available |
| Parquet supports `_seq <= S` predicate pushdown | Already available |
| Compaction can selectively skip files | Needs reference counting added, does not affect file format |

## 12. Dependencies

```
pyarrow >= 14.0
numpy >= 1.24
```

## 13. Implementation Priority

| Phase | Content                                          |
| ---- | --------------------------------------------- |
| P0   | Collection Schema (definition, validation, Arrow Schema generation, persistence) |
| P1   | Manifest (loading, atomic save, Partition-level file list management)  |
| P2   | MemTable (Schema-driven, with _partition routing), WAL (Arrow IPC dual files)|
| P3   | Data file read/write, Delta Log read/write, Flush pipeline (split by Partition + Manifest update)|
| P4   | Partition management (create / drop / list, _default auto-created)|
| P5   | Insert / Delete / Get end-to-end integration (internal engine API)        |
| P6   | Vector search (Brute-Force + bitmap pipeline + Partition Pruning)|
| P7   | Compaction Manager (Size-Tiered, independent per Partition + Manifest update)|
| P8   | DB layer + Collection layer (multi-DB, multi-Collection management)|
| P9   | Crash recovery (Manifest loading + WAL replay + deleted_map rebuild)|
| P10  | Schema changes (add_field + version number increment)            |
| P11  | End-to-end tests and edge cases                             |
