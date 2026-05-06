# Deep Dive Design: WAL · Segment · Search Architecture

## 1. Overview

WAL is the core guarantee for MilvusLite's crash safety. Any write must be persisted to WAL before entering the MemTable.
After a system crash, replaying the WAL can recover data in the MemTable that hasn't been flushed.

**Design Goals**:
- **Durability**: No data loss after process crash (see §8 for OS-level crash fsync discussion)
- **Low Write Amplification**: Arrow IPC binary format, avoiding the 3x bloat of text encoding for vectors
- **Simple and Reliable**: Append-only writes, never modify already-written content, delete entire files
- **Fast Recovery**: Sequential reads, Arrow zero-copy deserialization

---

## 2. Arrow IPC Streaming Format

### 2.1 Why Arrow IPC

| | JSONL | Arrow IPC Streaming |
|--|-------|---------------------|
| Vector encoding | Text `[0.1, 0.2, ...]` ≈ 3x bloat | Binary direct write, 1x |
| Parse overhead | JSON parse + float convert | Zero-copy mmap / direct deserialization |
| Schema validation | None (type errors discovered only at runtime) | Schema embedded in file header, auto-validated on read |
| Batch IO | One line per record, N IO ops | One RecordBatch = one IO op |
| Partial write recovery | Can truncate by line | Must truncate at RecordBatch boundary (see §9) |

### 2.2 Internal File Structure

```
Arrow IPC Streaming file layout:

┌─────────────────────────────────┐
│  Schema Message                 │  ← Written at file creation (new_stream)
│  (field names/types/metadata)   │
├─────────────────────────────────┤
│  RecordBatch Message #1         │  ← 1st write_batch
│  (metadata + body)              │
├─────────────────────────────────┤
│  RecordBatch Message #2         │  ← 2nd write_batch
│  (metadata + body)              │
├─────────────────────────────────┤
│  ...                            │
├─────────────────────────────────┤
│  RecordBatch Message #N         │  ← Nth write_batch
├─────────────────────────────────┤
│  EOS (End-of-Stream) Marker     │  ← Written at close()
│  (4 bytes: 0x00000000)          │
└─────────────────────────────────┘
```

- **Schema Message**: Contains the complete Arrow Schema (field names, types, nullable, metadata)
- **RecordBatch Message**: Contains all column data for one batch, compactly arranged in binary
- **EOS Marker**: 4 bytes of zeros, marking normal end of stream
- **No Footer**: Unlike Arrow IPC File format, Streaming format has no Footer and does not support random access

### 2.3 PyArrow Core API

```python
import pyarrow as pa

# ── Write ──
sink = pa.OSFile(path, "wb")                           # Open file
writer = pa.ipc.new_stream(sink, schema)               # Write Schema Message
writer.write_batch(record_batch)                       # Append RecordBatch
writer.close()                                         # Write EOS + close file

# ── Read ──
source = pa.OSFile(path, "rb")                         # Open file
reader = pa.ipc.open_stream(source)                    # Read Schema Message
for batch in reader:                                   # Read RecordBatch one by one
    process(batch)
# Or read all at once:
table = reader.read_all()                              # Merge all batches into Table
```

---

## 3. Dual-File Structure

### 3.1 File Separation

```
wal/
  wal_data_000001.arrow     # Insert/Update operations → corresponds to MemTable.insert_buf
  wal_delta_000001.arrow    # Delete operations        → corresponds to MemTable.delete_buf
```

**Why two separate files instead of one?**

1. **Different Schemas**: Data file contains all user fields + `$meta` + `_seq` + `_partition`; delta file only contains primary key + `_seq` + `_partition`
2. **Lazy Initialization**: If a round only has inserts without deletes, the delta file is never created (and vice versa)
3. **Symmetric with MemTable**: MemTable also has two independent buffers internally (insert_buf / delete_buf), corresponding one-to-one during recovery
4. **Symmetric with Parquet**: Flush produces two types of files: data Parquet and delta Parquet

### 3.2 WAL Schema

**wal_data_schema** (insert/update):

```
Field          Type                   Description
────           ────                   ────
_seq           uint64                 Independently assigned sequence number per record
_partition     utf8                   Target Partition name (used for routing during recovery)
{pk_field}     (determined by Schema) Primary key
{vector_field} list<float32>          Vector
{field_1}      ...                    User-defined fields
{field_N}      ...
$meta          utf8 (nullable)        Dynamic fields JSON serialization
```

**wal_delta_schema** (delete):

```
Field          Type                   Description
────           ────                   ────
{pk_field}     (determined by Schema) Primary key
_seq           uint64                 Batch delete shares a single _seq
_partition     utf8                   Target Partition ("_all" means across all Partitions)
```

**Note**: WAL schema has one extra `_partition` column compared to Parquet schema. Because WAL is shared at the Collection level,
it needs to record which Partition each piece of data belongs to, so it can be correctly routed to the MemTable during recovery. Parquet files are already isolated by Partition directory, so this is not needed.

---

## 4. WAL Internal State Model

### 4.1 Instance Attributes

```python
class WAL:
    wal_dir: str                           # WAL directory path
    _wal_data_schema: pa.Schema            # Arrow Schema for data file
    _wal_delta_schema: pa.Schema           # Arrow Schema for delta file
    _number: int                           # Current round WAL number N
    _data_writer: Optional[pa.ipc.RecordBatchStreamWriter]   # Lazily initialized
    _delta_writer: Optional[pa.ipc.RecordBatchStreamWriter]  # Lazily initialized
    _data_sink: Optional[pa.OSFile]        # Data file handle
    _delta_sink: Optional[pa.OSFile]       # Delta file handle
    _closed: bool                          # Whether already closed
```

### 4.2 State Transitions

data_writer and delta_writer are **independent**, each having three states:

```
         First write_insert()             close_and_delete()
  None ──────────────────────→ Active ──────────────────────→ Closed
  (file not created)           (file created, writer writable) (file closed+deleted)
```

Combined state matrix:

```
                        delta_writer
                  None      Active     Closed
            ┌──────────┬──────────┬──────────┐
    None    │ Initial   │ Delete   │    -     │
data_       │  state    │  only    │          │
writer      ├──────────┼──────────┼──────────┤
    Active  │ Insert   │  Both    │    -     │
            │  only    │  active  │          │
            ├──────────┼──────────┼──────────┤
    Closed  │    -     │    -     │ Closed   │
            └──────────┴──────────┴──────────┘

Note: close_and_delete() transitions both writers to Closed at once
    There is no intermediate state where data=Active + delta=Closed
```

### 4.3 File Paths

```python
@property
def data_path(self) -> Optional[str]:
    if self._data_writer is None:
        return None
    return os.path.join(self.wal_dir, f"wal_data_{self._number:06d}.arrow")

@property
def delta_path(self) -> Optional[str]:
    if self._delta_writer is None:
        return None
    return os.path.join(self.wal_dir, f"wal_delta_{self._number:06d}.arrow")
```

---

## 5. Core Method Implementations

### 5.1 `__init__`

```python
def __init__(self, wal_dir, wal_data_schema, wal_delta_schema, wal_number):
    self.wal_dir = wal_dir
    self._wal_data_schema = wal_data_schema
    self._wal_delta_schema = wal_delta_schema
    self._number = wal_number
    self._data_writer = None
    self._delta_writer = None
    self._data_sink = None
    self._delta_sink = None
    self._closed = False

    os.makedirs(wal_dir, exist_ok=True)
```

- **Does not create files**: Deferred to first write
- **Creates directory**: Ensures wal_dir exists

### 5.2 `write_insert`

```python
def write_insert(self, record_batch: pa.RecordBatch) -> None:
    """
    Flow:
    1. Check state: raise error if already closed
    2. Lazy initialization: create file + writer on first call
    3. Write RecordBatch
    """
    assert not self._closed, "WAL already closed"

    # ── Lazy initialization ──
    if self._data_writer is None:
        path = os.path.join(self.wal_dir, f"wal_data_{self._number:06d}.arrow")
        self._data_sink = pa.OSFile(path, "wb")
        self._data_writer = pa.ipc.new_stream(self._data_sink, self._wal_data_schema)

    # ── Write ──
    self._data_writer.write_batch(record_batch)
```

**Key Design Points**:

- **No Schema validation**: `write_batch()` already validates internally, schema mismatch raises `ArrowInvalid`
- **No flush/fsync**: Relies on OS buffer cache (see §8 discussion)
- **One RecordBatch corresponds to one insert() call**: Batch insert of N records = 1 RecordBatch with N rows

### 5.3 `write_delete`

```python
def write_delete(self, record_batch: pa.RecordBatch) -> None:
    """Symmetric with write_insert, writes to wal_delta file."""
    assert not self._closed, "WAL already closed"

    if self._delta_writer is None:
        path = os.path.join(self.wal_dir, f"wal_delta_{self._number:06d}.arrow")
        self._delta_sink = pa.OSFile(path, "wb")
        self._delta_writer = pa.ipc.new_stream(self._delta_sink, self._wal_delta_schema)

    self._delta_writer.write_batch(record_batch)
```

### 5.4 `close_and_delete`

```python
def close_and_delete(self) -> None:
    """
    Invocation timing: After flush succeeds.
    Flow:
    1. Close data_writer (write EOS + close file handle)
    2. Close delta_writer (write EOS + close file handle)
    3. Delete data file
    4. Delete delta file
    5. Mark as closed
    """
    if self._closed:
        return  # Idempotent

    # ── Close writers ──
    if self._data_writer is not None:
        self._data_writer.close()      # Write EOS marker
        self._data_sink.close()        # Close file handle
    if self._delta_writer is not None:
        self._delta_writer.close()
        self._delta_sink.close()

    # ── Delete files ──
    data_path = os.path.join(self.wal_dir, f"wal_data_{self._number:06d}.arrow")
    delta_path = os.path.join(self.wal_dir, f"wal_delta_{self._number:06d}.arrow")

    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists(delta_path):
        os.remove(delta_path)

    self._closed = True
```

**Idempotency**: Multiple calls to `close_and_delete()` are safe with no side effects.

**Partial success**: If the data file deletion succeeds but the delta file deletion fails (extreme case), recovery next time will find an orphan delta file; replaying it won't cause data inconsistency (_seq deduplication guarantee).

### 5.5 `find_wal_files`

```python
@staticmethod
def find_wal_files(wal_dir: str) -> List[int]:
    """
    Scan wal_dir, find all existing WAL numbers.
    Extract numbers by matching wal_data_NNNNNN.arrow and wal_delta_NNNNNN.arrow filenames.
    Return a deduplicated, sorted list of numbers.
    """
    if not os.path.exists(wal_dir):
        return []

    numbers = set()
    pattern = re.compile(r"^wal_(data|delta)_(\d{6})\.arrow$")
    for filename in os.listdir(wal_dir):
        m = pattern.match(filename)
        if m:
            numbers.add(int(m.group(2)))

    return sorted(numbers)
```

**Why match both file name patterns?** Because data and delta files exist independently:
- There may be only wal_data (only inserts were done, no deletes)
- There may be only wal_delta (only deletes were done, no inserts)
- Or both may exist

As long as either file exists, that number needs to be recovered.

### 5.6 `recover`

```python
@staticmethod
def recover(wal_dir, wal_number, wal_data_schema, wal_delta_schema):
    """
    Read WAL files of the specified number, return (data_batches, delta_batches).

    Flow:
    1. Construct file paths
    2. Read data and delta files separately
    3. For each file: exists → read all RecordBatches; doesn't exist → return empty list
    4. Truncated RecordBatch → discard, return complete batches before truncation (see §9 error handling)
    """
    data_path = os.path.join(wal_dir, f"wal_data_{wal_number:06d}.arrow")
    delta_path = os.path.join(wal_dir, f"wal_delta_{wal_number:06d}.arrow")

    data_batches = _read_wal_file(data_path)
    delta_batches = _read_wal_file(delta_path)

    return data_batches, delta_batches


def _read_wal_file(path: str) -> List[pa.RecordBatch]:
    """
    Read a single WAL file, return a list of RecordBatches.
    Handles three cases:
    - File doesn't exist → []
    - File is complete → all batches
    - File is truncated → complete batches before truncation
    """
    if not os.path.exists(path):
        return []

    batches = []
    try:
        source = pa.OSFile(path, "rb")
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            batches.append(batch)
    except pa.ArrowInvalid:
        # File truncated: some RecordBatch after Schema is incomplete
        # Already-read batches are complete, discard the incomplete part
        pass
    except Exception:
        # Can't even read the Schema → file is severely corrupted
        # Return empty, let upper layer decide how to handle
        # (Don't raise WALCorruptedError, because recovery should try to recover as much as possible)
        pass

    return batches
```

**Design Decision -- Truncation Handling**:

Arrow IPC Streaming characteristic: `pa.ipc.open_stream()` succeeds → Schema is complete. Then reading RecordBatches one by one,
if a batch is half-read when the file ends → raises `ArrowInvalid`. Already successfully read batches are all complete.

We choose **best-effort recovery**:
- Read as much as possible, discard the truncated part
- This is better than "discard everything if the file has any corruption"
- What's discarded is the data from the last write_batch call (the batch being written when the crash occurred)

---

## 6. WAL Numbering and Lifecycle Management

### 6.1 Numbering Rules

```
WAL number N is a monotonically increasing integer, starting from 1.
After each flush, N += 1.

Source: Manifest.active_wal_number
  - Collection first created: N = 1
  - Each flush: Manifest updates active_wal_number = N + 1
  - Recovery: Read from Manifest, combined with wal/ directory scan results, take max
```

### 6.2 Complete Lifecycle

```
                   Collection initialization
                         │
                         ▼
              ┌─────────────────────┐
              │  WAL(wal_dir, ...,  │
              │       number=N)     │
              │  State: both        │
              │  writers are None   │
              └─────────┬───────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              │              ▼
   write_insert()       │        write_delete()
   First: create        │        First: create
     file+writer        │          file+writer
   Later: append batch  │        Later: append batch
         │              │              │
         └──────────────┼──────────────┘
                        │
                  MemTable is full
                        │
                        ▼
              ┌─────────────────────┐
              │  Flush triggered     │
              │  1. Freeze WAL(N)   │
              │  2. Create WAL(N+1) │
              └─────────┬───────────┘
                        │
        ┌───────────────┤
        ▼               ▼
   New WAL(N+1)     Frozen WAL(N) enters flush pipeline
   Continues to          │
   accept writes         ▼
              ┌─────────────────────┐
              │  Flush pipeline      │
              │  Step 1-4: Write     │
              │    Parquet           │
              │  Step 5: Manifest   │
              │    active_wal=N+1   │
              │    manifest.save()  │
              │  Step 6:            │
              │    WAL(N).close_    │
              │    and_delete()     │
              └─────────────────────┘
```

### 6.3 Synchronization with Manifest

```
Timeline      WAL State                       Manifest.active_wal_number
──────        ────────                         ──────────────────────────
T0            WAL(1) created                   1
T1            WAL(1) writing...                1
T2            WAL(1) frozen, WAL(2) created    1  ← Not yet updated
T3            Flush: Parquet write complete     1
T4            Flush: Manifest updated           2  ← Updated at this moment
T5            WAL(1) deleted                    2
T6            WAL(2) writing...                 2
```

**Key Invariant**: Manifest update (T4) precedes WAL deletion (T5).
This guarantees that: if a crash occurs between T4 and T5, WAL(1) remains on disk,
but Manifest already points to the new data files → duplicate data produced by replaying WAL(1) is eliminated through _seq deduplication.

---

## 7. Crash Recovery Details

### 7.1 WAL State Analysis During Recovery

At Recovery startup, the following situations may be found in the wal/ directory:

```
Scenario A: No WAL files
  Cause: Last shutdown was normal, or normal exit after flush completed
  Handling: No replay needed, start directly

Scenario B: Only WAL(N), N == Manifest.active_wal_number
  Cause: Crash during normal writing (flush not triggered or manifest update not completed)
  Handling: Replay WAL(N) → MemTable

Scenario C: WAL(N) + WAL(N+1), Manifest.active_wal_number == N+1
  Cause: Flush completed Manifest update (active_wal=N+1), but WAL(N) was not deleted
  Handling: Replay WAL(N) (data already in Parquet, _seq deduplication eliminates duplicates)
           Replay WAL(N+1) (newly written data)

Scenario D: WAL(N) + WAL(N+1), Manifest.active_wal_number == N
  Cause: Flush in progress, new WAL already created but Manifest not updated before crash
  Handling: Replay WAL(N) (complete data)
           Replay WAL(N+1) (newly written data)
           Orphan Parquet files cleaned up by recovery

Scenario E: Only data file or delta file of WAL (not paired)
  Cause: That round only had inserts or only had deletes
  Handling: Normal, missing file treated as empty (_read_wal_file returns [])
```

### 7.2 WAL Processing Flow in Recovery

```python
def execute_recovery(data_dir, schema, manifest):
    wal_dir = os.path.join(data_dir, "wal")
    memtable = MemTable(schema)

    # ── Step 1: Discover WAL files ──
    wal_numbers = WAL.find_wal_files(wal_dir)
    if not wal_numbers:
        # Scenario A: No recovery needed
        return memtable, ...

    # ── Step 2: Replay all WALs in ascending order by number ──
    max_seq = manifest.current_seq
    for n in sorted(wal_numbers):
        data_batches, delta_batches = WAL.recover(
            wal_dir, n, wal_data_schema, wal_delta_schema
        )
        for batch in data_batches:
            for row in range(batch.num_rows):
                seq = batch.column("_seq")[row].as_py()
                partition = batch.column("_partition")[row].as_py()
                fields = {col: batch.column(col)[row].as_py()
                          for col in batch.schema.names
                          if col not in ("_seq", "_partition")}
                memtable.put(seq, partition, **fields)
                max_seq = max(max_seq, seq)

        for batch in delta_batches:
            for row in range(batch.num_rows):
                pk = batch.column(pk_name)[row].as_py()
                seq = batch.column("_seq")[row].as_py()
                partition = batch.column("_partition")[row].as_py()
                memtable.delete(pk, seq, partition)
                max_seq = max(max_seq, seq)

    # ── Step 3: Do not delete WAL files (see §7.3) ──
    next_seq = max_seq + 1
    return memtable, delta_log, next_seq
```

### 7.3 WAL File Handling Strategy After Recovery

**Core Question**: After Recovery replays WAL to MemTable, should old WAL files be immediately deleted?

**Approach Comparison**:

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| A. Delete immediately | `os.remove()` after replay | Clean disk | If crash again after recovery (before flush), data permanently lost |
| B. Retain, don't delete | Leave as-is, clean during flush | Safe against second crash | Need to handle cleanup of "multiple WAL rounds + current WAL" |

**Choose Approach B**: Retain old WAL files, clean up uniformly during flush.

**Implementation**:
- After Recovery, Collection knows that old WAL files may remain in the wal/ directory
- Collection creates a new WAL (number = max(all existing numbers) + 1)
- When MemTable flushes, the flush pipeline in Step 6 not only deletes the frozen WAL,
  but also cleans up all old WAL files with numbers < current active_wal_number

```python
# flush.py Step 6 enhanced
def _cleanup_wal_files(wal_dir: str, max_number_to_delete: int) -> None:
    """Delete all WAL files with number <= max_number_to_delete."""
    for n in WAL.find_wal_files(wal_dir):
        if n <= max_number_to_delete:
            data_path = os.path.join(wal_dir, f"wal_data_{n:06d}.arrow")
            delta_path = os.path.join(wal_dir, f"wal_delta_{n:06d}.arrow")
            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(delta_path):
                os.remove(delta_path)
```

### 7.4 WAL Number After Recovery

```python
# Determining the new WAL number after recovery
found_numbers = WAL.find_wal_files(wal_dir)       # Remaining on disk
manifest_number = manifest.active_wal_number       # Recorded in Manifest

new_wal_number = max(
    manifest_number,
    max(found_numbers) + 1 if found_numbers else 1
)
```

**This ensures the new WAL number will not conflict with any existing or previously existing WAL files.**

---

## 8. fsync and Durability

### 8.1 Crash Types

| Crash Type | Example | OS buffer cache | Data loss risk without fsync |
|---------|------|----------------|------------|
| Process crash (immediately taken over within same OS) | SIGKILL, abnormal exit | Retained (still held by OS) | Low -- but still present |
| Container/process killed then immediately taken over by new process | OOM-kill → restart | Still in cache, new process read hits | **High** -- see §8.2 counterexample |
| OS crash | Kernel panic, power loss | Lost | Present |

### 8.2 Default `sync_mode="close"`: fsync Once at Close

```python
class WAL:
    def __init__(self, ..., sync_mode: str = "close"):
        """
        sync_mode:
          "none"   - No fsync at all (for testing / performance benchmarks only)
          "close"  - Default. One os.fsync on sink before close_and_delete
          "batch"  - fsync after every write_batch (strongest consistency, slowest)
        """
```

**Why "none" is not the default**:

Consider this counterexample:

```
T0: WAL.write_insert(batch_X)   # batch X enters OS buffer cache, not flushed to disk
T1: Collection.insert returns success, client receives success response
T2: Container OOM-killed
T3: Orchestration system immediately starts new container, mounts same volume
T4: New process Collection.__init__ → recovery → reads WAL
```

At T4:
- The OS buffer cache held by the old process has already disappeared with the old process (container isolation)
- batch_X hasn't been flushed to disk yet
- The new process reads a **WAL file without batch_X**
- But the client was already told "success" -- **data lost, durability promise violated**

`sync_mode="close"` calls `os.fsync(sink.fileno())` once before `close_and_delete` to plug this window:
- WAL is closed when flush is triggered, fsync before close persists the entire file
- Frequency = flush frequency = once per `MEMTABLE_SIZE_LIMIT` rows, very sparse, negligible performance impact
- Critical path: durability on the "WAL switch → Manifest update" commit path is now ensured

**Cost of `sync_mode="batch"`**:

fsync on every `write_batch`. For an embedded vector database (vectors are large), this cost is significant, but kept as an option for users who want the strongest durability. **Not enabled by default**.

### 8.3 Implementation Details

```python
def write_insert(self, record_batch):
    assert not self._closed
    if self._data_writer is None:
        # ... lazy init
    self._data_writer.write_batch(record_batch)
    if self._sync_mode == "batch":
        os.fsync(self._data_sink.fileno())

def close_and_delete(self):
    if self._closed:
        return
    try:
        if self._sync_mode in ("close", "batch") and self._data_writer is not None:
            self._data_writer.close()                      # Write EOS marker
            os.fsync(self._data_sink.fileno())             # ← Force flush to disk
            self._data_sink.close()
        # ... same for _delta_writer
    finally:
        self._closed = True
        # ... delete files
```

**Note**: fsync must happen after `_data_writer.close()` and before `_data_sink.close()`. First close the writer to write the EOS marker into the OS buffer, then fsync to persist the entire file including EOS to disk.

---

## 9. Error Handling

### 9.1 Errors During Write

| Error Type | Trigger Scenario | Handling |
|---------|---------|---------|
| `ArrowInvalid` | RecordBatch schema mismatch | Raise directly (caller bug, should not reach WAL layer) |
| `OSError` | Disk full, permission denied | Raise directly, Collection layer handles (write fails, MemTable not updated) |
| `AssertionError` | Writing to an already closed WAL | Programming error, should not occur |

**Impact when WAL write fails**:

```
Collection.insert() flow:
  1. validate + allocate _seq     ← Succeeds
  2. build RecordBatch            ← Succeeds
  3. WAL.write_insert(batch)      ← Fails! Exception raised
  4. MemTable.put(...)            ← Not executed

Result: _seq is wasted (gaps), but data consistency is not affected.
WAL and MemTable stay in sync -- neither has written this data.
```

### 9.2 Errors During Recovery

| Error Type | Trigger Scenario | Handling |
|---------|---------|---------|
| File doesn't exist | Only data without delta (or vice versa) | Normal, missing one treated as empty |
| Schema read failure | File severely corrupted (only a few bytes written) | Skip the file, log warning |
| RecordBatch truncated | Crash during write | Return complete batches before truncation |
| Data validation failure | Extreme cases like disk bit flip | Skip the file, log warning |

**Recovery Principle: Best-effort recovery, don't abandon all data due to partial corruption.**

### 9.3 Safety of _seq Gaps

WAL write failures can cause _seq gaps, for example:

```
Normal:    _seq = [1, 2, 3, 4, 5]
With gap:  _seq = [1, 2, 4, 5]      ← 3 skipped due to WAL write failure
```

This is safe because:
- _seq is only used for "same PK, keep max_seq" deduplication, does not depend on continuity
- delta_log's `is_deleted(pk, data_seq)` comparison is "delete_seq > data_seq", does not depend on continuity

---

## 10. Interaction with Upper-Layer Components

### 10.1 Insert Flow

```
Collection.insert(records, partition_name)
  │
  ├─ 1. validate_record(record, schema)       # Validate each record
  ├─ 2. separate_dynamic_fields(record)       # Separate dynamic fields → $meta
  ├─ 3. seq = self._alloc_seq()               # Assign independent _seq to each record
  ├─ 4. Build RecordBatch (with _seq, _partition)
  │
  ├─ 5. self.wal.write_insert(batch)          ← WAL writes first
  │
  ├─ 6. for each record:                      ← Then MemTable
  │       self.memtable.put(seq, partition, **fields)
  │
  └─ 7. if self.memtable.size() >= LIMIT:     ← Check if flush needed
           self._trigger_flush()
```

### 10.2 Delete Flow

```
Collection.delete(pks, partition_name)
  │
  ├─ 1. shared_seq = self._alloc_seq()        # Batch shares a single _seq
  ├─ 2. partition = partition_name or "_all"
  ├─ 3. Build RecordBatch (pk_values, shared_seq, partition)
  │
  ├─ 4. self.wal.write_delete(batch)          ← WAL writes first
  │
  ├─ 5. for pk in pks:                        ← Then MemTable
  │       self.memtable.delete(pk, shared_seq, partition)
  │
  └─ 6. if self.memtable.size() >= LIMIT:
           self._trigger_flush()
```

### 10.3 WAL Switching During Flush

```
Collection._trigger_flush()
  │
  ├─ 1. frozen_memtable = self.memtable       # Freeze
  ├─ 2. frozen_wal = self.wal                 # Freeze
  │
  ├─ 3. self.memtable = MemTable(schema)      # Create new empty MemTable
  ├─ 4. new_number = frozen_wal.number + 1
  ├─ 5. self.wal = WAL(wal_dir, ..., new_number)  # Create new empty WAL
  │
  └─ 6. execute_flush(                        # Execute flush pipeline in background
  │       frozen_memtable,
  │       frozen_wal,                         # ← Pass in frozen WAL
  │       ...
  │     )
  │
  └─ Inside flush pipeline:
       Step 1-4: Write Parquet files
       Step 5:   manifest.active_wal_number = new_number
                 manifest.save()              # Atomic update
       Step 6:   _cleanup_wal_files(wal_dir, frozen_wal.number)
                                              # Delete frozen WAL + earlier residual WALs
```

### 10.4 Recovery Flow

```
Collection.__init__(name, data_dir, schema)
  │
  ├─ 1. manifest = Manifest.load(data_dir)
  │
  ├─ 2. memtable, delta_log, next_seq = execute_recovery(
  │       data_dir, schema, manifest          # Recovery internals:
  │     )                                     #   find_wal_files → recover → replay to memtable
  │
  ├─ 3. self.memtable = memtable              # Use the recovered MemTable
  ├─ 4. self._seq_counter = next_seq          # Restore _seq counter
  │
  ├─ 5. new_wal_number = ...                  # See §7.4 for number calculation
  └─ 6. self.wal = WAL(wal_dir, ..., new_wal_number)  # New WAL for subsequent writes
                                              # Old WAL files remain on disk, await flush cleanup
```

---

## 11. Complete Interface (Final Version)

Combining all the above design, the complete WAL interface is updated:

```python
class WAL:
    """Write-Ahead Log, Arrow IPC Streaming format, dual-file (data + delta).

    Each write round corresponds to a pair of WAL files (wal_data_{N}.arrow + wal_delta_{N}.arrow),
    deleted entirely after flush succeeds. Writers are lazily initialized (file created on first write).
    """

    def __init__(
        self,
        wal_dir: str,
        wal_data_schema: pa.Schema,
        wal_delta_schema: pa.Schema,
        wal_number: int,
        sync_mode: str = "close",
    ) -> None:
        """Initialize WAL (does not create files, deferred to first write).

        Args:
            wal_dir: Directory for WAL files (created if doesn't exist)
            wal_data_schema: Arrow Schema for wal_data file (with _seq, _partition, user fields, $meta)
            wal_delta_schema: Arrow Schema for wal_delta file (with pk, _seq, _partition)
            wal_number: Current round WAL number N (the N in filename, derived from Manifest or recovery)
            sync_mode: Durability strategy, see §8 for details.
                - "none"  No fsync at all (for testing / performance benchmarks only)
                - "close" Default. One fsync before close_and_delete
                - "batch" fsync after every write_batch
        """

    def write_insert(self, record_batch: pa.RecordBatch) -> None:
        """Append a RecordBatch to the wal_data file.

        Creates file and StreamWriter on first call (lazy initialization).
        record_batch's schema must match wal_data_schema (validated internally by PyArrow).

        When sync_mode="batch", fsync once after write_batch.

        Note: WAL doesn't know about Operation types -- it accepts raw RecordBatch.
        Operation dispatch is done in Collection._apply (due to dependency layering,
        see modules.md §9.16.5).

        Raises:
            ArrowInvalid: schema mismatch
            OSError: disk full or permission denied
            AssertionError: WAL already closed
        """

    def write_delete(self, record_batch: pa.RecordBatch) -> None:
        """Append a RecordBatch to the wal_delta file. Semantics symmetric with write_insert."""

    def close_and_delete(self) -> None:
        """Close writers and delete both WAL files. Idempotent operation.

        Invocation timing: Flush Step 6 (after Manifest has been updated).

        Flow (each writer wrapped in its own try/finally, independent of each other):
            for writer in (data_writer, delta_writer):
                writer.close()                              # Write EOS marker
                if sync_mode in ("close", "batch"):
                    os.fsync(sink.fileno())                 # Force flush to disk
                sink.close()
            for file in (data_path, delta_path):
                if exists: os.remove(file)
            self._closed = True

        Idempotency: Second call returns directly; failure of one writer.close doesn't affect the other.
        Silently skips deletion step if file doesn't exist.
        """

    @staticmethod
    def find_wal_files(wal_dir: str) -> List[int]:
        """Scan directory, return all WAL numbers (deduplicated, ascending).

        Matches wal_data_NNNNNN.arrow and wal_delta_NNNNNN.arrow,
        includes the number as long as either file exists.

        Returns:
            WAL number list in ascending order, [] if directory doesn't exist
        """

    @staticmethod
    def recover(
        wal_dir: str,
        wal_number: int,
    ) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        """Read WAL files of the specified number, return (data_batches, delta_batches).

        - File doesn't exist → corresponding list is empty
        - File truncated → return complete batches before truncation, discard incomplete part
        - File severely corrupted (Schema unreadable) → corresponding list is empty, log warning

        Operation wrapping is done in engine/recovery.py's replay_wal_operations()
        (due to dependency layering, see modules.md §9.16.5 for details).
        """

    @property
    def number(self) -> int:
        """Current WAL number N."""

    @property
    def data_path(self) -> Optional[str]:
        """wal_data file path. Returns None before lazy initialization (no insert written yet)."""

    @property
    def delta_path(self) -> Optional[str]:
        """wal_delta file path. Returns None before lazy initialization (no delete written yet)."""


# ── Module-level helper functions (not exported) ──

def _read_wal_file(path: str) -> List[pa.RecordBatch]:
    """Read a single WAL file, with fault-tolerant handling for truncation and corruption."""

def _cleanup_old_wals(wal_dir: str, up_to_number: int) -> None:
    """Delete all WAL files with number <= up_to_number. Called by flush Step 6."""
```

---

## 12. WAL Design Decision Summary

| Decision | Choice | Rationale |
|------|------|------|
| File format | Arrow IPC Streaming | No write amplification for vectors, zero-copy parsing, built-in Schema |
| File structure | Dual-file (data + delta) | Different schemas, lazy initialization, symmetric with MemTable |
| Writer initialization | Lazy (created on first write) | Avoid empty files |
| **fsync** | **Default `sync_mode="close"`, one fsync before close** | Covers container OOM-kill then immediate takeover crash scenario; frequency = flush frequency, negligible overhead. See §8 for details |
| Truncation handling | Best-effort recovery (read as much as possible) | Maximize data recovery |
| Old WAL after Recovery | Retain, wait for flush cleanup | Safe against second crash |
| WAL numbering | Monotonically increasing, recovered from Manifest | No duplicates, traceable |
| close_and_delete | Idempotent, each writer has independent try/finally | Safe for crash retry; one writer's close failure doesn't affect the other |
| Cleanup scope | Flush cleans all WALs with number <= frozen number | Unified handling of residual + frozen |
| **Write entry point** | **Raw `write_insert / write_delete`** | WAL doesn't know about Operations; dispatch is done in Collection.\_apply (due to dependency layering, see modules.md §9.16.5) |
| **Read entry point** | **`recover() → (data_batches, delta_batches)`** | engine/recovery.py's `replay_wal_operations` wraps it into an Operation stream sorted by _seq |

---

# Part II: Segment and Search Architecture

---

## 13. Upsert and Its Relationship with WAL

### 13.1 Upsert Is Not Delete + Insert

Traditional databases typically implement upsert as "delete then insert", involving two write steps.
MilvusLite uses an LSM-Tree style where **upsert is purely an insert, only writing to `wal_data`, never touching `wal_delta`**.

```
Insert("doc_1", new_data)   ← PK "doc_1" already exists in Parquet on disk

WAL:       wal_data appends one RecordBatch (_seq=new value)  ✅ Only writes one file
MemTable:  put("doc_1", ...) → dict overwrites old entry by PK  ✅ In-memory overwrite
Old data on disk: Still in Parquet, untouched                  ✅ No explicit deletion needed
```

"Overwrite" is implicitly achieved through **_seq deduplication**:

```
Disk Parquet:  doc_1, _seq=100, embedding=[0.1, ...]    ← Old version
MemTable:      doc_1, _seq=500, embedding=[0.9, ...]    ← New version

During search, bitmap dedup: same PK keeps max_seq → _seq=500 wins → old version invisible
During Compaction: old record (_seq=100) is physically purged
```

### 13.2 WAL Files Involved in Each API Operation

| Operation | wal_data | wal_delta | Atomicity risk |
|------|----------|-----------|-----------|
| insert (new PK) | Write | - | None (single file) |
| insert (existing PK = upsert) | Write | - | None (single file) |
| delete | - | Write | None (single file) |

**No single API call needs to write to both WAL files simultaneously**, so there is no "dual-file half-write" consistency issue.

### 13.3 LSM-Tree Style Trade-off

Simple writes (append-only) → but search needs extra filtering of expired data → solved by bitmap dedup → Compaction ultimately cleans up physical data.

---

## 14. Segment: In-Memory Cache of Data Files

### 14.1 Problem: Reading Disk on Every Search Is Unacceptable

MVP brute-force search needs to read all data from all Parquet files. If every search reads from disk:

```
1 million records × 128 dims × 4 bytes = 488 MB
Reading 488 MB on every search → latency of several seconds → unacceptable
```

### 14.2 Segment Concept

Each Parquet data file has a corresponding **Segment** (sealed segment) in memory.
Parquet files are immutable → Segment never invalidates once loaded, no cache eviction strategy needed.

```python
class Segment:
    """In-memory cache of a Parquet data file. File is immutable → cache never invalidates."""

    file_path: str              # Source file path
    partition: str              # Belonging Partition

    # ── Pre-extracted NumPy arrays (for search hot path) ──
    pks: np.ndarray             # Primary key array, used for bitmap dedup
    seqs: np.ndarray            # _seq array, used for bitmap dedup
    vectors: np.ndarray         # (N, dim) float32 matrix, used for distance calculation

    # ── Complete records (for returning search results) ──
    table: pa.Table             # Original Arrow Table, retrieve rows corresponding to top-k results

    # ── Future ──
    # faiss_index: faiss.Index  # FAISS index
```

### 14.3 Why Pre-extract NumPy Arrays

```
Approach A: Extract from pa.Table on every search
  pa.Table → .column("vec") → .to_numpy() → has conversion overhead
  Vector column is list<float32> (variable length), needs stack into (N, dim) contiguous array
  Doing this on every search → repeated waste

Approach B: One-time pre-extraction at load time (chosen approach)
  At Segment creation → extract pks / seqs / vectors into NumPy arrays
  During search → use directly, zero extra overhead
```

### 14.4 Segment Complete Interface

```python
# storage/segment.py

class Segment:
    """In-memory representation of an immutable data file. Pre-extracts NumPy arrays needed for search hot path at load time."""

    def __init__(self, file_path: str, partition: str, table: pa.Table,
                 pk_field: str, vector_field: str):
        self.file_path = file_path
        self.partition = partition
        self.table = table

        # One-time pre-extraction
        self.pks = table.column(pk_field).to_numpy()
        self.seqs = table.column("_seq").to_numpy()
        self.vectors = self._extract_vectors(table, vector_field)

    @staticmethod
    def load(file_path: str, partition: str,
             pk_field: str, vector_field: str) -> "Segment":
        """Load from Parquet file into Segment."""
        table = read_data_file(file_path)
        return Segment(file_path, partition, table, pk_field, vector_field)

    @staticmethod
    def from_table(file_path: str, partition: str, table: pa.Table,
                   pk_field: str, vector_field: str) -> "Segment":
        """Build Segment from existing Arrow Table (skip disk read during Flush)."""
        return Segment(file_path, partition, table, pk_field, vector_field)

    def search(self, query_vector: np.ndarray, valid_mask: np.ndarray,
               top_k: int, metric_type: str) -> List["Hit"]:
        """Search within this Segment, only compute distances for rows where valid_mask=True.

        Returns:
            Hit list sorted by distance in ascending order (at most top_k)
        """
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_vectors = self.vectors[valid_indices]
        distances = compute_distances(query_vector, valid_vectors, metric_type)

        k = min(top_k, len(distances))
        top_k_pos = np.argpartition(distances, k - 1)[:k]
        top_k_pos = top_k_pos[np.argsort(distances[top_k_pos])]

        return [
            Hit(pk=self.pks[valid_indices[i]],
                distance=float(distances[i]),
                row_index=int(valid_indices[i]),
                segment=self)
            for i in top_k_pos
        ]

    def get_record(self, row_index: int) -> dict:
        """Get complete record by row index (for top-k result lookback)."""
        return {col: self.table.column(col)[row_index].as_py()
                for col in self.table.schema.names
                if col != "_seq"}

    @staticmethod
    def _extract_vectors(table: pa.Table, vector_field: str) -> np.ndarray:
        """list<float32> column → (N, dim) float32 contiguous array."""
        vec_col = table.column(vector_field)
        return np.stack([v.as_py() for v in vec_col]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.table)
```

### 14.5 Cache Lifecycle

Segment lifecycle follows the Manifest exactly, no LRU or other eviction strategies needed:

```
Event                              Cache Operation
────                               ────────
Collection startup / Recovery      Load all data files from Manifest → Segment list
Flush complete                     New Parquet → create new Segment and add to cache
Compaction complete                Old files deleted → evict old Segments; new file → create new Segment
Collection.close()                 Release all Segments
```

```python
# Segment management in engine/collection.py

class Collection:
    _segments: Dict[str, List[Segment]]   # partition_name → [Segment, ...]

    def __init__(self, name, data_dir, schema):
        # Load all Segments at startup
        self._segments = {}
        for partition in manifest.list_partitions():
            self._segments[partition] = [
                Segment.load(f, partition, pk_field, vec_field)
                for f in manifest.get_data_files(partition)
            ]

    def _on_flush_complete(self, partition, new_file, table):
        """Flush callback: build Segment directly from in-memory table (skip disk read)"""
        seg = Segment.from_table(new_file, partition, table, pk_field, vec_field)
        self._segments.setdefault(partition, []).append(seg)

    def _on_compaction_complete(self, partition, old_files, new_file):
        """Compaction callback: evict old Segments, load new Segment"""
        old_set = set(old_files)
        self._segments[partition] = [
            s for s in self._segments[partition] if s.file_path not in old_set
        ]
        seg = Segment.load(new_file, partition, pk_field, vec_field)
        self._segments[partition].append(seg)
```

### 14.6 Flush Optimization: Zero-Copy Segment Building

During Flush, data is already in memory (frozen MemTable → Arrow Table).
After writing to Parquet, **use this Table directly to build a Segment, no need to read it back from disk**:

```
Normal path:   MemTable → pa.Table → write_data_file() → read_data_file() → Segment
                                          write to disk        read from disk (wasteful)

Optimized path: MemTable → pa.Table ──┬── write_data_file()     write to disk
                                      └── Segment.from_table()  directly reuse memory
```

### 14.7 Memory Budget

For an embedded local database, MVP assumes the dataset fits in memory:

```
1 million records × 128 dims float32 = 488 MB (vectors)
Plus PK / _seq / other fields ≈ 600-800 MB total

Perfectly acceptable for local machines (8-16 GB RAM)
```

In the future, if data exceeds memory, LRU eviction can be added at Segment granularity -- but MVP doesn't need it.

---

## 15. Segmented Search Architecture

### 15.1 Two Types of Segments

```
Sealed Segment    Loaded from Parquet files, immutable, can build FAISS index
                  Data source: storage/segment.py

Growing Segment   Unflushed data in MemTable, continuously changing, can only brute-force search
                  Data source: storage/memtable.py
```

Both participate in search with consistent behavior: accept valid_mask, return local top-k.

### 15.2 Core Contradiction: Dedup Is Global, Search Is Segmented

The same PK may exist in multiple Segments (caused by upsert):

```
Segment A (old Parquet):  doc_1  _seq=100  vec=[0.1, ...]
Segment B (new Parquet):  doc_1  _seq=500  vec=[0.9, ...]   ← Flushed after upsert
MemTable:                 doc_1  _seq=800  vec=[0.5, ...]   ← Yet another upsert
```

If each Segment deduplicates independently, Segment A wouldn't know doc_1 has been overwritten.
Therefore: **global dedup first, then segmented search**.

### 15.3 Complete Search Flow

```
Step 1: Global dedup → generate per-segment mask (lightweight, only uses pre-extracted pk/seq arrays)
──────────────────────────────────────────────────────────────────────────────

  segments:
    seg_A.pks  = [doc_1, doc_2]     seg_A.seqs = [100, 101]
    seg_B.pks  = [doc_1, doc_3]     seg_B.seqs = [500, 501]
    memtable   = [doc_1, doc_4]     seqs       = [800, 802]

  Global dedup:
    doc_1 → seg_A(100) vs seg_B(500) vs memtable(800) → memtable wins
    doc_2 → seg_A(101) unique → valid
    doc_3 → seg_B(501) unique → valid
    doc_4 → memtable(802) unique → valid

  Delete filtering: delta_log.is_deleted(pk, seq)

  Output per-segment masks:
    seg_A_mask = [False, True ]     ← doc_1 overwritten, doc_2 valid
    seg_B_mask = [False, True ]     ← doc_1 overwritten, doc_3 valid
    mem_mask   = [True,  True ]     ← doc_1 latest version valid, doc_4 valid

Step 2: Each Segment searches independently (only compute distances for valid rows)
──────────────────────────────────────────────────────────────────────────────

  seg_A.search(query, seg_A_mask, top_k) → [(doc_2, dist=0.3)]
  seg_B.search(query, seg_B_mask, top_k) → [(doc_3, dist=0.5)]
  memtable.search(query, mem_mask, top_k) → [(doc_1, dist=0.1), (doc_4, dist=0.7)]

Step 3: Merge all local top-k → global top-k
──────────────────────────────────────────────────────────────────────────────

  all_hits = [(doc_1, 0.1), (doc_2, 0.3), (doc_3, 0.5), (doc_4, 0.7)]
  sort by distance → take first top_k

Step 4: Retrieve complete records (only query top-k rows)
──────────────────────────────────────────────────────────────────────────────

  for hit in global_top_k:
      hit.segment.get_record(hit.row_index)   ← Look back in corresponding Segment
```

### 15.4 Global Dedup Implementation

```python
# search/bitmap.py

def build_segment_masks(
    segments: List[Segment],
    memtable: MemTable,
    delta_log: DeltaLog,
    pk_field: str,
) -> List[np.ndarray]:
    """Global dedup + delete filtering, output per-segment masks.

    Returns N+1 boolean masks: first N correspond to segments, last one to memtable.
    Each mask length equals the row count of the corresponding segment.
    """
    # ── Global dedup: PK → (max_seq, segment_idx, row_idx) ──
    best = {}  # pk → (max_seq, seg_idx, row_idx)

    for seg_idx, seg in enumerate(segments):
        for row_idx in range(len(seg.pks)):
            pk = seg.pks[row_idx]
            seq = int(seg.seqs[row_idx])
            if pk not in best or seq > best[pk][0]:
                best[pk] = (seq, seg_idx, row_idx)

    # MemTable as the last "segment"
    mem_idx = len(segments)
    mem_pks, mem_seqs = memtable.get_pk_seq_arrays()
    for row_idx in range(len(mem_pks)):
        pk = mem_pks[row_idx]
        seq = int(mem_seqs[row_idx])
        if pk not in best or seq > best[pk][0]:
            best[pk] = (seq, mem_idx, row_idx)

    # ── Build per-segment masks ──
    masks = [np.zeros(len(seg.pks), dtype=bool) for seg in segments]
    masks.append(np.zeros(len(mem_pks), dtype=bool))  # MemTable mask

    for pk, (seq, seg_idx, row_idx) in best.items():
        if not delta_log.is_deleted(pk, seq):
            masks[seg_idx][row_idx] = True

    return masks
```

### 15.5 MemTable Search Support

MemTable is the Growing Segment and needs to provide a search interface symmetric with Sealed Segment:

```python
# New methods added to storage/memtable.py

class MemTable:
    def get_pk_seq_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (pks, seqs) arrays of all active records in insert_buf.
        Used for bitmap global dedup."""

    def get_vectors(self, vector_field: str) -> np.ndarray:
        """Return vector matrix (N, dim) of all active records in insert_buf.
        Used for brute-force search."""

    def search(self, query_vector: np.ndarray, valid_mask: np.ndarray,
               top_k: int, metric_type: str) -> List["Hit"]:
        """Brute-force search records in insert_buf where valid_mask=True.
        Logic same as Segment.search()."""

    def get_record(self, row_index: int) -> dict:
        """Get complete record by row index (for top-k lookback)."""
```

### 15.6 Merge Strategy

```python
# search/merge.py

@dataclass
class Hit:
    """A single search result."""
    pk: Any
    distance: float
    row_index: int
    segment: Any        # Reference to Segment or MemTable, for looking back complete records

    def to_record(self) -> dict:
        """Look back complete record and attach distance."""
        record = self.segment.get_record(self.row_index)
        record["_distance"] = self.distance
        return record

def merge_results(segment_hits: List[List[Hit]], top_k: int) -> List[Hit]:
    """Merge local top-k from multiple segments, take global top-k.

    MVP: Simple concatenation + sort (few segments, fast enough)
    Future: k-way merge with heap (more efficient when many segments)
    """
    all_hits = []
    for hits in segment_hits:
        all_hits.extend(hits)
    all_hits.sort(key=lambda h: h.distance)
    return all_hits[:top_k]
```

### 15.7 Updated Executor

```python
# search/executor.py

def execute_search(
    query_vectors: List[np.ndarray],
    segments: List[Segment],
    memtable: MemTable,
    delta_log: DeltaLog,
    top_k: int,
    metric_type: str,
    pk_field: str,
) -> List[List[dict]]:
    """Search entry point. Global dedup → segmented search → merge → record lookback."""

    results = []
    for query in query_vectors:
        # ── Step 1: Global dedup → per-segment mask ──
        masks = build_segment_masks(segments, memtable, delta_log, pk_field)

        # ── Step 2: Segmented search ──
        all_hits = []
        for seg, mask in zip(segments, masks[:-1]):
            all_hits.append(seg.search(query, mask, top_k, metric_type))

        mem_mask = masks[-1]
        all_hits.append(memtable.search(query, mem_mask, top_k, metric_type))

        # ── Step 3: Merge ──
        merged = merge_results(all_hits, top_k)

        # ── Step 4: Look back complete records ──
        results.append([hit.to_record() for hit in merged])

    return results
```

---

## 16. data_file.py Interface Extension

Currently `data_file.py` only has full reads. For search optimization and future FAISS support, column projection and row selection are added:

```python
# storage/data_file.py

# ── Existing ──
def write_data_file(table, partition_dir, seq_min, seq_max) -> str:
    """Write Parquet data file."""

def read_data_file(path) -> pa.Table:
    """Full read of Parquet file. Called by Segment.load()."""

def parse_seq_range(filename) -> Tuple[int, int]:
    """Parse seq range from filename."""

def get_file_size(path) -> int:
    """Get file size (for Compaction bucketing)."""

# ── New ──
def read_columns(path: str, columns: List[str]) -> pa.Table:
    """Column projection read. Only read specified columns, skip the rest (especially vector column).

    Under Parquet columnar storage, skipping the vector column saves 90%+ IO.
    Use case: In future FAISS scenario, bitmap building only needs PK + _seq.

    Example: read_columns(path, ["doc_id", "_seq"])
    """
    return pq.read_table(path, columns=columns)

def read_rows(path: str, row_indices: List[int],
              columns: Optional[List[str]] = None) -> pa.Table:
    """Row selection read. Only read specified rows (with optional column projection).

    Use case: In future when FAISS returns top-k row_indices,
    only read complete records for those few rows, rather than loading the entire file.
    """
```

**MVP vs FAISS data_file call comparison**:

```
MVP (full data cached in Segment):
  At startup: read_data_file(path) → Segment          ← One-time load, resident in memory
  During search: Directly access Segment's NumPy arrays ← No disk reads

FAISS (future, on-demand reads):
  At startup: read_data_file(path) → Segment + build FAISS index
  During search: Segment provides pk/seq (bitmap)
          FAISS index provides vector search
          read_rows(path, top_k_indices) → look back complete records  ← Only read top-k rows
```

---

## 17. Module Structure Update

### 17.1 New and Changed Modules

```
storage/
  segment.py        ← New: In-memory representation of Parquet files (Segment class)
  data_file.py      ← Extended: New read_columns(), read_rows()
  memtable.py       ← Extended: New get_pk_seq_arrays(), get_vectors(), search()

search/
  bitmap.py         ← Changed: build_valid_mask() → build_segment_masks()
  distance.py       ← Unchanged
  executor.py       ← Changed: Input from raw arrays → segments + memtable
  merge.py          ← New: Hit data class + merge_results()
```

### 17.2 Updated Complete Structure

```
milvus_lite/
├── storage/
│   ├── wal.py           # WAL read/write
│   ├── memtable.py      # In-memory buffer (with search support)
│   ├── segment.py       # ★ In-memory cache of Parquet files (with search capability)
│   ├── data_file.py     # Parquet disk IO (with column projection/row selection)
│   ├── delta_log.py     # Delete records
│   └── manifest.py      # Global state
│
├── search/
│   ├── bitmap.py        # Global dedup → per-segment mask
│   ├── distance.py      # Distance computation
│   ├── executor.py      # Segmented search orchestration
│   └── merge.py         # ★ Hit + multi-way merge
│
├── engine/
│   ├── collection.py    # Core engine (manages Segment list)
│   ├── flush.py         # Flush pipeline (with Segment creation callback)
│   ├── recovery.py      # Crash recovery
│   └── compaction.py    # Compaction (with Segment replacement callback)
│
├── schema/              # Unchanged
├── db.py                # Unchanged
├── constants.py         # Unchanged
└── exceptions.py        # Unchanged
```

### 17.3 Updated Dependency Graph

```
Level 0:  constants.py, exceptions.py
Level 1:  schema/*
Level 2:  storage/wal, storage/memtable, storage/data_file,
          storage/delta_log, storage/manifest
Level 3:  storage/segment                         ← Depends on data_file + schema
Level 4:  search/bitmap, search/distance          ← Depends on segment + delta_log
Level 5:  search/merge                            ← Depends on Hit definition
Level 6:  search/executor                         ← Depends on L3-L5
Level 7:  engine/flush, recovery, compaction       ← Depends on L2-L6
Level 8:  engine/collection                        ← Depends on L2-L7
Level 9:  db.py                                    ← Depends on L8
```

---

## 18. Complete Design Decision Summary

### WAL Decisions (§1-§12)

| Decision | Choice | Rationale |
|------|------|------|
| File format | Arrow IPC Streaming | No write amplification for vectors, zero-copy parsing, built-in Schema |
| File structure | Dual-file (data + delta) | Different schemas, lazy initialization, symmetric with MemTable |
| Writer initialization | Lazy (created on first write) | Avoid empty files |
| fsync | MVP no fsync | Embedded scenario, process crash protected by OS cache |
| Truncation handling | Best-effort recovery (read as much as possible) | Maximize data recovery |
| Old WAL after Recovery | Retain, wait for flush cleanup | Safe against second crash |
| WAL numbering | Monotonically increasing, recovered from Manifest | No duplicates, traceable |
| close_and_delete | Idempotent | Safe for crash retry |
| Cleanup scope | Flush cleans all WALs with number <= frozen number | Unified handling of residual + frozen |

### Segment and Search Decisions (§13-§17)

| Decision | Choice | Rationale |
|------|------|------|
| Upsert implementation | Pure insert, _seq dedup | Single file write, no atomicity risk |
| Data caching | Segment resident in memory | Avoid disk reads on every search |
| Cache invalidation | Follows Manifest (no LRU) | Files are immutable, additions/removals are explicit |
| NumPy pre-extraction | One-time conversion at load | Zero conversion overhead on search hot path |
| Segment building during Flush | Reuse in-memory Table, don't re-read disk | Save one disk IO |
| Search model | Global dedup → segmented search → merge | Dedup must be global; search can be segmented in parallel + FAISS-compatible |
| MemTable search | Brute-force search (Growing Segment) | Small data volume, no index needed |
| Merge strategy | MVP concatenation + sort | Few segments, fast enough; can switch to k-way merge in the future |

### Dedup and Evolution Decisions (§19-§21)

| Decision | Choice | Rationale |
|------|------|------|
| delta_log.is_deleted comparison | `delete_seq > data_seq` (strictly greater than) | _seq monotonically increasing, precise semantics |
| Global dedup data source | Pre-extracted pk/seq arrays in Segment memory | Zero disk IO |
| Milvus-style optimization timing | Not introduced in MVP, gradually introduced in Phase 2 | Ensure correctness first, then optimize performance |
| Upsert atomicity (future) | Approach A single WAL file as the final form | RecordBatch-level atomicity, fundamental solution |
| Role of _seq dedup | Correctness safety net, always retained | Even with bitset optimization, remains as a fallback |
| Dedup timing (MVP) | Pre-dedup (before search) | Brute-force search requires precise top-k, cannot miss results |
| Dedup timing (FAISS) | Can switch to post-dedup | ANN is inherently approximate, over-fetch compensation is sufficient |
| PK uniqueness scope | Collection-level (cross-Partition) | _seq global dedup naturally achieves this, stronger guarantee than Milvus |

---

# Part III: Global Dedup Details and Evolution

---

## 19. Global Dedup Detailed Logic

### 19.1 Dedup Rules

```
For each PK:
  1. Find all occurrences of that PK across all Sealed Segments + MemTable
  2. Retain the one with the largest _seq (latest version), discard the rest
  3. Check delta_log for the retained version:
     If delete_seq > data_seq → deleted → filter out
```

### 19.2 delta_log.is_deleted Judgment Logic

```python
class DeltaLog:
    _deleted_map: Dict[Any, int]   # pk → max_delete_seq

    def is_deleted(self, pk, data_seq: int) -> bool:
        """Determine whether a data record has been deleted.

        Rule: A delete_seq exists and delete_seq > data_seq
              i.e., the delete operation occurred after the data was written.
        """
        if pk not in self._deleted_map:
            return False
        return self._deleted_map[pk] > data_seq
```

**Why `>` instead of `>=`?**

_seq is monotonically increasing, insert and delete will never get the same _seq:
- insert: each record gets an independent _seq (batch of N records = N different _seq values)
- delete: batch shares a single _seq, but it won't duplicate insert _seq values (comes from the same counter)

So `>` and `>=` are effectively equivalent, but `>` has more precise semantics:
"The delete operation's _seq is strictly greater than the data record's _seq" = "the delete happened after the write".

### 19.3 Step-by-Step Scenario Analysis

**Scenario 1: Normal upsert (cross-Segment dedup)**

```
Segment A (early flush):  doc_1  _seq=100  vec=[0.1, ...]
Segment B (later flush):  doc_1  _seq=500  vec=[0.9, ...]

best["doc_1"] = max(100, 500) = 500 → Segment B's version wins
delta_log.is_deleted("doc_1", 500) → "doc_1" not in deleted_map → False

seg_A_mask: doc_1 → False   ← Old version, skipped
seg_B_mask: doc_1 → True    ← Latest version, participates in search
```

**Scenario 2: Segment + MemTable dedup**

```
Segment A:  doc_1  _seq=100
MemTable:   doc_1  _seq=800   ← Recently upserted again

best["doc_1"] = max(100, 800) = 800 → MemTable wins

seg_A_mask: doc_1 → False
mem_mask:   doc_1 → True
```

**Scenario 3: Delete first, then insert (delete + re-insert)**

```
Timeline:
  T1: insert doc_1 → _seq=100 → flushed to Segment A
  T2: delete doc_1 → _seq=300 → delta_log: {doc_1: 300}
  T3: insert doc_1 → _seq=500 → flushed to Segment B

Dedup: best["doc_1"] = max(100, 500) = 500 → Segment B

Delete check:
  delta_log._deleted_map = {"doc_1": 300}
  is_deleted("doc_1", data_seq=500)
  → 300 > 500?  → False → Not deleted ✅

Result: doc_1 is visible, using Segment B's version
      (deletion occurred before re-insertion, does not affect the new version)
```

**Scenario 4: Upsert first, then delete**

```
Timeline:
  T1: insert doc_1 → _seq=100 → Segment A
  T2: insert doc_1 → _seq=500 → Segment B (upsert)
  T3: delete doc_1 → _seq=700 → delta_log: {doc_1: 700}

Dedup: best["doc_1"] = max(100, 500) = 500 → Segment B

Delete check:
  is_deleted("doc_1", data_seq=500)
  → 700 > 500?  → True → Deleted

Result: doc_1 is not visible (Segment A's old version was eliminated during dedup phase,
      Segment B's new version was filtered out during delete check phase)
```

**Scenario 5: Delete in MemTable covering Segment data**

```
Segment A:  doc_1  _seq=100  (old data on disk)
MemTable delete_buf:  delete doc_1, _seq=200

delta_log._deleted_map (includes MemTable delete_buf and on-disk delta files):
  {"doc_1": 200}

Dedup: best["doc_1"] = 100 → Segment A (only insert version)

Delete check:
  is_deleted("doc_1", 100) → 200 > 100 → True → Deleted

Result: doc_1 is not visible ✅
```

**Scenario 6: Multiple deletes, retain max delete_seq**

```
Timeline:
  T1: insert doc_1 → _seq=100
  T2: delete doc_1 → _seq=200
  T3: insert doc_1 → _seq=300   ← re-insert
  T4: delete doc_1 → _seq=400   ← delete again

delta_log._deleted_map = {"doc_1": 400}  ← Retains max(200, 400) = 400

Dedup: best["doc_1"] = 300
Delete check: is_deleted("doc_1", 300) → 400 > 300 → True
Result: doc_1 is not visible ✅
```

### 19.4 An Important Property: PK Uniqueness Within a Segment

Within a single Segment, each PK appears only once, because:
- **MemTable flush**: insert_buf is a dict (deduped by PK), output Table has only one row per PK
- **Compaction merge**: Same PK retains max_seq, output is unique

Therefore global dedup only handles **cross-Segment duplicates**, no need to handle intra-segment duplicates.
The size of the `best` dict = number of unique PKs (≤ total rows), no extra bloat.

### 19.5 Algorithm Complexity

```
Phase 1 (collect max_seq):
  Traverse all Segments' pks + seqs arrays → O(N)  N = total rows
  dict lookup/update → O(1) per entry

Phase 2 (generate mask):
  Traverse best dict → O(U)  U = number of unique PKs
  delta_log.is_deleted() → O(1) per PK (dict lookup)

Total: O(N) time, O(U) space
```

### 19.6 Source of Global Information

No extra query mechanism needed during search -- **all data is already in memory**:

```
Collection._segments = {
    "_default": [seg_A, seg_B, seg_C],     ← Segment objects resident in memory
}
Collection.memtable                         ← In memory
Collection.delta_log._deleted_map           ← In memory

Each Segment pre-extracted NumPy arrays at load time:
  seg_A.pks  = np.array(["doc_1", "doc_2", ...])
  seg_A.seqs = np.array([100, 101, ...])

Global dedup = traverse these in-memory arrays + query delta_log's in-memory dict
         Zero disk IO throughout
```

---

## 20. Milvus Dedup Approach Comparison

### 20.1 Milvus's Approach: Explicit Deletion at Write Time

Milvus shifts the dedup cost from search time to write time:

```
Milvus upsert("doc_1", new_vec):
  → Step 1: delete("doc_1")        ← Explicitly delete all old versions first
  → Step 2: insert("doc_1", ...)   ← Then insert new version
  → At search time: each Segment only checks its own bitset, no cross-Segment dedup needed
```

### 20.2 Key Mechanism: Bloom Filter

Milvus maintains a Bloom Filter for each Sealed Segment, recording which PKs the Segment contains:

```
Segment A: bloom_filter_A = {fingerprints of doc_1, doc_2, doc_3}
Segment B: bloom_filter_B = {fingerprints of doc_4, doc_5}
Segment C: bloom_filter_C = {fingerprints of doc_1, doc_6}
```

When delete("doc_1") arrives:

```
Check each Segment's Bloom Filter:
  bloom_filter_A.might_contain("doc_1") → True   → Mark doc_1 in A as deleted
  bloom_filter_B.might_contain("doc_1") → False  → Skip (doc_1 definitely not in B)
  bloom_filter_C.might_contain("doc_1") → True   → Mark doc_1 in C as deleted

Complexity: O(number of Segments), each Bloom Filter query O(1)
```

Bloom Filter characteristics:
- Says "not present" → **definitely not present** (no false negatives)
- Says "might be present" → **could be a false positive** (but harmless -- just records one extra invalid deletion)

### 20.3 Per-Segment Bitset

Each Segment has its own bitset, search only checks the local Segment's bitset:

```
Segment A (1000 rows):
  bitset = [1,1,0,1,1,0,1,...]     ← 0 = deleted/expired, 1 = valid

When searching Segment A:
  index.search(query, bitset=bitset)   ← FAISS directly skips rows where bitset=0
  No need to know what other Segments have ← Fundamental difference from our approach
```

### 20.4 Full Comparison of Two Approaches

```
                        Our Design                      Milvus
                        (Implicit dedup)                (Explicit deletion)
                        ─────────                       ─────────

upsert implementation   Only write wal_data             delete(old) + insert(new)
Write complexity        O(1)                            O(S) S=number of Segments (Bloom Filter queries)
Extra write structures  None                            Each Segment maintains Bloom Filter
Search-time dedup       Global scan O(N)                Not needed (old versions explicitly deleted)
                        N=total rows                    O(1) per row (only check local bitset)
Compaction              Clean old versions + deleted     Clean deleted
Correctness guarantee   _seq comparison                  Bloom Filter + explicit delete log
```

Core trade-off:

```
Ours: Simple writes O(1), search cost O(N)
      → Suitable for write-heavy, read-light, small data embedded scenarios

Milvus: Write cost O(S), search cost O(1) per row
        → Suitable for read-heavy, write-light, large data production environments
```

### 20.5 Milvus Search Flow vs Our Search Flow

```
Milvus search:                              Our search:
  │                                           │
  ├─ for each segment:                        ├─ Global scan of all seg.pks + seg.seqs
  │    ├─ Read this segment's bitset          │   Build best = {pk → max_seq}
  │    │   (already contains delete+dedup     │   O(N)
  │    │    info, no cross-segment query      │
  │    │    needed)                            │
  │    ├─ index.search(query, bitset)         ├─ Generate per-segment masks
  │    └─ Return local top-k                  │
  │                                           ├─ for each segment:
  ├─ merge local top-k → global top-k        │    seg.search(query, mask, top_k)
  │                                           │
  └─ done                                    ├─ merge → global top-k
                                              └─ done

  Zero global coordination during search       Global dedup scan needed during search
```

---

## 21. Future Evolution: Upsert Atomicity Problem and Solutions

### 21.1 Problem: Explicit Deletion Introduces Dual-File Writes

Once Milvus-style upsert = delete + insert is adopted, a single upsert needs to write both wal_data and wal_delta simultaneously:

```
upsert("doc_1", new_vec)
  │
  ├─ wal_delta.write_delete(doc_1)     ← Delete old version
  │                               ← ⚡ Crash
  └─ wal_data.write_insert(doc_1)      ← Insert new version
```

**Crash between the two steps: delete written but insert not written → doc_1 is deleted, new version lost → data lost.**

Reversed order, insert first then delete:

```
  ├─ wal_data.write_insert(doc_1, _seq=500)    ← Write insert first
  │                                       ← ⚡ Crash
  └─ wal_delta.write_delete(doc_1)              ← Then write delete

Crash: insert exists, delete doesn't
→ Old version (_seq=100) not explicitly deleted
→ New version (_seq=500) exists
→ Same PK appears twice (if relying only on per-segment bitset without _seq dedup)
```

**Regardless of order, dual-file writes cannot guarantee atomicity.**

### 21.2 Approach A: Merge into Single WAL File (Fundamental Solution)

Merge wal_data and wal_delta into a single WAL file, using an `_op` column to distinguish operation types.

**Merged WAL schema**:

```
_op:        utf8          "INSERT" | "DELETE"
_seq:       uint64
_partition: utf8
{pk_field}: ...
{vec_field}: list<f32>    (DELETE rows filled with null)
{other fields}: ...       (DELETE rows filled with null)
$meta:      utf8          (DELETE rows filled with null)
```

**Upsert writes one RecordBatch** containing two rows:

```
One write_batch() call, two rows in the same RecordBatch:

  Row 0: _op="DELETE", _seq=499, pk="doc_1", vec=null,       ...
  Row 1: _op="INSERT", _seq=500, pk="doc_1", vec=[0.9,...],  ...
```

**Atomicity guarantee**:

```
Arrow IPC write_batch() either writes a complete RecordBatch, or truncates and discards it.
Two rows in the same RecordBatch → live and die together:
  Write succeeds → both delete + insert are present
  Crash during write → recovery discards incomplete batch → neither is present
  ✅ RecordBatch-level atomicity
```

During Recovery, split by `_op` column:

```python
for batch in wal_batches:
    for row in range(batch.num_rows):
        op = batch.column("_op")[row].as_py()
        if op == "INSERT":
            memtable.put(seq, partition, **fields)
        elif op == "DELETE":
            memtable.delete(pk, seq, partition)
            # + Update related Segment bitsets via Bloom Filter
```

### 21.3 Approach B: Keep Dual Files + Write Order + _seq Fallback (Incremental Approach)

Keep the dual-file structure unchanged, use _seq as the ultimate correctness guarantee:

```
Write order: insert first, delete second

upsert("doc_1", new_vec):
  ├─ Step A: wal_data.write_insert(doc_1, _seq=500)    ← Write insert first
  └─ Step B: wal_delta.write_delete(doc_1, _seq=499)   ← Then write delete

Crash between A and B:
  insert exists, delete doesn't
  → Old version (_seq=100) not explicitly deleted
  → New version (_seq=500) exists
  → But _seq dedup still works: same PK keeps max_seq → 500 > 100 → new version wins ✅
```

**Essence**: Bloom Filter + per-segment bitset is the **performance optimization layer**, _seq dedup is the **correctness guarantee layer**. The two layers work together:

```
Normal case (99.9%):
  bitset fast filtering (O(1) per row) → no global scan needed

Abnormal case (crash causes incomplete bitset):
  _seq dedup as fallback → correctness unaffected
  Compaction repairs bitset → eventually returns to normal state
```

### 21.4 Approach C: Recovery Compensation (Enhancement of Approach B)

On top of Approach B, Recovery proactively detects and completes incomplete upserts:

```python
def _fix_incomplete_upserts(wal_data_batches, wal_delta_batches, segments):
    """Detect cases where insert exists but corresponding delete is missing, issue compensating delete."""

    # Collect PKs inserted via WAL
    inserted_pks = set()
    for batch in wal_data_batches:
        for row in range(batch.num_rows):
            inserted_pks.add(batch.column(pk_field)[row].as_py())

    # Collect PKs deleted via WAL
    deleted_pks = set()
    for batch in wal_delta_batches:
        for row in range(batch.num_rows):
            deleted_pks.add(batch.column(pk_field)[row].as_py())

    # PKs that were inserted but not deleted → possibly incomplete upsert due to crash
    missing_deletes = inserted_pks - deleted_pks

    for pk in missing_deletes:
        for seg in segments:
            if seg.bloom_filter.might_contain(pk):
                seg.mark_deleted(pk)    # Complete the per-segment bitset
```

### 21.5 Three Approaches Comparison

| | Approach A: Single WAL File | Approach B: Dual Files + _seq Fallback | Approach C: Recovery Compensation |
|--|-------------------|------------------------|-------------------|
| Atomicity | RecordBatch-level atomic | Not atomic, relies on _seq fallback | Not atomic, relies on Recovery repair |
| WAL changes | Large (merge to single file, schema adds _op + nullable) | None | None |
| Search correctness | Perfect | _seq fallback guarantee (occasionally needs global dedup) | Perfect after Recovery |
| Complexity | Schema changes | Search must retain _seq dedup path | Recovery adds detection logic |
| Performance | DELETE rows' null fields waste minimal space | Search degrades in abnormal cases | No extra search overhead |

### 21.6 Recommended Evolution Path

```
MVP (current design)
  upsert = pure insert, _seq implicit dedup
  Dual WAL files, no atomicity issue (each API only writes single file)
  Global scan dedup at search time, O(N)
  ✅ Simple and correct

Phase 2 (add Bloom Filter + bitset to optimize search performance)
  Adopt Approach B: upsert = insert first then delete, _seq fallback
  Normal case: bitset fast filtering; abnormal case: _seq fallback
  Optional Approach C for Recovery compensation
  ✅ Gradual optimization, doesn't break existing correctness

Phase 3 (pursue perfect atomicity)
  Adopt Approach A: merge into single WAL file + _op column
  RecordBatch-level atomicity, completely eliminates inconsistency window
  Can safely reduce weight of _seq global dedup (still retained as defensive check)
  ✅ Final form
```

### 21.7 Role of _seq Dedup

**_seq dedup is the system's safety net, present across all evolution stages, and should not be removed.**

```
Even when Phase 3 achieves perfect single WAL atomicity + Bloom Filter + per-segment bitset:
  → Still retain _seq dedup as a defensive check
  → Guards against Bloom Filter false positives + code bugs + unforeseen edge cases
  → Very low cost (already have _seq column, comparison operation is O(1))
  → Very high benefit (last line of defense for correctness)
```

---

## 22. Pre-Dedup vs Post-Dedup

### 22.1 Post-Dedup Approach

Instead of doing global dedup before search, filter after search:

```
Pre-dedup (current approach):
  Global scan PK+_seq → generate mask → only search valid rows → merge

Post-dedup (alternative approach):
  Each Segment searches independently (including expired data) → merge → dedup → take top-k
```

Post-dedup is simpler -- each Segment is completely independent, no global scan needed before search.

### 22.2 Problem: Expired Data Occupies Local Top-k Slots

```
Scenario: top_k = 3

Segment A (old):
  doc_1  _seq=100  vec=[0.1,...]  dist=0.05  ← Expired version, very close to query
  doc_2                           dist=0.30
  doc_7                           dist=0.31
  doc_4                           dist=0.32  ← 4th place
  ...

Segment B (new):
  doc_1  _seq=500  vec=[0.9,...]  dist=0.80  ← Latest version, far from query
  doc_3                           dist=0.40
```

**Pre-dedup (correct result)**:

```
Dedup: doc_1 → Segment B wins (_seq=500)
Segment A valid row search: doc_2(0.30), doc_7(0.31), doc_4(0.32) → local top-3
Segment B valid row search: doc_1(0.80), doc_3(0.40) → local top-2

Merge top-3: [doc_2(0.30), doc_7(0.31), doc_4(0.32)]  ✅
```

**Post-dedup (missing results)**:

```
Segment A searches all (including expired doc_1):
  local top-3: [doc_1(0.05), doc_2(0.30), doc_7(0.31)]
                  ↑ Expired data took a slot, doc_4(0.32) ranked 4th and was truncated

Segment B:
  local top-3: [doc_3(0.40), doc_1(0.80)]

Merge: [doc_1(A,0.05), doc_2(0.30), doc_7(0.31), doc_3(0.40), doc_1(B,0.80)]
Dedup: doc_1 retains B version(0.80), discards A version(0.05)
Result: [doc_2(0.30), doc_7(0.31), doc_3(0.40)]

Correct answer: [doc_2(0.30), doc_7(0.31), doc_4(0.32)]
                                       ↑ doc_4 is missing!
```

**Root cause**: The expired doc_1 (dist=0.05) displaced doc_4 (dist=0.32) in Segment A's local top-3.
After dedup, doc_1 is discarded, but doc_4 was already lost during local top-k truncation and cannot be recovered.

### 22.3 Mitigation: Over-Fetch

Each Segment takes `top_k × factor` instead of `top_k`, leaving headroom for dedup:

```
factor = 2: Segment A takes top-6 instead of top-3
  → [doc_1(0.05), doc_2(0.30), doc_7(0.31), doc_4(0.32), ...]
  → doc_4 is preserved ✅

Merge → dedup → take top-3
  → [doc_2(0.30), doc_7(0.31), doc_4(0.32)] ✅
```

**But how large should the factor be?**

```
Depends on "how many PKs are duplicated across Segments":
  Just flushed, almost no duplicates        → factor=1.1 is enough
  Many upserts, lots of duplicates          → Need factor=3, 5, or even larger
  Extreme case: all PKs have been upserted  → Need factor≥2
  Cannot know in advance → can only guess, wrong guess means missing results
```

### 22.4 Two Approaches Comparison

| | Pre-dedup | Post-dedup + over-fetch |
|--|--------|---------------------|
| Correctness | Precise (no missing results) | May miss results (when factor isn't large enough) |
| Implementation complexity | Requires global scan of PK+_seq | Each segment independent, simpler |
| Distance computation volume | Only compute for valid rows | Expired rows also computed (waste) |
| Applicable scenario | Brute-force search (requires precision) | ANN approximate search (inherently imprecise) |
| Distributed friendliness | Poor (requires global coordination) | Good (each segment independent) |

### 22.5 Conclusion: Depends on Search Precision Requirements

```
Brute-force search (MVP) → must use pre-dedup
  Users expect precise top-k, missing results is unacceptable
  All data is in memory, O(N) global scan cost is acceptable

FAISS ANN (future) → post-dedup is acceptable
  ANN is inherently approximate, missing one borderline result has little impact
  Each segment has an independent FAISS index, pre-dedup provides less benefit
    (FAISS search complexity doesn't grow linearly with data volume, filtering a few expired records saves little)
  over-fetch factor=2 is sufficient for most scenarios
  More natural for distributed search (segments can be on different nodes)
```

### 22.6 Relationship with Evolution Path

```
MVP:     Pre-dedup (precise) + brute-force search
Phase 2: Pre-dedup (precise) + Bloom Filter optimization on write side
Phase 3: Can optionally switch to post-dedup + FAISS ANN
         _seq dedup still serves as safety net at this point:
         Even if post-dedup misses borderline results (precision issue),
         _seq mechanism at least guarantees no incorrect content from expired data is returned
```

---

## 23. Milvus Cross-Partition Duplicate PK Investigation

### 23.1 Milvus's Behavior: No PK Uniqueness Check at All

Milvus official documentation clearly states:

> "Milvus does not support primary key de-duplication for now. Therefore, there can be duplicate primary keys in the same collection."

Writing the same PK to two different Partitions, **both records will be saved**, no error, no overwrite.

### 23.2 Read Behavior (Inconsistent)

| Operation | Behavior | Issue |
|------|------|------|
| query(pk=X) | Returns **the earliest inserted one** (post-reduce dedup, retains first found) | Not necessarily the latest version |
| search(ANN) | **May return multiple results with the same PK**, takes the most vector-similar one | Semantics unclear |
| count(\*) | **Includes all duplicates**, inserting a PK twice means count=2 | Inaccurate count |
| Limit=10 | Takes 10 first then dedup → actual return may be **< 10** | Unreliable pagination |

### 23.3 Upsert Also Doesn't Cross Partitions

Milvus's `upsert()` is **Partition-level** delete + insert:

```
Partition A has PK=1
Call upsert(PK=1, partition=B)

→ The record in Partition A will NOT be deleted
→ Result: Both Partition A and B each have one record with PK=1
```

To clean across Partitions, you must manually `delete(pk=1, partition_name=None)` (None means scan all Partitions).

### 23.4 Compaction Also Doesn't Cross Partitions

Compaction merges Segments within a Partition, **does not handle duplicate PKs across Partitions**.

### 23.5 Milvus's Official Stance

- Issue #36199 marked as **"resolution/by-design"** -- acknowledged as known behavior, not considered a bug
- **Global PK Dedup** added to Roadmap, targeting v2.6 / v3.0 (not yet implemented)
- Current official recommendation: Use `auto_id=True` to avoid the problem, or deduplicate at the application layer

Related Issue Summary:

| Issue / PR | Title | Key Points |
|------------|------|------|
| Issue #36199 | Duplicate primary key values leads to inconsistent query results | Marked by-design; query returns earliest version, search returns most similar version |
| Issue #28615 | Enhance constraints for inserting duplicate primary key data | Limit parameter gets fewer results after dedup |
| Issue #31552 | Support primary key dedup and vector dedup when insert | Feature request, assigned to milestone 3.0 |
| Issue #37389 | Does Milvus actually perform deduplication? | Only upsert deduplicates, insert does not |
| Issue #33353 | Insert data with duplicate primary key | Silently accepts duplicate PKs |
| Discussion #18202 | The Scope of Primary Key | PK scope is Collection-level, but uniqueness not guaranteed |
| Discussion #18201 | What happens if I insert data with the same id several times? | query returns the first one; search may return multiple with same PK |
| PR #10967 | Remove primary key duplicated query result on proxy | Proxy-layer post-reduce dedup implementation |

### 23.6 Design Implications for MilvusLite

**Three choices for PK uniqueness scope**:

| Approach | PK Uniqueness Scope | Pros | Cons |
|------|-------------|------|------|
| A: Collection-level unique | Across all Partitions | Clear semantics, consistent search/query results | Requires global check during insert (or _seq dedup fallback) |
| B: Partition-level unique | Within Partition only | Simple implementation, Partitions fully isolated | Same problems as Milvus: duplicates in cross-Partition queries |
| C: No guarantee (same as Milvus) | None | Simplest | Unpredictable results, inaccurate counts |

**We choose Approach A, and it's already naturally implemented**:

```
MilvusLite's _seq global dedup mechanism naturally provides Collection-level PK uniqueness:

1. No extra check needed during insert
   → Write directly, don't query old Partitions

2. During search, build_segment_masks globally scans Segments across all Partitions
   → Traverses all seg.pks + seg.seqs
   → Only retains max_seq → same PK across Partitions automatically takes latest version
   → Expired versions automatically filtered, regardless of which Partition they're in

3. delete(partition_name=None) is already designed for all-Partition scan
   → Cross-Partition deletion is natively supported

4. get(pks, partition_names=None) similarly
   → Global _seq dedup, returns latest version
```

**This is a stronger semantic guarantee than Milvus, with no extra overhead**:

```
                        Milvus                     MilvusLite
                        ────────                   ──────────
PK uniqueness scope     Not guaranteed              Collection-level (cross-Partition)
upsert across Partitions Does not handle (only      _seq dedup handles automatically
                        delete+insert within
                        target Partition)
search across Partitions May return duplicate PKs    Global dedup, only returns latest version per PK
                        query/search behavior
                        inconsistent
count accuracy          Inaccurate (includes dupes)  Accurate (counts after dedup)
Extra implementation    Needs Global PK Dedup        None (_seq naturally supports)
overhead                (Roadmap v3.0)
```

**But there is a trade-off**:

```
Cost: Global scan O(N) for dedup during search
     → Acceptable for MVP (brute-force search is already O(N), dedup scan is negligible in comparison)
     → After Phase 2 introduces Bloom Filter + bitset, dedup can be moved from search hot path to write side
```
