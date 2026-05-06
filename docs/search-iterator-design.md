# Search Iterator Design Document

## 1. Background and Goals

pymilvus provides two SearchIterator implementations:

- **V1 (client-side)**: Adaptive pagination based on distance ranges. The client repeatedly calls Search RPC with `radius` + `range_filter` parameters, progressively expanding the search range, using PK exclusion for deduplication.
- **V2 (server-side)**: Token-based pagination. The server executes a search with `top_k=batch_size` each time, using the `last_bound` distance threshold to skip already-returned results, achieving batch-by-batch progression.

### Current Status

MilvusLite already supports all the underlying capabilities that V1 depends on:
- ✅ Range Search (`radius` + `range_filter`) — Phase 14
- ✅ Offset pagination — Phase 17
- ✅ PK `not in [...]` filter expressions — Phase 8
- ✅ `search_iterator` tests in test_iterator.py have passed

**Conclusion: V1 already works.** pymilvus's SearchIterator V1 is a pure client-side implementation. As long as the Search RPC correctly supports `radius`, `range_filter`, `offset`, and filter expressions, V1 can run.

### Design Goals

Implement **V2 server-side iterator**, for the following reasons:
1. V2 is the default path in pymilvus (it tries V2 first, falling back to V1 on failure)
2. V2 avoids V1's multiple adaptive retries, resulting in lower latency
3. V2 avoids the overhead of the client constructing PK exclusion expressions (which can be very long for large batches)
4. V1 may enter an infinite loop when there are many records with the same `distance` (the PK exclusion list explodes)

---

## 2. Milvus Implementation Analysis

### 2.1 Milvus V2 Actual Architecture

Milvus's V2 is **not** a full search + cache approach, but rather **batch-by-batch search + last_bound distance filtering**:

1. Each request performs a real search with `topK = batch_size`
2. The `CachedSearchIterator` at the C++ segcore layer caches the **vector index iterator state** (HNSW graph traversal position), not search results
3. Uses `last_bound` (the distance value of the last entry from the previous batch) to filter out results where `distance ≤ last_bound`
4. The token does not correspond to server-side state storage; the real "cursor" is the `last_bound` value

```
Round 1: Search(batch_size=100, last_bound=None)
  → Returns the nearest 100 entries + last_bound=0.85

Round 2: Search(batch_size=100, last_bound=0.85)
  → Skips distance ≤ 0.85, returns next batch + last_bound=1.23

Round 3: Search(batch_size=100, last_bound=1.23)
  → Skips distance ≤ 1.23, returns next batch...

Empty result → Client terminates
```

### 2.2 Why Milvus Does Not Use Full Caching

Full caching means setting `top_k` to a very large value (e.g., 16384), which for ANN indexes like HNSW is equivalent to degrading to brute-force search, completely wasting the benefit of index acceleration. Batch-by-batch search fetches only `batch_size` entries each time, preserving the early termination advantage of the index.

### 2.3 pymilvus V2 Protocol

**Handshake flow**:

```
Client                                    Server
  |                                          |
  |-- Search(batch_size=1, iter_v2=true) --> |   # probe call
  |<--- SearchResults + {token, last_bound} -|   # returns token
  |                                          |
  |-- Search(token=T, last_bound=B) -------> |   # subsequent calls
  |<--- SearchResults + {token, last_bound} -|
  |                                          |
  |       ... repeat until results are empty ...  |
```

**Key parameters** (embedded in search_params KV):

| Key | Type | Description |
|-----|------|-------------|
| `iterator` | bool | Marks as iterator mode |
| `search_iter_v2` | bool | Uses V2 protocol |
| `search_iter_batch_size` | int | Number of results per batch |
| `search_iter_id` | str | Iterator token (absent on first request) |
| `search_iter_last_bound` | float | Distance value of the last entry from the previous batch |
| `guarantee_timestamp` | int | MVCC timestamp (ensures consistent snapshot) |

**SearchResultData extended fields**:

```protobuf
message SearchIteratorV2Results {
    string token = 1;       // Iterator unique identifier
    float last_bound = 2;   // Distance value of the last entry in this batch
}

message SearchResultData {
    ...
    optional SearchIteratorV2Results search_iterator_v2_results = 11;
}
```

**pymilvus client behavior**:

1. **Probe**: First call with `batch_size=1`, obtains token + last_bound
2. **Iterate**: Subsequent calls carry `token` + `last_bound`, `limit = batch_size`
3. **Terminate**: Stops when empty results are returned
4. **Timestamp**: Obtains `session_ts` from the first response, subsequent requests carry `guarantee_timestamp` to ensure snapshot consistency

**Constraints**:
- Only supports nq=1 (single-vector query)
- Cannot be combined with groupBy / offset / order_by

---

## 3. MilvusLite Design

### 3.1 Core Idea

Align with Milvus: **batch-by-batch search + last_bound distance filtering**.

Each iteration request executes a `Collection.search(top_k=batch_size)`, filtering out already-returned results through the `last_bound` parameter (`distance ≤ last_bound`), achieving batch-by-batch progression.

```
pymilvus SearchIteratorV2
    │
    ▼
gRPC Search RPC (with iterator parameters)
    │
    ▼
MilvusServicer.Search()
    │  Detects search_iter_v2=true
    ▼
Collection.search(top_k=batch_size, last_bound=B)
    │  Filters results with distance ≤ last_bound during search
    ▼
Returns SearchResults + SearchIteratorV2Results{token, last_bound}
```

### 3.2 Core Design Decisions

#### Decision 1: Batch-by-Batch Search + last_bound Filtering (Aligned with Milvus)

**Approach**: Each iteration request executes `search(top_k=batch_size)`, and the engine layer filters candidates with `distance ≤ last_bound` after distance computation.

**Rationale**:
- Preserves HNSW index's early termination advantage, avoids degradation to brute-force search
- Only computes distances for batch_size results per batch, low memory overhead
- No need to manage server-side cache and its lifecycle
- Consistent with Milvus behavior

#### Decision 2: last_bound Filtering Implemented at the Executor Layer

The filtering happens after search results are sorted but before returning — entries with `distance ≤ last_bound` are removed.

**Distance conventions** (internally unified as "smaller = more similar"):
- `COSINE`: `1 - similarity`, range [0, 2], smaller means more similar
- `L2`: Euclidean distance, smaller means more similar
- `IP`: `-dot_product` (internally negated), smaller means more similar

The filtering rule is unified as: **skip results with `distance ≤ last_bound`** (since internal distances are unified as "small = similar", results before last_bound have smaller distances and should be skipped).

Note: The `last_bound` returned to pymilvus is the **external distance** (IP distance is negated back to positive), which differs in sign from the internal distance. Conversion is needed at the adapter layer.

#### Decision 3: Token Is a Stateless Identifier

Consistent with Milvus, the token does not correspond to server-side state. A `uuid4` is generated on the first request and returned to the client, which passes it back unchanged in subsequent requests. The real cursor is the `last_bound` value.

Stateless server = no need to manage TTL / cache cleanup / memory leaks.

#### Decision 4: Snapshot Isolation Based on `_seq`

**Problem**: Each `.next()` call of the iterator is an independent `Collection.search()`. If there are insert/delete operations between two calls, results become inconsistent:

```
iterator.next()  → returns batch 1 (distance 0.1~0.5)
    ↓
collection.insert(new_record)  # new record with distance=0.3
    ↓
iterator.next()  → returns batch 2 (distance > 0.5)
                   new record distance=0.3 ≤ last_bound, filtered → permanently lost
```

**Approach**: Use `_seq` (globally monotonically increasing sequence number) for snapshot isolation, aligned with Milvus's `session_ts` mechanism.

This is essentially **MVCC (Multi-Version Concurrency Control)**:
- `_seq` = version number (each record carries one, monotonically increasing)
- `snapshot_seq` = upper bound of the read snapshot's version (equivalent to classic MVCC's `read_ts`)
- `seq_mask = (seqs <= snapshot_seq)` = version visibility check

MilvusLite's `_seq` mechanism inherently has MVCC capabilities. Previously it was only used for dedup and tombstone determination; here it is extended for snapshot reads.

Based on this underlying capability, Consistency Levels can theoretically be implemented:
- **Strong**: `snapshot_seq = current_seq` (the current default behavior, naturally satisfied by single-process synchronous execution)
- **Session**: `snapshot_seq = the _seq of the last write in this session`
- **Eventually**: `snapshot_seq = the _seq at the last flush` (reads only persisted data)

However, for a single-process embedded architecture, Strong is natural, and other levels provide no practical benefit. Therefore **Consistency Levels are not exposed as a user-facing feature**. `snapshot_seq` is only used internally within Search Iterator.

**Flow**:

1. **First iteration request**: Capture the current `Collection._seq` as `snapshot_seq`
2. **Return `session_ts`**: Return `snapshot_seq` via the `session_ts` field of SearchResults
3. **pymilvus auto-passes it back**: Subsequent requests carry `guarantee_timestamp = snapshot_seq`
4. **Subsequent searches**: Add `seq_mask = (seqs <= snapshot_seq)` in the bitmap pipeline, merged into valid_mask

**Implementation location**: In `build_valid_mask()` of `search/bitmap.py`, after the existing dedup + tombstone + scalar filter, append seq filtering:

```python
if snapshot_seq is not None:
    seq_mask = (all_seqs <= snapshot_seq)
    valid_mask &= seq_mask
```

This way, records inserted after the snapshot (`_seq > snapshot_seq`) and delete operations (`delete_seq > snapshot_seq`) are invisible, ensuring a consistent data view throughout the iteration process.

**Advantages**:
- Zero additional storage — `_seq` already exists in every record, MVCC is "free"
- Zero server-side state — `snapshot_seq` is passed back by the client via `guarantee_timestamp`
- Reuses pymilvus's existing `session_ts` / `guarantee_timestamp` protocol fields
- No impact on non-iterator requests (`snapshot_seq=None` skips filtering)

#### Decision 5: Handling Records with Equal Distances

When multiple records have exactly the same distance, `last_bound` may cause some equal-distance records to be skipped. Handling strategies:

- Using **strict less than** `distance < last_bound` may cause duplicates
- Using **less than or equal** `distance ≤ last_bound` may cause omissions

Milvus uses `distance > last_bound` (strict greater than, equivalent to skipping ≤), accepting equal-distance omissions. We align with this behavior. Rationale:
- Vector search itself is approximate (ANN), minor equal-distance omissions are acceptable
- Better than duplicates — duplicates would complicate client-side logic

---

## 4. Implementation Details

### 4.1 Engine Layer Changes: Adding Iterator Parameters to Collection.search

```python
def search(
    self,
    query_vectors,
    top_k=10,
    metric_type="COSINE",
    ...,
    last_bound=None,       # New: iterator distance threshold
    snapshot_seq=None,      # New: snapshot sequence number (iterator consistency)
) -> List[List[dict]]:
```

**last_bound filtering** — after search results are sorted, before top_k truncation:

```python
# At the executor layer, after top-k selection:
if last_bound is not None:
    hits = [h for h in hits if h["distance"] > last_bound]
```

**snapshot_seq filtering** — in the bitmap pipeline:

```python
# In build_valid_mask() of search/bitmap.py:
if snapshot_seq is not None:
    seq_mask = (all_seqs <= snapshot_seq)
    valid_mask &= seq_mask
```

### 4.2 Distance Conversion

The `last_bound` passed in by pymilvus is the **external distance** (the value returned to the user). The engine internally uses a unified convention (smaller = more similar). Conversion is needed at the adapter layer:

| metric | External distance | Internal distance | Conversion |
|--------|-------------------|-------------------|------------|
| COSINE | `1 - sim` | `1 - sim` | No conversion needed |
| L2 | `euclidean` | `euclidean` | No conversion needed |
| IP | `dot_product` (positive) | `-dot_product` (negative) | `internal = -external` |

Reverse conversion when returning `last_bound`.

### 4.3 Filtering Implementation Location

In the global merge stage of `search/executor_indexed.py`, top-k results are already sorted by distance. Append one filtering step:

```python
def _apply_last_bound(hits: List[dict], last_bound: float) -> List[dict]:
    """Filter out results with distance ≤ last_bound (already in internal distance space)."""
    return [h for h in hits if h["distance"] > last_bound]
```

Note: This filtering happens **after** top-k selection. Therefore the actual flow is:
1. Normal search with `top_k = batch_size + margin` (over-fetch to compensate for filtered entries)
2. Filter `distance ≤ last_bound`
3. Truncate to the first `batch_size` entries

Over-fetch margin: Milvus's C++ layer has the index iterator naturally skip results ≤ last_bound (skipped directly during HNSW graph traversal). MilvusLite's FAISS does not support this customization, so over-fetch + post-filter is used instead. The over-fetch multiplier is set to 2x (i.e., `top_k = batch_size * 2`). If fewer than batch_size entries remain after filtering, no retry is performed — the actual count is returned, and the client decides whether to continue.

### 4.4 Servicer Changes

```python
def Search(self, request, context):
    # ... existing parsing logic ...
    parsed = parse_search_request(request, ...)

    # ── New: detect V2 iterator mode ──
    if parsed.get("search_iter_v2"):
        return self._handle_search_iterator_v2(request, col, parsed)

    # ── Original search logic ──
    results = col.search(...)
    ...

def _handle_search_iterator_v2(self, request, col, parsed):
    batch_size = parsed.get("search_iter_batch_size", 1000)
    last_bound_external = parsed.get("search_iter_last_bound")  # external distance
    token = parsed.get("search_iter_id")
    guarantee_ts = parsed.get("guarantee_timestamp", 0)

    # First request: generate token, capture snapshot _seq
    is_first = (token is None)
    if is_first:
        import uuid
        token = str(uuid.uuid4())
        snapshot_seq = col._seq          # Capture current sequence number as snapshot
    else:
        snapshot_seq = guarantee_ts if guarantee_ts > 0 else None

    # Convert last_bound to internal distance space
    metric = parsed["metric_type"]
    last_bound_internal = None
    if last_bound_external is not None:
        if metric == "IP":
            last_bound_internal = -last_bound_external
        else:
            last_bound_internal = last_bound_external

    # Execute search (over-fetch 2x to compensate for last_bound filtering)
    results = col.search(
        query_vectors=parsed["query_vectors"],
        top_k=batch_size * 2,
        metric_type=metric,
        partition_names=parsed["partition_names"],
        expr=parsed["expr"],
        output_fields=parsed["output_fields"],
        anns_field=parsed.get("anns_field"),
        last_bound=last_bound_internal,
        snapshot_seq=snapshot_seq,        # Snapshot isolation
    )

    # Truncate to batch_size
    for i, hits in enumerate(results):
        results[i] = hits[:batch_size]

    # Compute this batch's last_bound (external distance of the last entry)
    new_last_bound = 0.0
    if results and results[0]:
        last_hit_distance = results[0][-1]["distance"]
        # Internal distance → external distance
        if metric == "IP":
            new_last_bound = -last_hit_distance
        else:
            new_last_bound = last_hit_distance

    # Build response
    result_data = build_search_result_data(...)

    # Set V2 iterator info
    result_data.search_iterator_v2_results.token = token
    result_data.search_iterator_v2_results.last_bound = new_last_bound

    # First request: return session_ts, pymilvus will auto-pass it back as guarantee_timestamp
    session_ts = snapshot_seq if is_first else 0

    return milvus_pb2.SearchResults(
        status=common_pb2.Status(**success_status_kwargs()),
        results=result_data,
        session_ts=session_ts,
    )
```

### 4.5 parse_search_request Extension

Add the following to the existing `parse_search_request` return value:

```python
# V2 iterator parameters
"search_iter_v2": bool(raw_params.get("search_iter_v2", False)),
"search_iter_batch_size": int(raw_params.get("search_iter_batch_size", 1000)),
"search_iter_id": raw_params.get("search_iter_id"),        # str or None
"search_iter_last_bound": raw_params.get("search_iter_last_bound"),  # float or None
"iterator": bool(raw_params.get("iterator", False)),
```

---

## 5. Module Breakdown

| File | Change | Description |
|------|--------|-------------|
| `search/bitmap.py` | Modified | build_valid_mask() adds `snapshot_seq` filtering |
| `search/executor_indexed.py` | Modified | Adds last_bound filtering after top-k, passes through snapshot_seq |
| `engine/collection.py` | Modified | search() adds `last_bound` + `snapshot_seq` parameters |
| `adapter/grpc/servicer.py` | Modified | Search() adds V2 branch, snapshot _seq capture |
| `adapter/grpc/translators/search.py` | Modified | Extracts V2 iterator parameters + guarantee_timestamp |
| `adapter/grpc/translators/result.py` | Modified | Sets search_iterator_v2_results field |

No new files needed. The server is stateless (snapshot_seq is passed back by the client via guarantee_timestamp).

---

## 6. Edge Cases

| Scenario | Handling |
|----------|----------|
| Probe request (batch_size=1) | Normal search with top_k=2, returns 1 entry after filtering |
| First request (no last_bound) | No filtering, returns top batch_size normally; captures _seq as snapshot |
| Empty results | Returns empty results + token + last_bound=0, pymilvus terminates |
| Still fewer than batch_size after over-fetch | Returns actual count, no retry |
| Many records with equal distances | Partial omission, consistent with Milvus behavior |
| nq > 1 | Rejected, returns error (V2 only supports single-vector) |
| Combined with groupBy / offset | Rejected, returns error |
| Insert during iteration | New records with `_seq > snapshot_seq` are filtered by seq_mask, invisible |
| Delete during iteration | Delete operations with `_seq > snapshot_seq` are ignored, already-returned records are unaffected |

---

## 7. Test Plan

| Test | Content |
|------|---------|
| **Unit: last_bound filtering** | Verify results with distance ≤ last_bound are correctly filtered |
| **Unit: snapshot_seq filtering** | Verify records with _seq > snapshot_seq are invisible |
| **Unit: distance conversion** | Verify correctness of internal/external distance conversion for IP metric |
| **Integration: pymilvus search_iterator** | Complete iteration flow, verify no duplicates, results in ascending distance order |
| **Integration: different batch_size** | 1, 10, 100, larger than total count |
| **Integration: with filter** | Scalar filter + iterator |
| **Integration: different metrics** | COSINE / L2 / IP |
| **Integration: limit parameter** | pymilvus limit < total count |
| **Integration: insert during iteration** | Insert new records during iteration, verify iteration results are unaffected |
| **Integration: sparse scenario** | Most results filtered by filter, verify over-fetch behavior |
| **Regression: normal search unaffected** | Non-iterator requests follow the original path |

---

## 8. Query Iterator

Query Iterator already works correctly in Phase 16 through pymilvus's client-side implementation (based on offset pagination calling Query RPC). No server-side optimization is needed for now.

---

## 9. Implementation Steps

1. **Step 1**: `search/bitmap.py` — build_valid_mask() adds `snapshot_seq` filtering (with unit tests)
2. **Step 2**: `search/executor_indexed.py` — Adds `last_bound` filtering logic, passes through snapshot_seq (with unit tests)
3. **Step 3**: `engine/collection.py` — search() adds `last_bound` + `snapshot_seq` parameters
4. **Step 4**: `adapter/grpc/translators/search.py` — Extracts V2 parameters + guarantee_timestamp
5. **Step 5**: `adapter/grpc/servicer.py` — Search() adds V2 branch, snapshot capture, distance conversion
6. **Step 6**: `adapter/grpc/translators/result.py` — Sets SearchIteratorV2Results + session_ts
7. **Step 7**: Integration tests (pymilvus search_iterator + snapshot isolation + multiple scenarios)
