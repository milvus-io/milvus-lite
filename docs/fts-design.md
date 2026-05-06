# Deep Design: Full Text Search Subsystem (Phase 11)

## 1. Overview

MilvusLite Phase 11 introduces Full Text Search (FTS), allowing users to perform semantic relevance search through natural language text. **The core implementation is BM25 scoring + sparse inverted index**, fully compatible with the Milvus Full Text Search API.

**Why now**:
- Full text search is a core new capability of Milvus 2.5; a "local version of Milvus" must keep up
- BM25 search complements dense vector retrieval — dense vectors excel at semantic understanding, BM25 excels at precise keyword matching
- Phases 9/10 established the index state machine + gRPC adapter layer; Phase 11 extends on this foundation without requiring a new architectural paradigm

**Core positioning**: Compatible with the Milvus FTS API — pymilvus users can directly use `Function(type=BM25)` + `text_match` + sparse vector search without any code changes.

---

## 2. Architecture Decisions

### 2.1 Full Text Search Pipeline: Function-Driven (Decision)

**Decision: Adopt the Milvus Function mechanism — users declare a BM25 Function in the schema, and the engine automatically tokenizes + generates sparse vectors at insert time.**

Data flow:

```
Insert:
  User record {"text": "machine learning"}
  → Analyzer tokenizes → ["machine", "learning"]
  → term hash → {2847: 1, 9134: 1}  (term_hash → TF)
  → Stored as SPARSE_FLOAT_VECTOR column

Search:
  query "learning algorithm"
  → Tokenize → ["learning", "algorithm"]
  → Look up inverted index → Accumulate BM25 score for each term's posting list
  → top-k
```

**Why not use dense embedding**:
- BM25 is a deterministic algorithm, independent of external models, zero inference cost
- In exact keyword matching scenarios, BM25 >> dense vectors
- Consistent with the Milvus API (FunctionType.BM25)

### 2.2 Tokenization Strategy: Built-in Lightweight Analyzer (Decision)

**Decision: Built-in StandardAnalyzer (regex tokenization) as the default implementation, with jieba as an optional Chinese dependency.**

Candidate comparison:

| Dimension | Built-in Lightweight (Selected) | Depends on NLTK/spaCy | Depends on External Service |
|---|---|---|---|
| Deployment simplicity | Zero additional dependencies | Requires NLTK data packages / spaCy models | Requires starting a tokenization service |
| Chinese support | jieba (optional extra) | spaCy Chinese model | Flexible but complex |
| Alignment with Milvus | Milvus has built-in standard/jieba/icu | Not aligned | Not aligned |
| Performance | Sufficient (regex is not the bottleneck) | Slower (model loading) | Network overhead |

**Implementation**:
- `StandardAnalyzer`: `re.findall(r'\w+', text.lower())` — split by non-word characters + lowercase
- `JiebaAnalyzer`: `jieba.cut(text)` — optional exact/search mode
- Supports stop words filter

### 2.3 Sparse Vector Storage: Packed Binary Format (Decision)

**Decision: SPARSE_FLOAT_VECTOR is stored in Arrow/Parquet as a `pa.binary()` column, where each row is packed bytes (interleaved uint32 index + float32 value), consistent with the Milvus SparseFloatArray.contents format.**

Candidate comparison:

| Format | Pros | Cons |
|---|---|---|
| **packed binary (Selected)** | Compact; consistent with proto format, zero-copy conversion | Not human-readable |
| JSON string | Readable | 3x size, serialization/deserialization overhead |
| Arrow Map<int,float> | Native type | Parquet map support is unstable |

**Encoding format**:
```
Each row = N groups of (uint32_le index, float32_le value), sorted by index in ascending order
Empty sparse vector = empty bytes b""
```

### 2.4 Inverted Index: Per-segment (Decision)

**Decision: Each segment owns an independent inverted index (SparseInvertedIndex), consistent with the Phase 9 VectorIndex 1:1 binding strategy.**

| Dimension | Per-segment (Selected) | Global |
|---|---|---|
| Compatibility with LSM immutable architecture | Perfect — segment is immutable → index is immutable | Poor — requires maintaining a mutable global index |
| IDF accuracy | Per-segment IDF (possible deviation; Elasticsearch does the same) | Exact |
| Incremental update cost | Zero — new segment builds new index | Must update global index on every flush |
| Coordination with compaction | Natural — merge segments and rebuild index | Complex |
| Implementation complexity | Low | High |

**IDF accuracy compensation**: Elasticsearch in production also uses per-segment IDF, which is sufficiently accurate for most scenarios. IDF deviation in small segments can be mitigated through compaction merging.

### 2.5 BM25 Scoring Strategy: Computed at Query Time (Decision)

**Decision: Only store term frequency (TF) at insert time; complete BM25 scoring is computed in real-time at query time based on per-segment statistics.**

**Rationale**:
- IDF changes as documents are added/deleted; pre-computed BM25 scores would become inaccurate due to IDF drift
- The overhead of query-time computation is concentrated in inverted index lookups (O(postings_per_term)), which is not a bottleneck
- Each segment pre-computes doc_count, avgdl, and df_map at build time; query time only requires one division and multiplication

### 2.6 Term ID Mapping: Hash (Decision)

**Decision: Use a deterministic hash function to map term strings to uint32 IDs, without maintaining a global vocabulary file.**

- Pros: Stateless, no need to share vocabulary across segments, simplifies storage
- Collision risk: 32-bit hash space ~4.3 billion; with actual vocabulary < 1 million, collision probability is extremely low (< 0.01%)
- Uses `mmh3` (MurmurHash3) or built-in `hash` + modulo

### 2.7 Multi-Vector Field Support: anns_field Parameter (Decision)

**Decision: The search API adds an `anns_field` parameter to specify which vector field to search, breaking the MVP limitation of "only one vector field".**

- When the schema contains both FLOAT_VECTOR + SPARSE_FLOAT_VECTOR, users select the search target via `anns_field`
- Default value: the first FLOAT_VECTOR field (maintains backward compatibility)
- The gRPC layer extracts anns_field from the SearchRequest and passes it to the engine

---

## 3. Module Structure

### 3.1 New Modules

```
milvus_lite/
├── analyzer/                      # Phase 11.2: Tokenization subsystem
│   ├── __init__.py
│   ├── protocol.py                # Analyzer ABC
│   ├── standard.py                # StandardAnalyzer (regex)
│   ├── jieba_analyzer.py          # JiebaAnalyzer (optional)
│   ├── factory.py                 # create_analyzer(params) → Analyzer
│   └── hash.py                    # term_to_id(term) → uint32
│
├── index/
│   └── sparse_inverted.py         # Phase 11.5: SparseInvertedIndex
│
└── adapter/grpc/translators/
    └── sparse.py                  # Phase 11.7: Sparse vector encoding/decoding
```

### 3.2 Modified Modules

| Module | Changes |
|---|---|
| `schema/types.py` | ✅ (11.1 completed) DataType.SPARSE_FLOAT_VECTOR, Function, FunctionType, FieldSchema new attributes |
| `schema/validation.py` | ✅ (11.1 completed) Vector field constraints relaxed, BM25 function validation, sparse vector validation |
| `schema/arrow_builder.py` | ✅ (11.1 completed) SPARSE_FLOAT_VECTOR → pa.binary() |
| `schema/persistence.py` | ✅ (11.1 completed) schema.json supports functions and new attributes |
| `engine/collection.py` | Insert auto-generates function output; search supports anns_field; sparse vector serialization |
| `storage/segment.py` | Attach SparseInvertedIndex |
| `search/filter/` | text_match / phrase_match functions |
| `adapter/grpc/servicer.py` | FunctionSchema handling; anns_field forwarding |
| `adapter/grpc/translators/` | schema/records/search layer extensions |

---

## 4. BM25 Algorithm Details

### 4.1 Scoring Formula

```
score(D, Q) = Σ_{qi ∈ Q} IDF(qi) · f(qi, D) · (k1 + 1) / (f(qi, D) + k1 · (1 - b + b · |D| / avgdl))
```

Where:
- `f(qi, D)` = term frequency (TF) of term qi in document D
- `|D|` = total token count of document D
- `avgdl` = average token count across all documents in the segment
- `k1` = saturation parameter (default 1.5, controls TF growth rate)
- `b` = length normalization parameter (default 0.75, controls long document penalty)
- `IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)`
  - `N` = total number of documents in the segment
  - `df(qi)` = number of documents in the segment containing term qi

### 4.2 Data Stored at Write Time

| Storage Item | Location | Format |
|---|---|---|
| term hash → TF (per-row) | Sparse vector column (Parquet binary) | packed uint32+float32 |
| doc_length (per-row) | Inverted index metadata | Derived from the sum of sparse vector values |
| doc_count, avgdl, df_map | SparseInvertedIndex in-memory state | Computed at build time, persisted at save time |
| posting_lists | SparseInvertedIndex | term_hash → [(local_id, tf), ...] |

### 4.3 Search-Time Flow

```python
def search_bm25(query_text, index, top_k):
    terms = analyzer.analyze(query_text)  # → List[int] (term hashes)
    scores = {}  # doc_id → accumulated score

    for term_hash in set(terms):
        posting = index.posting_lists.get(term_hash, [])
        df = len(posting)
        idf = math.log((index.doc_count - df + 0.5) / (df + 0.5) + 1)

        for doc_id, tf in posting:
            dl = index.doc_lengths[doc_id]
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / index.avgdl))
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm

    return top_k_by_score(scores, top_k)
```

---

## 5. Analyzer Subsystem

### 5.1 Analyzer ABC

```python
class Analyzer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split text into a list of tokens."""

    def analyze(self, text: str) -> List[int]:
        """tokenize + hash → list of term_ids."""
        return [term_to_id(t) for t in self.tokenize(text)]
```

### 5.2 StandardAnalyzer

```python
class StandardAnalyzer(Analyzer):
    def __init__(self, stop_words: Optional[Set[str]] = None):
        self._stop_words = stop_words or set()
        self._pattern = re.compile(r'\w+')

    def tokenize(self, text: str) -> List[str]:
        tokens = self._pattern.findall(text.lower())
        if self._stop_words:
            tokens = [t for t in tokens if t not in self._stop_words]
        return tokens
```

### 5.3 JiebaAnalyzer

```python
class JiebaAnalyzer(Analyzer):
    def __init__(self, mode: str = "search", stop_words=None, user_dict=None):
        import jieba  # optional dependency
        self._mode = mode
        # ...

    def tokenize(self, text: str) -> List[str]:
        if self._mode == "search":
            return list(jieba.cut_for_search(text))
        return list(jieba.cut(text))
```

### 5.4 Factory

```python
def create_analyzer(params: Optional[dict]) -> Analyzer:
    if params is None:
        return StandardAnalyzer()
    tokenizer = params.get("tokenizer", "standard")
    if tokenizer == "standard":
        return StandardAnalyzer(stop_words=_parse_stop_words(params))
    if tokenizer == "jieba" or (isinstance(tokenizer, dict) and tokenizer.get("type") == "jieba"):
        return JiebaAnalyzer(...)
    raise SchemaValidationError(f"unknown tokenizer: {tokenizer}")
```

---

## 6. Sparse Vector Encoding/Decoding

### 6.1 Python dict ↔ packed bytes

```python
import struct

def sparse_to_bytes(sv: dict[int, float]) -> bytes:
    """dict[int, float] → packed bytes (sorted by index)."""
    if not sv:
        return b""
    pairs = sorted(sv.items())
    return struct.pack(f"<{len(pairs) * 2}I",
                       *[x for idx, val in pairs
                         for x in (idx, struct.unpack('I', struct.pack('f', val))[0])])

def bytes_to_sparse(b: bytes) -> dict[int, float]:
    """packed bytes → dict[int, float]."""
    if not b:
        return {}
    n = len(b) // 8  # 8 bytes per pair (uint32 + float32)
    result = {}
    for i in range(n):
        idx = struct.unpack_from('<I', b, i * 8)[0]
        val = struct.unpack_from('<f', b, i * 8 + 4)[0]
        result[idx] = val
    return result
```

### 6.2 Mapping to Milvus SparseFloatArray

| Milvus Proto | MilvusLite |
|---|---|
| `SparseFloatArray.contents[i]` (bytes) | `sparse_to_bytes(row_dict)` |
| `SparseFloatArray.dim` | `max(all_indices) + 1` across all rows |

---

## 7. SparseInvertedIndex

### 7.1 Class Signature

```python
class SparseInvertedIndex(VectorIndex):
    """Per-segment inverted index, implementing the VectorIndex protocol."""

    def build(self, vectors: list[dict], valid_mask: np.ndarray) -> None:
        """Build the inverted index from a list of sparse vectors.

        vectors: list of dict[int, float] (term_hash → tf)
        valid_mask: boolean array, True = the row is valid

        State after building:
        - posting_lists: dict[int, list[tuple[int, float]]]  # term → [(local_id, tf)]
        - doc_count: int
        - doc_lengths: np.ndarray  # per-doc token count
        - avgdl: float
        - df_map: dict[int, int]  # term → document frequency
        """

    def search(self, query_vectors, top_k, valid_mask=None):
        """query_vectors: list of dict[int, float] (query term hashes → weight)

        Returns: (ids, distances) — distances are the negation of BM25 scores
        (consistent with the VectorIndex protocol: smaller = more similar)
        """

    def save(self, f) -> None: ...
    def load(self, f) -> None: ...
```

### 7.2 BM25 Distance Convention

The VectorIndex protocol requires "smaller distance = more similar". Higher BM25 scores indicate greater relevance, so:
- **distance = -bm25_score** (negated)
- Search results sorted by distance ascending = sorted by BM25 score descending
- Consistent with Milvus behavior (pymilvus negates the score back when retrieving from SearchResults)

### 7.3 Persistence

Stored as `.sidx` files (JSON or pickle), containing:
- posting_lists (serialized as {term_hash: [[local_id, tf], ...]})
- doc_count, avgdl, df_map
- doc_lengths
- bm25_k1, bm25_b parameters

---

## 8. text_match Filter

### 8.1 Syntax

```
text_match(field_name, 'token1 token2 token3')
```

- Multiple tokens use **OR** logic: matches documents containing any of the tokens
- Tokens are processed through the Analyzer associated with the field
- Requires the field to have `enable_match=True` + `enable_analyzer=True`

### 8.2 Implementation Approach

Add a new `TextMatchNode` AST node in the filter subsystem:

```python
class TextMatchNode(ASTNode):
    field_name: str
    query_text: str  # raw query text
```

During evaluation:
1. Use the field's corresponding Analyzer to tokenize query_text → query_tokens
2. Tokenize each row's text field value → doc_tokens
3. Return `bool(set(query_tokens) & set(doc_tokens))`

---

## 9. gRPC Adapter Layer Extensions

### 9.1 Schema Translation

| Milvus Proto | MilvusLite |
|---|---|
| `FieldSchema.data_type = 104` (SparseFloatVector) | `DataType.SPARSE_FLOAT_VECTOR` |
| `FieldSchema.type_params["enable_analyzer"]` | `FieldSchema.enable_analyzer` |
| `FieldSchema.type_params["analyzer_params"]` | `FieldSchema.analyzer_params` (JSON decoded) |
| `FieldSchema.type_params["enable_match"]` | `FieldSchema.enable_match` |
| `CollectionSchema.functions` | `CollectionSchema.functions` |
| `FunctionSchema.type = BM25 (1)` | `FunctionType.BM25` |
| `FieldSchema.is_function_output = True` | `FieldSchema.is_function_output` |

### 9.2 FieldData Encoding/Decoding

**SparseFloatArray decoding** (insert path):
```python
sfa = fd.vectors.sparse_float_vector
for content_bytes in sfa.contents:
    sv = bytes_to_sparse(content_bytes)  # → dict[int, float]
    column.append(sv)
```

**SparseFloatArray encoding** (response path):
```python
sfa = schema_pb2.SparseFloatArray()
max_dim = 0
for sv in column:
    sfa.contents.append(sparse_to_bytes(sv))
    if sv:
        max_dim = max(max_dim, max(sv.keys()) + 1)
sfa.dim = max_dim
fd.vectors.sparse_float_vector.CopyFrom(sfa)
```

### 9.3 Search Request Handling

When performing BM25 search, pymilvus sends text queries as follows:
- `PlaceholderGroup` contains `PlaceholderValue.type = 104` (SPARSE_FLOAT_VECTOR)
- But the actual data is text — pymilvus internally tokenizes and generates sparse vectors before encoding
- MilvusLite needs to support two search entry points:
  1. Client-tokenized sparse vector search
  2. Direct text search (engine tokenizes internally)

### 9.4 Index Parameters

| Milvus index_type | MilvusLite Handling |
|---|---|
| `SPARSE_INVERTED_INDEX` | Maps to SparseInvertedIndex |
| `metric_type = BM25` | Uses BM25 scoring |
| `bm25_k1`, `bm25_b` | Passed to SparseInvertedIndex |

---

## 10. Phased Implementation Plan

### Phase 11.1 — Schema Extensions ✅

Completed. Added SPARSE_FLOAT_VECTOR DataType, Function/FunctionType, FieldSchema new attributes, schema validation, Arrow type mapping, persistence. 30 new tests.

### Phase 11.2 — Analyzer Tokenization Subsystem

Create new `milvus_lite/analyzer/` package:
- `protocol.py` — Analyzer ABC
- `standard.py` — StandardAnalyzer
- `jieba_analyzer.py` — JiebaAnalyzer (optional)
- `factory.py` — create_analyzer
- `hash.py` — term_to_id

Validation: Unit tests covering tokenization results, hash determinism, stop words.

### Phase 11.3 — Sparse Vector Storage

- `sparse_to_bytes` / `bytes_to_sparse` encoding/decoding functions
- engine/collection.py's `_build_wal_data_batch` supports sparse vector column serialization
- validate_record skips function output fields at insert time

Validation: Sparse vector round-trip (dict → bytes → dict), WAL write/read.

### Phase 11.4 — BM25 Function Engine

- engine/collection.py insert path: detect BM25 function → tokenize → generate TF sparse vector → inject into record
- Text field + sparse vector field written to WAL / Parquet simultaneously

Validation: insert + flush → read Parquet to verify sparse vector column exists and is correct.

### Phase 11.5 — Sparse Inverted Index + BM25 Search

- `index/sparse_inverted.py` — SparseInvertedIndex implementation
- Integration with segment attach_index / load state machine
- Search path supports anns_field selection + BM25 distance

Validation:
- Standalone BM25 search accuracy tests
- Insert + search end-to-end tests
- BM25 score correctness verification (manual calculation comparison)

### Phase 11.6 — text_match Filter

- Extend filter subsystem to support `text_match(field, 'tokens')` function
- Add TextMatchNode AST node
- Three backend implementations

Validation: text_match used standalone + combined with vector search.

### Phase 11.7 — gRPC Adapter Layer Extensions

- FunctionSchema ↔ Function translation
- SparseFloatArray FieldData encoding/decoding
- Search request handling (sparse vector / text query)
- CreateIndex supports SPARSE_INVERTED_INDEX / BM25

Validation: pymilvus end-to-end full text search flow.

### Phase 11.8 — Integration Tests

Extract key scenarios from the Milvus test suite:
1. Basic BM25 search (English)
2. text_match filtering
3. BM25 + FLOAT_VECTOR mixed schema
4. Index rebuild after Flush/Compaction
5. Load/Release state machine with BM25 index

---

## 11. Dependency Changes

```toml
[project.optional-dependencies]
chinese = ["jieba>=0.42"]
```

jieba is an optional dependency and does not affect the base installation. StandardAnalyzer has zero additional dependencies.

---

## 12. Out of Scope for Phase 11

- Multi-Analyzer (multi-language dynamic selection) → Future
- phrase_match (phrase matching + slop control) → Future
- LexicalHighlighter (search result highlighting) → Future
- TextEmbedding Function (calling external embedding models) → Future
- ICU tokenizer → Future
- BM25 global IDF statistics (cross-segment IDF merging) → Future
- **Hybrid Search RPC (multi-path ANN re-ranking) → Phase 12** (see roadmap.md)

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Hash collisions leading to inaccurate BM25 scoring | 32-bit space is sufficient; can upgrade to 64-bit or add collision detection in the future |
| Per-segment IDF deviation | Same strategy as Elasticsearch; mitigated after compaction merges segments |
| jieba installation issues | Optional extra, does not affect English scenarios |
| Sparse vector Parquet size | Binary column is already the most compact format; naturally merged during compaction |
| Adding anns_field to search API breaks backward compatibility | Default value = first FLOAT_VECTOR field, no behavior change |
