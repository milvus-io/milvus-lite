# Deep Design: FuncChain (Function Chain)

## 1. Overview

The current function execution logic in LiteVecDB is scattered throughout the Collection class: 4 per-type lists (`_bm25_functions`, `_embedding_functions`, `_rerank_functions`, `_decay_functions`) plus 4 apply methods. Adding a new function type requires touching initialization, insert, and search in three separate places with if/elif branches.

**Goal of FuncChain**: Inspired by Milvus `internal/util/function/chain/`, replace per-type branches with a unified **Operator pipeline** that allows multiple Functions to be **composed and executed in series**.

**Why now**:
- Milvus has already unified reranking with FuncChain (Merge → Map → Sort → Limit → Select), and has reserved `StageIngestion` for migrating BM25/Embedding into the chain system
- LiteVecDB has accumulated 4 function types (BM25, TEXT_EMBEDDING, RERANK, DECAY); maintaining scattered logic is increasingly costly
- Future scenarios require serial chaining (e.g., text → Embedding A → dense_vec_a → DimReduce → dense_vec_b), which demands a general-purpose chain execution framework

**Core positioning**: Maintain conceptual alignment with Milvus chain (FunctionExpr / Operator / FuncChain / Stage), but use Python native data structures (`List[dict]`) instead of Arrow DataFrame to fit the embedded scenario.

---

## 2. Alignment with Milvus Chain

| Milvus (Go) | LiteVecDB (Python) | Differences |
|---|---|---|
| `types.FunctionExpr` interface | `FunctionExpr` ABC | Same semantics: stateless column computation |
| `DataFrame` (Arrow Chunked) | `DataFrame` (List[List[dict]]) | Python GC replaces Arrow Allocator; chunk = per-query results |
| `Operator` interface | `Operator` ABC | Same semantics: `execute(ctx, df) → df` |
| `MapOp` + `BaseOp` | `MapOp` | Column mapping + FunctionExpr invocation |
| `MergeOp` (5 strategies) | `MergeOp` (RRF/Weighted/Max/Sum/Avg) | Multi-path search result merging |
| `SortOp` (per-chunk sort) | `SortOp` (per-chunk sort) | Sort by column |
| `LimitOp` (per-chunk) | `LimitOp` (per-chunk) | offset + limit |
| `SelectOp` | `SelectOp` | Column projection |
| `GroupByOp` | `GroupByOp` | Grouped search |
| `FilterOp` | `FilterOp` | Boolean expression row filtering |
| `FuncChain` | `FuncChain` | Ordered pipeline + stage validation + fluent API |
| `rerank_builder.go` | `builder.py` | Build chain from schema.functions |
| `types.FuncContext` | `FuncContext` | Execution context (stage) |
| `types.FunctionFactory` + Registry | `create_function_expr()` factory | Embedded: no need for dynamic registry, direct factory |
| `StageIngestion / StageL2Rerank / ...` | `STAGE_INGESTION / STAGE_L2_RERANK` | Simplified to 2 stages (embedded has no distributed multi-level rerank) |

---

## 3. Architecture Decisions

### 3.1 Data Container: Python DataFrame (Decision)

**Decision: Use `List[List[dict]]` as the DataFrame internal representation, where each inner list is a chunk (corresponding to one query's result set).**

Candidate comparison:

| Dimension | Python List[List[dict]] (Selected) | PyArrow Table | Custom Columnar Structure |
|---|---|---|---|
| Implementation complexity | Zero dependencies, Python native | Requires Arrow type dispatch | Requires boilerplate code |
| Integration with existing code | insert records are already List[dict] | Requires dict↔Arrow conversion | Requires dict↔custom conversion |
| Per-chunk semantics | 2D list naturally supported | Requires ChunkedArray | Must build chunk mechanism |
| Performance | Sufficient (embedded scale) | Better for large data | Middle ground |
| Operator implementation | Standard Python list operations | Requires Arrow compute | Depends on structure |

**Rationale**:
- LiteVecDB is an embedded single-process system with limited data scale; Python native structures are sufficient
- Existing insert/search interfaces are already `List[dict]`; no conversion layer needed
- Operator implementations are more intuitive and easier to debug

**Chunk semantics**:
- Ingestion stage: single chunk — `chunks = [records]`
- Rerank stage: nq chunks — `chunks[i]` = search results for the i-th query

### 3.2 Stage Design: Two Stages (Decision)

**Decision: Define only two stages — `ingestion` and `rerank`.**

Milvus has 6 stages (Ingestion, L2_Rerank, L1_Rerank, L0_Rerank, PreProcess, PostProcess) because in a distributed system reranking can execute at Proxy / QueryNode / Segment levels. LiteVecDB is a single-process embedded system where all reranking happens in one place; multi-level rerank stages are unnecessary.

```python
STAGE_INGESTION = "ingestion"   # executed during insert/upsert
STAGE_L2_RERANK    = "rerank"      # executed during search post-processing
```

### 3.3 Operator Set: 6 Types (Decision)

**Decision: Implement Map, Merge, Sort, Limit, Select, and GroupBy — six Operator types.**

This is the complete Operator set actually used by Milvus `rerank_builder.go`. Taking the Decay reranker chain as an example:

```
Merge(strategy) → Map(DecayExpr) → Map(ScoreCombineExpr) → Sort($score, DESC) → Limit(limit, offset) → [Map(RoundDecimal)] → Select($id, $score)
```

Map alone cannot express Merge (multi-path merging), Sort (ordering), Limit (pagination), or Select (projection) — these are essential search post-processing operations. All four reranker types (RRF / Weighted / Decay / Model) share the Sort → Limit → Select tail; only the Merge strategy and intermediate Map steps differ.

FilterOp is not in MVP scope (scalar filtering is currently handled independently on the search side), but the interface is reserved.

### 3.4 FunctionExpr vs Operator Layering (Decision)

**Decision: FunctionExpr is responsible only for pure column computation (input columns → output columns). Column mapping (which columns to read from DataFrame, which to write to) is handled by the Operator.**

This is consistent with Milvus's design:

```
MapOp (Operator layer)   ─┬─ Responsible for: reading input_cols from DataFrame, writing output_cols
                           └─ Invokes: FunctionExpr.execute(inputs) → outputs

FunctionExpr (Compute layer) ── Responsible for: pure computation logic (e.g., BM25 tokenization, decay)
                                 Unaware of DataFrame existence
```

Benefit: The same FunctionExpr can be reused with different column mappings (e.g., DecayExpr applied to different fields).

---

## 4. Core Interface Design

### 4.1 FunctionExpr — Stateless Column Computation Unit

```python
# function/types.py

from abc import ABC, abstractmethod
from typing import List, FrozenSet

STAGE_INGESTION = "ingestion"
STAGE_L2_RERANK    = "rerank"

# Virtual column names used internally by DataFrame
ID_FIELD    = "$id"
SCORE_FIELD = "$score"


class FunctionExpr(ABC):
    """Stateless column-level computation unit.

    Responsible only for pure computation: input columns → output columns.
    Unaware of DataFrame structure; column name mapping is handled by MapOp.

    Corresponds to Milvus: internal/util/function/chain/types.FunctionExpr
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Function name (e.g., "bm25", "decay", "score_combine")."""

    @property
    @abstractmethod
    def supported_stages(self) -> FrozenSet[str]:
        """Stages where this function can execute."""

    @abstractmethod
    def execute(self, inputs: List[list]) -> List[list]:
        """Execute computation.

        Args:
            inputs: List of input columns. inputs[i] is a list of values,
                    length = number of records in the chunk.
        Returns:
            List of output columns. Length defined by the function
            (typically = number of output fields).
        """

    def is_runnable(self, stage: str) -> bool:
        """Check if this function supports the given stage."""
        return stage in self.supported_stages
```

### 4.2 DataFrame — Lightweight Data Container

```python
# function/dataframe.py

from typing import List, Optional


class DataFrame:
    """Lightweight columnar data container.

    Internal storage is List[List[dict]], where each inner list is a chunk.

    - Ingestion stage: single chunk, chunks = [records]
    - Rerank stage: nq chunks, chunks[i] = search results for the i-th query

    Corresponds to Milvus: internal/util/function/chain/dataframe.go
    """

    __slots__ = ("_chunks",)

    def __init__(self, chunks: List[List[dict]]):
        self._chunks = chunks

    # ── Factory Methods ──

    @classmethod
    def from_records(cls, records: List[dict]) -> "DataFrame":
        """Create from insert records (single chunk)."""
        return cls([records])

    @classmethod
    def from_search_results(cls, results: List[List[dict]]) -> "DataFrame":
        """Create from search return value (per-query chunks)."""
        return cls(results)

    # ── Export ──

    def to_records(self) -> List[dict]:
        """Export as flat records (single chunk only)."""
        assert len(self._chunks) == 1, "to_records() requires single chunk"
        return self._chunks[0]

    def to_search_results(self) -> List[List[dict]]:
        """Export as per-query search results."""
        return self._chunks

    # ── Accessors ──

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def chunk(self, idx: int) -> List[dict]:
        return self._chunks[idx]

    def column(self, name: str, chunk_idx: int) -> list:
        """Read all values of a column from the specified chunk."""
        return [r.get(name) for r in self._chunks[chunk_idx]]

    def set_column(self, name: str, chunk_idx: int, values: list) -> None:
        """Write a column of values back to the specified chunk (in-place)."""
        chunk = self._chunks[chunk_idx]
        for r, v in zip(chunk, values):
            r[name] = v

    def column_names(self, chunk_idx: int = 0) -> List[str]:
        """Get column names (from the first record's keys)."""
        chunk = self._chunks[chunk_idx]
        return list(chunk[0].keys()) if chunk else []
```

### 4.3 FuncContext — Execution Context

```python
# function/types.py (continued)

class FuncContext:
    """Execution context for function chains.

    Corresponds to Milvus: internal/util/function/chain/types.FuncContext
    """

    __slots__ = ("_stage",)

    def __init__(self, stage: str):
        self._stage = stage

    @property
    def stage(self) -> str:
        return self._stage
```

### 4.4 Operator — Base Class

```python
# function/operator.py

from abc import ABC, abstractmethod
from typing import List


class Operator(ABC):
    """Base class for operators.

    Operators work on DataFrames: receive an input DataFrame, return an output DataFrame.
    Each Operator declares the column names it reads (inputs) and produces (outputs).

    Corresponds to Milvus: internal/util/function/chain/chain.go Operator interface
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Operator name (e.g., "Map", "Sort", "Merge")."""

    @abstractmethod
    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        """Execute the operator.

        Args:
            ctx: Execution context
            df:  Input DataFrame
        Returns:
            Output DataFrame (may be the same object modified in-place,
            or a new object)
        """
```

### 4.5 FuncChain — Ordered Pipeline

```python
# function/chain.py

from typing import List, Optional


class FuncChain:
    """Ordered Operator pipeline.

    - Fluent API: chain.merge(...).map(...).sort(...).limit(...)
    - Execute runs all Operators sequentially
    - If the first Operator is MergeOp, supports multi-path inputs

    Corresponds to Milvus: internal/util/function/chain/chain.go FuncChain
    """

    def __init__(self, name: str, stage: str):
        self._name = name
        self._stage = stage
        self._operators: List[Operator] = []

    @property
    def stage(self) -> str:
        return self._stage

    # ── Fluent API ──

    def add(self, op: Operator) -> "FuncChain":
        """Append an Operator to the end of the pipeline."""
        self._operators.append(op)
        return self

    def map(self, expr: FunctionExpr,
            input_cols: List[str], output_cols: List[str]) -> "FuncChain":
        """Add a MapOp."""
        if not expr.is_runnable(self._stage):
            raise ValueError(
                f"FunctionExpr '{expr.name}' does not support "
                f"stage '{self._stage}'"
            )
        return self.add(MapOp(expr, input_cols, output_cols))

    def merge(self, strategy: str, **kwargs) -> "FuncChain":
        """Add a MergeOp (must be the first Operator in the chain)."""
        return self.add(MergeOp(strategy, **kwargs))

    def sort(self, column: str, desc: bool = True) -> "FuncChain":
        """Add a SortOp."""
        return self.add(SortOp(column, desc))

    def limit(self, limit: int, offset: int = 0) -> "FuncChain":
        """Add a LimitOp."""
        return self.add(LimitOp(limit, offset))

    def select(self, *columns: str) -> "FuncChain":
        """Add a SelectOp."""
        return self.add(SelectOp(list(columns)))

    def group_by(self, field: str, group_size: int,
                 limit: int, offset: int = 0,
                 scorer: str = "max") -> "FuncChain":
        """Add a GroupByOp."""
        return self.add(GroupByOp(field, group_size, limit, offset, scorer))

    # ── Execution ──

    def execute(self, *inputs: DataFrame) -> DataFrame:
        """Execute the entire chain.

        If the first Operator is MergeOp, accepts multi-path inputs;
        otherwise accepts only a single input.
        """
        ctx = FuncContext(self._stage)
        start_idx = 0

        if self._operators and isinstance(self._operators[0], MergeOp):
            result = self._operators[0].execute_multi(ctx, list(inputs))
            start_idx = 1
        else:
            if len(inputs) != 1:
                raise ValueError(
                    f"Chain expects 1 input but got {len(inputs)} "
                    f"(first operator is not MergeOp)"
                )
            result = inputs[0]

        for op in self._operators[start_idx:]:
            result = op.execute(ctx, result)

        return result

    # ── Debug ──

    def __repr__(self) -> str:
        ops = " → ".join(op.name for op in self._operators)
        return f"FuncChain({self._name}, stage={self._stage}): {ops}"
```

---

## 5. Operator Detailed Design

### 5.1 MapOp — Column Transformation

**Responsibility**: Read input_cols from DataFrame, call FunctionExpr.execute(), write results to output_cols.

**Corresponds to Milvus**: `operator_map.go`

```python
class MapOp(Operator):
    """Execute FunctionExpr column transformation independently per chunk.

    input_cols: column names to read from DataFrame
    output_cols: column names to write FunctionExpr output back to DataFrame

    output_cols may overlap with input_cols (e.g., ScoreCombine merges
    $score and _decay_score, writes back to $score).
    """

    name = "Map"

    def __init__(self, expr: FunctionExpr,
                 input_cols: List[str], output_cols: List[str]):
        self._expr = expr
        self._input_cols = input_cols
        self._output_cols = output_cols

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        for chunk_idx in range(df.num_chunks):
            # 1. Read input columns
            inputs = [df.column(col, chunk_idx) for col in self._input_cols]
            # 2. Execute function
            outputs = self._expr.execute(inputs)
            # 3. Write output columns back
            for col_name, col_data in zip(self._output_cols, outputs):
                df.set_column(col_name, chunk_idx, col_data)
        return df
```

### 5.2 MergeOp — Multi-Path Merging

**Responsibility**: Merge multiple search result sets into one. This is the core of hybrid search — multiple ANN sub-searches return independent results, and MergeOp merges and deduplicates them by strategy.

**Corresponds to Milvus**: `operator_merge.go`

**5 merge strategies**:

| Strategy | Formula | Description |
|---|---|---|
| `rrf` | `score = Σ 1/(k + rank_i)` | Rank-based fusion, independent of score magnitude |
| `weighted` | `score = Σ weight_i × normalize(score_i)` | Weighted sum, requires score normalization |
| `max` | `score = max(score_i)` | Take highest score (e.g., pre-merge for decay) |
| `sum` | `score = Σ score_i` | Sum of scores |
| `avg` | `score = mean(score_i)` | Average of scores |

```python
class MergeOp(Operator):
    """Merge multi-path search results.

    - Must be the first Operator in the chain
    - execute_multi() accepts multiple DataFrames, deduplicates by pk and merges
    - Aligned with Milvus MergeStrategy
    """

    name = "Merge"

    def __init__(self, strategy: str, **kwargs):
        self._strategy = strategy      # "rrf" | "weighted" | "max" | "sum" | "avg"
        self._weights = kwargs.get("weights", [])
        self._rrf_k = kwargs.get("rrf_k", 60.0)
        self._metric_types = kwargs.get("metric_types", [])
        self._normalize = kwargs.get("normalize", False)

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        raise RuntimeError("MergeOp requires execute_multi()")

    def execute_multi(self, ctx: FuncContext,
                      inputs: List[DataFrame]) -> DataFrame:
        """Merge multiple DataFrames.

        Each input DataFrame should have the same number of chunks (= nq).
        When the same pk appears in multiple paths, scores are merged by strategy.
        """
        if not inputs:
            raise ValueError("MergeOp requires at least one input")
        if len(inputs) == 1:
            return inputs[0]

        nq = inputs[0].num_chunks
        merged_chunks = []

        for q in range(nq):
            # Collect (pk, score, hit) + rank from all paths
            pk_map = {}  # pk → {hit, entries: [(path_idx, score, rank)]}
            for path_idx, inp in enumerate(inputs):
                chunk = inp.chunk(q)
                for rank, hit in enumerate(chunk):
                    pk = hit.get(ID_FIELD)
                    if pk not in pk_map:
                        pk_map[pk] = {"hit": dict(hit), "entries": []}
                    pk_map[pk]["entries"].append(
                        (path_idx, hit.get(SCORE_FIELD, 0.0), rank)
                    )

            # Compute final score by strategy
            results = []
            for pk, info in pk_map.items():
                score = self._compute_score(info["entries"], len(inputs))
                merged_hit = info["hit"]
                merged_hit[SCORE_FIELD] = score
                results.append(merged_hit)

            merged_chunks.append(results)

        return DataFrame(merged_chunks)

    def _compute_score(self, entries, num_paths):
        """Compute merged score by strategy."""
        if self._strategy == "rrf":
            return sum(1.0 / (self._rrf_k + rank) for _, _, rank in entries)
        elif self._strategy == "weighted":
            total = 0.0
            for path_idx, score, _ in entries:
                w = self._weights[path_idx] if path_idx < len(self._weights) else 1.0
                total += w * score
            return total
        elif self._strategy == "max":
            return max(score for _, score, _ in entries)
        elif self._strategy == "sum":
            return sum(score for _, score, _ in entries)
        elif self._strategy == "avg":
            scores = [score for _, score, _ in entries]
            return sum(scores) / len(scores)
        else:
            raise ValueError(f"Unknown merge strategy: {self._strategy}")
```

### 5.3 SortOp — Sorting

**Responsibility**: Sort records within each chunk by a specified column.

**Corresponds to Milvus**: `operator_sort.go`

```python
class SortOp(Operator):
    """Per-chunk sorting.

    Each chunk (query) is sorted independently.
    Supports tie-break by $id ASC (aligned with Milvus).
    """

    name = "Sort"

    def __init__(self, column: str, desc: bool = True,
                 tie_break_col: str = ID_FIELD):
        self._column = column
        self._desc = desc
        self._tie_break_col = tie_break_col

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            chunk.sort(
                key=lambda r: (
                    self._sort_key(r.get(self._column)),
                    r.get(self._tie_break_col, 0)
                ),
                reverse=self._desc,
            )
        return df

    @staticmethod
    def _sort_key(val):
        """None sorts to the end."""
        if val is None:
            return (1, 0)  # (is_none=1, val) — ensures None goes last
        return (0, val)
```

### 5.4 LimitOp — Pagination

**Responsibility**: Apply offset + limit to each chunk.

**Corresponds to Milvus**: `operator_limit.go`

```python
class LimitOp(Operator):
    """Per-chunk offset + limit."""

    name = "Limit"

    def __init__(self, limit: int, offset: int = 0):
        self._limit = limit
        self._offset = offset

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            start = min(self._offset, len(chunk))
            end = min(start + self._limit, len(chunk))
            new_chunks.append(chunk[start:end])
        return DataFrame(new_chunks)
```

### 5.5 SelectOp — Column Projection

**Responsibility**: Keep only specified columns, remove all other fields.

**Corresponds to Milvus**: `operator_select.go`

```python
class SelectOp(Operator):
    """Keep specified columns, remove all others."""

    name = "Select"

    def __init__(self, columns: List[str]):
        self._columns = set(columns)

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            new_chunks.append([
                {k: v for k, v in r.items() if k in self._columns}
                for r in chunk
            ])
        return DataFrame(new_chunks)
```

### 5.6 GroupByOp — Grouped Search

**Responsibility**: Group by field, keep top-N per group, return the first `limit` groups.

**Corresponds to Milvus**: `operator_group_by.go`

```python
class GroupByOp(Operator):
    """Per-chunk grouped search.

    1. Group by group_by_field
    2. Within each group, sort by $score DESC, keep top group_size
    3. Compute group_score using scorer (max/sum/avg)
    4. Sort groups by group_score DESC
    5. Skip offset groups, take limit groups
    6. Add $group_score column
    """

    name = "GroupBy"

    def __init__(self, field: str, group_size: int,
                 limit: int, offset: int = 0,
                 scorer: str = "max"):
        self._field = field
        self._group_size = group_size
        self._limit = limit
        self._offset = offset
        self._scorer = scorer  # "max" | "sum" | "avg"

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)

            # 1. Group
            groups = {}  # field_val → [hits]
            for hit in chunk:
                key = hit.get(self._field)
                groups.setdefault(key, []).append(hit)

            # 2. Sort within group + truncate to top group_size
            scored_groups = []
            for key, hits in groups.items():
                hits.sort(key=lambda r: r.get(SCORE_FIELD, 0), reverse=True)
                top_hits = hits[:self._group_size]
                group_score = self._compute_group_score(top_hits)
                scored_groups.append((group_score, key, top_hits))

            # 3. Sort groups + offset + limit
            scored_groups.sort(key=lambda g: g[0], reverse=True)
            selected = scored_groups[self._offset:self._offset + self._limit]

            # 4. Flatten + add $group_score
            result = []
            for group_score, key, hits in selected:
                for hit in hits:
                    hit["$group_score"] = group_score
                    result.append(hit)

            new_chunks.append(result)
        return DataFrame(new_chunks)

    def _compute_group_score(self, hits):
        scores = [h.get(SCORE_FIELD, 0) for h in hits]
        if not scores:
            return 0.0
        if self._scorer == "max":
            return max(scores)
        elif self._scorer == "sum":
            return sum(scores)
        elif self._scorer == "avg":
            return sum(scores) / len(scores)
        return max(scores)
```

---

## 6. FunctionExpr Implementations

### 6.1 BM25Expr — Text → Sparse Vector (Ingestion)

```python
class BM25Expr(FunctionExpr):
    """text → analyze → compute_tf → sparse vector dict"""

    name = "bm25"
    supported_stages = frozenset({STAGE_INGESTION})

    def __init__(self, analyzer):
        self._analyzer = analyzer

    def execute(self, inputs: List[list]) -> List[list]:
        from milvus_lite.analyzer.sparse import compute_tf
        texts = inputs[0]
        sparse_vecs = []
        for text in texts:
            if text is None or not isinstance(text, str):
                sparse_vecs.append({})
            else:
                term_ids = self._analyzer.analyze(text)
                sparse_vecs.append(compute_tf(term_ids))
        return [sparse_vecs]
```

### 6.2 EmbeddingExpr — Text → Dense Vector (Ingestion)

```python
class EmbeddingExpr(FunctionExpr):
    """text → embedding provider → dense vector"""

    name = "text_embedding"
    supported_stages = frozenset({STAGE_INGESTION})

    def __init__(self, provider):
        self._provider = provider

    def execute(self, inputs: List[list]) -> List[list]:
        texts = inputs[0]
        # Batch process non-null texts
        indices = []
        batch = []
        for i, text in enumerate(texts):
            if text is not None and isinstance(text, str) and text:
                indices.append(i)
                batch.append(text)

        vectors = [None] * len(texts)
        if batch:
            embeddings = self._provider.embed_documents(batch)
            for i, emb in zip(indices, embeddings):
                vectors[i] = emb

        # Fill null values with zero vectors
        zero_vec = [0.0] * self._provider.dimension
        for i in range(len(vectors)):
            if vectors[i] is None:
                vectors[i] = zero_vec

        return [vectors]
```

### 6.3 DecayExpr — Numeric → Decay Factor (Rerank)

**Corresponds to Milvus**: `expr/decay_expr.go`

```python
class DecayExpr(FunctionExpr):
    """numeric column → decay factor [0, 1]

    Three decay functions (aligned with Milvus):
    - gauss:  exp(-0.5 * ((max(0, |val-origin|-offset)) / scale)^2)
    - exp:    exp(ln(decay) * max(0, |val-origin|-offset) / scale)
    - linear: max(0, (scale - max(0, |val-origin|-offset)) / scale)
    """

    name = "decay"
    supported_stages = frozenset({STAGE_L2_RERANK})

    def __init__(self, function: str, origin: float, scale: float,
                 offset: float = 0.0, decay: float = 0.5):
        self._function = function  # "gauss" | "exp" | "linear"
        self._origin = origin
        self._scale = scale
        self._offset = offset
        self._decay = decay

    def execute(self, inputs: List[list]) -> List[list]:
        import math
        values = inputs[0]
        factors = []
        for val in values:
            if val is None:
                factors.append(0.0)
                continue
            dist = max(0.0, abs(float(val) - self._origin) - self._offset)
            if self._function == "gauss":
                factor = math.exp(-0.5 * (dist / self._scale) ** 2)
            elif self._function == "exp":
                factor = math.exp(
                    math.log(self._decay) * dist / self._scale
                )
            elif self._function == "linear":
                factor = max(0.0, (self._scale - dist) / self._scale)
            else:
                factor = 0.0
            factors.append(factor)
        return [factors]
```

### 6.4 ScoreCombineExpr — Score Combination (Rerank)

**Corresponds to Milvus**: `expr/score_combine_expr.go`

```python
class ScoreCombineExpr(FunctionExpr):
    """($score, factor) → $score * factor

    Combine multiple score columns into a single final score.
    mode="multiply" is the default behavior for decay reranker.
    """

    name = "score_combine"
    supported_stages = frozenset({STAGE_L2_RERANK})

    def __init__(self, mode: str = "multiply"):
        self._mode = mode  # "multiply" | "sum" | "max" | "min" | "avg"

    def execute(self, inputs: List[list]) -> List[list]:
        n = len(inputs[0])
        results = []
        for row_idx in range(n):
            vals = [col[row_idx] for col in inputs]
            if None in vals:
                results.append(0.0)
                continue
            if self._mode == "multiply":
                r = 1.0
                for v in vals:
                    r *= v
                results.append(r)
            elif self._mode == "sum":
                results.append(sum(vals))
            elif self._mode == "max":
                results.append(max(vals))
            elif self._mode == "min":
                results.append(min(vals))
            elif self._mode == "avg":
                results.append(sum(vals) / len(vals))
            else:
                results.append(0.0)
        return [results]
```

### 6.5 RoundDecimalExpr — Distance Rounding (Rerank)

**Corresponds to Milvus**: `expr/round_decimal_expr.go`

```python
class RoundDecimalExpr(FunctionExpr):
    """$score → round($score, decimal)"""

    name = "round_decimal"
    supported_stages = frozenset({STAGE_INGESTION, STAGE_L2_RERANK})

    def __init__(self, decimal: int):
        self._decimal = decimal

    def execute(self, inputs: List[list]) -> List[list]:
        scores = inputs[0]
        return [[round(s, self._decimal) if s is not None else None
                 for s in scores]]
```

### 6.6 RerankModelExpr — Semantic Reranking (Rerank)

```python
class RerankModelExpr(FunctionExpr):
    """document_text column → relevance_score column

    Calls an external rerank model (e.g., Cohere rerank) to rescore.
    Query texts must be bound at creation time.
    """

    name = "rerank_model"
    supported_stages = frozenset({STAGE_L2_RERANK})

    def __init__(self, provider, query_texts: List[str]):
        self._provider = provider
        self._query_texts = query_texts

    def execute(self, inputs: List[list]) -> List[list]:
        # inputs[0] = document texts for this chunk
        doc_texts = inputs[0]
        # Simplified: assumes each execute call processes one chunk.
        # query_text must come from FuncContext or external binding.
        # Here accessed via _query_texts[chunk_idx].
        # Implementation should pass chunk_idx via FuncContext.
        rerank_results = self._provider.rerank(
            self._query_texts[0], doc_texts, top_n=len(doc_texts)
        )
        scores = [0.0] * len(doc_texts)
        for r in rerank_results:
            scores[r.index] = r.relevance_score
        return [scores]
```

> **Note**: RerankModelExpr needs to know which query text corresponds to the current chunk. Two approaches: (1) FuncContext carries chunk_idx → query_texts mapping; (2) create a separate expr for each nq when building the chain. Implementation will use approach (1), adding a chunk_idx field to FuncContext.

---

## 7. Chain Builder — Build Chains from schema.functions

### 7.1 Ingestion Chain Builder

```python
# function/builder.py

def build_ingestion_chain(schema, field_by_name) -> Optional[FuncChain]:
    """Build ingestion chain from schema.functions.

    Iterates all functions in the schema, adding those that support
    the ingestion stage to the chain in declaration order.

    Returns:
        FuncChain or None (when no ingestion functions exist)
    """
    if not schema.functions:
        return None

    chain = FuncChain("ingestion", STAGE_INGESTION)
    has_steps = False

    for func in schema.functions:
        if func.function_type == FunctionType.BM25:
            in_name = func.input_field_names[0]
            out_name = func.output_field_names[0]
            in_field = field_by_name[in_name]
            analyzer = create_analyzer(in_field.analyzer_params)
            chain.map(BM25Expr(analyzer), [in_name], [out_name])
            has_steps = True

        elif func.function_type == FunctionType.TEXT_EMBEDDING:
            in_name = func.input_field_names[0]
            out_name = func.output_field_names[0]
            provider = create_embedding_provider(func.params)
            chain.map(EmbeddingExpr(provider), [in_name], [out_name])
            has_steps = True

        # Future: other ingestion-stage functions

    return chain if has_steps else None
```

### 7.2 Rerank Chain Builder

**Directly mirrors the 4 chain patterns from Milvus `rerank_builder.go`**:

```python
def build_rerank_chain(
    schema,
    search_params: dict,      # {limit, offset, round_decimal, group_by_field, group_size}
    search_metrics: List[str], # metric type for each search path
) -> Optional[FuncChain]:
    """Build rerank chain from RERANK/DECAY functions in schema.functions.

    4 chain patterns (aligned with Milvus rerank_builder.go):

    RRF:      Merge(RRF) → Sort → Limit → [RoundDecimal] → Select
    Weighted: Merge(Weighted) → Sort → Limit → [RoundDecimal] → Select
    Decay:    Merge(strategy) → Map(Decay) → Map(ScoreCombine) → Sort → Limit → [RoundDecimal] → Select
    Model:    Merge(Max) → Map(RerankModel) → Sort → Limit → [RoundDecimal] → Select
    """
    rerank_func = _find_rerank_function(schema)
    if rerank_func is None:
        return None

    chain = FuncChain("rerank", STAGE_L2_RERANK)
    reranker_type = _get_reranker_type(rerank_func)

    # ── Head: Merge ──
    if reranker_type == "rrf":
        rrf_k = rerank_func.params.get("k", 60.0)
        chain.merge("rrf", rrf_k=rrf_k)

    elif reranker_type == "weighted":
        weights = rerank_func.params.get("weights", [])
        normalize = rerank_func.params.get("norm_score", False)
        chain.merge("weighted", weights=weights,
                    metric_types=search_metrics, normalize=normalize)

    elif reranker_type == "decay":
        score_mode = rerank_func.params.get("score_mode", "max")
        chain.merge(score_mode, metric_types=search_metrics)
        # Map(DecayExpr)
        in_name = rerank_func.input_field_names[0]
        decay_expr = DecayExpr(
            function=rerank_func.params["function"],
            origin=rerank_func.params["origin"],
            scale=rerank_func.params["scale"],
            offset=rerank_func.params.get("offset", 0.0),
            decay=rerank_func.params.get("decay", 0.5),
        )
        chain.map(decay_expr, [in_name], ["_decay_score"])
        # Map(ScoreCombineExpr)
        chain.map(ScoreCombineExpr("multiply"),
                  [SCORE_FIELD, "_decay_score"], [SCORE_FIELD])

    elif reranker_type == "model":
        chain.merge("max")
        in_name = rerank_func.input_field_names[0]
        provider = create_rerank_provider(rerank_func.params)
        model_expr = RerankModelExpr(provider, query_texts=[])  # query_texts injected at execute time
        chain.map(model_expr, [in_name], [SCORE_FIELD])

    # ── Tail: Sort / GroupBy → [RoundDecimal] → Select ──
    group_by_field = search_params.get("group_by_field")
    limit = search_params.get("limit", 10)
    offset = search_params.get("offset", 0)
    round_decimal = search_params.get("round_decimal", -1)

    if group_by_field:
        group_size = search_params.get("group_size", 1)
        chain.group_by(group_by_field, group_size, limit, offset)
    else:
        chain.sort(SCORE_FIELD, desc=True)
        chain.limit(limit, offset)

    if round_decimal >= 0:
        chain.map(RoundDecimalExpr(round_decimal),
                  [SCORE_FIELD], [SCORE_FIELD])

    select_cols = [ID_FIELD, SCORE_FIELD]
    if group_by_field:
        select_cols.extend([group_by_field, "$group_score"])
    chain.select(*select_cols)

    return chain
```

---

## 8. Collection Refactoring

### 8.1 Initialization: 4 Lists → 2 Chains

**Before**:
```python
# ~40 lines of if/elif branches in Collection.__init__
self._bm25_functions: List[Tuple[str, str, Any]] = []
self._embedding_functions: List[Tuple[str, str, Any]] = []
self._rerank_functions: List[Tuple[str, Any]] = []
self._decay_functions: List[Tuple[str, Any]] = []
for func in schema.functions:
    if func.function_type == FunctionType.BM25: ...
    elif func.function_type == FunctionType.TEXT_EMBEDDING: ...
    elif func.function_type == FunctionType.RERANK:
        if reranker_type == "decay": ...
        else: ...
```

**After**:
```python
# Collection.__init__
from milvus_lite.function.builder import build_ingestion_chain

field_by_name = {f.name: f for f in schema.fields}
self._ingestion_chain = build_ingestion_chain(schema, field_by_name)
# rerank chain is built on-demand at search time (depends on search_params)
```

### 8.2 Insert: 4 Lines of Apply → 1 Line of Execute

**Before**:
```python
# in insert()
if self._bm25_functions:
    self._apply_bm25_functions(records)
if self._embedding_functions:
    self._apply_embedding_functions(records)
```

**After**:
```python
# in insert()
if self._ingestion_chain:
    df = DataFrame.from_records(records)
    self._ingestion_chain.execute(df)
    # records are modified in-place, no additional work needed
```

### 8.3 Search Post-Processing: Scattered Rerank/Decay → Rerank Chain

**Before**:
```python
# ~30 lines in search()
if self._query_texts is not None:
    raw_results = self._apply_rerank(raw_results, self._query_texts, ...)
    scores_replaced = True
if self._decay_functions:
    raw_results = self._apply_decay(raw_results, metric_type, scores_replaced)
    scores_replaced = True
# + group_by post-processing
# + offset handling
```

**After**:
```python
# in search() — rerank chain (Hybrid Search scenario)
if rerank_chain:
    # Multi-path search results → DataFrames
    dfs = [DataFrame.from_search_results(r) for r in per_path_results]
    merged = rerank_chain.execute(*dfs)
    raw_results = merged.to_search_results()
    # Sort + Limit + GroupBy + RoundDecimal + Select all handled within the chain
```

### 8.4 Methods to Remove

After chain refactoring, the following Collection methods can be deleted:

| Method | Replacement |
|---|---|
| `_apply_bm25_functions()` | `BM25Expr` + `MapOp` |
| `_apply_embedding_functions()` | `EmbeddingExpr` + `MapOp` |
| `_apply_rerank()` | `RerankModelExpr` + `MapOp` |
| `_apply_decay()` | `DecayExpr` + `ScoreCombineExpr` + `MapOp` |

---

## 9. Data Flow Diagrams

### 9.1 Ingestion Chain Data Flow

```
User records: [{"text": "hello world", "id": 1}, ...]
    │
    ▼
DataFrame.from_records(records)
    │ chunks = [[{"text": "hello world", "id": 1}, ...]]
    │
    ├─ MapOp(BM25Expr, ["text"] → ["sparse_vec"])
    │   │  extract: texts = ["hello world", ...]
    │   │  compute: BM25Expr.execute([texts]) → [sparse_vecs]
    │   │  write:   records[i]["sparse_vec"] = sparse_vecs[i]
    │   ▼
    ├─ MapOp(EmbeddingExpr, ["text"] → ["dense_vec"])
    │   │  extract: texts = ["hello world", ...]
    │   │  compute: EmbeddingExpr.execute([texts]) → [vectors]
    │   │  write:   records[i]["dense_vec"] = vectors[i]
    │   ▼
    ├─ [Future: MapOp(DimReduceExpr, ["dense_vec"] → ["reduced_vec"])]
    │   │  chaining: consumes dense_vec output from previous step
    │   ▼
    │
df.to_records() → modified records, continue to validate → WAL → MemTable
```

### 9.2 Rerank Chain Data Flow (Decay Example)

```
Multi-path search results:
  path_0: [[{$id: 1, $score: 0.9, ts: 100}, {$id: 2, $score: 0.8, ts: 200}], ...]  (dense)
  path_1: [[{$id: 2, $score: -3.5, ts: 200}, {$id: 3, $score: -4.1, ts: 50}], ...]  (BM25)
    │
    ▼
MergeOp(strategy="max", metric_types=["COSINE", "BM25"])
    │  per-query: deduplicate by pk + take highest $score via max strategy
    │  result: [[{$id: 1, $score: 0.9, ts: 100}, {$id: 2, $score: 0.8, ts: 200}, {$id: 3, ...}], ...]
    ▼
MapOp(DecayExpr(gauss, origin=now, scale=86400), ["ts"] → ["_decay_score"])
    │  per-row: _decay_score = gauss(|ts - now|)
    │  result: each hit gains a _decay_score field
    ▼
MapOp(ScoreCombineExpr("multiply"), ["$score", "_decay_score"] → ["$score"])
    │  per-row: $score = $score * _decay_score
    │  result: $score updated
    ▼
SortOp("$score", desc=True)
    │  per-chunk: sort by $score descending
    ▼
LimitOp(limit=10, offset=0)
    │  per-chunk: take first 10
    ▼
MapOp(RoundDecimalExpr(4), ["$score"] → ["$score"])     [optional]
    │  per-row: $score = round($score, 4)
    ▼
SelectOp("$id", "$score")
    │  per-row: keep only $id and $score
    ▼
DataFrame.to_search_results() → final return value
```

---

## 10. Module Structure

### New `function/` Package

```
milvus_lite/function/
├── __init__.py           # Public API: FuncChain, build_ingestion_chain, build_rerank_chain
├── types.py              # FunctionExpr ABC, FuncContext, Stage constants, column name constants
├── dataframe.py          # DataFrame
├── chain.py              # FuncChain
├── operator.py           # Operator ABC
├── ops/
│   ├── __init__.py
│   ├── map_op.py         # MapOp
│   ├── merge_op.py       # MergeOp
│   ├── sort_op.py        # SortOp
│   ├── limit_op.py       # LimitOp
│   ├── select_op.py      # SelectOp
│   └── group_by_op.py    # GroupByOp
├── expr/
│   ├── __init__.py
│   ├── bm25_expr.py      # BM25Expr
│   ├── embedding_expr.py # EmbeddingExpr
│   ├── decay_expr.py     # DecayExpr
│   ├── score_combine.py  # ScoreCombineExpr
│   ├── round_decimal.py  # RoundDecimalExpr
│   └── rerank_model.py   # RerankModelExpr
└── builder.py            # build_ingestion_chain, build_rerank_chain
```

### Modified Existing Modules

| Module | Changes |
|---|---|
| `engine/collection.py` | Remove 4 function lists + 4 apply methods; use `_ingestion_chain` instead |
| `engine/collection.py` | Refactor search post-processing logic to use rerank chain |

---

## 11. Sub-Phase Breakdown

| Sub-Phase | Content | Deliverables |
|---|---|---|
| **FC-1** | Core framework | `types.py` (FunctionExpr, FuncContext, constants), `dataframe.py` (DataFrame), `operator.py` (Operator ABC), `chain.py` (FuncChain), unit tests |
| **FC-2** | MapOp + Ingestion Exprs | `ops/map_op.py`, `expr/bm25_expr.py`, `expr/embedding_expr.py`, `builder.py` (build_ingestion_chain), unit tests |
| **FC-3** | Collection insert refactor | Remove `_bm25_functions` / `_embedding_functions` / `_apply_*` methods, use ingestion chain instead; **all regression tests green** |
| **FC-4** | Rerank Operators | `ops/merge_op.py`, `ops/sort_op.py`, `ops/limit_op.py`, `ops/select_op.py`, `ops/group_by_op.py`, unit tests |
| **FC-5** | Rerank Exprs | `expr/decay_expr.py`, `expr/score_combine.py`, `expr/round_decimal.py`, `expr/rerank_model.py`, unit tests |
| **FC-6** | Rerank Chain Builder | `builder.py` (build_rerank_chain), integration tests for all 4 chain patterns |
| **FC-7** | Collection search refactor | Remove `_apply_rerank` / `_apply_decay` / inline group_by / offset logic, use rerank chain instead; **all regression tests green** |
| **FC-8** | gRPC Hybrid Search integration | `servicer.py` HybridSearch RPC uses rerank chain instead of inline reranker; pymilvus end-to-end tests |

### Verification Strategy

| Sub-Phase | Test Focus |
|---|---|
| FC-1 | DataFrame create/export/column read-write; FuncChain empty chain / single op / multi op execution |
| FC-2 | BM25Expr tokenization+TF correctness; EmbeddingExpr batch + null handling; MapOp column mapping correctness |
| FC-3 | **All existing insert-related tests pass without regression** (BM25 insert, embedding insert, mixed insert) |
| FC-4 | MergeOp 5 strategies: dedup + score merging correctness; SortOp ascending/descending/None handling; LimitOp offset boundaries; SelectOp column filtering; GroupByOp grouping + scorer |
| FC-5 | DecayExpr 3 decay functions compared with hand-computed values; ScoreCombineExpr multiply/sum/max; RoundDecimalExpr precision |
| FC-6 | 4 rerank chains (RRF/Weighted/Decay/Model) end-to-end input/output verification |
| FC-7 | **All existing search/rerank tests pass without regression**; chain path results match original path results |
| FC-8 | pymilvus hybrid_search end-to-end tests |

### Completion Criteria

- `milvus_lite/function/` package fully implemented
- No per-type function branch code in Collection
- 4 rerank chains (RRF / Weighted / Decay / Model) automatically built by builder
- Adding a new function type requires only: new FunctionExpr subclass + registration in builder
- **1529+ tests green, 0 regressions**
