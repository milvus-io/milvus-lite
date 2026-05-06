# Deep Design: Scalar Filter Expression System (Scalar Filter)

## 1. Overview

MilvusLite Phase 8 introduces a Milvus-style scalar filter expression system, allowing `Collection.search` /
`get` / `query` to accept predicate expressions in string form (e.g., `"age > 18 and category == 'tech'"`),
enabling hybrid queries combining "vector recall + scalar filtering".

**Why build it ourselves**: pymilvus only passes the expression string through to the Milvus server — all lex/parse/eval
is done server-side. MilvusLite is embedded, with no "server", so we must implement the complete lexer + parser
+ type checker + evaluator ourselves.

**Why "Milvus-inspired" rather than binary compatible**:
- Milvus grammar changes across versions (expressions differ between 2.3 and 2.4)
- The full grammar includes JSON paths, array operations, UDFs and other rarely-used features
- "Milvus-like" + documenting the subset we support is sufficient for users migrating from pymilvus
- Strict compatibility will only be considered from F3+ onward (at which point we may switch to an ANTLR-generated parser)

---

## 2. Three-Stage Compilation Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  source string                                               │
│      "age > 18 and category in ['tech', 'news']"             │
│                              │                               │
│                              │  parse_expr(s)                │
│                              ▼                               │
│  Expr (raw AST)                                              │
│      And(operands=(CmpOp("==", ...), InOp(...)))             │
│                              │                               │
│                              │  compile_expr(expr, schema)   │
│                              ▼                               │
│  CompiledExpr  ─── field binding + type checking + backend selection │
│                              │                               │
│                              │  evaluate(compiled, table)    │
│                              ▼                               │
│  pa.BooleanArray (length == table.num_rows)                  │
└──────────────────────────────────────────────────────────────┘
```

**Why three stages**:
- **parse is schema-independent**: the same expression can be cached (F2c optimization)
- **compile is data-independent**: bind once, execute many times
- **evaluate is the hot path**: only performs backend dispatch, zero parsing overhead

---

## 3. Grammar Subset (Tier 1, aligned with Milvus Plan.g4)

### 3.1 Operators + Precedence

| Prec | Operator | Associativity | Notes |
|---|---|---|---|
| 1 | `or`, `OR`, `\|\|` | left | |
| 2 | `and`, `AND`, `&&` | left | |
| 3 | `not`, `NOT`, `!` | right (prefix) | |
| 4 | `==`, `!=`, `<`, `<=`, `>`, `>=` | left | Chained comparisons accepted by parser, rejected by semantic check |
| 4 | `in [...]`, `not in [...]` | non-assoc | RHS must be a literal array |
| 4 | `like` | non-assoc | SQL LIKE wildcards `%` / `_` (F2a) |
| 4 | `is null`, `is not null` | non-assoc | Null check (F2a) |
| 5 | `+`, `-`, `*`, `/`, `%` | left | Arithmetic operations (F2a) |
| 5 | `-` (unary) | right | Only `Unary(SUB, expr)` |
| 6 | `text_match(field, query)` | — | Full-text search match (Phase 11) |
| 6 | `array_contains(field, val)` | — | Array contains single value (F3) |
| 6 | `array_contains_all(field, [vals])` | — | Array contains all values (F3) |
| 6 | `array_contains_any(field, [vals])` | — | Array contains any value (F3) |
| 6 | `array_length(field)` | — | Array length (F3) |
| 7 | literal / ident / `(...)` | — | |

### 3.2 Literals

| Type | Syntax | Examples |
|---|---|---|
| Integer | Decimal | `42`, `0`, `-7` (negative sign is unary) |
| Float | Decimal + scientific notation | `3.14`, `1e3`, `1.5e-2`, `-0.5` |
| String | Double/single quotes + C-style escape | `"hello"`, `'world'`, `"a\"b"` |
| Boolean | Three forms (consistent with Milvus) | `true`/`True`/`TRUE`, `false`/`False`/`FALSE` |
| Array | `[lit, lit, ...]` with trailing comma | `[1, 2, 3]`, `["a", "b",]` |

**Note**: Milvus accepts `True`/`true`/`TRUE` in three forms but **does not accept** `tRuE`. F1 is consistent with Milvus:
mixed-case forms other than these 6 are rejected at the lex stage with a did-you-mean hint.

### 3.3 String Escape

| Escape | Meaning |
|---|---|
| `\"` | Double quote |
| `\'` | Single quote |
| `\\` | Backslash |
| `\n` | Newline |
| `\r` | Carriage return |
| `\t` | Tab |

F1 does not yet support `\xHH`, `\uXXXX`, `\OOO` octal and other rare escapes (Milvus supports them, deferred to F2).

### 3.4 Identifier Rules

- `[a-zA-Z_][a-zA-Z_0-9]*`
- **Case-sensitive** (consistent with Milvus)
- Keyword case is **not** sensitive (`and == AND == &&`, but `And` is not accepted)
- **Reserved prefixes** `_seq` / `_partition`: allowed at parse stage, rejected at semantic stage
- **`$meta`**: Completely rejected in F1 (deferred to F2b)

### 3.5 Whitespace and Comments

- Whitespace: ` `, `\t`, `\r`, `\n` all skipped
- **No comments** (consistent with Milvus)

### 3.6 Complete BNF (subset implemented in F1)

```
expr            : or_expr ;
or_expr         : and_expr (OR and_expr)* ;
and_expr        : not_expr (AND not_expr)* ;
not_expr        : NOT not_expr | term ;
term            : cmp_term | in_term | unary | primary ;
cmp_term        : (unary | primary) CMP_OP (unary | primary) ;
in_term         : Identifier (NOT)? IN array_literal ;
unary           : SUB primary ;        // -7, -age
primary         : literal
                | Identifier
                | '(' expr ')'
                ;
literal         : INT | FLOAT | STRING | BOOL ;
array_literal   : '[' (literal (',' literal)* (',')?)? ']' ;

CMP_OP          : '==' | '!=' | '<' | '<=' | '>' | '>=' ;

// Lexer
INT             : [1-9][0-9]* | '0' ;
FLOAT           : [0-9]+ '.' [0-9]+ ([eE][+-]?[0-9]+)?
                | [0-9]+ [eE][+-]?[0-9]+ ;
STRING          : '"' DoubleStrChar* '"' | "'" SingleStrChar* "'" ;
BOOL            : 'true' | 'True' | 'TRUE' | 'false' | 'False' | 'FALSE' ;

AND             : 'and' | 'AND' | '&&' ;
OR              : 'or'  | 'OR'  | '||' ;
NOT             : 'not' | 'NOT' | '!' ;
IN              : 'in'  | 'IN' ;
SUB             : '-' ;

Identifier      : [a-zA-Z_][a-zA-Z_0-9]* ;
Whitespace      : [ \t\r\n]+ -> skip ;
```

---

## 4. AST Nodes

20 frozen dataclasses, all with value semantics, hashable, automatic `__eq__`. See `modules.md §9.21` for details.

```
Literal:    IntLit, FloatLit, StringLit, BoolLit
List:       ListLit
Reference:  FieldRef
Operations: CmpOp, InOp, And, Or, Not
Arithmetic: ArithOp
Pattern:    LikeOp
Null test:  IsNullOp
Dynamic:    MetaAccess, JsonAccess
FTS:        TextMatchOp
Array:      ArrayContainsOp, ArrayLengthOp, ArrayAccessOp
```

**Key design decisions**:
- Use `tuple` instead of `list` (frozen-friendly)
- No common base class — use `Union` + `isinstance` dispatch (consistent with the Operation abstraction)
- No methods — behavior lives in backends
- Each node carries `pos` for error message traceability

---

## 5. Compilation Stage: semantic.py

### 5.1 Compilation Steps

```
1. Walk AST → collect all FieldRef
2. For each FieldRef:
   - lookup in schema.fields
   - if not found → FilterFieldError with did-you-mean
   - if reserved (_seq / _partition / $meta) → FilterFieldError
   - if FLOAT_VECTOR → FilterTypeError
3. Walk AST again → infer + check types
4. Choose backend:
   - F1: always "arrow"
   - F2b/F3+: contains $meta reference → "hybrid" (per-batch JSON preprocessing then delegate to arrow_backend)
   - F3: contains UDF → "python"
5. Wrap in CompiledExpr
```

### 5.2 Type Inference + Compatibility

```
int  ≈ int     ✓
int  ≈ float   ✓ (promotion)
str  ≈ str     ✓
bool ≈ bool    ✓
other          ✗
```

Chained comparisons `a == b == c` are **not** rejected at the parse stage (consistent with Milvus); the semantic stage
reports a type error: "left side is bool (result of `a == b`), right side is int — comparison
between bool and int not supported".

### 5.3 Error Message Requirements

Error messages are the face of the parser. F1 must achieve:

```
>>> col.search([[...]], expr="age >> 18")
FilterParseError: unexpected token '>' at column 5
  age >> 18
      ^
expected: expression after '>'
```

```
>>> col.search([[...]], expr="ag > 18")
FilterFieldError: unknown field 'ag' at column 1
  ag > 18
  ^^
available fields: [id, age, category, score]
did you mean 'age'?
```

```
>>> col.search([[...]], expr="age > 'eighteen'")
FilterTypeError: type mismatch at column 7
  age > 'eighteen'
        ^^^^^^^^^
left side is int (field 'age'), right side is string
```

Implementation key points:
- All exceptions inherit from `MilvusLiteError`, allowing users to catch with a single handler
- Exceptions carry `source: str` + `pos: int`, `__str__` automatically renders caret
- "did you mean" uses `difflib.get_close_matches` (standard library)
- Each error specifies **field name** and **type**, not just "type mismatch"

---

## 6. Backend Design

### 6.1 Three-Backend Decision

| Backend | Purpose | Speed (100K rows) |
|---|---|---|
| `arrow_backend` | Pure schema field expressions (all of F1+F2a) | ~5ms |
| `hybrid_backend` | Expressions containing `$meta` dynamic fields (F3+) | ~50–100ms |
| `python_backend` | Differential test baseline + hybrid fallback + future UDF | ~500ms |

**Backend is statically determined at compile time** — no dispatch on the evaluate hot path.
- Pure schema fields → `arrow`
- Contains `$meta` → `hybrid` (per-batch JSON preprocessing then delegate to arrow_backend)
- `python` is never automatically selected by the dispatcher, serving only as:
  1. Differential test baseline in test_e2e
  2. Runtime fallback for hybrid_backend when encountering heterogeneous JSON types / incompatible arrow kernels
  3. Final destination for future F3 UDF / truly dynamic semantics

### 6.2 arrow_backend Implementation Strategy

**AST → pyarrow.compute call tree**. pyarrow.compute is a vectorized C++ implementation with ready-made
kernels for all comparisons / boolean operations / IN.

| AST | pyarrow operation |
|---|---|
| Literal | `pa.scalar(value)` |
| FieldRef | `table.column(name)` |
| CmpOp | `pc.equal / less / ...` |
| InOp | `pc.is_in(col, value_set=values)` + optional `pc.invert` |
| And | `functools.reduce(pc.and_kleene, masks)` |
| Or | `functools.reduce(pc.or_kleene, masks)` |
| Not | `pc.invert` |

**Key details**:
- Use `and_kleene` / `or_kleene` instead of `and_` / `or_`: pyarrow recommends Kleene
  three-valued logic for nullable data
- Literals use `pa.scalar`, compute kernels accept array vs scalar with automatic broadcasting
- Top-level result calls `pc.fill_null(False)`: null means "no information", treated as false in filter semantics

### 6.3 python_backend Implementation Strategy

Row-wise interpreter: converts pa.Table to list of dicts, calls Python eval on each row.

```python
def evaluate_python(compiled, data) -> pa.BooleanArray:
    rows = data.to_pylist()
    out = [False] * len(rows)
    for i, row in enumerate(rows):
        result = _eval_row(compiled.ast, row)
        out[i] = bool(result) if result is not None else False
    return pa.array(out, type=pa.bool_())
```

NULL three-valued logic: uses Kleene implementation for AND/OR/NOT, finally None → False.

**Performance**: 100K rows ~500ms. Slow but general-purpose. From F3+ onward, no longer automatically selected by the dispatcher, serving only as differential baseline + hybrid fallback.

### 6.3a hybrid_backend Implementation Strategy (F3+)

`$meta["key"]` was initially implemented in F2b as `python_backend` doing direct row-wise interpretation — paying
the dual cost of "AST walk + JSON parse" per row, 100K rows ~500ms. F3+ introduces hybrid_backend:

**Idea**: Lift JSON parsing and column materialization to a one-time per-batch operation, letting comparisons/arithmetic/boolean logic still run through arrow vectorization.

**Steps**:
1. `collect_meta_keys(ast)` scans the AST once to collect the set of all `$meta["key"]` keys
2. `_augment_table(data, keys)`:
   - One `to_pylist()` call to extract the `$meta` column
   - `json.loads` once per row (tolerating None / dict / malformed JSON)
   - For each key, use `pa.array([d.get(key) for d in parsed])` to materialize an Arrow column
   - Append to the original table, with column naming convention `__meta__<key>` (double underscore prefix to avoid conflicts)
3. `_rewrite_meta_access(ast, keys)`: replaces all `MetaAccess(key)` nodes in the AST with
   `FieldRef("__meta__<key>")`, producing a new AST
4. Use `dataclasses.replace` to temporarily change the backend to `"arrow"`, then call `evaluate_arrow`

**Performance**: 100K rows ~50–100ms (bottleneck shifts from row-wise Python to JSON parsing), roughly 5–10x improvement.

**Fallback**: The entire augment + arrow eval is wrapped in try/except. Any failure (heterogeneous types / all-null
columns without matching kernel / arrow-unsupported type conversions) → falls back to `python_backend` for this single evaluate call.
Fallback is per-evaluate rather than per-row, keeping overhead manageable.

**Correctness**: Differential tests (`test_meta_hybrid_vs_python_parity`) run through 14 `$meta` expressions,
comparing hybrid and python output row by row. The semantic source of truth is `python_backend`; any hybrid deviation causes test failure.

### 6.4 Differential Testing

In `test_e2e.py`, each case **runs both backends simultaneously**, asserting equal results:

```python
@pytest.mark.parametrize("expr_str", [...50+ cases...])
def test_arrow_python_equivalence(expr_str, sample_table, sample_schema):
    expr = parse_expr(expr_str)
    compiled = compile_expr(expr, sample_schema)

    arrow_result = evaluate_arrow(compiled, sample_table)

    py_compiled = CompiledExpr(
        ast=compiled.ast, fields=compiled.fields, backend="python",
    )
    py_result = evaluate_python(py_compiled, sample_table)

    assert arrow_result.equals(py_result), \
        f"backend mismatch on '{expr_str}'"
```

**Why differential testing is critical**:
1. Two implementations cross-validate each other — bugs in either side are exposed by the other
2. NULL three-valued logic, type promotion, boundary values — these error-prone areas are caught by symmetry
3. After F2b introduces `$meta`, differential testing naturally extends to verify "whether backend selection chose correctly"

---

## 7. Integration with the Existing Search Pipeline

### 7.1 Data Flow

```
        ┌──────────────────────────────────┐
        │  Collection.search(expr=...)     │
        └────────────┬─────────────────────┘
                     │ parse + compile
                     ▼
        ┌──────────────────────────────────┐
        │  assemble_candidates(            │
        │    filter_compiled=...           │
        │  )                               │
        │                                  │
        │  per source (segment / memtable):│
        │    bool_arr = evaluate(          │
        │      compiled, source_table      │
        │    )                             │
        │    chunks.append(bool_arr)       │
        │                                  │
        │  filter_mask = concat(chunks)    │
        └────────────┬─────────────────────┘
                     │ returns + filter_mask
                     ▼
        ┌──────────────────────────────────┐
        │  execute_search(                 │
        │    filter_mask=...               │
        │  )                               │
        │                                  │
        │  build_valid_mask(               │
        │    dedup + tombstone             │
        │      + filter_mask               │
        │  )                               │
        │                                  │
        │  for each query: distance + topk │
        └──────────────────────────────────┘
```

### 7.2 bitmap.py Changes

```python
def build_valid_mask(
    all_pks, all_seqs, delta_index,
    filter_mask: Optional[np.ndarray] = None,  # NEW
) -> np.ndarray:
    mask = ...  # existing dedup + tombstone
    if filter_mask is not None:
        mask = mask & filter_mask
    return mask
```

### 7.3 assembler.py Changes

```python
def assemble_candidates(
    segments, memtable, vector_field,
    partition_names=None,
    filter_compiled=None,  # NEW
):
    ...
    filter_chunks = []
    for segment in scoped_segments:
        if filter_compiled:
            bool_arr = filter.evaluate(filter_compiled, segment.table)
            filter_chunks.append(bool_arr.to_numpy(zero_copy_only=False))
        ...
    if filter_compiled and mt_pks:
        mt_table = _memtable_to_table(memtable, partition_names)
        bool_arr = filter.evaluate(filter_compiled, mt_table)
        filter_chunks.append(bool_arr.to_numpy(zero_copy_only=False))

    filter_mask = np.concatenate(filter_chunks) if filter_chunks else None
    return all_pks, all_seqs, all_vectors, all_records, filter_mask
```

**Why filter is in assembler rather than bitmap**:
- Data is already in pa.Table form (segments hold the original Table)
- pyarrow.compute needs columnar input; the bitmap stage has already been numpy-ified
- Keeps bitmap.py as pure numpy (consistent with distance / executor)

`assembler` is the **only module in the search subsystem that knows both the storage types (Segment, MemTable) and
the filter subsystem**.

---

## 8. Collection API Upgrade

```python
class Collection:
    def search(
        self,
        query_vectors: List[list],
        top_k: int = 10,
        metric_type: str = "COSINE",
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,        # NEW
    ) -> List[List[dict]]: ...

    def get(
        self,
        pks: List[Any],
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,        # NEW
    ) -> List[dict]: ...

    def query(                              # NEW METHOD
        self,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Pure scalar query — no vector. Returns all matching rows."""

    def _compile_filter(self, expr_str: str) -> CompiledExpr:
        from milvus_lite.search.filter import parse_expr, compile_expr
        return compile_expr(parse_expr(expr_str), self._schema)
```

`query()` is a new method — pure scalar query, equivalent to `search(query=None, expr=...)` but does not require
a query vector, does not compute distances, and directly returns all matching rows (with optional `limit`).

---

## 9. Phase 8 Sub-Phase Breakdown

| Phase | Target Grammar | Backend | Status |
|---|---|---|---|
| **F1** | Tier 1: comparison + boolean + IN + literals + field reference + parentheses | arrow_backend only; python_backend only for differential testing | ✅ done |
| **F2a** | + `like` + arithmetic (`+ - * / %`) + `is null` | Still arrow_backend | ✅ done |
| **F2b** | + `$meta["key"]` dynamic fields | Introduces python_backend dispatch | ✅ done |
| **F2c** | filter LRU cache + `query()` integration | Backend-independent | ✅ done |
| **F3+** | Performance optimization: per-batch JSON preprocessing → arrow_backend; hybrid replaces python as default $meta dispatch | Introduces hybrid_backend | ✅ done |
| **F3** | + `json_contains` / `array_contains` / UDF / strict Milvus compatibility | Extends python_backend; optional ANTLR parser swap | — |

---

## 10. About ANTLR

Milvus uses ANTLR4 + Plan.g4 to generate a C++ parser. For F1, we chose a **hand-written Pratt parser** rather
than introducing the ANTLR Python target, for these reasons:

1. **F1 grammar is small** (10 operators), hand-written ~300 lines of Python, debug-friendly
2. **Zero dependencies** (no need to introduce antlr4-python3-runtime + 1500 lines of generated code)
3. **Error messages are controllable** (hand-written caret + did-you-mean is simpler than ANTLR's override BaseErrorListener)
4. **AST is a stable interface** — when switching to an ANTLR backend in F3, the type checker / evaluator remain unchanged

However, we **borrow from Milvus Plan.g4's grammar design**: the operator precedence table, keyword casing, literal syntax,
and AST node shapes all align with Milvus (making it easier to swap the parser implementation if true binary compatibility is ever needed).

Reference: [milvus-io/milvus Plan.g4](https://github.com/milvus-io/milvus/blob/master/internal/parser/planparserv2/Plan.g4)

---

## 11. Not in Phase F1 Scope

| Feature | Deferred to |
|---|---|
| `like` operator | F2a |
| Arithmetic (`+, -, *, /, %`) | F2a |
| `is null` / `is not null` | F2a |
| `$meta` dynamic fields | F2b |
| JSON / array functions | F3 |
| UDF | F3 |
| Expression cache | F2c |
| ANTLR-based parser | F3+ |
| DuckDB backend | F3+ |

Operators outside the F1 grammar are rejected at the lex/parse stage with a "Phase F2/F3 will support" hint,
rather than failing silently.

---

## 12. Completion Criteria

- **F1 done**:
  - `col.search([[...]], expr="age > 18 and category in ['tech', 'news']")` runs successfully
  - Differential testing with 50+ cases all green
  - `examples/m8_demo.py` passes
  - `Collection.search` / `get` / `query` — all three methods accept expr
  - Error messages include caret + did-you-mean

- **F2 done**:
  - `col.search(expr="title like 'AI%' and $meta['priority'] > 5")` runs successfully
  - filter LRU cache + `query()` integration

- **F3+ done**:
  - hybrid_backend replaces python_backend as default `$meta` dispatch
  - Differential testing hybrid vs python is consistent across all `$meta` expressions
  - Heterogeneous JSON types / all-null columns and other anomalies automatically fall back to python_backend

- **F3 done**:
  - Passes a subset of the pymilvus expression test suite
  - Optional ANTLR backend
