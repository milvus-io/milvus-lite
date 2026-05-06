"""Row-wise Python backend for filter evaluation.

Slow but flexible — used as the differential-test baseline in F1 and
as the actual fallback in F2b ($meta dynamic field) and F3 (UDFs).

NULL semantics: Kleene three-valued logic. The internal _eval_row
returns Python `bool`, `True`, `False`, or `None` (= unknown). The
top-level entry collapses None → False so the caller gets a clean
BooleanArray with no null entries.
"""

from __future__ import annotations

import functools
import json
import operator
import re
from typing import TYPE_CHECKING, Any, Optional, Union

import pyarrow as pa

from milvus_lite.search.filter.ast import (
    And,
    ArithOp,
    BoolLit,
    CmpOp,
    FieldRef,
    FloatLit,
    InOp,
    IntLit,
    IsNullOp,
    LikeOp,
    ListLit,
    MetaAccess,
    Not,
    Or,
    StringLit,
    JsonAccess,
    TextMatchOp,
    ArrayContainsOp,
    ArrayLengthOp,
    ArrayAccessOp,
)

if TYPE_CHECKING:
    from milvus_lite.search.filter.semantic import CompiledExpr


_CMP_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<":  operator.lt,
    "<=": operator.le,
    ">":  operator.gt,
    ">=": operator.ge,
}

# Phase F2a — arithmetic
_ARITH_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def _like_to_regex(pattern: str) -> "re.Pattern[str]":
    """Translate a SQL LIKE pattern to a Python regex (anchored).

    Wildcards:
        '%' → '.*'
        '_' → '.'
    All other characters are regex-escaped. Escape character support
    (e.g. `\\%` for a literal %) is deferred to F3.
    """
    out: list[str] = []
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return re.compile("^" + "".join(out) + "$", re.DOTALL)


# Cache compiled LIKE patterns to avoid recompiling per row.
# Use functools.lru_cache to bound memory in long-running processes.
@functools.lru_cache(maxsize=1024)
def _get_like_regex(pattern: str) -> "re.Pattern[str]":
    return _like_to_regex(pattern)


def evaluate_python(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Row-wise interpreter. Returns BooleanArray of length data.num_rows.

    None (Kleene unknown) at the top level becomes False (no row matches).
    """
    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data])

    rows = data.to_pylist()
    out = [False] * len(rows)
    for i, row in enumerate(rows):
        result = _eval_row(compiled.ast, row)
        if result is None:
            out[i] = False
        else:
            out[i] = bool(result)
    return pa.array(out, type=pa.bool_())


def _eval_row(node, row: dict) -> Any:
    """Evaluate a single AST node against a single row dict.

    Returns:
        - Python int / float / str / bool for literals and field refs
        - True / False for boolean operations
        - None for Kleene "unknown" (e.g., comparing a NULL field)
    """
    if isinstance(node, IntLit):
        return node.value
    if isinstance(node, FloatLit):
        return node.value
    if isinstance(node, StringLit):
        return node.value
    if isinstance(node, BoolLit):
        return node.value

    if isinstance(node, FieldRef):
        return row.get(node.name)

    if isinstance(node, CmpOp):
        left = _eval_row(node.left, row)
        right = _eval_row(node.right, row)
        if left is None or right is None:
            return None  # NULL propagation (Kleene)
        try:
            return _CMP_OPS[node.op](left, right)
        except TypeError:
            # Cross-type comparison at runtime — fall back to None.
            # (Should not happen if semantic.py did its job, but robust.)
            return None

    if isinstance(node, InOp):
        val = _eval_row(node.field, row)
        if val is None:
            return None  # NULL propagation (Kleene logic)
        members = {el.value for el in node.values.elements}
        result = val in members
        return (not result) if node.negate else result

    if isinstance(node, And):
        # Kleene AND:
        #   any False → False
        #   else any None → None
        #   else True
        seen_null = False
        for op in node.operands:
            r = _eval_row(op, row)
            if r is False:
                return False
            if r is None:
                seen_null = True
        return None if seen_null else True

    if isinstance(node, Or):
        # Kleene OR:
        #   any True  → True
        #   else any None → None
        #   else False
        seen_null = False
        for op in node.operands:
            r = _eval_row(op, row)
            if r is True:
                return True
            if r is None:
                seen_null = True
        return None if seen_null else False

    if isinstance(node, Not):
        r = _eval_row(node.operand, row)
        if r is None:
            return None
        return not r

    # ── Phase F2a ───────────────────────────────────────────────
    if isinstance(node, ArithOp):
        left = _eval_row(node.left, row)
        right = _eval_row(node.right, row)
        if left is None or right is None:
            return None
        try:
            return _ARITH_OPS[node.op](left, right)
        except (TypeError, ZeroDivisionError):
            return None

    if isinstance(node, LikeOp):
        value = _eval_row(node.value, row)
        if value is None:
            return None
        if not isinstance(value, str):
            return False
        regex = _get_like_regex(node.pattern.value)
        return regex.match(value) is not None

    if isinstance(node, IsNullOp):
        # IS NULL on a missing key in row dict counts as null too.
        val = row.get(node.field.name)
        is_null = val is None
        return (not is_null) if node.negate else is_null

    # ── Phase F2b: $meta dynamic field access ───────────────────
    if isinstance(node, MetaAccess):
        meta_str = row.get("$meta")
        if meta_str is None:
            return None
        # The $meta column may be a pre-parsed dict (in test fixtures)
        # or a JSON string (the production wal/parquet representation).
        if isinstance(meta_str, dict):
            return meta_str.get(node.key)
        if not isinstance(meta_str, str):
            return None
        try:
            d = json.loads(meta_str)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(d, dict):
            return None
        return d.get(node.key)

    # ── JSON field path access ─────────────────────────────────
    if isinstance(node, JsonAccess):
        field_val = row.get(node.field_name)
        if field_val is None:
            return None
        # field_val may be a JSON string or already-parsed dict
        if isinstance(field_val, str):
            try:
                field_val = json.loads(field_val)
            except (json.JSONDecodeError, ValueError):
                return None
        # Walk the keys tuple for chained access: info["a"]["b"]
        for key in node.keys:
            if isinstance(field_val, dict):
                field_val = field_val.get(key)
            else:
                return None
        return field_val

    # ── Phase 11.6: text_match ──────────────────────────────────
    if isinstance(node, TextMatchOp):
        field_val = row.get(node.field.name)
        if field_val is None or not isinstance(field_val, str):
            return False
        # Tokenize both field value and query using StandardAnalyzer
        from milvus_lite.analyzer.standard import StandardAnalyzer
        analyzer = StandardAnalyzer()
        doc_tokens = set(analyzer.tokenize(field_val))
        query_tokens = set(analyzer.tokenize(node.query.value))
        if not query_tokens:
            return False
        # OR logic: match if any query token is in doc tokens
        return bool(doc_tokens & query_tokens)

    # ── Array functions ─────────────────────────────────────────
    if isinstance(node, ArrayContainsOp):
        arr = _eval_row(node.field, row)
        if arr is None or not isinstance(arr, (list, tuple)):
            return False
        if node.mode == "any_one":
            # array_contains(field, single_value)
            target = _eval_row(node.values, row)
            return target in arr
        # array_contains_all / array_contains_any with list
        if isinstance(node.values, ListLit):
            targets = [_eval_row(e, row) for e in node.values.elements]
        else:
            v = _eval_row(node.values, row)
            targets = v if isinstance(v, (list, tuple)) else [v]
        if node.mode == "all":
            return all(t in arr for t in targets)
        # mode == "any"
        return any(t in arr for t in targets)

    if isinstance(node, ArrayLengthOp):
        arr = row.get(node.field.name)
        if arr is None or not isinstance(arr, (list, tuple)):
            return 0
        return len(arr)

    if isinstance(node, ArrayAccessOp):
        arr = row.get(node.field_name)
        if arr is None or not isinstance(arr, (list, tuple)):
            return None
        idx = node.index
        if 0 <= idx < len(arr):
            return arr[idx]
        return None

    raise TypeError(f"unknown AST node: {type(node).__name__}")
