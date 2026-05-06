"""Hybrid backend — per-batch JSON preprocessing + arrow_backend.

Phase F3+ optimization for $meta dynamic-field expressions.

Why: python_backend evaluates $meta row-by-row, paying ~AST-walk +
JSON-parse overhead for every row. For 100K rows on a typical
expression that's ~500-1000ms. The per-batch approach amortizes
JSON parsing into a single pass and lets the bulk of the comparison
/ arithmetic / boolean work run vectorized in arrow_backend.

Strategy:
    1. Collect every distinct ``$meta["key"]`` reference in the AST
    2. Pull the $meta column out of the table once (Arrow → Python)
    3. json.loads each row's $meta string ONCE
    4. For each referenced key, materialize an Arrow column from
       parsed[i].get(key)
    5. Append those columns to a copy of the table with synthetic
       names like ``__meta__category``
    6. Rewrite the AST: MetaAccess(key) → FieldRef("__meta__key")
    7. Delegate to arrow_backend on the augmented table

Performance: 100K rows + simple expression goes from ~500ms (python
backend) to ~50-100ms (this backend). Bottleneck shifts from per-row
Python overhead to JSON parsing — still better than per-row + AST walk.

Robustness: if pyarrow can't infer a uniform type for a key (e.g., the
JSON has mixed int/string values), we catch the exception at column-
build time and fall back to evaluating that single expression with
python_backend. The fallback is per-evaluate(), not per-row.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Set, Union

import pyarrow as pa

from milvus_lite.search.filter.ast import (
    And,
    ArithOp,
    BoolLit,
    CmpOp,
    Expr,
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
)

if TYPE_CHECKING:
    from milvus_lite.search.filter.semantic import CompiledExpr


# Synthetic column-name prefix for materialized $meta keys. The double
# underscore is unusual enough that it won't collide with user fields
# (we already reject schema field names that start with `_`-style
# reserved tokens like `_seq` / `_partition`).
_META_PREFIX = "__meta__"


def evaluate_hybrid(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Evaluate a $meta-bearing expression via per-batch preprocessing.

    On any preprocessing failure (e.g. malformed JSON, mixed types),
    falls back to python_backend for this evaluation only.
    """
    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data])

    keys = collect_meta_keys(compiled.ast)
    if not keys and not _has_python_only_nodes(compiled.ast):
        # Defensive: dispatcher selected hybrid but the expression has
        # no $meta or python-only nodes. Just delegate to arrow_backend.
        from milvus_lite.search.filter.eval.arrow_backend import evaluate_arrow
        return evaluate_arrow(compiled, data)

    from milvus_lite.search.filter.eval.arrow_backend import evaluate_arrow
    from milvus_lite.search.filter.eval.python_backend import evaluate_python

    try:
        augmented = _augment_table(data, keys)
        rewritten = _rewrite_meta_access(compiled.ast, keys)
        # Build a temporary CompiledExpr with the rewritten AST and an
        # arrow backend tag, then call evaluate_arrow. We don't mutate
        # the original CompiledExpr.
        from dataclasses import replace
        tmp = replace(compiled, ast=rewritten, backend="arrow")
        return evaluate_arrow(tmp, augmented)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError,
            TypeError, NotImplementedError, KeyError) as _exc:
        # Any failure on the fast path — heterogeneous JSON types,
        # all-null synthetic column with no compatible arrow kernel,
        # or any other type-mismatch surfacing inside evaluate_arrow —
        # falls back to the row-wise python_backend for this single
        # call. python_backend is the semantic source of truth and the
        # differential test in tests/search/filter/test_e2e.py keeps
        # the two paths in agreement on every supported expression.
        return evaluate_python(compiled, data)


# ---------------------------------------------------------------------------
# AST walking helpers
# ---------------------------------------------------------------------------

def collect_meta_keys(node: Expr) -> Set[str]:
    """Walk the AST and return the set of all $meta keys referenced."""
    keys: Set[str] = set()
    _collect(node, keys)
    return keys


def _collect(node: Expr, keys: Set[str]) -> None:
    if isinstance(node, MetaAccess):
        keys.add(node.key)
        return
    if isinstance(node, (CmpOp, ArithOp)):
        _collect(node.left, keys)
        _collect(node.right, keys)
        return
    if isinstance(node, (And, Or)):
        for op in node.operands:
            _collect(op, keys)
        return
    if isinstance(node, Not):
        _collect(node.operand, keys)
        return
    if isinstance(node, InOp):
        _collect(node.field, keys)
        _collect(node.values, keys)
        return
    if isinstance(node, ListLit):
        for el in node.elements:
            _collect(el, keys)
        return
    if isinstance(node, LikeOp):
        _collect(node.value, keys)
        return
    if isinstance(node, IsNullOp):
        _collect(node.field, keys)
        return
    # Leaves: literals and FieldRef have no children with MetaAccess
    return


def _has_python_only_nodes(node: Expr) -> bool:
    """Check if the AST contains nodes that require python_backend."""
    from milvus_lite.search.filter.ast import ArrayContainsOp, ArrayLengthOp, ArrayAccessOp
    if isinstance(node, (TextMatchOp, JsonAccess, ArrayContainsOp, ArrayLengthOp, ArrayAccessOp)):
        return True
    if isinstance(node, (CmpOp, ArithOp)):
        return _has_python_only_nodes(node.left) or _has_python_only_nodes(node.right)
    if isinstance(node, (And, Or)):
        return any(_has_python_only_nodes(op) for op in node.operands)
    if isinstance(node, Not):
        return _has_python_only_nodes(node.operand)
    if isinstance(node, InOp):
        return _has_python_only_nodes(node.field)
    if isinstance(node, LikeOp):
        return _has_python_only_nodes(node.value)
    if isinstance(node, IsNullOp):
        return _has_python_only_nodes(node.field)
    return False


def _rewrite_meta_access(node: Expr, keys: Set[str]) -> Expr:
    """Return a new AST with every MetaAccess replaced by FieldRef.

    The synthetic field names are ``__meta__<key>``; the augmented
    table has matching columns built by _augment_table.
    """
    if isinstance(node, MetaAccess):
        return FieldRef(name=_synthetic_name(node.key), pos=node.pos)
    if isinstance(node, CmpOp):
        return CmpOp(
            op=node.op,
            left=_rewrite_meta_access(node.left, keys),
            right=_rewrite_meta_access(node.right, keys),
            pos=node.pos,
        )
    if isinstance(node, ArithOp):
        return ArithOp(
            op=node.op,
            left=_rewrite_meta_access(node.left, keys),
            right=_rewrite_meta_access(node.right, keys),
            pos=node.pos,
        )
    if isinstance(node, And):
        return And(
            operands=tuple(_rewrite_meta_access(op, keys) for op in node.operands),
            pos=node.pos,
        )
    if isinstance(node, Or):
        return Or(
            operands=tuple(_rewrite_meta_access(op, keys) for op in node.operands),
            pos=node.pos,
        )
    if isinstance(node, Not):
        return Not(operand=_rewrite_meta_access(node.operand, keys), pos=node.pos)
    if isinstance(node, InOp):
        # InOp.field is a FieldRef per parser contract — won't be a
        # MetaAccess in F2b. The rewriter still recurses for safety.
        new_field = _rewrite_meta_access(node.field, keys)
        # If somehow the field became a non-FieldRef (shouldn't happen),
        # return the unchanged node — the arrow path will fail loudly.
        if isinstance(new_field, FieldRef):
            return InOp(field=new_field, values=node.values, negate=node.negate, pos=node.pos)
        return node
    if isinstance(node, LikeOp):
        return LikeOp(
            value=_rewrite_meta_access(node.value, keys),
            pattern=node.pattern,
            pos=node.pos,
        )
    if isinstance(node, IsNullOp):
        new_field = _rewrite_meta_access(node.field, keys)
        if new_field is not node.field:
            return IsNullOp(field=new_field, negate=node.negate, pos=node.pos)
        return node
    # Literals and FieldRef are leaves — return unchanged.
    return node


def _synthetic_name(key: str) -> str:
    return f"{_META_PREFIX}{key}"


# ---------------------------------------------------------------------------
# Per-batch JSON preprocessing
# ---------------------------------------------------------------------------

def _augment_table(table: pa.Table, keys: Set[str]) -> pa.Table:
    """Build a copy of *table* with one extra column per referenced $meta key.

    The column name for key ``"category"`` is ``"__meta__category"``.
    Each column's value at row i is ``parsed_meta[i].get(key)`` —
    pyarrow infers the type from the values.

    If the table has no $meta column at all, all extracted columns are
    null-typed. (This shouldn't happen in practice — semantic check
    enforced enable_dynamic_field=True before letting the expression
    compile — but we don't crash if it does.)

    Raises:
        pa.ArrowInvalid: heterogeneous types within a key (caller falls
            back to python_backend).
    """
    if "$meta" not in table.column_names:
        # No $meta column — all extracted columns are entirely null.
        n = table.num_rows
        augmented = table
        for key in sorted(keys):
            null_col = pa.nulls(n, type=pa.null())
            augmented = augmented.append_column(_synthetic_name(key), null_col)
        return augmented

    # Pull $meta as a Python list (one Arrow → Python conversion).
    meta_list = table.column("$meta").to_pylist()

    # Parse each row's JSON exactly once. Tolerate null and bad JSON.
    parsed: List[Any] = []
    for s in meta_list:
        if s is None:
            parsed.append(None)
        elif isinstance(s, dict):
            # Test fixtures sometimes pass dicts directly.
            parsed.append(s)
        elif isinstance(s, str):
            try:
                parsed.append(json.loads(s))
            except (json.JSONDecodeError, ValueError):
                parsed.append(None)
        else:
            parsed.append(None)

    # Build one Arrow column per referenced key.
    augmented = table
    for key in sorted(keys):
        values = [
            (d.get(key) if isinstance(d, dict) else None)
            for d in parsed
        ]
        col = pa.array(values)  # type inferred; may raise on mixed types
        augmented = augmented.append_column(_synthetic_name(key), col)

    return augmented
