"""Filter expression AST nodes.

20 frozen dataclass nodes + Expr Union type. All nodes are pure data
(no methods) — behavior lives in semantic.py and the eval/* backends.

Each node carries a `pos` field (column in the source string) so error
messages can render caret-style pointers back to the original input.

Design notes:
    - Frozen + tuple operands → values are hashable, equality is structural
    - No common base class — Union + isinstance dispatch (consistent with
      engine/operation.py Operation pattern)
    - Tuples not lists for And/Or operands and ListLit elements
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union


# ── Literals ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IntLit:
    value: int
    pos: int


@dataclass(frozen=True)
class FloatLit:
    value: float
    pos: int


@dataclass(frozen=True)
class StringLit:
    value: str
    pos: int


@dataclass(frozen=True)
class BoolLit:
    value: bool
    pos: int


# Literal alias for use in ListLit (a list contains only simple literals).
Literal = Union[IntLit, FloatLit, StringLit, BoolLit]


@dataclass(frozen=True)
class ListLit:
    """Homogeneous literal list — used inside `in [...]`.

    Element type homogeneity is enforced at semantic-check time, not at
    parse time, so the parser can produce a clean AST node and let
    semantic.py emit a precise type error.
    """
    elements: Tuple[Literal, ...]
    pos: int


# ── Reference ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldRef:
    """Reference to a schema field by name."""
    name: str
    pos: int


# ── Operations ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CmpOp:
    """Binary comparison: ==, !=, <, <=, >, >=.

    Both sides can be any Expr (literal, field ref, or — in F2 — an
    arithmetic sub-expression). Type compatibility is checked in
    semantic.compile_expr.
    """
    op: str           # "==", "!=", "<", "<=", ">", ">="
    left: "Expr"
    right: "Expr"
    pos: int


@dataclass(frozen=True)
class InOp:
    """Membership: `field in [a, b, c]` or `field not in [...]`."""
    field: FieldRef
    values: ListLit
    negate: bool       # True for "not in"
    pos: int


@dataclass(frozen=True)
class And:
    """Logical AND of two or more operands.

    Multi-arity (not strictly binary) so the parser can flatten chains
    like `a and b and c and d` into a single node, which is more
    cache-friendly for the evaluator.
    """
    operands: Tuple["Expr", ...]
    pos: int


@dataclass(frozen=True)
class Or:
    operands: Tuple["Expr", ...]
    pos: int


@dataclass(frozen=True)
class Not:
    operand: "Expr"
    pos: int


# ── Phase F2a: arithmetic / LIKE / IS NULL ──────────────────────────────────

@dataclass(frozen=True)
class ArithOp:
    """Binary arithmetic: +, -, *, /.

    Operand types must be numeric (int or float). Result type is float
    if either operand is float, else int. Modulo and power are deferred
    to a later F2 sub-phase.
    """
    op: str           # "+", "-", "*", "/"
    left: "Expr"
    right: "Expr"
    pos: int


@dataclass(frozen=True)
class LikeOp:
    """SQL LIKE: `value LIKE 'pattern'`.

    The pattern uses SQL wildcards: '%' matches any sequence (incl. empty),
    '_' matches a single character. Escape support is deferred to F3.

    The value side must be a string-typed expression (typically a
    FieldRef of a VARCHAR field). The pattern must be a string literal.
    """
    value: "Expr"
    pattern: StringLit
    pos: int


@dataclass(frozen=True)
class IsNullOp:
    """`field IS NULL` or `field IS NOT NULL`.

    The operand must be a FieldRef. Returns bool. Phase F2a only supports
    plain field refs; JSON path access (`$meta["key"] is null`) lands in F2b.
    """
    field: FieldRef
    negate: bool      # True for "IS NOT NULL"
    pos: int


# ── Phase F2b: dynamic field access ─────────────────────────────────────────

@dataclass(frozen=True)
class MetaAccess:
    """`$meta["key"]` — dynamic-field value lookup.

    The schema must have ``enable_dynamic_field=True`` for this node to
    be allowed. The result type is "dynamic" (not known until runtime),
    and any expression containing a MetaAccess forces backend selection
    to "hybrid" — pyarrow.compute has no built-in JSON path kernel, so
    hybrid_backend does per-batch JSON preprocessing to materialize
    dynamic keys as Arrow columns, then delegates to arrow_backend.
    Falls back to python_backend on heterogeneous JSON types.
    """
    key: str
    pos: int


# ── JSON field path access ────────────────────────────────────────────────

@dataclass(frozen=True)
class JsonAccess:
    """``field_name["key"]`` or ``field_name["a"]["b"]`` — JSON path lookup.

    Unlike MetaAccess (which is specifically for ``$meta["key"]``),
    JsonAccess works on any JSON-typed schema field. The result type
    is dynamic (not known until runtime). Forces python backend.

    ``keys`` is a tuple of one or more string keys for chained access:
    ``info["a"]["b"]`` → ``keys=("a", "b")``.
    """
    field_name: str
    keys: Tuple[str, ...]
    pos: int


# ── Phase 11.6: text_match function ──────────────────────────────────────

@dataclass(frozen=True)
class TextMatchOp:
    """``text_match(field_name, 'query tokens')``.

    Tokenizes the query string using the field's analyzer, then checks
    if any query token appears in the document's tokenized field value.
    Multiple query tokens use OR logic (match any).
    """
    field: FieldRef
    query: StringLit
    pos: int


# ── Array filter functions ────────────────────────────────────────────────

@dataclass(frozen=True)
class ArrayContainsOp:
    """``array_contains(field, value)`` / ``array_contains_all`` / ``array_contains_any``."""
    field: FieldRef
    values: "Expr"  # single value or ListLit
    mode: str       # "any_one" | "all" | "any"
    pos: int


@dataclass(frozen=True)
class ArrayLengthOp:
    """``array_length(field)`` — returns the length of an array field."""
    field: FieldRef
    pos: int


@dataclass(frozen=True)
class ArrayAccessOp:
    """``field[index]`` — integer index access on an array field."""
    field_name: str
    index: int
    pos: int


# ── Type alias for the union of all node types ──────────────────────────────

Expr = Union[
    IntLit, FloatLit, StringLit, BoolLit,
    ListLit,
    FieldRef,
    CmpOp, InOp, And, Or, Not,
    ArithOp, LikeOp, IsNullOp,
    MetaAccess, JsonAccess,
    TextMatchOp,
    ArrayContainsOp, ArrayLengthOp, ArrayAccessOp,
]
