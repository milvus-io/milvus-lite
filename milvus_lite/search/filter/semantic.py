"""Schema-bound semantic check + type inference for filter expressions.

Phase 8 stage 2: parse_expr produces an AST that's schema-agnostic;
compile_expr binds it to a CollectionSchema, checks every field
reference, infers a type for every node, and selects an evaluation
backend. The result is a CompiledExpr that the evaluator can run on
any pa.Table matching the schema.

F1 always picks the "arrow" backend (pyarrow.compute path). Future
phases will switch to "python" when the expression contains $meta or
UDFs that arrow_backend can't handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
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
    Literal,
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
from milvus_lite.search.filter.exceptions import (
    FilterFieldError,
    FilterTypeError,
)


# ── Semantic types ──────────────────────────────────────────────────────────

# A small internal type lattice for filter operands. We don't reuse
# DataType because we don't care about INT8 vs INT64 here — int promotion
# rules collapse all integers to a single "int" semantic type.
SEM_INT = "int"
SEM_FLOAT = "float"
SEM_STRING = "string"
SEM_BOOL = "bool"
# Phase F2b: $meta["key"] returns a value whose type is unknown until
# runtime. SEM_DYNAMIC is compatible with any other type for the purpose
# of comparisons / IN / arithmetic — runtime semantics decide.
SEM_DYNAMIC = "dynamic"

_SEM_TYPES = {SEM_INT, SEM_FLOAT, SEM_STRING, SEM_BOOL, SEM_DYNAMIC}

# Reserved field names that must not be referenced from filter expressions.
_RESERVED_FIELDS = frozenset({"_seq", "_partition", "$meta"})


def _datatype_to_sem(dtype: DataType) -> Optional[str]:
    """Map a schema DataType to the filter semantic type, or None if
    the field type is not allowed in scalar filters (e.g. FLOAT_VECTOR)."""
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        return SEM_INT
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return SEM_FLOAT
    if dtype == DataType.VARCHAR:
        return SEM_STRING
    if dtype == DataType.BOOL:
        return SEM_BOOL
    if dtype == DataType.JSON:
        return SEM_STRING  # JSON column is stored as string in Phase F1
    if dtype == DataType.ARRAY:
        return SEM_DYNAMIC  # handled by python backend
    if dtype == DataType.FLOAT_VECTOR:
        return None  # not allowed in scalar filter
    return None


def _types_compatible(left: str, right: str) -> bool:
    """Comparison-compatible: same type, or int↔float promotion, or
    one side is dynamic ($meta access)."""
    if left == right:
        return True
    if {left, right} == {SEM_INT, SEM_FLOAT}:
        return True
    if SEM_DYNAMIC in (left, right):
        return True
    return False


def _common_type(left: str, right: str) -> Optional[str]:
    """Common type when comparing — float wins over int. Dynamic wins
    over everything (defers type resolution to runtime)."""
    if left == right:
        return left
    if {left, right} == {SEM_INT, SEM_FLOAT}:
        return SEM_FLOAT
    if SEM_DYNAMIC in (left, right):
        return SEM_DYNAMIC
    return None


@dataclass(frozen=True)
class FieldInfo:
    """Schema-bound field descriptor passed to evaluator backends."""
    name: str
    dtype: DataType
    sem_type: str
    nullable: bool


@dataclass(frozen=True)
class CompiledExpr:
    """A type-checked, schema-bound, evaluation-ready expression.

    The same Expr tree is reused; this wrapper carries the metadata
    that the evaluator backends need.
    """
    ast: Expr
    fields: Dict[str, FieldInfo]
    backend: str   # "arrow" | "python"
    source: str    # original expression string (for error messages)


# ---------------------------------------------------------------------------
# Dynamic field rewriting
# ---------------------------------------------------------------------------

_RESERVED_FIELD_NAMES = frozenset({"_seq", "_partition", "$meta"})


def _rewrite_dynamic_field_refs(
    node: Expr,
    schema_fields: Dict[str, FieldSchema],
) -> Expr:
    """Replace FieldRef nodes for unknown fields with MetaAccess nodes.

    When ``enable_dynamic_field=True``, a bare ``color == "red"`` should
    behave like ``$meta["color"] == "red"``.  This function walks the AST
    and performs that rewrite BEFORE semantic checking so the existing
    MetaAccess handling (type inference + hybrid backend) works unchanged.
    """
    if isinstance(node, FieldRef):
        if node.name not in schema_fields and node.name not in _RESERVED_FIELD_NAMES:
            return MetaAccess(key=node.name, pos=node.pos)
        return node
    if isinstance(node, CmpOp):
        return CmpOp(
            op=node.op,
            left=_rewrite_dynamic_field_refs(node.left, schema_fields),
            right=_rewrite_dynamic_field_refs(node.right, schema_fields),
            pos=node.pos,
        )
    if isinstance(node, ArithOp):
        return ArithOp(
            op=node.op,
            left=_rewrite_dynamic_field_refs(node.left, schema_fields),
            right=_rewrite_dynamic_field_refs(node.right, schema_fields),
            pos=node.pos,
        )
    if isinstance(node, (And, Or)):
        cls = type(node)
        return cls(
            operands=tuple(
                _rewrite_dynamic_field_refs(op, schema_fields)
                for op in node.operands
            ),
            pos=node.pos,
        )
    if isinstance(node, Not):
        return Not(
            operand=_rewrite_dynamic_field_refs(node.operand, schema_fields),
            pos=node.pos,
        )
    if isinstance(node, InOp):
        new_field = _rewrite_dynamic_field_refs(node.field, schema_fields)
        return InOp(
            field=new_field, values=node.values,
            negate=node.negate, pos=node.pos,
        )
    if isinstance(node, LikeOp):
        return LikeOp(
            value=_rewrite_dynamic_field_refs(node.value, schema_fields),
            pattern=node.pattern,
            pos=node.pos,
        )
    if isinstance(node, IsNullOp):
        new_field = _rewrite_dynamic_field_refs(node.field, schema_fields)
        return IsNullOp(field=new_field, negate=node.negate, pos=node.pos)
    # Leaves (literals, MetaAccess, JsonAccess, TextMatchOp, Array*Op)
    # are returned unchanged.
    return node


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def compile_expr(
    expr: Expr,
    schema: CollectionSchema,
    source: str = "",
) -> CompiledExpr:
    """Bind field references, check types, choose backend.

    Args:
        expr: AST from parse_expr
        schema: target CollectionSchema
        source: original expression string (used in error messages — pass
            it through from your call site if you parsed via parse_expr)

    Raises:
        FilterFieldError: unknown field reference
        FilterTypeError:  type mismatch in operand or non-bool top-level
    """
    # Build a map of all schema field names → FieldSchema for fast lookup.
    schema_fields: Dict[str, FieldSchema] = {f.name: f for f in schema.fields}
    field_names = [f.name for f in schema.fields]

    # When dynamic fields are enabled, rewrite bare FieldRef nodes for
    # unknown field names to MetaAccess(key=name) so the hybrid backend
    # can extract them from the $meta JSON column automatically.
    # This matches Milvus behavior where `color == "red"` is equivalent
    # to `$meta["color"] == "red"`.
    if schema.enable_dynamic_field:
        expr = _rewrite_dynamic_field_refs(expr, schema_fields)

    fields_used: Dict[str, FieldInfo] = {}
    ctx = _CompileCtx(
        schema_fields=schema_fields,
        field_names=field_names,
        fields_used=fields_used,
        source=source,
        enable_dynamic_field=schema.enable_dynamic_field,
        has_meta_access=False,
    )

    # Walk the AST: type-check + collect referenced fields.
    top_type = _check_node(expr, ctx)

    # Top-level expression must be boolean.
    if top_type != SEM_BOOL:
        raise FilterTypeError(
            f"top-level filter expression must evaluate to bool, got {top_type}",
            source, getattr(expr, "pos", 0),
        )

    # Backend selection (Phase F3+):
    #   - "arrow"  for pure schema field expressions (fast path)
    #   - "hybrid" for $meta expressions (per-batch JSON preprocessing
    #     then arrow path; ~10x faster than pure row-wise python)
    #   - "python" reserved for future UDF / truly dynamic things that
    #     hybrid can't handle. Not selected automatically in F3+.
    backend = "hybrid" if ctx.has_meta_access else "arrow"

    return CompiledExpr(
        ast=expr,
        fields=fields_used,
        backend=backend,
        source=source,
    )


@dataclass
class _CompileCtx:
    """Internal walking context — packs the multiple parameters needed
    by _check_node so the recursive call signature stays small."""
    schema_fields: Dict[str, FieldSchema]
    field_names: List[str]
    fields_used: Dict[str, FieldInfo]
    source: str
    enable_dynamic_field: bool
    has_meta_access: bool


# ---------------------------------------------------------------------------
# Internal: recursive type checker
# ---------------------------------------------------------------------------

def _check_node(node: Expr, ctx: "_CompileCtx") -> str:
    """Recursively check a node and return its semantic type."""

    # ── Literals ────────────────────────────────────────────────
    if isinstance(node, IntLit):
        return SEM_INT
    if isinstance(node, FloatLit):
        return SEM_FLOAT
    if isinstance(node, StringLit):
        return SEM_STRING
    if isinstance(node, BoolLit):
        return SEM_BOOL

    # ── ListLit (only used inside InOp; pure-form is rare) ──────
    if isinstance(node, ListLit):
        if not node.elements:
            return "empty"
        elem_types = [_check_node(e, ctx) for e in node.elements]
        first = elem_types[0]
        for t in elem_types[1:]:
            if not _types_compatible(first, t):
                raise FilterTypeError(
                    f"list elements must be of compatible types, "
                    f"got {first} and {t}",
                    ctx.source, node.pos,
                )
        return _common_type(first, first) or first

    # ── FieldRef ────────────────────────────────────────────────
    if isinstance(node, FieldRef):
        if node.name in _RESERVED_FIELDS:
            raise FilterFieldError(
                f"reserved field {node.name!r} cannot be used in filter expressions",
                ctx.source, node.pos, node.name, available_fields=ctx.field_names,
            )
        if node.name not in ctx.schema_fields:
            raise FilterFieldError(
                f"unknown field {node.name!r}",
                ctx.source, node.pos, node.name, available_fields=ctx.field_names,
            )
        field = ctx.schema_fields[node.name]
        sem = _datatype_to_sem(field.dtype)
        if sem is None:
            raise FilterTypeError(
                f"field {node.name!r} of type {field.dtype.value} cannot be used "
                f"in scalar filter expressions",
                ctx.source, node.pos, span=len(node.name),
            )
        ctx.fields_used[node.name] = FieldInfo(
            name=node.name,
            dtype=field.dtype,
            sem_type=sem,
            nullable=field.nullable,
        )
        return sem

    # ── CmpOp ───────────────────────────────────────────────────
    if isinstance(node, CmpOp):
        left_type = _check_node(node.left, ctx)
        right_type = _check_node(node.right, ctx)
        if not _types_compatible(left_type, right_type):
            left_desc = _describe_operand(node.left, left_type)
            right_desc = _describe_operand(node.right, right_type)
            raise FilterTypeError(
                f"comparison '{node.op}' between incompatible types",
                ctx.source, node.pos, span=len(node.op),
                left_desc=left_desc, right_desc=right_desc,
            )
        return SEM_BOOL

    # ── InOp ────────────────────────────────────────────────────
    if isinstance(node, InOp):
        field_type = _check_node(node.field, ctx)
        if node.values.elements:
            list_type = _check_node(node.values, ctx)
            if list_type != "empty" and not _types_compatible(field_type, list_type):
                raise FilterTypeError(
                    f"'in' list elements ({list_type}) incompatible with "
                    f"field {node.field.name!r} ({field_type})",
                    ctx.source, node.pos, span=2,
                )
        return SEM_BOOL

    # ── And / Or ────────────────────────────────────────────────
    if isinstance(node, And) or isinstance(node, Or):
        for op in node.operands:
            t = _check_node(op, ctx)
            if t != SEM_BOOL:
                op_name = "and" if isinstance(node, And) else "or"
                raise FilterTypeError(
                    f"operands of '{op_name}' must be boolean, got {t}",
                    ctx.source, getattr(op, "pos", node.pos),
                )
        return SEM_BOOL

    # ── Not ─────────────────────────────────────────────────────
    if isinstance(node, Not):
        t = _check_node(node.operand, ctx)
        if t != SEM_BOOL:
            raise FilterTypeError(
                f"operand of 'not' must be boolean, got {t}",
                ctx.source, node.pos,
            )
        return SEM_BOOL

    # ── ArithOp (Phase F2a) ─────────────────────────────────────
    if isinstance(node, ArithOp):
        left_type = _check_node(node.left, ctx)
        right_type = _check_node(node.right, ctx)
        # Numeric (int/float) AND dynamic ($meta) are allowed.
        for side, t in (("left", left_type), ("right", right_type)):
            if t not in (SEM_INT, SEM_FLOAT, SEM_DYNAMIC):
                operand = node.left if side == "left" else node.right
                raise FilterTypeError(
                    f"arithmetic '{node.op}' requires numeric operands, "
                    f"{side} side is {_describe_operand(operand, t)}",
                    ctx.source, node.pos, span=len(node.op),
                    left_desc=_describe_operand(operand, t),
                )
        # Result type: dynamic if either is dynamic; else float for / or
        # mixed; else int.
        if SEM_DYNAMIC in (left_type, right_type):
            return SEM_DYNAMIC
        if node.op == "/" or left_type == SEM_FLOAT or right_type == SEM_FLOAT:
            return SEM_FLOAT
        return SEM_INT

    # ── LikeOp (Phase F2a) ──────────────────────────────────────
    if isinstance(node, LikeOp):
        value_type = _check_node(node.value, ctx)
        if value_type not in (SEM_STRING, SEM_DYNAMIC):
            raise FilterTypeError(
                f"LIKE requires a string operand, got "
                f"{_describe_operand(node.value, value_type)}",
                ctx.source, node.pos,
                left_desc=_describe_operand(node.value, value_type),
            )
        return SEM_BOOL

    # ── IsNullOp (Phase F2a) ────────────────────────────────────
    if isinstance(node, IsNullOp):
        # The parser enforces FieldRef as the operand. We don't restrict
        # the field's type — IS NULL is meaningful for any field.
        _check_node(node.field, ctx)
        return SEM_BOOL

    # ── MetaAccess (Phase F2b) ──────────────────────────────────
    if isinstance(node, MetaAccess):
        if not ctx.enable_dynamic_field:
            raise FilterFieldError(
                f"$meta access requires schema enable_dynamic_field=True",
                ctx.source, node.pos,
                field_name="$meta",
                available_fields=ctx.field_names,
            )
        ctx.has_meta_access = True
        return SEM_DYNAMIC

    # ── JsonAccess (JSON field path) ────────────────────────
    if isinstance(node, JsonAccess):
        # Validate field exists in schema
        if node.field_name not in ctx.schema_fields:
            raise FilterFieldError(
                f"unknown field {node.field_name!r}",
                ctx.source, node.pos,
                field_name=node.field_name,
                available_fields=ctx.field_names,
            )
        ctx.has_meta_access = True  # force python backend
        return SEM_DYNAMIC

    # ── TextMatchOp (Phase 11.6) ─────────────────────────────
    if isinstance(node, TextMatchOp):
        _check_node(node.field, ctx)
        ctx.has_meta_access = True
        return SEM_BOOL

    # ── Array functions ──────────────────────────────────────
    if isinstance(node, ArrayContainsOp):
        _check_node(node.field, ctx)
        ctx.has_meta_access = True
        return SEM_BOOL

    if isinstance(node, ArrayLengthOp):
        _check_node(node.field, ctx)
        ctx.has_meta_access = True
        return SEM_INT

    if isinstance(node, ArrayAccessOp):
        if node.field_name not in ctx.schema_fields:
            raise FilterFieldError(
                f"unknown field {node.field_name!r}",
                ctx.source, node.pos,
                field_name=node.field_name,
                available_fields=ctx.field_names,
            )
        ctx.has_meta_access = True
        return SEM_DYNAMIC

    raise TypeError(f"unknown AST node type: {type(node).__name__}")


def _describe_operand(node: Expr, sem_type: str) -> str:
    """Build a human-readable description of an operand for type errors.

    Examples:
        '(int)'                          → "int"
        FieldRef('age', int)             → "int (field 'age')"
        IntLit(18)                       → "int"
        StringLit('hi')                  → "string"
    """
    if isinstance(node, FieldRef):
        return f"{sem_type} (field {node.name!r})"
    return sem_type
