"""PyArrow.compute backend for filter evaluation.

Walks the AST and translates each node into a pyarrow.compute call.
This is the F1 hot path — for Tier 1 grammar it's fully vectorized,
written in C++ inside pyarrow, and fast (~5ms for 100K rows).

NULL handling: pyarrow Kleene three-valued logic (and_kleene / or_kleene),
with `pc.fill_null(False)` at the top level so any null in operand chain
becomes "no row matches" rather than a three-valued result the caller
would have to interpret.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Union

import pyarrow as pa
import pyarrow.compute as pc

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
)

if TYPE_CHECKING:
    from milvus_lite.search.filter.semantic import CompiledExpr


_CMP_KERNELS = {
    "==": pc.equal,
    "!=": pc.not_equal,
    "<":  pc.less,
    "<=": pc.less_equal,
    ">":  pc.greater,
    ">=": pc.greater_equal,
}

# Phase F2a — arithmetic kernels
_ARITH_KERNELS = {
    "+": pc.add,
    "-": pc.subtract,
    "*": pc.multiply,
    "/": pc.divide,
}


def evaluate_arrow(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Evaluate the compiled expression as a single BooleanArray.

    The result has length == data.num_rows. Null operand propagation
    is collapsed to False at the top level.
    """
    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data])

    result = _eval(compiled.ast, data)

    # Top-level result must be boolean.
    if isinstance(result, pa.Scalar):
        # Edge case: top-level is a literal bool. Broadcast to a column.
        scalar_value = result.as_py()
        return pa.array(
            [bool(scalar_value)] * data.num_rows, type=pa.bool_()
        )

    if not pa.types.is_boolean(result.type):
        raise TypeError(
            f"top-level filter expression must be boolean, got {result.type}"
        )

    # Coerce ChunkedArray → Array, then null → False.
    if isinstance(result, pa.ChunkedArray):
        result = result.combine_chunks() if result.num_chunks > 0 else pa.array([], type=pa.bool_())
    return pc.fill_null(result, False)


def _to_float(value):
    """Cast a pyarrow scalar/array to float64 if it's integer-typed.

    Used to make `/` produce float division like Python's `/`.
    """
    if isinstance(value, pa.Scalar):
        if pa.types.is_integer(value.type):
            return pa.scalar(float(value.as_py()), type=pa.float64())
        return value
    # Array / ChunkedArray
    if pa.types.is_integer(value.type):
        return pc.cast(value, pa.float64())
    return value


def _eval(node, table):
    """Recursive AST → pyarrow result. Returns either:
        - pa.Scalar (for literals)
        - pa.Array / pa.ChunkedArray (for fields and operations)
    """
    if isinstance(node, IntLit):
        return pa.scalar(node.value, type=pa.int64())
    if isinstance(node, FloatLit):
        return pa.scalar(node.value, type=pa.float64())
    if isinstance(node, StringLit):
        return pa.scalar(node.value, type=pa.string())
    if isinstance(node, BoolLit):
        return pa.scalar(node.value, type=pa.bool_())

    if isinstance(node, FieldRef):
        col = table.column(node.name)
        # Combine chunks so downstream pc.* always sees a flat Array,
        # avoiding subtle type-promotion differences across chunks.
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks() if col.num_chunks > 0 else pa.array([], type=col.type)
        return col

    if isinstance(node, CmpOp):
        left = _eval(node.left, table)
        right = _eval(node.right, table)
        kernel = _CMP_KERNELS[node.op]
        return kernel(left, right)

    if isinstance(node, InOp):
        col = _eval(node.field, table)
        # Build the value_set as a typed Arrow array. The element type
        # comes from the literals; semantic.py has already verified
        # type compatibility with the field, so pyarrow's auto-cast
        # handles any int/float promotion.
        py_values = [el.value for el in node.values.elements]
        if not py_values:
            # Empty list: `in []` → all False, `not in []` → all True.
            val = node.negate  # False for `in`, True for `not in`
            return pa.array([val] * len(col), type=pa.bool_())
        value_set = pa.array(py_values)
        result = pc.is_in(col, value_set=value_set)
        if node.negate:
            result = pc.invert(result)
        return result

    if isinstance(node, And):
        masks = [_eval(op, table) for op in node.operands]
        return functools.reduce(pc.and_kleene, masks)

    if isinstance(node, Or):
        masks = [_eval(op, table) for op in node.operands]
        return functools.reduce(pc.or_kleene, masks)

    if isinstance(node, Not):
        operand = _eval(node.operand, table)
        return pc.invert(operand)

    # ── Phase F2a ───────────────────────────────────────────────
    if isinstance(node, ArithOp):
        left = _eval(node.left, table)
        right = _eval(node.right, table)
        # Match Python's `/` semantics: int / int → float (not floor div).
        # pc.divide on two int columns does integer division by default,
        # which differs from python_backend (operator.truediv) and from
        # what users expect in filter expressions.
        if node.op == "/":
            left = _to_float(left)
            right = _to_float(right)
        kernel = _ARITH_KERNELS[node.op]
        return kernel(left, right)

    if isinstance(node, LikeOp):
        value = _eval(node.value, table)
        # pc.match_like uses SQL LIKE syntax: % matches any sequence,
        # _ matches any single character. The pattern is the literal
        # value from the parser; pyarrow handles the rest.
        return pc.match_like(value, node.pattern.value)

    if isinstance(node, IsNullOp):
        col = _eval(node.field, table)
        result = pc.is_null(col)
        return pc.invert(result) if node.negate else result

    if isinstance(node, MetaAccess):
        # Defensive: compile_expr should never select arrow_backend for
        # an expression containing MetaAccess (that's the whole point of
        # the backend dispatcher). If we hit this, the dispatcher is
        # broken — fail loudly.
        raise NotImplementedError(
            "arrow_backend cannot evaluate $meta access; expressions with "
            "MetaAccess must be routed to python_backend by compile_expr"
        )

    raise TypeError(f"unknown AST node: {type(node).__name__}")
