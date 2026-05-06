"""Public API for the filter expression subsystem.

Three-stage compilation:

    parse_expr(s)              → Expr (raw AST)
    compile_expr(expr, schema) → CompiledExpr
    evaluate(compiled, table)  → pa.BooleanArray

Convenience: parse + compile in one call:

    compile_filter("age > 18", schema)
"""

from milvus_lite.search.filter.exceptions import (
    FilterError,
    FilterFieldError,
    FilterParseError,
    FilterTypeError,
)
from milvus_lite.search.filter.parser import parse_expr
from milvus_lite.search.filter.semantic import CompiledExpr, FieldInfo, compile_expr
from milvus_lite.search.filter.eval import evaluate


def compile_filter(source: str, schema) -> CompiledExpr:
    """Convenience: parse_expr + compile_expr in one call.

    Most call sites that already have the source string and the schema
    use this rather than the two-step API. Two-step is useful when you
    want to cache parsed AST across schemas (Phase F2c).
    """
    return compile_expr(parse_expr(source), schema, source=source)


__all__ = [
    "parse_expr",
    "compile_expr",
    "compile_filter",
    "evaluate",
    "CompiledExpr",
    "FieldInfo",
    "FilterError",
    "FilterParseError",
    "FilterFieldError",
    "FilterTypeError",
]
