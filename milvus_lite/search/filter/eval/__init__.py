"""Backend dispatcher for filter evaluation.

evaluate(compiled, table) → BooleanArray

The backend choice is made statically at compile_expr time and stored
in compiled.backend. Three backends in F3+:

    "arrow"  — pyarrow.compute fast path; pure schema field expressions
    "hybrid" — per-batch JSON preprocessing + arrow_backend; for $meta
               expressions (Phase F3+ optimization, ~10x faster than
               row-wise)
    "python" — row-wise interpreter; differential test baseline + the
               last-resort fallback for things hybrid can't handle
               (currently nothing in the public grammar; F3 UDFs would
               use this)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pyarrow as pa

from milvus_lite.search.filter.eval.arrow_backend import evaluate_arrow
from milvus_lite.search.filter.eval.hybrid_backend import evaluate_hybrid
from milvus_lite.search.filter.eval.python_backend import evaluate_python

if TYPE_CHECKING:
    from milvus_lite.search.filter.semantic import CompiledExpr


def evaluate(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Dispatch to the backend recorded on the compiled expression."""
    if compiled.backend == "arrow":
        return evaluate_arrow(compiled, data)
    if compiled.backend == "hybrid":
        return evaluate_hybrid(compiled, data)
    if compiled.backend == "python":
        return evaluate_python(compiled, data)
    raise ValueError(f"unknown filter backend: {compiled.backend!r}")


__all__ = ["evaluate", "evaluate_arrow", "evaluate_hybrid", "evaluate_python"]
