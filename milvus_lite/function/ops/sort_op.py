"""SortOp — per-chunk sorting.

Corresponds to Milvus: internal/util/function/chain/operator_sort.go
"""

from __future__ import annotations

import functools
import math

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import FuncContext


class SortOp(Operator):
    """Sort records within each chunk by a column.

    Supports descending/ascending order, with ``None`` and float ``NaN``
    values always sorted to the end.  An explicit tie-break column is sorted
    ascending; without one, equal values preserve their input order.
    """

    name = "Sort"

    def __init__(
        self,
        column: str,
        desc: bool = True,
        tie_break_col: str | None = None,
    ) -> None:
        self._column = column
        self._desc = desc
        self._tie_break_col = tie_break_col

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        col = self._column
        tb = self._tie_break_col
        desc = self._desc

        def _is_missing(value):
            return value is None or (
                isinstance(value, float) and math.isnan(value)
            )

        def _compare_tie(a, b):
            if tb is None:
                return 0
            ta = a.get(tb)
            tb_val = b.get(tb)
            ta_missing = _is_missing(ta)
            tb_missing = _is_missing(tb_val)
            if ta_missing and tb_missing:
                return 0
            if ta_missing:
                return 1
            if tb_missing:
                return -1
            return (ta > tb_val) - (ta < tb_val)

        def _cmp(a, b):
            va = a.get(col)
            vb = b.get(col)
            va_missing = _is_missing(va)
            vb_missing = _is_missing(vb)
            # Missing values always last
            if va_missing and vb_missing:
                return _compare_tie(a, b)
            if va_missing:
                return 1  # a goes after b
            if vb_missing:
                return -1  # a goes before b
            # Primary comparison
            if va != vb:
                if desc:
                    return (vb > va) - (vb < va)
                else:
                    return (va > vb) - (va < vb)
            # Tie-break: always ascending
            return _compare_tie(a, b)

        cmp_key = functools.cmp_to_key(_cmp)
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            chunk.sort(key=cmp_key)
        return df
