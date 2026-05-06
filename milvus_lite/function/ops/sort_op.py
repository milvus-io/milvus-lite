"""SortOp — per-chunk sorting.

Corresponds to Milvus: internal/util/function/chain/operator_sort.go
"""

from __future__ import annotations

import functools

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import ID_FIELD, FuncContext


class SortOp(Operator):
    """Sort records within each chunk by a column.

    Supports descending/ascending order, with ``None`` values always
    sorted to the end.  Ties are broken by ``$id`` ascending (aligned
    with Milvus).
    """

    name = "Sort"

    def __init__(
        self,
        column: str,
        desc: bool = True,
        tie_break_col: str = ID_FIELD,
    ) -> None:
        self._column = column
        self._desc = desc
        self._tie_break_col = tie_break_col

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        col = self._column
        tb = self._tie_break_col
        desc = self._desc

        def _cmp(a, b):
            va = a.get(col)
            vb = b.get(col)
            # None always last
            if va is None and vb is None:
                # tie-break by id ascending
                ta = a.get(tb, 0)
                tb_val = b.get(tb, 0)
                return (ta > tb_val) - (ta < tb_val)
            if va is None:
                return 1  # a goes after b
            if vb is None:
                return -1  # a goes before b
            # Primary comparison
            if va != vb:
                if desc:
                    return (vb > va) - (vb < va)
                else:
                    return (va > vb) - (va < vb)
            # Tie-break: always ascending
            ta = a.get(tb, 0)
            tb_val = b.get(tb, 0)
            return (ta > tb_val) - (ta < tb_val)

        cmp_key = functools.cmp_to_key(_cmp)
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            chunk.sort(key=cmp_key)
        return df
