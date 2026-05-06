"""SelectOp — column projection.

Corresponds to Milvus: internal/util/function/chain/operator_select.go
"""

from __future__ import annotations

from typing import List

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import FuncContext


class SelectOp(Operator):
    """Keep only the specified columns, remove all others."""

    name = "Select"

    def __init__(self, columns: List[str]) -> None:
        self._columns = set(columns)

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            new_chunks.append([
                {k: v for k, v in r.items() if k in self._columns}
                for r in chunk
            ])
        return DataFrame(new_chunks)
