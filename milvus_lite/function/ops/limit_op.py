"""LimitOp — per-chunk offset + limit.

Corresponds to Milvus: internal/util/function/chain/operator_limit.go
"""

from __future__ import annotations

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import FuncContext


class LimitOp(Operator):
    """Apply offset + limit to each chunk independently."""

    name = "Limit"

    def __init__(self, limit: int, offset: int = 0) -> None:
        self._limit = limit
        self._offset = offset

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            start = min(self._offset, len(chunk))
            end = min(start + self._limit, len(chunk))
            new_chunks.append(chunk[start:end])
        return DataFrame(new_chunks)
