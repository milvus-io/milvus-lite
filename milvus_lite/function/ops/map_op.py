"""MapOp — column transformation operator.

Reads input columns from a DataFrame, invokes a FunctionExpr, and writes
the results back as output columns.  Each chunk is processed independently.

Corresponds to Milvus: internal/util/function/chain/operator_map.go
"""

from __future__ import annotations

from typing import List

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import FuncContext, FunctionExpr


class MapOp(Operator):
    """Apply a :class:`FunctionExpr` to specified columns per chunk.

    ``output_cols`` may overlap with ``input_cols`` (e.g. ScoreCombine
    merges ``$score`` and ``_decay_score``, writes back to ``$score``).
    """

    name = "Map"

    def __init__(
        self,
        expr: FunctionExpr,
        input_cols: List[str],
        output_cols: List[str],
    ) -> None:
        self._expr = expr
        self._input_cols = input_cols
        self._output_cols = output_cols

    @property
    def expr(self) -> FunctionExpr:
        return self._expr

    @property
    def input_cols(self) -> List[str]:
        return self._input_cols

    @property
    def output_cols(self) -> List[str]:
        return self._output_cols

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        for chunk_idx in range(df.num_chunks):
            ctx.chunk_idx = chunk_idx
            # 1. Read input columns
            inputs = [df.column(col, chunk_idx) for col in self._input_cols]
            # 2. Execute function
            outputs = self._expr.execute(ctx, inputs)
            if len(outputs) != len(self._output_cols):
                raise ValueError(
                    f"MapOp({self._expr.name}): expected "
                    f"{len(self._output_cols)} output columns but got "
                    f"{len(outputs)}"
                )
            # 3. Write output columns back
            for col_name, col_data in zip(self._output_cols, outputs):
                df.set_column(col_name, chunk_idx, col_data)
        return df
