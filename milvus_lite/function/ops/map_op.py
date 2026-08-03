"""MapOp — column transformation operator.

Reads input columns from a DataFrame, invokes a FunctionExpr, and writes
the results back as output columns.  Each chunk is processed independently.

Corresponds to Milvus: internal/util/function/chain/operator_map.go
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypeAlias

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import FuncContext, FunctionExpr


@dataclass(frozen=True)
class ColumnBinding:
    """Bind a function input to a DataFrame column."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                "ColumnBinding.name must be a string, "
                f"got {type(self.name).__name__}"
            )


@dataclass(frozen=True)
class LiteralBinding:
    """Bind a function input to a literal repeated for each chunk row."""

    value: object


InputBinding: TypeAlias = ColumnBinding | LiteralBinding
InputSpec: TypeAlias = str | InputBinding


class MapOp(Operator):
    """Apply a :class:`FunctionExpr` to specified columns per chunk.

    ``output_cols`` may overlap with ``input_cols`` (e.g. ScoreCombine
    merges ``$score`` and ``_decay_score``, writes back to ``$score``).
    """

    name = "Map"

    def __init__(
        self,
        expr: FunctionExpr,
        input_cols: List[InputSpec],
        output_cols: List[str],
    ) -> None:
        self._expr = expr
        self._input_bindings = [
            self._normalize_input_spec(spec, idx)
            for idx, spec in enumerate(input_cols)
        ]
        self._output_cols = output_cols

    @staticmethod
    def _normalize_input_spec(spec: InputSpec, idx: int) -> InputBinding:
        if isinstance(spec, str):
            return ColumnBinding(spec)
        if isinstance(spec, (ColumnBinding, LiteralBinding)):
            return spec
        raise TypeError(
            f"MapOp input spec at index {idx} must be a column name string, "
            f"ColumnBinding, or LiteralBinding; got {type(spec).__name__}"
        )

    @property
    def expr(self) -> FunctionExpr:
        return self._expr

    @property
    def input_cols(self) -> List[str]:
        return [
            binding.name
            for binding in self._input_bindings
            if isinstance(binding, ColumnBinding)
        ]

    @property
    def input_bindings(self) -> List[InputBinding]:
        return list(self._input_bindings)

    @property
    def output_cols(self) -> List[str]:
        return self._output_cols

    @staticmethod
    def _resolve_input(
        binding: InputBinding,
        df: DataFrame,
        chunk_idx: int,
    ) -> list:
        if isinstance(binding, ColumnBinding):
            return df.column(binding.name, chunk_idx)
        return [binding.value] * len(df.chunk(chunk_idx))

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        for chunk_idx in range(df.num_chunks):
            ctx.chunk_idx = chunk_idx
            # 1. Read input columns
            inputs = [
                self._resolve_input(binding, df, chunk_idx)
                for binding in self._input_bindings
            ]
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
