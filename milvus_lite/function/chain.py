"""Ordered operator pipeline (FuncChain).

Corresponds to Milvus: internal/util/function/chain/chain.go FuncChain
"""

from __future__ import annotations

from typing import List

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import FuncContext, FunctionExpr


class FuncChain:
    """Ordered Operator pipeline with a fluent API.

    Usage::

        chain = FuncChain("ingestion", STAGE_INGESTION)
        chain.map(BM25Expr(analyzer), ["text"], ["sparse_vec"])
        chain.map(EmbeddingExpr(provider), ["text"], ["dense_vec"])
        result = chain.execute(DataFrame.from_records(records))
    """

    def __init__(self, name: str, stage: str) -> None:
        self._name = name
        self._stage = stage
        self._operators: List[Operator] = []

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def operators(self) -> List[Operator]:
        return list(self._operators)

    # ── Fluent API ───────────��────────────────────────────────

    def add(self, op: Operator) -> FuncChain:
        """Append an Operator to the end of the pipeline."""
        from milvus_lite.function.ops.merge_op import MergeOp

        if isinstance(op, MergeOp) and self._operators:
            raise ValueError("MergeOp must be the first operator in the chain")
        self._operators.append(op)
        return self

    def map(
        self,
        expr: FunctionExpr,
        input_cols: List[str],
        output_cols: List[str],
    ) -> FuncChain:
        """Add a :class:`MapOp`."""
        if not expr.is_runnable(self._stage):
            raise ValueError(
                f"FunctionExpr '{expr.name}' does not support "
                f"stage '{self._stage}'"
            )
        from milvus_lite.function.ops.map_op import MapOp

        return self.add(MapOp(expr, input_cols, output_cols))

    def merge(self, strategy: str, **kwargs) -> FuncChain:
        """Add a :class:`MergeOp` (must be the first Operator)."""
        if self._operators:
            raise ValueError("MergeOp must be the first operator in the chain")
        from milvus_lite.function.ops.merge_op import MergeOp

        return self.add(MergeOp(strategy, **kwargs))

    def sort(self, column: str, desc: bool = True) -> FuncChain:
        """Add a :class:`SortOp`."""
        from milvus_lite.function.ops.sort_op import SortOp

        return self.add(SortOp(column, desc))

    def limit(self, limit: int, offset: int = 0) -> FuncChain:
        """Add a :class:`LimitOp`."""
        from milvus_lite.function.ops.limit_op import LimitOp

        return self.add(LimitOp(limit, offset))

    def select(self, *columns: str) -> FuncChain:
        """Add a :class:`SelectOp`."""
        from milvus_lite.function.ops.select_op import SelectOp

        return self.add(SelectOp(list(columns)))

    def group_by(
        self,
        field: str,
        group_size: int,
        limit: int,
        offset: int = 0,
        scorer: str = "max",
        sort_descending: bool = True,
    ) -> FuncChain:
        """Add a :class:`GroupByOp`."""
        from milvus_lite.function.ops.group_by_op import GroupByOp

        return self.add(
            GroupByOp(
                field,
                group_size,
                limit,
                offset,
                scorer,
                sort_descending=sort_descending,
            )
        )

    # ── Execution ──────────���──────────────────────────────────

    def execute(self, *inputs: DataFrame) -> DataFrame:
        """Execute the entire chain.

        If the first Operator is a :class:`MergeOp`, multiple inputs are
        accepted; otherwise exactly one input is required.
        """
        from milvus_lite.function.ops.merge_op import MergeOp

        ctx = FuncContext(self._stage)
        start_idx = 0

        if self._operators and isinstance(self._operators[0], MergeOp):
            result = self._operators[0].execute_multi(ctx, list(inputs))
            start_idx = 1
        else:
            if len(inputs) != 1:
                raise ValueError(
                    f"Chain expects 1 input but got {len(inputs)} "
                    f"(first operator is not MergeOp)"
                )
            result = inputs[0]

        for op in self._operators[start_idx:]:
            result = op.execute(ctx, result)

        return result

    # ── Debug ───���─────────────────────────────────────��───────

    def __repr__(self) -> str:
        ops = " -> ".join(op.name for op in self._operators)
        return f"FuncChain({self._name}, stage={self._stage}): {ops}"
