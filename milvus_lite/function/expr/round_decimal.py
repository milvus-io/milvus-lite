"""RoundDecimalExpr — round score to N decimal places.

Corresponds to Milvus: internal/util/function/chain/expr/round_decimal_expr.go
"""

from __future__ import annotations

from typing import FrozenSet, List

from milvus_lite.function.types import (
    STAGE_L2_RERANK,
    FuncContext,
    FunctionExpr,
)


class RoundDecimalExpr(FunctionExpr):
    """``$score -> round($score, decimal)``."""

    name = "round_decimal"
    supported_stages: FrozenSet[str] = frozenset({STAGE_L2_RERANK})

    def __init__(self, decimal: int) -> None:
        self._decimal = decimal

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        scores = inputs[0]
        return [
            [round(s, self._decimal) if s is not None else None for s in scores]
        ]
