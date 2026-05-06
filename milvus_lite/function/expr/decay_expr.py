"""DecayExpr — numeric field to decay factor (rerank stage).

Delegates to :class:`DecayReranker` for the actual math to avoid
duplicating the gauss/exp/linear formulas.

Corresponds to Milvus: internal/util/function/chain/expr/decay_expr.go
"""

from __future__ import annotations

from typing import FrozenSet, List

from milvus_lite.function.types import STAGE_L2_RERANK, FuncContext, FunctionExpr


class DecayExpr(FunctionExpr):
    """numeric column -> decay factor [0, 1].

    Wraps :class:`~milvus_lite.rerank.decay.DecayReranker` as a
    columnar :class:`FunctionExpr`.
    """

    name = "decay"
    supported_stages: FrozenSet[str] = frozenset({STAGE_L2_RERANK})

    def __init__(
        self,
        function: str,
        origin: float,
        scale: float,
        offset: float = 0.0,
        decay: float = 0.5,
    ) -> None:
        from milvus_lite.rerank.decay import DecayReranker

        # DecayReranker validates scale>0, 0<decay<1, valid function name
        self._reranker = DecayReranker(
            function=function,
            origin=origin,
            scale=scale,
            offset=offset,
            decay=decay,
        )

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        compute = self._reranker.compute_factor
        values = inputs[0]
        factors: list = []
        for val in values:
            if val is None:
                factors.append(0.0)
            else:
                factors.append(compute(float(val)))
        return [factors]
