"""ScoreCombineExpr — combine multiple score columns into one.

Corresponds to Milvus: internal/util/function/chain/expr/score_combine_expr.go
"""

from __future__ import annotations

from typing import FrozenSet, List

from milvus_lite.function.types import STAGE_L2_RERANK, FuncContext, FunctionExpr


class ScoreCombineExpr(FunctionExpr):
    """Combine multiple score columns into a single final score.

    Modes: ``multiply`` (default for decay), ``sum``, ``max``, ``min``,
    ``avg``.
    """

    name = "score_combine"
    supported_stages: FrozenSet[str] = frozenset({STAGE_L2_RERANK})

    _VALID_MODES = frozenset({"multiply", "sum", "max", "min", "avg"})

    def __init__(self, mode: str = "multiply") -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"ScoreCombineExpr: unknown mode {mode!r}, "
                f"must be one of {sorted(self._VALID_MODES)}"
            )
        self._mode = mode

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        n = len(inputs[0])
        results: list = []
        for row_idx in range(n):
            vals = [col[row_idx] for col in inputs]
            if None in vals:
                results.append(0.0)
                continue
            if self._mode == "multiply":
                r = 1.0
                for v in vals:
                    r *= v
                results.append(r)
            elif self._mode == "sum":
                results.append(sum(vals))
            elif self._mode == "max":
                results.append(max(vals))
            elif self._mode == "min":
                results.append(min(vals))
            elif self._mode == "avg":
                results.append(sum(vals) / len(vals))
            else:
                results.append(0.0)
        return [results]
