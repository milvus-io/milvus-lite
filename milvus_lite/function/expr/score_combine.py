"""Legacy score-combination expression."""

from __future__ import annotations

from milvus_lite.function.expr.num_combine import NumCombineExpr


class ScoreCombineExpr(NumCombineExpr):
    """Backward-compatible numeric combination with multiply as default."""

    name = "score_combine"
    _MIN_INPUT_COLUMNS = 1

    def __init__(self, mode: str = "multiply") -> None:
        if mode == "weighted":
            raise ValueError("ScoreCombineExpr does not support weighted mode")
        super().__init__(mode=mode)
