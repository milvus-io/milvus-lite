"""NumCombineExpr — combine numeric columns into one score column."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import FrozenSet, List, Optional

from milvus_lite.function.types import STAGE_L2_RERANK, FuncContext, FunctionExpr


class NumCombineExpr(FunctionExpr):
    """Combine numeric input columns row by row."""

    name = "num_combine"
    supported_stages: FrozenSet[str] = frozenset({STAGE_L2_RERANK})

    _VALID_MODES = frozenset(
        {"multiply", "sum", "max", "min", "avg", "weighted"}
    )
    _MIN_INPUT_COLUMNS = 2
    _NON_FINITE_RESULT_ERROR = "num_combine produced a non-finite result"

    def __init__(
        self,
        mode: str = "sum",
        weights: Optional[Sequence[Real]] = None,
    ) -> None:
        if not isinstance(mode, str) or mode not in self._VALID_MODES:
            raise ValueError(f"unknown num_combine mode: {mode!r}")
        if mode == "weighted":
            if weights is None:
                raise ValueError("weighted num_combine requires weights")
            if isinstance(weights, (str, bytes)) or not isinstance(
                weights, Sequence
            ):
                raise ValueError(
                    "num_combine weights must be finite numeric values"
                )
            if not weights:
                raise ValueError("weighted num_combine requires weights")
            if any(
                isinstance(weight, bool)
                or not isinstance(weight, Real)
                or not self._is_finite(weight)
                for weight in weights
            ):
                raise ValueError(
                    "num_combine weights must be finite numeric values"
                )
            self._weights = [float(weight) for weight in weights]
        elif weights is not None:
            raise ValueError("num_combine weights require weighted mode")
        else:
            self._weights = None
        self._mode = mode

    @staticmethod
    def _is_finite(value: Real) -> bool:
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        if len(inputs) < self._MIN_INPUT_COLUMNS:
            raise ValueError("num_combine requires at least two input columns")

        column_length = len(inputs[0])
        if any(len(column) != column_length for column in inputs[1:]):
            raise ValueError("num_combine input columns must have equal lengths")
        if self._weights is not None and len(self._weights) != len(inputs):
            raise ValueError(
                "num_combine weights must match input column count"
            )

        output = []
        for row_index in range(column_length):
            values = [column[row_index] for column in inputs]
            if any(value is None for value in values):
                output.append(0.0)
                continue
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not self._is_finite(value)
                for value in values
            ):
                raise ValueError(
                    "num_combine row values must be finite numeric non-bool values"
                )

            try:
                if self._mode == "multiply":
                    result = 1.0
                    for value in values:
                        result *= value
                elif self._mode == "sum":
                    result = sum(values)
                elif self._mode == "max":
                    result = max(values)
                elif self._mode == "min":
                    result = min(values)
                elif self._mode == "avg":
                    result = math.fsum(
                        value / len(values) for value in values
                    )
                else:
                    result = sum(
                        value * weight
                        for value, weight in zip(values, self._weights)
                    )
                if not math.isfinite(result):
                    raise ValueError(self._NON_FINITE_RESULT_ERROR)
            except OverflowError as exc:
                raise ValueError(self._NON_FINITE_RESULT_ERROR) from exc
            output.append(result)
        return [output]
