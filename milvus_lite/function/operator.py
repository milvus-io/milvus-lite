"""Base class for function chain operators.

Corresponds to Milvus: internal/util/function/chain/chain.go Operator
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.types import FuncContext


class Operator(ABC):
    """Base class for operators.

    Operators receive an input :class:`DataFrame`, transform it, and
    return an output :class:`DataFrame`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Operator name (e.g., ``"Map"``, ``"Sort"``, ``"Merge"``)."""

    @abstractmethod
    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        """Execute the operator.

        Args:
            ctx: Execution context.
            df:  Input DataFrame.

        Returns:
            Output DataFrame (may be the same object modified in-place,
            or a new object).
        """
