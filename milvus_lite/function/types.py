"""Core types for the function chain system.

Provides FunctionExpr (stateless column computation), FuncContext
(execution context), and stage/column constants.

Corresponds to Milvus: internal/util/function/chain/types/types.go
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import FrozenSet, List

# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------

STAGE_INGESTION = "ingestion"  # insert / upsert
STAGE_L0_RERANK = "l0_rerank"  # segment / route-level search rerank
STAGE_L2_RERANK = "rerank"  # search / proxy-level post-processing

# ---------------------------------------------------------------------------
# Virtual column names used by the chain
# ---------------------------------------------------------------------------

ID_FIELD = "$id"
SCORE_FIELD = "$score"
DISTANCE_FIELD = "$distance"
GROUP_SCORE_FIELD = "$group_score"
DECAY_SCORE_FIELD = "_decay_score"


# ---------------------------------------------------------------------------
# FuncContext
# ---------------------------------------------------------------------------


class FuncContext:
    """Execution context for function chains.

    Corresponds to Milvus: internal/util/function/chain/types.FuncContext
    """

    __slots__ = ("_stage", "_chunk_idx")

    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._chunk_idx = 0

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def chunk_idx(self) -> int:
        return self._chunk_idx

    @chunk_idx.setter
    def chunk_idx(self, value: int) -> None:
        self._chunk_idx = value


# ---------------------------------------------------------------------------
# FunctionExpr
# ---------------------------------------------------------------------------


class FunctionExpr(ABC):
    """Stateless column-level computation unit.

    Responsible only for pure computation: input columns -> output columns.
    Unaware of DataFrame structure; column name mapping is handled by MapOp.

    Corresponds to Milvus: internal/util/function/chain/types.FunctionExpr
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Function name (e.g., ``"bm25"``, ``"decay"``)."""

    @property
    @abstractmethod
    def supported_stages(self) -> FrozenSet[str]:
        """Stages where this function can execute."""

    @abstractmethod
    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        """Execute computation.

        Args:
            ctx: Execution context (carries stage, chunk_idx, etc.).
            inputs: List of input columns.  ``inputs[i]`` is a list of
                values whose length equals the number of records in the
                current chunk.

        Returns:
            List of output columns.
        """

    def is_runnable(self, stage: str) -> bool:
        """Check whether this function supports *stage*."""
        return stage in self.supported_stages
