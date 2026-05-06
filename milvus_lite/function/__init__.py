"""Function chain system for unified function execution.

Inspired by Milvus ``internal/util/function/chain/``.
"""

from milvus_lite.function.builder import build_ingestion_chain
from milvus_lite.function.chain import FuncChain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.expr import BM25Expr, EmbeddingExpr
from milvus_lite.function.operator import Operator
from milvus_lite.function.ops import MapOp
from milvus_lite.function.types import (
    DECAY_SCORE_FIELD,
    DISTANCE_FIELD,
    STAGE_INGESTION,
    STAGE_L0_RERANK,
    STAGE_L2_RERANK,
    ID_FIELD,
    SCORE_FIELD,
    GROUP_SCORE_FIELD,
    FuncContext,
    FunctionExpr,
)

__all__ = [
    "FuncChain",
    "DataFrame",
    "Operator",
    "FuncContext",
    "FunctionExpr",
    "MapOp",
    "BM25Expr",
    "EmbeddingExpr",
    "build_ingestion_chain",
    "DECAY_SCORE_FIELD",
    "DISTANCE_FIELD",
    "STAGE_INGESTION",
    "STAGE_L0_RERANK",
    "STAGE_L2_RERANK",
    "ID_FIELD",
    "SCORE_FIELD",
    "GROUP_SCORE_FIELD",
]
