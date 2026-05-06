"""Rerank subsystem.

Supports RERANK Function type with two modes:

1. **Semantic reranking** (external API): re-scores results using a
   cross-encoder model (e.g. Cohere).
2. **Decay reranking** (local): adjusts scores based on a numeric
   field's proximity to an origin value (gauss/exp/linear curves).
3. **Boost reranking** (local): applies request-level metadata-driven
   weights to search candidates.

Public exports:
    RerankProvider     — abstract protocol for semantic rerankers
    RerankResult       — single rerank result
    DecayReranker      — local decay reranker
    apply_boost_ranker — local boost ranker executor
    create_rerank_provider — factory dispatch for semantic providers
"""

from milvus_lite.rerank.protocol import RerankProvider, RerankResult
from milvus_lite.rerank.decay import DecayReranker
from milvus_lite.rerank.boost import apply_boost_ranker, decode_boost_function_score
from milvus_lite.rerank.factory import create_rerank_provider

__all__ = [
    "RerankProvider",
    "RerankResult",
    "DecayReranker",
    "apply_boost_ranker",
    "decode_boost_function_score",
    "create_rerank_provider",
]
