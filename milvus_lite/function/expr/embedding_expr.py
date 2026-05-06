"""EmbeddingExpr — text to dense vector (ingestion stage).

Calls an embedding provider's batch API to convert text inputs into
float vectors.  Null / empty texts are mapped to zero vectors.
"""

from __future__ import annotations

from typing import FrozenSet, List

from milvus_lite.function.types import STAGE_INGESTION, FuncContext, FunctionExpr


class EmbeddingExpr(FunctionExpr):
    """text -> embedding provider -> dense vector."""

    name = "text_embedding"
    supported_stages: FrozenSet[str] = frozenset({STAGE_INGESTION})

    def __init__(self, provider) -> None:
        self._provider = provider

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        texts = inputs[0]
        # Batch process non-null, non-empty strings
        indices: List[int] = []
        batch: List[str] = []
        for i, text in enumerate(texts):
            if text is not None and isinstance(text, str) and text:
                indices.append(i)
                batch.append(text)

        vectors: list = [None] * len(texts)
        if batch:
            embeddings = self._provider.embed_documents(batch)
            for i, emb in zip(indices, embeddings):
                vectors[i] = emb

        # Fill null slots with zero vectors (fresh list per slot to avoid aliasing)
        dim = self._provider.dimension
        for i in range(len(vectors)):
            if vectors[i] is None:
                vectors[i] = [0.0] * dim

        return [vectors]
