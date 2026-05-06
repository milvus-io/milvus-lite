"""BM25Expr — text to sparse vector (ingestion stage).

Tokenizes text via an Analyzer and computes term frequencies,
producing a sparse vector dict ``{term_hash: frequency}``.
"""

from __future__ import annotations

from typing import FrozenSet, List

from milvus_lite.function.types import STAGE_INGESTION, FuncContext, FunctionExpr


class BM25Expr(FunctionExpr):
    """text -> analyze -> compute_tf -> sparse vector dict."""

    name = "bm25"
    supported_stages: FrozenSet[str] = frozenset({STAGE_INGESTION})

    def __init__(self, analyzer) -> None:
        self._analyzer = analyzer

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        from milvus_lite.analyzer.sparse import compute_tf

        texts = inputs[0]
        sparse_vecs: list = []
        for text in texts:
            if text is None or not isinstance(text, str):
                sparse_vecs.append({})
            else:
                term_ids = self._analyzer.analyze(text)
                sparse_vecs.append(compute_tf(term_ids))
        return [sparse_vecs]
