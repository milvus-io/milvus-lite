"""MergeOp — multi-path search result merging.

Combines results from multiple search paths (hybrid search) into a
single result set, deduplicating by ``$id`` and computing a merged
score according to the selected strategy.

Corresponds to Milvus: internal/util/function/chain/operator_merge.go
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD, FuncContext


class MergeOp(Operator):
    """Merge multi-path search results.

    Must be the first Operator in a chain.  ``execute_multi()`` accepts
    multiple :class:`DataFrame` objects (one per search path) and returns
    a single merged :class:`DataFrame`.

    Strategies:
        rrf      — Reciprocal Rank Fusion: ``Σ 1/(k + rank_i)``
        weighted — Weighted sum with optional metric-aware normalization
        max      — Maximum score across routes
        sum      — Sum of scores across routes
        avg      — Average of scores across routes
    """

    name = "Merge"

    def __init__(self, strategy: str, **kwargs) -> None:
        self._strategy = strategy
        self._weights: List[float] = kwargs.get("weights", [])
        self._rrf_k: float = kwargs.get("rrf_k", 60.0)
        self._normalize: bool = kwargs.get("normalize", True)
        self._metric_types: List[str] = list(kwargs.get("metric_types", []))
        self._force_descending: bool = kwargs.get("force_descending", False)
        self._sort_descending, self._score_norm_funcs = _resolve_merge_behavior(
            normalize=self._normalize,
            force_descending=self._force_descending,
            metric_types=self._metric_types,
        )

    @property
    def sort_descending(self) -> bool:
        return self._sort_descending

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        raise RuntimeError("MergeOp requires execute_multi()")

    def execute_multi(
        self, ctx: FuncContext, inputs: List[DataFrame]
    ) -> DataFrame:
        if not inputs:
            raise ValueError("MergeOp requires at least one input")

        nq = inputs[0].num_chunks
        for idx, inp in enumerate(inputs[1:], start=1):
            if inp.num_chunks != nq:
                raise ValueError(
                    "MergeOp requires all inputs to have the same number "
                    f"of chunks: input 0 has {nq}, input {idx} has "
                    f"{inp.num_chunks}"
                )
        if self._strategy == "weighted" and len(self._weights) != len(inputs):
            raise ValueError(
                f"weighted merge requires {len(inputs)} weights, "
                f"got {len(self._weights)}"
            )

        merged_chunks: List[List[dict]] = []

        for q in range(nq):
            if self._strategy == "rrf":
                merged_chunks.append(self._merge_rrf(inputs, q))
            elif self._strategy == "weighted":
                merged_chunks.append(self._merge_weighted(inputs, q))
            else:
                merged_chunks.append(
                    self._merge_simple(inputs, q, self._strategy)
                )

        return DataFrame(merged_chunks)

    # ── RRF ──────────────────────────────────────────────────

    def _merge_rrf(
        self, inputs: List[DataFrame], q: int
    ) -> List[dict]:
        pk_entity: Dict[Any, dict] = {}
        pk_score: Dict[Any, float] = {}

        for inp in inputs:
            chunk = inp.chunk(q) if q < inp.num_chunks else []
            for rank, hit in enumerate(chunk):
                pk = hit.get(ID_FIELD)
                if pk not in pk_entity:
                    pk_entity[pk] = dict(hit)
                rrf = 1.0 / (self._rrf_k + rank + 1)
                pk_score[pk] = pk_score.get(pk, 0.0) + rrf

        results: List[dict] = []
        for pk, score in pk_score.items():
            merged = pk_entity[pk]
            merged[SCORE_FIELD] = score
            results.append(merged)
        return results

    # ── Weighted ─────────────────────────────────────────────

    def _merge_weighted(
        self, inputs: List[DataFrame], q: int
    ) -> List[dict]:
        num_routes = len(inputs)
        weights = list(self._weights)

        pk_entity: Dict[Any, dict] = {}
        pk_route_scores: Dict[Any, List[Optional[float]]] = {}

        for route_idx, inp in enumerate(inputs):
            chunk = inp.chunk(q) if q < inp.num_chunks else []
            for hit in chunk:
                pk = hit.get(ID_FIELD)
                score = hit.get(SCORE_FIELD, 0.0)
                if pk not in pk_route_scores:
                    pk_route_scores[pk] = [None] * num_routes
                    pk_entity[pk] = dict(hit)
                pk_route_scores[pk][route_idx] = score

        if not pk_route_scores:
            return []

        results: List[dict] = []
        for pk, scores in pk_route_scores.items():
            final = 0.0
            for r in range(num_routes):
                if scores[r] is not None:
                    score = scores[r]
                    norm_func = self._score_norm_func(r)
                    if norm_func is not None:
                        score = norm_func(score)
                    final += weights[r] * score
            merged = pk_entity[pk]
            merged[SCORE_FIELD] = final
            results.append(merged)
        return results

    # ── Simple strategies (max / sum / avg) ──────────────────

    def _merge_simple(
        self,
        inputs: List[DataFrame],
        q: int,
        strategy: str,
    ) -> List[dict]:
        pk_entity: Dict[Any, dict] = {}
        pk_scores: Dict[Any, List[float]] = {}

        for input_idx, inp in enumerate(inputs):
            chunk = inp.chunk(q) if q < inp.num_chunks else []
            for hit in chunk:
                pk = hit.get(ID_FIELD)
                score = hit.get(SCORE_FIELD, 0.0)
                norm_func = self._score_norm_func(input_idx)
                if norm_func is not None:
                    score = norm_func(score)
                if pk not in pk_entity:
                    pk_entity[pk] = dict(hit)
                pk_scores.setdefault(pk, []).append(score)

        results: List[dict] = []
        for pk, scores in pk_scores.items():
            if strategy == "max":
                final = max(scores)
            elif strategy == "sum":
                final = sum(scores)
            elif strategy == "avg":
                final = sum(scores) / len(scores)
            else:
                raise ValueError(f"Unknown merge strategy: {strategy}")
            merged = pk_entity[pk]
            merged[SCORE_FIELD] = final
            results.append(merged)
        return results

    def _score_norm_func(self, input_idx: int):
        if input_idx < len(self._score_norm_funcs):
            return self._score_norm_funcs[input_idx]
        return None


def _resolve_merge_behavior(
    *,
    normalize: bool,
    force_descending: bool,
    metric_types: List[str],
):
    norm_funcs = [None] * len(metric_types)
    if not metric_types:
        return True, norm_funcs

    if normalize:
        return True, [_normalize_func(metric_type) for metric_type in metric_types]

    mixed, sort_descending = _classify_metric_order(metric_types)
    if mixed or force_descending:
        return True, [
            _direction_convert_func(metric_type) for metric_type in metric_types
        ]

    return sort_descending, norm_funcs


def _classify_metric_order(metric_types: List[str]) -> Tuple[bool, bool]:
    positive = sum(1 for metric_type in metric_types if _positively_related(metric_type))
    negative = len(metric_types) - positive
    if positive and negative:
        return True, True
    return False, negative == 0


def _positively_related(metric_type: str) -> bool:
    return metric_type.upper() in {
        "IP",
        "COSINE",
        "BM25",
        "MHJACCARD",
        "MAX_SIM",
        "MAXSIM",
        "MAX_SIM_IP",
        "MAXSIMIP",
        "MAX_SIM_COSINE",
        "MAXSIMCOSINE",
    }


def _direction_convert_func(metric_type: str):
    if _positively_related(metric_type):
        return None
    return lambda distance: 1.0 - 2.0 * math.atan(float(distance)) / math.pi


def _normalize_func(metric_type: str):
    upper = metric_type.upper()
    if upper == "COSINE":
        return lambda score: (1.0 + float(score)) * 0.5
    if upper == "IP":
        return lambda score: 0.5 + math.atan(float(score)) / math.pi
    if upper == "BM25":
        return lambda score: 2.0 * math.atan(float(score)) / math.pi
    if _positively_related(metric_type):
        return lambda score: 0.5 + math.atan(float(score)) / math.pi
    return lambda distance: 1.0 - 2.0 * math.atan(float(distance)) / math.pi
