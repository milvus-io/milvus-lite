"""Reranker implementations for HybridSearch.

Two strategies:

    WeightedRanker — per-query min-max normalize each route's distances
        to [0,1], flip to scores (1 - normalized), then weighted sum.
    RRFRanker — Reciprocal Rank Fusion: score = Σ 1/(k + rank_i).
        Rank-based, ignores distance magnitude.

Both produce a merged list of (pk, final_score, entity) sorted by
final_score descending, then sliced to the requested limit.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def parse_rank_params(kv_pairs) -> dict:
    """Decode rank_params KeyValuePair list from HybridSearchRequest.

    Returns dict with keys: strategy, params, limit, offset.
    """
    raw: dict = {}
    for kv in kv_pairs:
        try:
            raw[kv.key] = json.loads(kv.value)
        except (json.JSONDecodeError, ValueError):
            raw[kv.key] = kv.value

    group_by_field = raw.get("group_by_field")
    if isinstance(group_by_field, str) and group_by_field:
        pass
    else:
        group_by_field = None
    group_size = raw.get("group_size")
    strict_group_size = raw.get("strict_group_size")

    return {
        "strategy": raw.get("strategy", "rrf"),
        "params": raw.get("params", {}),
        "limit": int(raw.get("limit", 10)),
        "offset": int(raw.get("offset", 0)),
        "group_by_field": group_by_field,
        "group_size": int(group_size) if group_size is not None else None,
        "strict_group_size": bool(strict_group_size) if strict_group_size is not None else None,
    }


def rerank(
    strategy: str,
    params: dict,
    all_results: List[List[List[dict]]],
    limit: int,
    offset: int = 0,
) -> List[List[dict]]:
    """Merge multiple search result sets using the specified strategy.

    Args:
        strategy: "weighted" or "rrf"
        params: strategy-specific parameters
        all_results: list of per-route results. Each route is
            List[List[dict]] (nq × hits), where each hit has
            {"id", "distance", "entity"}.
        limit: final top-k after merging
        offset: skip first N merged results

    Returns:
        List[List[dict]] — nq × merged_hits, each hit has
        {"id", "distance", "entity"} where distance = -final_score
        (smaller = better, consistent with engine convention).
    """
    if not all_results:
        return []

    nq = len(all_results[0])
    num_routes = len(all_results)

    if strategy == "weighted":
        weights = params.get("weights", [1.0 / num_routes] * num_routes)
        if len(weights) < num_routes:
            weights.extend([0.0] * (num_routes - len(weights)))
        return _weighted_rerank(all_results, weights, nq, limit, offset)

    # Default: RRF
    k = params.get("k", 60)
    return _rrf_rerank(all_results, k, nq, limit, offset)


def _weighted_rerank(
    all_results: List[List[List[dict]]],
    weights: List[float],
    nq: int,
    limit: int,
    offset: int,
) -> List[List[dict]]:
    """Weighted score fusion with per-query min-max normalization."""
    merged: List[List[dict]] = []

    for qi in range(nq):
        # Collect per-route scores for normalization
        pk_entity: Dict[Any, dict] = {}  # pk → best entity
        pk_route_scores: Dict[Any, List[Optional[float]]] = {}

        for route_idx, route_results in enumerate(all_results):
            hits = route_results[qi] if qi < len(route_results) else []
            for hit in hits:
                pk = hit["id"]
                dist = hit["distance"]
                # distance is "smaller = more similar", so score = -distance
                score = -dist

                if pk not in pk_route_scores:
                    pk_route_scores[pk] = [None] * len(all_results)
                    pk_entity[pk] = hit.get("entity", {})
                pk_route_scores[pk][route_idx] = score

        if not pk_route_scores:
            merged.append([])
            continue

        # Per-route min-max normalization
        num_routes = len(all_results)
        route_mins = [float("inf")] * num_routes
        route_maxs = [float("-inf")] * num_routes

        for scores in pk_route_scores.values():
            for r in range(num_routes):
                if scores[r] is not None:
                    route_mins[r] = min(route_mins[r], scores[r])
                    route_maxs[r] = max(route_maxs[r], scores[r])

        # Compute weighted scores
        pk_final: List[Tuple[Any, float, dict]] = []
        for pk, scores in pk_route_scores.items():
            final = 0.0
            for r in range(num_routes):
                if scores[r] is not None:
                    rng = route_maxs[r] - route_mins[r]
                    if rng > 0:
                        norm = (scores[r] - route_mins[r]) / rng
                    else:
                        norm = 1.0  # all scores equal → max
                    final += weights[r] * norm
            pk_final.append((pk, final, pk_entity[pk]))

        # Sort by final score descending
        pk_final.sort(key=lambda x: x[1], reverse=True)

        # Apply offset + limit
        sliced = pk_final[offset: offset + limit]

        hits = []
        for pk, score, entity in sliced:
            hits.append({
                "id": pk,
                "distance": -score,  # back to "smaller = better"
                "entity": entity,
            })
        merged.append(hits)

    return merged


def _rrf_rerank(
    all_results: List[List[List[dict]]],
    k: int,
    nq: int,
    limit: int,
    offset: int,
) -> List[List[dict]]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)."""
    merged: List[List[dict]] = []

    for qi in range(nq):
        pk_entity: Dict[Any, dict] = {}
        pk_rrf_score: Dict[Any, float] = {}

        for route_results in all_results:
            hits = route_results[qi] if qi < len(route_results) else []
            for rank, hit in enumerate(hits):
                pk = hit["id"]
                if pk not in pk_entity:
                    pk_entity[pk] = hit.get("entity", {})
                rrf = 1.0 / (k + rank + 1)  # rank is 0-based, +1 for 1-based
                pk_rrf_score[pk] = pk_rrf_score.get(pk, 0.0) + rrf

        if not pk_rrf_score:
            merged.append([])
            continue

        # Sort by RRF score descending
        sorted_items = sorted(pk_rrf_score.items(), key=lambda x: x[1], reverse=True)
        sliced = sorted_items[offset: offset + limit]

        hits = []
        for pk, score in sliced:
            hits.append({
                "id": pk,
                "distance": -score,
                "entity": pk_entity[pk],
            })
        merged.append(hits)

    return merged
