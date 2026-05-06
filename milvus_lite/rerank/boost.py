"""Boost Ranker support.

Boost Ranker is a request-level RERANK function used by Milvus search.
It adjusts candidate scores with metadata-driven rules before the final
top-k is selected.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from milvus_lite.exceptions import SchemaValidationError


def decode_kv_pairs(kv_pairs) -> Dict[str, Any]:
    """Decode proto KeyValuePair values using JSON when possible."""
    out: Dict[str, Any] = {}
    for kv in kv_pairs:
        try:
            out[kv.key] = json.loads(kv.value)
        except (json.JSONDecodeError, TypeError, ValueError):
            out[kv.key] = kv.value
    return out


def decode_boost_function_score(function_score) -> Optional[dict]:
    """Decode a SearchRequest.function_score if it contains Boost Rankers."""
    functions = []
    for fn in getattr(function_score, "functions", []):
        params = decode_kv_pairs(fn.params)
        reranker = str(params.get("reranker", "")).lower()
        if reranker != "boost":
            raise SchemaValidationError(
                "search ranker only supports Boost Ranker functions "
                f"(got reranker={params.get('reranker')!r})"
            )
        if list(getattr(fn, "input_field_names", [])):
            raise SchemaValidationError(
                f"Boost Ranker function {fn.name!r} requires empty input_field_names"
            )
        functions.append({
            "name": fn.name,
            "params": _validate_boost_params(fn.name, params),
        })

    if not functions:
        return None

    return {
        "functions": functions,
        "params": _normalize_function_score_params(
            decode_kv_pairs(getattr(function_score, "params", []))
        ),
    }


def decode_hybrid_function_score(function_score) -> dict:
    """Split HybridSearch FunctionScore into query-node and proxy rerankers.

    Boost rankers execute as L0 functions at the route/engine level.
    RRF/Weighted/Decay/Model execute as L2 functions at the global
    hybrid merge level.
    """
    score_params = _normalize_function_score_params(
        decode_kv_pairs(getattr(function_score, "params", []))
    )
    boost_functions = []
    l2_func = None

    for fn in getattr(function_score, "functions", []):
        params = decode_kv_pairs(fn.params)
        reranker = str(params.get("reranker", "")).lower()

        if reranker == "boost":
            if list(getattr(fn, "input_field_names", [])):
                raise SchemaValidationError(
                    f"Boost Ranker function {fn.name!r} requires empty input_field_names"
                )
            boost_functions.append({
                "name": fn.name,
                "params": _validate_boost_params(fn.name, params),
            })
            continue

        if l2_func is not None:
            raise SchemaValidationError(
                "Hybrid FunctionScore supports at most one non-boost "
                "rerank function"
            )

        if reranker not in ("rrf", "weighted", "decay", "model") and not params.get("provider"):
            raise SchemaValidationError(
                f"Unsupported Hybrid FunctionScore reranker {reranker!r}"
            )
        l2_func = SimpleNamespace(
            name=fn.name,
            input_field_names=list(getattr(fn, "input_field_names", [])),
            output_field_names=list(getattr(fn, "output_field_names", [])),
            params=_normalize_l2_params(reranker, params),
        )

    boost = None
    if boost_functions:
        boost = {"functions": boost_functions, "params": score_params}

    return {"boost": boost, "rerank": l2_func}


def merge_boost_rankers(*rankers: Optional[dict]) -> Optional[dict]:
    """Combine route-level and top-level boost specs."""
    functions = []
    params = None
    for ranker in rankers:
        if not ranker:
            continue
        functions.extend(ranker.get("functions") or [])
        params = ranker.get("params") or params
    if not functions:
        return None
    return {
        "functions": functions,
        "params": params or {"boost_mode": "multiply", "function_mode": "multiply"},
    }


def apply_boost_ranker(
    results: List[List[dict]],
    ranker: dict,
    *,
    metric_type: str,
    pk_name: str,
    compile_filter,
    row_matches_filter,
) -> List[List[dict]]:
    """Apply Boost Ranker through the L0 FuncChain."""
    if not ranker:
        return results

    from milvus_lite.function.chain import FuncChain
    from milvus_lite.function.dataframe import DataFrame
    from milvus_lite.function.ops.boost_op import BoostOp
    from milvus_lite.function.types import (
        DISTANCE_FIELD,
        ID_FIELD,
        STAGE_L0_RERANK,
    )

    chain_chunks: List[List[dict]] = []
    for hits in results:
        chunk = []
        for hit in hits:
            flat = {
                ID_FIELD: hit["id"],
                DISTANCE_FIELD: hit["distance"],
            }
            flat.update(hit.get("entity") or {})
            chunk.append(flat)
        chain_chunks.append(chunk)

    chain = FuncChain("l0_rerank", STAGE_L0_RERANK)
    chain.add(
        BoostOp(
            ranker,
            metric_type=metric_type,
            pk_name=pk_name,
            compile_filter=compile_filter,
            row_matches_filter=row_matches_filter,
        )
    )
    chain.sort(DISTANCE_FIELD, desc=False)
    result_df = chain.execute(DataFrame(chain_chunks))

    virtual = {ID_FIELD, DISTANCE_FIELD}
    boosted: List[List[dict]] = []
    for ci in range(result_df.num_chunks):
        out_hits = []
        for row in result_df.chunk(ci):
            entity = {k: v for k, v in row.items() if k not in virtual}
            out_hits.append({
                "id": row[ID_FIELD],
                "distance": row[DISTANCE_FIELD],
                "entity": entity,
            })
        boosted.append(out_hits)

    return boosted


def _validate_boost_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if params.get("reranker", "").lower() != "boost":
        raise SchemaValidationError(
            f"Boost Ranker function {name!r} requires params.reranker='boost'"
        )

    if "weight" not in params:
        raise SchemaValidationError(
            f"Boost Ranker function {name!r} requires params.weight"
        )
    try:
        params["weight"] = float(params["weight"])
    except (TypeError, ValueError):
        raise SchemaValidationError(
            f"Boost Ranker function {name!r} params.weight must be a number"
        )

    filt = params.get("filter")
    if filt is not None and not isinstance(filt, str):
        raise SchemaValidationError(
            f"Boost Ranker function {name!r} params.filter must be a string"
        )

    random_score = params.get("random_score")
    if random_score is not None:
        if isinstance(random_score, str):
            try:
                random_score = json.loads(random_score)
            except (json.JSONDecodeError, TypeError, ValueError):
                raise SchemaValidationError(
                    f"Boost Ranker function {name!r} params.random_score "
                    "must be an object"
                )
        if not isinstance(random_score, dict):
            raise SchemaValidationError(
                f"Boost Ranker function {name!r} params.random_score must be an object"
            )
        params["random_score"] = random_score

    return params


def _normalize_function_score_params(params: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for key, default in (("boost_mode", "multiply"), ("function_mode", "multiply")):
        value = str(params.get(key, default)).lower()
        if value not in ("multiply", "sum"):
            raise SchemaValidationError(
                f"FunctionScore params.{key} must be 'Multiply' or 'Sum'"
            )
        out[key] = value
    return out


def _normalize_l2_params(reranker: str, params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if reranker == "rrf":
        if "k" in out:
            out["k"] = float(out["k"])
        return out

    if reranker == "weighted":
        weighted = _normalize_proxy_params(reranker, out)
        out.update(weighted)
        out.setdefault("norm_score", False)
        return out

    if reranker == "model" or out.get("provider"):
        out["queries"] = _parse_model_queries(out)
        return out

    return out


def _normalize_proxy_params(reranker: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if reranker == "rrf":
        out = {}
        if "k" in params:
            out["k"] = float(params["k"])
        return out

    out = {}
    if "weights" in params:
        weights = params["weights"]
        if isinstance(weights, str):
            try:
                weights = json.loads(weights)
            except (json.JSONDecodeError, TypeError, ValueError):
                raise SchemaValidationError(
                    "Weighted reranker params.weights must be an array"
                )
        if not isinstance(weights, list):
            raise SchemaValidationError(
                "Weighted reranker params.weights must be an array"
            )
        out["weights"] = [float(w) for w in weights]
    if "norm_score" in params:
        out["norm_score"] = _parse_bool(params["norm_score"])
    return out


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    raise SchemaValidationError("Weighted reranker params.norm_score must be bool")


def _parse_model_queries(params: Dict[str, Any]) -> List[str]:
    if "queries" not in params:
        raise SchemaValidationError(
            "Model reranker requires params.queries"
        )

    queries = params["queries"]
    if isinstance(queries, str):
        try:
            queries = json.loads(queries)
        except (json.JSONDecodeError, TypeError, ValueError):
            raise SchemaValidationError(
                "Model reranker params.queries must be a JSON array"
            )

    if not isinstance(queries, list) or not queries:
        raise SchemaValidationError(
            "Model reranker params.queries must be a non-empty array"
        )
    if not all(isinstance(q, str) for q in queries):
        raise SchemaValidationError(
            "Model reranker params.queries must contain only strings"
        )
    return list(queries)


def _combine(values: List[float], mode: str) -> float:
    if mode == "sum":
        return sum(values)
    product = 1.0
    for value in values:
        product *= value
    return product


def _apply_boost_to_distance(
    distance: float,
    value: float,
    boost_mode: str,
    metric_type: str,
) -> float:
    if boost_mode == "multiply":
        return distance * value

    # Sum mode operates on the metric's natural score.  IP and BM25 use
    # higher-is-better scores internally represented as negative distances.
    if metric_type in ("IP", "BM25"):
        return distance - value
    return distance + value


def _stable_random_score(hit: dict, random_score: dict, pk_name: str) -> float:
    seed = random_score.get("seed", 0)
    field = random_score.get("field")
    if field:
        entity = hit.get("entity") or {}
        if field == pk_name:
            value = hit.get("id")
        else:
            value = entity.get(field)
    else:
        value = hit.get("id")

    payload = f"{seed}:{value!r}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)
