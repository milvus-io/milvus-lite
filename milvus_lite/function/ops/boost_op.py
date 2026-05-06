"""BoostOp - L0 Boost Ranker operator."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import DISTANCE_FIELD, ID_FIELD, FuncContext


class BoostOp(Operator):
    """Apply Boost Ranker functions to per-route search hits.

    Boost runs before the L2 merge/rerank stage.  It operates on the
    engine distance representation where smaller values are better.
    """

    name = "Boost"

    def __init__(
        self,
        ranker: dict,
        *,
        metric_type: str,
        pk_name: str,
        compile_filter,
        row_matches_filter,
        distance_col: str = DISTANCE_FIELD,
    ) -> None:
        self._functions = ranker.get("functions") or []
        params = ranker.get("params") or {}
        self._boost_mode = params.get("boost_mode", "multiply")
        self._function_mode = params.get("function_mode", "multiply")
        self._metric_type = metric_type
        self._pk_name = pk_name
        self._row_matches_filter = row_matches_filter
        self._distance_col = distance_col

        self._compiled_filters: Dict[str, Any] = {}
        for fn in self._functions:
            filt = fn["params"].get("filter")
            if filt:
                self._compiled_filters[filt] = compile_filter(filt)

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        for chunk_idx in range(df.num_chunks):
            for row in df.chunk(chunk_idx):
                values = []
                for fn in self._functions:
                    fn_params = fn["params"]
                    filt = fn_params.get("filter")
                    if filt:
                        eval_row = dict(row)
                        eval_row[self._pk_name] = row.get(self._pk_name, row.get(ID_FIELD))
                        if not self._row_matches_filter(
                            eval_row, self._compiled_filters[filt]
                        ):
                            continue

                    value = float(fn_params["weight"])
                    random_score = fn_params.get("random_score")
                    if random_score is not None:
                        value *= self._stable_random_score(row, random_score)
                    values.append(value)

                if not values:
                    continue

                combined = _combine(values, self._function_mode)
                row[self._distance_col] = _apply_boost_to_distance(
                    float(row[self._distance_col]),
                    combined,
                    self._boost_mode,
                    self._metric_type,
                )

        return df

    def _stable_random_score(self, row: dict, random_score: dict) -> float:
        seed = random_score.get("seed", 0)
        field = random_score.get("field")
        if field:
            if field == self._pk_name:
                value = row.get(ID_FIELD)
            else:
                value = row.get(field)
        else:
            value = row.get(ID_FIELD)

        payload = f"{seed}:{value!r}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _combine(values: list[float], mode: str) -> float:
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
    metric = metric_type.upper()
    if boost_mode == "multiply":
        if metric == "COSINE":
            score = 1.0 - distance
            return 1.0 - score * value
        return distance * value

    if metric in ("COSINE", "IP", "BM25"):
        return distance - value
    return distance + value
