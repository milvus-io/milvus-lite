"""GroupByOp — grouped search.

Corresponds to Milvus: internal/util/function/chain/operator_group_by.go
"""

from __future__ import annotations

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.operator import Operator
from milvus_lite.function.types import GROUP_SCORE_FIELD, SCORE_FIELD, FuncContext


class GroupByOp(Operator):
    """Per-chunk grouped search.

    1. Group by ``field``
    2. Within each group, sort by ``$score`` DESC, keep top ``group_size``
    3. Compute ``$group_score`` using *scorer* (max / sum / avg)
    4. Sort groups by ``$group_score`` DESC
    5. Skip *offset* groups, take *limit* groups
    6. Flatten and return
    """

    name = "GroupBy"

    _VALID_SCORERS = frozenset({"max", "sum", "avg"})

    def __init__(
        self,
        field: str,
        group_size: int,
        limit: int,
        offset: int = 0,
        scorer: str = "max",
        sort_descending: bool = True,
    ) -> None:
        if scorer not in self._VALID_SCORERS:
            raise ValueError(f"Unknown group scorer: {scorer!r}")
        self._field = field
        self._group_size = group_size
        self._limit = limit
        self._offset = offset
        self._scorer = scorer
        self._sort_descending = sort_descending

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)

            # 1. Group
            groups: dict = {}
            for hit in chunk:
                key = hit.get(self._field)
                groups.setdefault(key, []).append(hit)

            # 2. Per-group sort + truncate
            scored_groups = []
            for key, hits in groups.items():
                hits.sort(
                    key=lambda r: r.get(SCORE_FIELD, 0),
                    reverse=self._sort_descending,
                )
                top_hits = hits[: self._group_size]
                group_score = self._compute_group_score(top_hits)
                scored_groups.append((group_score, key, top_hits))

            # 3. Sort groups + offset + limit
            scored_groups.sort(key=lambda g: g[0], reverse=self._sort_descending)
            selected = scored_groups[self._offset : self._offset + self._limit]

            # 4. Flatten + add $group_score
            result = []
            for group_score, _key, hits in selected:
                for hit in hits:
                    hit[GROUP_SCORE_FIELD] = group_score
                    result.append(hit)

            new_chunks.append(result)
        return DataFrame(new_chunks)

    def _compute_group_score(self, hits):
        scores = [h.get(SCORE_FIELD, 0) for h in hits]
        if not scores:
            return 0.0
        if self._scorer == "max":
            return max(scores)
        elif self._scorer == "sum":
            return sum(scores)
        elif self._scorer == "avg":
            return sum(scores) / len(scores)
        raise ValueError(f"Unknown group scorer: {self._scorer!r}")
