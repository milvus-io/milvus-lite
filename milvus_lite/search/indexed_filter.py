"""Helpers for scalar-index-backed filter evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from milvus_lite.index.scalar import IndexedFilterPlan
    from milvus_lite.search.filter.semantic import CompiledExpr
    from milvus_lite.storage.segment import Segment


def evaluate_segment_filter(
    seg: "Segment",
    compiled_filter: "CompiledExpr",
    indexed_filter_plan: Optional["IndexedFilterPlan"],
) -> np.ndarray:
    if indexed_filter_plan is not None:
        if indexed_filter_plan.required_fields <= seg.scalar_indexes.keys():
            return indexed_filter_plan.evaluate(seg.scalar_indexes).astype(bool, copy=False)
    from milvus_lite.search.filter import evaluate_mask

    return evaluate_mask(compiled_filter, seg.table)


