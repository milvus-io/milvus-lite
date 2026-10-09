"""Mask and result helpers shared by the separate sparse scorers."""

import numpy as np


def validate_mask(mask, num_rows: int):
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.shape != (num_rows,) or array.dtype != np.bool_:
        raise ValueError(f"valid_mask must be a boolean array of shape ({num_rows},)")
    return array


def empty_results(num_queries: int, top_k: int):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    return (
        np.full((num_queries, top_k), -1, dtype=np.int64),
        np.full((num_queries, top_k), float("inf"), dtype=np.float32),
    )
