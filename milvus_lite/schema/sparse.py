"""Shared sparse value validation, independent of scoring and storage IO."""

from __future__ import annotations

import math
import struct


def normalize_sparse_vector(value: object, *, allow_empty: bool = True) -> dict[int, float]:
    """Validate and canonicalize a vector to nonzero float32 weights.

    The maximum uint32 ID is reserved by the Milvus sparse protocol. Raise
    ValueError here; public schema/Engine callers add field context.
    """
    if not isinstance(value, dict):
        raise ValueError(f"sparse vector must be a dict, got {type(value).__name__}")
    result = {}
    for dimension, weight in value.items():
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise ValueError(f"sparse dimension {dimension!r} must be int")
        if not 0 <= dimension < 2**32 - 1:
            raise ValueError(f"sparse dimension {dimension} must be non-negative and less than 2^32-1")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(f"sparse weight at dimension {dimension} must be numeric")
        try:
            number = float(weight)
            if not math.isfinite(number) or number < 0:
                raise ValueError("weight must be finite and non-negative")
            rounded = struct.unpack("<f", struct.pack("<f", number))[0]
            if not math.isfinite(rounded):
                raise ValueError("weight must be finite as float32")
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"sparse weight at dimension {dimension} must be finite, non-negative float32") from exc
        if rounded != 0:
            result[dimension] = rounded
    if not result and not allow_empty:
        raise ValueError("sparse vector must contain at least one nonzero float32 weight")
    return result
