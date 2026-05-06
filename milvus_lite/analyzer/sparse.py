"""Sparse vector encoding / decoding and TF computation.

Sparse vectors are stored as packed binary in Arrow/Parquet columns
(``pa.binary()``).  The on-disk format matches Milvus's
``SparseFloatArray.contents`` encoding:

    Each row = N pairs of (uint32_le index, float32_le value),
    sorted by index in ascending order.

This module also provides ``compute_tf`` which converts a list of
term IDs (from ``Analyzer.analyze``) into a sparse vector dict
mapping term_hash → term_frequency (float).
"""

from __future__ import annotations

import struct
from collections import Counter
from typing import Dict, List


def sparse_to_bytes(sv: Dict[int, float]) -> bytes:
    """Encode a sparse vector dict to packed binary.

    Args:
        sv: mapping of term_id (uint32) → value (float32).
            Empty dict → empty bytes.

    Returns:
        Packed bytes: pairs of (uint32_le, float32_le) sorted by index.
    """
    if not sv:
        return b""
    pairs = sorted(sv.items())
    buf = bytearray(len(pairs) * 8)
    for i, (idx, val) in enumerate(pairs):
        struct.pack_into("<If", buf, i * 8, idx, val)
    return bytes(buf)


def bytes_to_sparse(b: bytes) -> Dict[int, float]:
    """Decode packed binary to a sparse vector dict.

    Args:
        b: packed bytes from ``sparse_to_bytes``.

    Returns:
        dict mapping term_id → value.
    """
    if not b:
        return {}
    if len(b) % 8 != 0:
        raise ValueError(
            f"sparse vector bytes length {len(b)} is not a multiple of 8"
        )
    n = len(b) // 8
    result: Dict[int, float] = {}
    for i in range(n):
        idx, val = struct.unpack_from("<If", b, i * 8)
        result[idx] = val
    return result


def compute_tf(term_ids: List[int]) -> Dict[int, float]:
    """Compute term frequency from a list of term IDs.

    Args:
        term_ids: output of ``Analyzer.analyze(text)`` — may contain
            duplicate IDs (one per token occurrence).

    Returns:
        Sparse vector dict: ``{term_hash: count}`` where count is the
        number of times the term appeared (as float for proto compat).
    """
    if not term_ids:
        return {}
    counts = Counter(term_ids)
    return {tid: float(c) for tid, c in counts.items()}
