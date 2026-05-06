"""Deterministic term-to-ID hash function.

Maps a token string to a uint32 integer for use as the sparse vector
index. Uses a simple FNV-1a variant that is:
- Deterministic across Python versions and platforms
- Fast (no external dependency)
- Well-distributed in the uint32 range

We avoid Python's built-in hash() because it is randomized by
PYTHONHASHSEED and platform-dependent.
"""

from __future__ import annotations

# FNV-1a parameters for 32-bit
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def term_to_id(term: str) -> int:
    """Map a token string to a uint32 term ID (0 .. 2^32-1).

    Uses FNV-1a hash on the UTF-8 bytes of *term*.
    """
    h = _FNV_OFFSET
    for b in term.encode("utf-8"):
        h ^= b
        h = (h * _FNV_PRIME) & _MASK32
    return h
