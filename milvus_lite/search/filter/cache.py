"""LRU cache for compiled filter expressions.

Why: in a typical search workload the same filter string is sent many
times in a row (a UI binding the same `expr` to every query in a user
session). Re-parsing + re-compiling each time is wasted work — the
result is fully determined by (schema, expr_string) and the schema is
immutable for a Collection's lifetime.

Design: a tiny OrderedDict-based LRU. Per-Collection instances; no
cross-Collection sharing because each Collection holds its own schema
binding. functools.lru_cache would work but it's awkward on instance
methods and doesn't expose stats / clear / size in a way that fits
our test plan.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Bounded LRU cache.

    Operations:
        get(key)   — None if absent; on hit, mark as most recently used.
        put(key, value) — insert/update, evict oldest if over capacity.
        clear()    — drop everything.
        __len__    — current size.
        hits / misses / evictions — counters for diagnostics.

    Not thread-safe. MilvusLite is single-writer-per-Collection by
    architectural invariant, and Python's dict / OrderedDict are
    GIL-protected for atomic ops, so this is fine for our use.
    """

    __slots__ = ("_cache", "_maxsize", "hits", "misses", "evictions")

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be > 0, got {maxsize}")
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: K) -> Optional[V]:
        """Look up *key*. Returns None on miss; promotes to MRU on hit."""
        try:
            value = self._cache[key]
        except KeyError:
            self.misses += 1
            return None
        # Move to end → most recently used.
        self._cache.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: K, value: V) -> None:
        """Insert or replace *key*. Evicts the oldest entry if over capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: K) -> bool:
        return key in self._cache

    @property
    def maxsize(self) -> int:
        return self._maxsize
