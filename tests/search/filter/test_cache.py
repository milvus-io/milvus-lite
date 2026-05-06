"""Tests for search/filter/cache.py — LRU correctness."""

import pytest

from milvus_lite.search.filter.cache import LRUCache


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_empty_cache():
    c = LRUCache(maxsize=4)
    assert len(c) == 0
    assert c.hits == 0
    assert c.misses == 0
    assert c.evictions == 0


def test_invalid_maxsize():
    with pytest.raises(ValueError):
        LRUCache(maxsize=0)
    with pytest.raises(ValueError):
        LRUCache(maxsize=-1)


# ---------------------------------------------------------------------------
# Basic put / get
# ---------------------------------------------------------------------------

def test_put_then_get():
    c = LRUCache(maxsize=4)
    c.put("a", 1)
    assert c.get("a") == 1
    assert len(c) == 1


def test_get_miss():
    c = LRUCache(maxsize=4)
    assert c.get("nope") is None
    assert c.misses == 1
    assert c.hits == 0


def test_put_replaces_existing():
    c = LRUCache(maxsize=4)
    c.put("a", 1)
    c.put("a", 2)
    assert c.get("a") == 2
    assert len(c) == 1


def test_contains_operator():
    c = LRUCache(maxsize=4)
    c.put("a", 1)
    assert "a" in c
    assert "b" not in c


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

def test_eviction_oldest_first():
    c = LRUCache(maxsize=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.put("d", 4)  # evicts "a"
    assert "a" not in c
    assert "b" in c
    assert "c" in c
    assert "d" in c
    assert len(c) == 3
    assert c.evictions == 1


def test_get_promotes_to_mru():
    c = LRUCache(maxsize=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    # Access "a" — now "b" is the LRU
    c.get("a")
    c.put("d", 4)  # evicts "b"
    assert "a" in c
    assert "b" not in c
    assert "c" in c
    assert "d" in c


def test_replace_promotes_to_mru():
    """put() on an existing key should also move it to MRU."""
    c = LRUCache(maxsize=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.put("a", 11)   # replace + promote
    c.put("d", 4)    # evicts "b" (now LRU)
    assert "a" in c
    assert "b" not in c
    assert c.get("a") == 11


def test_many_evictions():
    c = LRUCache(maxsize=2)
    for i in range(10):
        c.put(f"k{i}", i)
    assert len(c) == 2
    assert c.evictions == 8
    # Only the last two should be present
    assert c.get("k9") == 9
    assert c.get("k8") == 8


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_hit_miss_counters():
    c = LRUCache(maxsize=4)
    c.put("a", 1)
    c.get("a")
    c.get("a")
    c.get("nope")
    c.get("nope")
    c.get("nope")
    assert c.hits == 2
    assert c.misses == 3


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def test_clear():
    c = LRUCache(maxsize=4)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.get("nope")
    c.clear()
    assert len(c) == 0
    assert c.hits == 0
    assert c.misses == 0
    assert c.evictions == 0
    assert c.get("a") is None


# ---------------------------------------------------------------------------
# Generic typing — works with any hashable key
# ---------------------------------------------------------------------------

def test_int_keys():
    c = LRUCache(maxsize=4)
    c.put(1, "one")
    c.put(2, "two")
    assert c.get(1) == "one"
    assert c.get(2) == "two"


def test_tuple_keys():
    c = LRUCache(maxsize=4)
    c.put(("a", 1), "x")
    assert c.get(("a", 1)) == "x"
    assert c.get(("a", 2)) is None
