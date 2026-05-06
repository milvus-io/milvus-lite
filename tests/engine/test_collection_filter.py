"""Phase 8 / Phase F1 integration tests — Collection.search/get/query
with the `expr` parameter.

These tests run filters through the full Collection write/read path
(WAL → MemTable → flush → segments → search/get/query). They verify
both the happy path and that filter errors propagate cleanly.
"""

import os

import pytest

from milvus_lite import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    FilterFieldError,
    FilterParseError,
    FilterTypeError,
)


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="age", dtype=DataType.INT64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="active", dtype=DataType.BOOL),
        FieldSchema(name="category", dtype=DataType.VARCHAR),
    ])


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("c", str(tmp_path / "d"), schema)
    yield c
    c.close()


def _populate(col, rows=None):
    """Insert a small fixed dataset for filter tests."""
    if rows is None:
        rows = [
            {"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "age": 18, "title": "intro",
             "score": 0.5, "active": True, "category": "tech"},
            {"id": "b", "vec": [0.0, 1.0, 0.0, 0.0], "age": 25, "title": "ai",
             "score": 0.7, "active": False, "category": "tech"},
            {"id": "c", "vec": [0.0, 0.0, 1.0, 0.0], "age": 30, "title": "news",
             "score": 0.3, "active": True, "category": "news"},
            {"id": "d", "vec": [0.0, 0.0, 0.0, 1.0], "age": 50, "title": "blog",
             "score": 0.9, "active": False, "category": "blog"},
            {"id": "e", "vec": [1.0, 1.0, 0.0, 0.0], "age": 70, "title": "ai",
             "score": 1.0, "active": True, "category": "tech"},
        ]
    col.insert(rows)


# ---------------------------------------------------------------------------
# search() with expr
# ---------------------------------------------------------------------------

def test_search_with_filter_int(col):
    _populate(col)
    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=10,
        metric_type="L2",
        expr="age > 25",
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    # Only c, d, e have age > 25
    assert ids == {"c", "d", "e"}


def test_search_with_filter_string(col):
    _populate(col)
    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=10,
        metric_type="L2",
        expr="category == 'tech'",
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"a", "b", "e"}


def test_search_with_filter_in(col):
    _populate(col)
    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=10,
        metric_type="L2",
        expr="age in [18, 30, 70]",
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"a", "c", "e"}


def test_search_with_complex_filter(col):
    _populate(col)
    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=10,
        metric_type="L2",
        expr="age >= 25 and (category == 'tech' or category == 'news')",
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    # b(25,tech), c(30,news), e(70,tech) match
    assert ids == {"b", "c", "e"}


def test_search_filter_with_top_k_limit(col):
    """top_k still applies after filtering — only the closest ones."""
    _populate(col)
    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=2,
        metric_type="L2",
        expr="active == true",
    )
    [hits] = results
    assert len(hits) == 2  # capped by top_k
    # All hits must satisfy filter (active=true) — a, c, e are active
    for h in hits:
        assert h["id"] in ("a", "c", "e")


def test_search_filter_no_matches(col):
    _populate(col)
    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=10,
        metric_type="L2",
        expr="age > 999",
    )
    assert results == [[]]


def test_search_filter_after_flush(col):
    """Filter should work the same on segments as on MemTable."""
    _populate(col)
    col.flush()
    assert col.count() == 0  # MemTable empty, all in segment

    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=10,
        metric_type="L2",
        expr="category == 'tech'",
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"a", "b", "e"}


def test_search_filter_mixed_memtable_and_segment(col):
    """Half flushed, half in MemTable — filter must apply to both."""
    rows = [
        {"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "age": 18, "title": "x",
         "score": 0.5, "active": True, "category": "tech"},
        {"id": "b", "vec": [0.0, 1.0, 0.0, 0.0], "age": 25, "title": "x",
         "score": 0.7, "active": True, "category": "news"},
    ]
    col.insert(rows)
    col.flush()

    rows2 = [
        {"id": "c", "vec": [0.0, 0.0, 1.0, 0.0], "age": 30, "title": "x",
         "score": 0.3, "active": True, "category": "tech"},
    ]
    col.insert(rows2)

    results = col.search(
        [[0.5, 0.5, 0.5, 0.5]],
        top_k=10,
        metric_type="L2",
        expr="category == 'tech'",
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"a", "c"}


# ---------------------------------------------------------------------------
# get() with expr
# ---------------------------------------------------------------------------

def test_get_with_filter_match(col):
    _populate(col)
    out = col.get(["a", "b", "c"], expr="age > 20")
    ids = {r["id"] for r in out}
    assert ids == {"b", "c"}


def test_get_with_filter_no_match(col):
    _populate(col)
    out = col.get(["a", "b", "c"], expr="age > 999")
    assert out == []


def test_get_with_filter_unknown_pk(col):
    _populate(col)
    out = col.get(["a", "missing"], expr="age >= 0")
    ids = {r["id"] for r in out}
    assert ids == {"a"}


def test_get_filter_after_flush(col):
    _populate(col)
    col.flush()
    out = col.get(["a", "b", "c", "d", "e"], expr="active")
    ids = {r["id"] for r in out}
    assert ids == {"a", "c", "e"}


# ---------------------------------------------------------------------------
# query() — pure scalar
# ---------------------------------------------------------------------------

def test_query_basic(col):
    _populate(col)
    out = col.query("age > 25")
    ids = {r["id"] for r in out}
    assert ids == {"c", "d", "e"}


def test_query_with_in(col):
    _populate(col)
    out = col.query("category in ['tech', 'news']")
    ids = {r["id"] for r in out}
    assert ids == {"a", "b", "c", "e"}


def test_query_complex(col):
    _populate(col)
    out = col.query("age >= 25 and active == true")
    ids = {r["id"] for r in out}
    # b active=False (excluded); c, e are active and >= 25
    assert ids == {"c", "e"}


def test_query_returns_full_records(col):
    _populate(col)
    out = col.query("id == 'a'")
    assert len(out) == 1
    rec = out[0]
    assert rec["id"] == "a"
    assert rec["age"] == 18
    assert rec["title"] == "intro"


def test_query_output_fields(col):
    _populate(col)
    out = col.query("age >= 25", output_fields=["title", "age"])
    for rec in out:
        # pk is always kept; output_fields selects others
        assert set(rec.keys()) == {"id", "title", "age"}


def test_query_limit(col):
    _populate(col)
    out = col.query("age >= 0", limit=2)
    assert len(out) == 2


def test_query_no_matches(col):
    _populate(col)
    out = col.query("age > 999")
    assert out == []


def test_query_empty_collection(col):
    out = col.query("age >= 0")
    assert out == []


def test_query_after_flush(col):
    _populate(col)
    col.flush()
    out = col.query("category == 'tech'")
    ids = {r["id"] for r in out}
    assert ids == {"a", "b", "e"}


def test_query_mixed_memtable_segment(col):
    """Half flushed, half in MemTable — query unions both."""
    col.insert([
        {"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "age": 18, "title": "x",
         "score": 0.5, "active": True, "category": "tech"},
    ])
    col.flush()
    col.insert([
        {"id": "b", "vec": [0.0, 1.0, 0.0, 0.0], "age": 25, "title": "y",
         "score": 0.7, "active": True, "category": "tech"},
    ])
    out = col.query("category == 'tech'")
    ids = {r["id"] for r in out}
    assert ids == {"a", "b"}


def test_query_after_delete(col):
    _populate(col)
    col.delete(["a", "b"])
    out = col.query("category == 'tech'")
    ids = {r["id"] for r in out}
    # a, b deleted; e remains
    assert ids == {"e"}


def test_query_accepts_empty_expr(col):
    """query() with empty or None expr returns all records (no error)."""
    results_empty = col.query("")
    results_none = col.query(None)
    assert isinstance(results_empty, list)
    assert isinstance(results_none, list)
    assert len(results_empty) == len(results_none)

def test_query_rejects_non_string_expr(col):
    with pytest.raises(TypeError, match="string or None"):
        col.query(123)


# ---------------------------------------------------------------------------
# Error propagation — filter parse / type errors
# ---------------------------------------------------------------------------

def test_search_invalid_expr_syntax(col):
    _populate(col)
    with pytest.raises(FilterParseError):
        col.search([[1.0, 0.0, 0.0, 0.0]], expr="age >> 18")


def test_search_unknown_field(col):
    _populate(col)
    with pytest.raises(FilterFieldError, match="unknown field"):
        col.search([[1.0, 0.0, 0.0, 0.0]], expr="agg > 18")


def test_search_type_mismatch(col):
    _populate(col)
    with pytest.raises(FilterTypeError):
        col.search([[1.0, 0.0, 0.0, 0.0]], expr="age > 'eighteen'")


def test_get_invalid_expr(col):
    _populate(col)
    with pytest.raises(FilterParseError):
        col.get(["a"], expr="age >> 18")


def test_query_unknown_field_did_you_mean(col):
    _populate(col)
    with pytest.raises(FilterFieldError) as exc:
        col.query("agg > 18")
    msg = str(exc.value)
    assert "did you mean 'age'" in msg


def test_search_reserved_field_rejected(col):
    _populate(col)
    with pytest.raises(FilterFieldError, match="reserved"):
        col.search([[1.0, 0.0, 0.0, 0.0]], expr="_seq > 0")


def test_search_vector_field_rejected(col):
    _populate(col)
    with pytest.raises(FilterTypeError, match="float_vector"):
        col.search([[1.0, 0.0, 0.0, 0.0]], expr="vec > 0")


# ---------------------------------------------------------------------------
# Search filter with persistence (across restart)
# ---------------------------------------------------------------------------

def test_filter_works_across_restart(tmp_path, schema):
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([
        {"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "age": 18, "title": "x",
         "score": 0.5, "active": True, "category": "tech"},
        {"id": "b", "vec": [0.0, 1.0, 0.0, 0.0], "age": 25, "title": "y",
         "score": 0.7, "active": True, "category": "news"},
    ])
    col1.close()  # flushes to segment

    col2 = Collection("c", data_dir, schema)
    out = col2.query("category == 'tech'")
    assert len(out) == 1
    assert out[0]["id"] == "a"
    col2.close()


# ===========================================================================
# Phase F2a — arithmetic, LIKE, IS NULL through Collection
# ===========================================================================

def test_search_with_arithmetic(col):
    _populate(col)
    # age + 1 > 25  matches age >= 25 → b(25), c(30), d(50), e(70)
    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]], top_k=10, metric_type="L2",
        expr="age + 1 > 25",
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"b", "c", "d", "e"}


def test_query_with_arithmetic(col):
    _populate(col)
    out = col.query("age * 2 >= 50")
    ids = {r["id"] for r in out}
    # 2*age >= 50 → age >= 25 → b, c, d, e
    assert ids == {"b", "c", "d", "e"}


def test_query_with_division(col):
    _populate(col)
    out = col.query("age / 2 > 12")
    ids = {r["id"] for r in out}
    # age/2 > 12 → age > 24 → b(25), c(30), d(50), e(70)
    assert ids == {"b", "c", "d", "e"}


def test_query_with_unary_minus(col):
    _populate(col)
    out = col.query("-age < -25")
    ids = {r["id"] for r in out}
    # -age < -25 → age > 25 → c, d, e
    assert ids == {"c", "d", "e"}


def test_query_with_like_prefix(col):
    _populate(col)
    out = col.query("title like 'a%'")
    ids = {r["id"] for r in out}
    # titles starting with 'a': b='ai', e='ai'
    assert ids == {"b", "e"}


def test_query_with_like_contains(col):
    _populate(col)
    out = col.query("title like '%i%'")
    ids = {r["id"] for r in out}
    # titles containing 'i': a='intro', b='ai', e='ai'
    assert ids == {"a", "b", "e"}


def test_query_with_like_single_char(col):
    _populate(col)
    out = col.query("title like '_i'")
    ids = {r["id"] for r in out}
    # 2-char titles ending with 'i': b='ai', e='ai'
    assert ids == {"b", "e"}


def test_query_with_is_null(tmp_path, schema):
    """Need to insert nullable values to test IS NULL."""
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([
        {"id": "with", "vec": [0.5, 0.25, 0.125, 0.75], "age": 10,
         "title": "hello", "score": 0.5, "active": True, "category": "tech"},
        {"id": "without", "vec": [0.5, 0.25, 0.125, 0.75], "age": 20,
         "score": 0.5, "active": True, "category": "tech"},  # title omitted → null
    ])
    out = col.query("title is null")
    ids = {r["id"] for r in out}
    assert ids == {"without"}

    out = col.query("title is not null")
    ids = {r["id"] for r in out}
    assert ids == {"with"}
    col.close()


def test_query_with_complex_f2a(col):
    _populate(col)
    out = col.query("age + 1 > 20 and title like '%i%'")
    ids = {r["id"] for r in out}
    # age+1 > 20 → age >= 20 → b,c,d,e
    # AND title like '%i%' → b='ai', e='ai' (and a='intro' but a has age=18)
    assert ids == {"b", "e"}


def test_query_arithmetic_with_string_field_rejected(col):
    _populate(col)
    with pytest.raises(FilterTypeError, match="numeric"):
        col.query("category + 1 > 0")


def test_query_like_on_int_rejected(col):
    _populate(col)
    with pytest.raises(FilterTypeError, match="string"):
        col.query("age like '1%'")


# ===========================================================================
# Phase F2b — $meta dynamic field through Collection
# ===========================================================================

@pytest.fixture
def schema_dynamic():
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
            FieldSchema(name="age", dtype=DataType.INT64),
        ],
        enable_dynamic_field=True,
    )


@pytest.fixture
def col_dynamic(tmp_path, schema_dynamic):
    c = Collection("c", str(tmp_path / "d"), schema_dynamic)
    yield c
    c.close()


def _populate_dynamic(col):
    col.insert([
        {"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "age": 18,
         "category": "tech", "priority": 1, "score": 0.5},
        {"id": "b", "vec": [0.0, 1.0, 0.0, 0.0], "age": 25,
         "category": "news", "priority": 5, "score": 0.7},
        {"id": "c", "vec": [0.0, 0.0, 1.0, 0.0], "age": 30,
         "category": "tech", "priority": 3, "score": 0.9},
        {"id": "d", "vec": [0.0, 0.0, 0.0, 1.0], "age": 50,
         "category": "blog", "priority": 2, "score": 0.3},
    ])


def test_query_meta_string_eq(col_dynamic):
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('$meta["category"] == "tech"')
    ids = {r["id"] for r in out}
    assert ids == {"a", "c"}


def test_query_meta_int_compare(col_dynamic):
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('$meta["priority"] > 2')
    ids = {r["id"] for r in out}
    # b(5), c(3) match
    assert ids == {"b", "c"}


def test_query_meta_combined_with_regular(col_dynamic):
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('age > 20 and $meta["category"] == "tech"')
    ids = {r["id"] for r in out}
    # a fails on age, b is news, c matches, d is blog
    assert ids == {"c"}


def test_query_meta_after_flush(col_dynamic):
    """Meta access works on segments too — JSON column survives flush."""
    _populate_dynamic(col_dynamic)
    col_dynamic.flush()
    assert col_dynamic.count() == 0
    out = col_dynamic.query('$meta["category"] == "tech"')
    ids = {r["id"] for r in out}
    assert ids == {"a", "c"}


def test_query_meta_mixed_memtable_and_segment(col_dynamic):
    """Half flushed, half in MemTable — meta filter must apply to both."""
    col_dynamic.insert([
        {"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "age": 18,
         "category": "tech", "priority": 1},
    ])
    col_dynamic.flush()
    col_dynamic.insert([
        {"id": "b", "vec": [0.0, 1.0, 0.0, 0.0], "age": 25,
         "category": "news", "priority": 5},
    ])
    out = col_dynamic.query('$meta["priority"] >= 1')
    ids = {r["id"] for r in out}
    assert ids == {"a", "b"}


def test_search_with_meta_filter(col_dynamic):
    _populate_dynamic(col_dynamic)
    results = col_dynamic.search(
        [[1.0, 0.0, 0.0, 0.0]], top_k=10, metric_type="L2",
        expr='$meta["category"] == "tech"',
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"a", "c"}


def test_get_with_meta_filter(col_dynamic):
    _populate_dynamic(col_dynamic)
    out = col_dynamic.get(
        ["a", "b", "c", "d"],
        expr='$meta["priority"] > 2',
    )
    ids = {r["id"] for r in out}
    assert ids == {"b", "c"}


def test_query_meta_arithmetic(col_dynamic):
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('$meta["score"] * 2 > 1.0')
    ids = {r["id"] for r in out}
    # 0.5*2=1.0 (no), 0.7*2=1.4, 0.9*2=1.8, 0.3*2=0.6
    assert ids == {"b", "c"}


def test_query_meta_missing_key(col_dynamic):
    """Querying a key that doesn't exist in any record — empty result."""
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('$meta["nonexistent"] == "x"')
    assert out == []


def test_query_meta_partial_records(col_dynamic):
    """Some records have the key, others don't — only matching ones."""
    col_dynamic.insert([
        {"id": "with", "vec": [1.0, 0.0, 0.0, 0.0], "age": 18,
         "extra_field": "value1"},
        {"id": "without", "vec": [0.0, 1.0, 0.0, 0.0], "age": 25,
         "other_field": "value2"},
    ])
    out = col_dynamic.query('$meta["extra_field"] == "value1"')
    ids = {r["id"] for r in out}
    assert ids == {"with"}


def test_query_meta_persists_across_restart(tmp_path, schema_dynamic):
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema_dynamic)
    col1.insert([
        {"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "age": 18, "category": "tech"},
        {"id": "b", "vec": [0.0, 1.0, 0.0, 0.0], "age": 25, "category": "news"},
    ])
    col1.close()  # flushes to segment

    col2 = Collection("c", data_dir, schema_dynamic)
    out = col2.query('$meta["category"] == "tech"')
    assert len(out) == 1
    assert out[0]["id"] == "a"
    col2.close()


def test_meta_without_dynamic_field_rejected(col):
    """Schema without enable_dynamic_field → meta access fails at compile."""
    _populate(col)
    with pytest.raises(FilterFieldError, match="enable_dynamic_field"):
        col.query('$meta["x"] == 1')


def test_meta_complex_expression(col_dynamic):
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query(
        '($meta["category"] == "tech" or $meta["category"] == "news") '
        'and $meta["priority"] > 2'
    )
    ids = {r["id"] for r in out}
    # b (news, prio 5), c (tech, prio 3)
    assert ids == {"b", "c"}


# ===========================================================================
# Issue #10 — bare dynamic field names (auto-rewrite to $meta)
# ===========================================================================

def test_query_dynamic_bare_field_eq(col_dynamic):
    """color == 'tech' should work without $meta["..."] wrapper."""
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('category == "tech"')
    ids = {r["id"] for r in out}
    assert ids == {"a", "c"}


def test_query_dynamic_bare_field_compare(col_dynamic):
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('priority > 2')
    ids = {r["id"] for r in out}
    assert ids == {"b", "c"}


def test_query_dynamic_bare_combined_with_schema_field(col_dynamic):
    """Mix schema field (age) and bare dynamic field (category)."""
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('age > 20 and category == "tech"')
    ids = {r["id"] for r in out}
    assert ids == {"c"}


def test_query_dynamic_bare_in_list(col_dynamic):
    _populate_dynamic(col_dynamic)
    out = col_dynamic.query('category in ["tech", "blog"]')
    ids = {r["id"] for r in out}
    assert ids == {"a", "c", "d"}


def test_search_dynamic_bare_filter(col_dynamic):
    _populate_dynamic(col_dynamic)
    results = col_dynamic.search(
        [[1.0, 0.0, 0.0, 0.0]], top_k=10, metric_type="L2",
        expr='category == "tech"',
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"a", "c"}


def test_query_dynamic_bare_after_flush(col_dynamic):
    """Bare dynamic field filter works on flushed segments too."""
    _populate_dynamic(col_dynamic)
    col_dynamic.flush()
    out = col_dynamic.query('category == "tech"')
    ids = {r["id"] for r in out}
    assert ids == {"a", "c"}


# ===========================================================================
# Phase F2c — filter expression cache
# ===========================================================================

def test_filter_cache_hit_returns_same_instance(col):
    """Repeated _compile_filter on the same expression returns the same
    CompiledExpr object — proves the cache is working."""
    _populate(col)
    first = col._compile_filter("age > 18")
    second = col._compile_filter("age > 18")
    assert first is second


def test_filter_cache_distinct_expressions(col):
    _populate(col)
    a = col._compile_filter("age > 18")
    b = col._compile_filter("age > 25")
    assert a is not b
    assert len(col._filter_cache) == 2


def test_filter_cache_hits_recorded(col):
    _populate(col)
    col._compile_filter("age > 18")  # miss
    col._compile_filter("age > 18")  # hit
    col._compile_filter("age > 18")  # hit
    assert col._filter_cache.hits == 2
    assert col._filter_cache.misses == 1


def test_filter_cache_through_search(col):
    _populate(col)
    col.search([[1.0, 0.0, 0.0, 0.0]], top_k=10, expr="age > 18")
    col.search([[1.0, 0.0, 0.0, 0.0]], top_k=10, expr="age > 18")
    col.search([[1.0, 0.0, 0.0, 0.0]], top_k=10, expr="age > 18")
    assert col._filter_cache.hits == 2
    assert col._filter_cache.misses == 1


def test_filter_cache_through_query(col):
    _populate(col)
    col.query("category == 'tech'")
    col.query("category == 'tech'")
    assert col._filter_cache.hits == 1


def test_filter_cache_through_get(col):
    _populate(col)
    col.get(["a"], expr="age > 0")
    col.get(["a"], expr="age > 0")
    assert col._filter_cache.hits == 1


def test_filter_cache_lru_eviction(col):
    """Force a small cache and verify LRU eviction."""
    from milvus_lite.search.filter.cache import LRUCache
    col._filter_cache = LRUCache(maxsize=2)

    _populate(col)
    col._compile_filter("age > 1")
    col._compile_filter("age > 2")
    col._compile_filter("age > 3")  # evicts age > 1
    assert len(col._filter_cache) == 2
    assert "age > 1" not in col._filter_cache
    assert "age > 2" in col._filter_cache
    assert "age > 3" in col._filter_cache


def test_filter_cache_does_not_cache_errors(col):
    """A bad expression should never get cached and never poison subsequent
    calls."""
    _populate(col)
    with pytest.raises(FilterParseError):
        col._compile_filter("age >> 18")
    # Cache should still be empty
    assert len(col._filter_cache) == 0
    # And a subsequent valid call works and gets cached.
    col._compile_filter("age > 18")
    assert len(col._filter_cache) == 1


def test_filter_cache_meta_expression(col_dynamic):
    """$meta expressions should also be cached."""
    _populate_dynamic(col_dynamic)
    col_dynamic._compile_filter('$meta["category"] == "tech"')
    col_dynamic._compile_filter('$meta["category"] == "tech"')
    assert col_dynamic._filter_cache.hits == 1


# ===========================================================================
# Issue #17 — chained JSON path access (info["a"]["b"])
# ===========================================================================

@pytest.fixture
def schema_json():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="info", dtype=DataType.JSON),
    ])


@pytest.fixture
def col_json(tmp_path, schema_json):
    c = Collection("c", str(tmp_path / "d"), schema_json)
    yield c
    c.close()


def _populate_json(col):
    col.insert([
        {"id": 1, "vec": [1, 0, 0, 0], "info": {"a": {"b": 1, "c": "x"}}},
        {"id": 2, "vec": [0, 1, 0, 0], "info": {"a": {"b": 5, "c": "y"}}},
        {"id": 3, "vec": [0, 0, 1, 0], "info": {"a": {"b": 10, "c": "z"}}},
    ])


def test_query_json_chained_access(col_json):
    """info["a"]["b"] >= 5 should work."""
    _populate_json(col_json)
    col_json.load()
    out = col_json.query('info["a"]["b"] >= 5')
    ids = {r["id"] for r in out}
    assert ids == {2, 3}


def test_query_json_chained_string(col_json):
    """info["a"]["c"] == 'z' should work."""
    _populate_json(col_json)
    col_json.load()
    out = col_json.query('info["a"]["c"] == "z"')
    ids = {r["id"] for r in out}
    assert ids == {3}


def test_query_json_single_level_still_works(col_json):
    """info["a"] should still work (returns dict, compared dynamically)."""
    _populate_json(col_json)
    col_json.load()
    # Single-level access still works
    out = col_json.query('info["a"]["b"] > 0')
    assert len(out) == 3


def test_query_json_chained_after_flush(col_json):
    """Chained JSON access works on flushed segments too."""
    _populate_json(col_json)
    col_json.flush()
    col_json.load()
    out = col_json.query('info["a"]["b"] >= 5')
    ids = {r["id"] for r in out}
    assert ids == {2, 3}


def test_search_json_chained_filter(col_json):
    """Chained JSON access works in search expr."""
    _populate_json(col_json)
    col_json.load()
    results = col_json.search(
        [[1, 0, 0, 0]], top_k=10, metric_type="L2",
        expr='info["a"]["b"] >= 5',
    )
    ids = {h["id"] for h in results[0]}
    assert ids == {2, 3}


def test_query_json_three_levels(col_json):
    """Three-level chained access: info["x"]["y"]["z"]."""
    col_json.insert([
        {"id": 10, "vec": [1, 0, 0, 0], "info": {"x": {"y": {"z": 42}}}},
        {"id": 11, "vec": [0, 1, 0, 0], "info": {"x": {"y": {"z": 99}}}},
    ])
    col_json.load()
    out = col_json.query('info["x"]["y"]["z"] > 50')
    ids = {r["id"] for r in out}
    assert ids == {11}
