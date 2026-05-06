"""End-to-end filter evaluation tests, including the differential
arrow_backend == python_backend property check.

This is the safety net for Phase F1 backend correctness. Each test
expression runs through both backends and asserts the resulting
BooleanArrays are equal. Any divergence catches a bug in either
implementation — and gives us confidence that future F2b/F3 work
that introduces python_backend dispatch will preserve semantics.
"""

from dataclasses import replace

import pyarrow as pa
import pytest

from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.search.filter.eval.arrow_backend import evaluate_arrow
from milvus_lite.search.filter.eval.python_backend import evaluate_python
from milvus_lite.search.filter.parser import parse_expr
from milvus_lite.search.filter.semantic import compile_expr


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="age", dtype=DataType.INT64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="active", dtype=DataType.BOOL),
        FieldSchema(name="category", dtype=DataType.VARCHAR),
    ])


@pytest.fixture
def sample_table():
    """A test table with a mix of values, including some nulls."""
    return pa.table({
        "id": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "age": [10, 18, 25, 30, 30, 50, 70, 100],
        "title": ["intro", "ai", "news", None, "blog", "tech", "ai", None],
        "score": [0.1, 0.5, 0.7, 0.3, 0.9, 0.5, 1.0, 0.0],
        "active": [True, False, True, True, False, True, True, False],
        "category": ["news", "tech", "tech", "blog", "news", "tech", "ai", "blog"],
    })


def compile_str(s: str, schema):
    return compile_expr(parse_expr(s), schema, source=s)


def force_python(compiled):
    return replace(compiled, backend="python")


# ===========================================================================
# Differential arrow == python tests
# ===========================================================================

DIFFERENTIAL_EXPRESSIONS = [
    # ── Simple comparisons ──────────────────────────────────────
    "age == 25",
    "age != 25",
    "age < 25",
    "age <= 25",
    "age > 25",
    "age >= 25",
    "age == 0",
    "age == 100",
    # ── Float comparisons ───────────────────────────────────────
    "score > 0.5",
    "score <= 0.5",
    "score == 0.0",
    "score != 1.0",
    # ── Int↔float promotion ────────────────────────────────────
    "age > 25.5",
    "age >= 18",
    "score > 0",
    "score <= 1",
    # ── String comparisons ──────────────────────────────────────
    "category == 'tech'",
    "category != 'tech'",
    "category == 'news'",
    # ── Boolean field ───────────────────────────────────────────
    "active",
    "active == true",
    "active == false",
    "not active",
    # ── IN ──────────────────────────────────────────────────────
    "age in [10, 25, 50]",
    "age in [99, 100]",
    "age in []",
    "age not in [10, 25, 50]",
    "category in ['tech', 'news']",
    "category not in ['blog']",
    # ── Logical AND / OR ────────────────────────────────────────
    "age > 18 and category == 'tech'",
    "age > 18 or category == 'tech'",
    "age > 18 and age < 50",
    "category == 'tech' or category == 'news'",
    # ── NOT ─────────────────────────────────────────────────────
    "not (age > 18)",
    "not (age == 25 and category == 'tech')",
    # ── Complex / nested ────────────────────────────────────────
    "age > 18 and (category == 'tech' or category == 'news')",
    "(age > 50 or score > 0.8) and active",
    "not (active and age < 30)",
    "age > 18 and category in ['tech', 'news'] and score >= 0.5",
    "age in [10, 30, 70] or category in ['blog']",
    # ── Reversed LHS literal ────────────────────────────────────
    "18 < age",
    "0.5 < score",
    "'tech' == category",
    # ── True/False top level ────────────────────────────────────
    "true",
    "false",
    # ── Nullable field interactions ─────────────────────────────
    "title == 'ai'",
    "title != 'ai'",
    "title == 'nonexistent'",
    # ── Edge: empty result ──────────────────────────────────────
    "age == 999",
    "age > 100 and age < 0",
    # ── Edge: matches everything ────────────────────────────────
    "age >= 0",
    "age >= 0 or age < 0",
    # ── Phase F2a: arithmetic ───────────────────────────────────
    "age + 1 > 25",
    "age - 5 > 20",
    "age * 2 >= 50",
    "age / 2 < 30",
    "score * 2 > 1.0",
    "score + 0.1 < 0.6",
    "age + score > 25.5",     # int + float promotion
    "age * 2 + 1 > 50",       # nested arithmetic
    "(age + 5) * 2 > 60",     # parens
    "-age + 100 > 50",        # unary minus
    "age + 1 > age",          # field on both sides
    "age - 1 == 24",
    # ── Phase F2a: LIKE ─────────────────────────────────────────
    "title like 'a%'",
    "title like '%i%'",
    "title like 'ai'",        # exact match
    "title like '_i'",        # single-char wildcard
    "category like 'tech'",
    "category like 'te%'",
    "title like 'AI%'",       # case-sensitive
    # ── Phase F2a: IS NULL / IS NOT NULL ────────────────────────
    "title is null",
    "title is not null",
    "age is null",            # always False (non-nullable)
    "age is not null",        # always True (non-nullable)
    "title is null and age > 18",
    "title is not null or category == 'tech'",
    # ── Phase F2a: combined ─────────────────────────────────────
    "age + 1 > 20 and title like '%i%'",
    "title is not null and age * 2 > 30",
    "(age - 10) * 2 < 50 or score / 2 > 0.4",
]


@pytest.mark.parametrize("expr_str", DIFFERENTIAL_EXPRESSIONS)
def test_differential_arrow_python(expr_str, sample_table, schema):
    """The arrow and python backends must produce the same boolean mask
    for every expression on every row."""
    compiled = compile_str(expr_str, schema)
    arrow_result = evaluate_arrow(compiled, sample_table)
    py_result = evaluate_python(force_python(compiled), sample_table)

    assert arrow_result.equals(py_result), (
        f"backend mismatch on {expr_str!r}:\n"
        f"  arrow:  {arrow_result.to_pylist()}\n"
        f"  python: {py_result.to_pylist()}"
    )


# ===========================================================================
# Specific result-value tests (no differential — assert exact rows)
# ===========================================================================

def test_simple_filter(sample_table, schema):
    compiled = compile_str("age > 25", schema)
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [
        False,  # age=10
        False,  # age=18
        False,  # age=25 (not strictly >)
        True,   # age=30
        True,   # age=30
        True,   # age=50
        True,   # age=70
        True,   # age=100
    ]


def test_in_filter(sample_table, schema):
    compiled = compile_str("age in [10, 30, 100]", schema)
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [
        True,   # 10
        False,  # 18
        False,  # 25
        True,   # 30
        True,   # 30
        False,  # 50
        False,  # 70
        True,   # 100
    ]


def test_complex_and_or(sample_table, schema):
    compiled = compile_str(
        "age >= 30 and (category == 'tech' or category == 'news')",
        schema,
    )
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [
        False,  # age=10
        False,  # age=18 (< 30)
        False,  # age=25 (< 30)
        False,  # age=30, category=blog
        True,   # age=30, category=news
        True,   # age=50, category=tech
        False,  # age=70, category=ai
        False,  # age=100, category=blog
    ]


def test_not_in_filter(sample_table, schema):
    compiled = compile_str("category not in ['tech', 'news']", schema)
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [
        False,  # news (excluded)
        False,  # tech
        False,  # tech
        True,   # blog
        False,  # news
        False,  # tech
        True,   # ai
        True,   # blog
    ]


def test_bool_field_filter(sample_table, schema):
    compiled = compile_str("active", schema)
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [True, False, True, True, False, True, True, False]


def test_top_level_true(sample_table, schema):
    """Top-level `true` should match every row."""
    compiled = compile_str("true", schema)
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [True] * 8


def test_top_level_false(sample_table, schema):
    compiled = compile_str("false", schema)
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [False] * 8


def test_null_field_does_not_match(sample_table, schema):
    """For nullable fields, null values should never match a filter
    (NULL → False at top level)."""
    compiled = compile_str("title == 'ai'", schema)
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [
        False,  # 'intro'
        True,   # 'ai'
        False,  # 'news'
        False,  # NULL
        False,  # 'blog'
        False,  # 'tech'
        True,   # 'ai'
        False,  # NULL
    ]


def test_null_field_negation(sample_table, schema):
    """`title != 'ai'` should also be False for nulls (not True!).
    This tests Kleene three-valued logic."""
    compiled = compile_str("title != 'ai'", schema)
    result = evaluate_arrow(compiled, sample_table)
    assert result.to_pylist() == [
        True,   # 'intro'
        False,  # 'ai'
        True,   # 'news'
        False,  # NULL → False (not True)
        True,   # 'blog'
        True,   # 'tech'
        False,  # 'ai'
        False,  # NULL → False
    ]


def test_empty_table(schema):
    empty = pa.table({
        "id": pa.array([], type=pa.string()),
        "age": pa.array([], type=pa.int64()),
        "title": pa.array([], type=pa.string()),
        "score": pa.array([], type=pa.float32()),
        "active": pa.array([], type=pa.bool_()),
        "category": pa.array([], type=pa.string()),
    })
    compiled = compile_str("age > 0", schema)
    result = evaluate_arrow(compiled, empty)
    assert len(result) == 0


def test_record_batch_input(sample_table, schema):
    """The evaluator should accept a RecordBatch as well as a Table."""
    batch = sample_table.to_batches()[0]
    compiled = compile_str("age > 25", schema)
    result = evaluate_arrow(compiled, batch)
    assert len(result) == batch.num_rows
    assert result.to_pylist()[3] is True  # age=30


# ===========================================================================
# Dispatcher tests
# ===========================================================================

def test_dispatcher_arrow(sample_table, schema):
    from milvus_lite.search.filter.eval import evaluate
    compiled = compile_str("age > 25", schema)
    result = evaluate(compiled, sample_table)
    assert pa.types.is_boolean(result.type)


def test_dispatcher_python(sample_table, schema):
    from milvus_lite.search.filter.eval import evaluate
    compiled = force_python(compile_str("age > 25", schema))
    result = evaluate(compiled, sample_table)
    assert pa.types.is_boolean(result.type)


def test_dispatcher_unknown_backend(sample_table, schema):
    from milvus_lite.search.filter.eval import evaluate
    bad = replace(compile_str("age > 25", schema), backend="quantum")
    with pytest.raises(ValueError, match="unknown filter backend"):
        evaluate(bad, sample_table)


# ===========================================================================
# Phase F2b — $meta dynamic field access (python_backend only)
# ===========================================================================

@pytest.fixture
def schema_dynamic():
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
            FieldSchema(name="age", dtype=DataType.INT64),
        ],
        enable_dynamic_field=True,
    )


@pytest.fixture
def meta_table():
    """Sample table with $meta column holding JSON strings."""
    return pa.table({
        "id": ["a", "b", "c", "d", "e"],
        "age": [10, 25, 30, 50, 70],
        "$meta": [
            '{"category": "tech", "priority": 1, "score": 0.5}',
            '{"category": "news", "priority": 5, "score": 0.7}',
            '{"category": "tech", "priority": 3, "score": 0.9}',
            None,                                        # null $meta
            '{"category": "blog", "score": 0.2}',         # missing 'priority'
        ],
    })


def test_meta_string_eq(meta_table, schema_dynamic):
    compiled = compile_expr(
        parse_expr('$meta["category"] == "tech"'),
        schema_dynamic,
        source='$meta["category"] == "tech"',
    )
    assert compiled.backend == "hybrid"
    from milvus_lite.search.filter.eval import evaluate
    result = evaluate(compiled, meta_table)
    assert result.to_pylist() == [True, False, True, False, False]


def test_meta_int_compare(meta_table, schema_dynamic):
    compiled = compile_expr(
        parse_expr('$meta["priority"] > 2'),
        schema_dynamic,
    )
    from milvus_lite.search.filter.eval import evaluate
    result = evaluate(compiled, meta_table)
    # a (1, no), b (5, yes), c (3, yes), d (null), e (missing)
    assert result.to_pylist() == [False, True, True, False, False]


def test_meta_float_compare(meta_table, schema_dynamic):
    compiled = compile_expr(
        parse_expr('$meta["score"] > 0.5'),
        schema_dynamic,
    )
    from milvus_lite.search.filter.eval import evaluate
    result = evaluate(compiled, meta_table)
    # a (0.5, no — strictly >), b (0.7), c (0.9), d (null), e (0.2)
    assert result.to_pylist() == [False, True, True, False, False]


def test_meta_combined_with_regular_field(meta_table, schema_dynamic):
    compiled = compile_expr(
        parse_expr('age > 20 and $meta["category"] == "tech"'),
        schema_dynamic,
    )
    from milvus_lite.search.filter.eval import evaluate
    result = evaluate(compiled, meta_table)
    # a (10, fail), b (25, news), c (30, tech), d (null), e (70, blog)
    assert result.to_pylist() == [False, False, True, False, False]


def test_meta_arithmetic(meta_table, schema_dynamic):
    compiled = compile_expr(
        parse_expr('$meta["score"] * 2 > 1.0'),
        schema_dynamic,
    )
    from milvus_lite.search.filter.eval import evaluate
    result = evaluate(compiled, meta_table)
    # 0.5*2=1.0 (no, strict >), 0.7*2=1.4 (yes), 0.9*2=1.8, null, 0.2*2=0.4
    assert result.to_pylist() == [False, True, True, False, False]


def test_meta_null_meta_column(meta_table, schema_dynamic):
    """Row with $meta=null should never match any expression."""
    compiled = compile_expr(
        parse_expr('$meta["anything"] == "x"'),
        schema_dynamic,
    )
    from milvus_lite.search.filter.eval import evaluate
    result = evaluate(compiled, meta_table)
    # d has $meta=null → False
    assert result.to_pylist()[3] is False


def test_meta_missing_key(meta_table, schema_dynamic):
    """$meta exists but the key is absent → False."""
    compiled = compile_expr(
        parse_expr('$meta["nonexistent"] == "x"'),
        schema_dynamic,
    )
    from milvus_lite.search.filter.eval import evaluate
    result = evaluate(compiled, meta_table)
    assert all(v is False for v in result.to_pylist())


def test_meta_in_or(meta_table, schema_dynamic):
    compiled = compile_expr(
        parse_expr('$meta["category"] == "tech" or $meta["category"] == "news"'),
        schema_dynamic,
    )
    from milvus_lite.search.filter.eval import evaluate
    result = evaluate(compiled, meta_table)
    # tech, news, tech, null, blog
    assert result.to_pylist() == [True, True, True, False, False]


def test_arrow_backend_rejects_meta(meta_table, schema_dynamic):
    """Force arrow backend on a $meta expression — should fail loudly."""
    from milvus_lite.search.filter.eval.arrow_backend import evaluate_arrow
    compiled = compile_expr(
        parse_expr('$meta["category"] == "tech"'),
        schema_dynamic,
    )
    forced = replace(compiled, backend="arrow")
    with pytest.raises(NotImplementedError, match="\\$meta"):
        evaluate_arrow(forced, meta_table)


# ===========================================================================
# Phase F3+ — hybrid_backend differential parity vs python_backend
# ===========================================================================
#
# Each $meta expression runs through BOTH the hybrid backend (per-batch JSON
# preprocessing → arrow_backend) and the python backend (row-wise interpreter)
# and the resulting BooleanArrays must agree on every row. This is the
# correctness safety net for the Phase F3+ optimization: any divergence means
# the hybrid path has misimplemented the semantics that the python_backend
# differential test already validated.

_META_DIFF_EXPRS = [
    '$meta["category"] == "tech"',
    '$meta["category"] != "tech"',
    '$meta["priority"] > 2',
    '$meta["priority"] >= 3',
    '$meta["priority"] < 5',
    '$meta["score"] > 0.5',
    '$meta["score"] * 2 > 1.0',
    '$meta["score"] + 0.1 >= 0.6',
    '$meta["category"] == "tech" or $meta["category"] == "news"',
    '$meta["category"] == "tech" and $meta["priority"] > 2',
    'age > 20 and $meta["category"] == "tech"',
    'age < 50 or $meta["score"] > 0.8',
    'not ($meta["category"] == "blog")',
    # NOTE: `$meta[...] in [...]` is rejected by the parser today —
    # `in` requires a FieldRef on the LHS, not a MetaAccess. When the
    # parser is extended to allow it (Phase F4+), add it here.
    '$meta["title"] like "AI%"',
    '$meta["nonexistent"] == "x"',
]


@pytest.mark.parametrize("source", _META_DIFF_EXPRS)
def test_meta_hybrid_vs_python_parity(source, meta_table, schema_dynamic):
    """hybrid backend == python backend on every $meta expression."""
    compiled = compile_expr(parse_expr(source), schema_dynamic, source=source)
    # Sanity: dispatcher should have selected hybrid for $meta exprs.
    assert compiled.backend == "hybrid"

    from milvus_lite.search.filter.eval.hybrid_backend import evaluate_hybrid
    hybrid_result = evaluate_hybrid(compiled, meta_table)

    py_compiled = replace(compiled, backend="python")
    python_result = evaluate_python(py_compiled, meta_table)

    assert hybrid_result.to_pylist() == python_result.to_pylist(), (
        f"hybrid/python divergence on: {source}\n"
        f"  hybrid={hybrid_result.to_pylist()}\n"
        f"  python={python_result.to_pylist()}"
    )


def test_meta_hybrid_handles_missing_meta_column(schema_dynamic):
    """If $meta column is absent entirely, hybrid degrades to all-null
    extracted columns and the result still matches python_backend."""
    table = pa.table({
        "id": ["a", "b"],
        "age": [10, 20],
    })
    compiled = compile_expr(
        parse_expr('$meta["x"] == "y"'),
        schema_dynamic,
        source='$meta["x"] == "y"',
    )
    from milvus_lite.search.filter.eval.hybrid_backend import evaluate_hybrid
    hybrid_result = evaluate_hybrid(compiled, table)
    python_result = evaluate_python(replace(compiled, backend="python"), table)
    assert hybrid_result.to_pylist() == python_result.to_pylist()


def test_meta_hybrid_falls_back_on_mixed_types(schema_dynamic):
    """When a $meta key has heterogeneous types across rows, pa.array
    raises and hybrid falls back to python_backend transparently."""
    table = pa.table({
        "id": ["a", "b", "c"],
        "age": [10, 20, 30],
        "$meta": [
            '{"mixed": 1}',
            '{"mixed": "two"}',
            '{"mixed": 3.0}',
        ],
    })
    compiled = compile_expr(
        parse_expr('$meta["mixed"] == 1'),
        schema_dynamic,
        source='$meta["mixed"] == 1',
    )
    from milvus_lite.search.filter.eval.hybrid_backend import evaluate_hybrid
    # Should not raise — fallback to python_backend.
    result = evaluate_hybrid(compiled, table)
    # Compare with python_backend directly to confirm fallback parity.
    python_result = evaluate_python(replace(compiled, backend="python"), table)
    assert result.to_pylist() == python_result.to_pylist()


def test_meta_hybrid_record_batch_input(meta_table, schema_dynamic):
    """RecordBatch (rather than Table) input should still be handled."""
    batch = meta_table.to_batches()[0]
    compiled = compile_expr(
        parse_expr('$meta["category"] == "tech"'),
        schema_dynamic,
    )
    from milvus_lite.search.filter.eval.hybrid_backend import evaluate_hybrid
    result = evaluate_hybrid(compiled, batch)
    assert result.to_pylist() == [True, False, True, False, False]
