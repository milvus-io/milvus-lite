"""Tests for search/filter/semantic.py — type checking + field binding."""

import pytest

from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.search.filter.exceptions import FilterFieldError, FilterTypeError
from milvus_lite.search.filter.parser import parse_expr
from milvus_lite.search.filter.semantic import compile_expr


@pytest.fixture
def schema():
    """Schema with a mix of types for type checking."""
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="age", dtype=DataType.INT64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="active", dtype=DataType.BOOL),
        FieldSchema(name="category", dtype=DataType.VARCHAR),
    ])


def compile_str(s: str, schema):
    return compile_expr(parse_expr(s), schema, source=s)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_simple_int_comparison(schema):
    c = compile_str("age > 18", schema)
    assert c.backend == "arrow"
    assert "age" in c.fields


def test_string_equality(schema):
    c = compile_str("category == 'tech'", schema)
    assert "category" in c.fields


def test_bool_field(schema):
    c = compile_str("active", schema)
    # active is bool field, top level is bool — OK
    assert "active" in c.fields


def test_int_float_promotion(schema):
    """age is INT64, comparing with a float literal is allowed (promote int → float)."""
    c = compile_str("age >= 18.5", schema)
    assert "age" in c.fields


def test_float_field_int_literal(schema):
    """score is FLOAT; comparing with an int literal is allowed."""
    c = compile_str("score > 0", schema)
    assert "score" in c.fields


def test_in_expression(schema):
    c = compile_str("age in [10, 20, 30]", schema)
    assert "age" in c.fields


def test_in_string(schema):
    c = compile_str("category in ['tech', 'news']", schema)


def test_in_empty_list(schema):
    """Empty list is allowed — it just never matches."""
    compile_str("age in []", schema)


def test_complex_expression(schema):
    c = compile_str(
        "age > 18 and category in ['tech', 'news'] or score >= 0.5",
        schema,
    )
    assert {"age", "category", "score"} <= set(c.fields.keys())


def test_negation(schema):
    compile_str("not (age > 18)", schema)


def test_records_pos(schema):
    """The compiled expression must record the source for error messages."""
    c = compile_str("age > 18", schema)
    assert c.source == "age > 18"


# ---------------------------------------------------------------------------
# Field errors
# ---------------------------------------------------------------------------

def test_unknown_field(schema):
    with pytest.raises(FilterFieldError) as exc:
        compile_str("ag > 18", schema)
    msg = str(exc.value)
    assert "unknown field 'ag'" in msg
    assert "did you mean 'age'" in msg
    assert "column 1" in msg


def test_unknown_field_no_close_match(schema):
    with pytest.raises(FilterFieldError) as exc:
        compile_str("xyzzy > 18", schema)
    msg = str(exc.value)
    assert "unknown field 'xyzzy'" in msg
    # No close match, so no "did you mean"
    assert "did you mean" not in msg


def test_reserved_field_seq(schema):
    with pytest.raises(FilterFieldError, match="reserved field '_seq'"):
        compile_str("_seq > 100", schema)


def test_reserved_field_partition(schema):
    with pytest.raises(FilterFieldError, match="reserved field '_partition'"):
        compile_str("_partition == 'a'", schema)


def test_vector_field_rejected(schema):
    with pytest.raises(FilterTypeError, match="vec.*float_vector"):
        compile_str("vec > 0", schema)


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------

def test_int_vs_string(schema):
    with pytest.raises(FilterTypeError) as exc:
        compile_str("age > 'eighteen'", schema)
    msg = str(exc.value)
    assert "incompatible" in msg
    assert "int" in msg
    assert "string" in msg
    assert "age" in msg  # field name in description


def test_string_vs_int(schema):
    with pytest.raises(FilterTypeError):
        compile_str("category == 18", schema)


def test_bool_vs_int(schema):
    with pytest.raises(FilterTypeError):
        compile_str("active == 1", schema)


def test_chained_comparison_type_error(schema):
    """`a == b == c`: parse-accepted, semantic-rejected because the LHS
    of the second '==' is bool (result of `a == b`)."""
    with pytest.raises(FilterTypeError):
        compile_str("age == 18 == 20", schema)


def test_in_list_wrong_type(schema):
    with pytest.raises(FilterTypeError, match="incompatible"):
        compile_str("age in ['a', 'b']", schema)


def test_heterogeneous_list(schema):
    with pytest.raises(FilterTypeError, match="compatible"):
        compile_str("age in [1, 'two', 3]", schema)


def test_top_level_must_be_bool(schema):
    """An int literal at the top level isn't a valid filter."""
    with pytest.raises(FilterTypeError, match="must evaluate to bool"):
        compile_str("age", schema)  # top is int, not bool


def test_top_level_field_int(schema):
    with pytest.raises(FilterTypeError, match="must evaluate to bool"):
        compile_str("18", schema)


def test_and_with_non_bool_operand(schema):
    with pytest.raises(FilterTypeError, match="and.*must be boolean"):
        compile_str("age and active", schema)  # age is int, not bool


def test_not_with_non_bool(schema):
    with pytest.raises(FilterTypeError, match="not.*must be boolean"):
        compile_str("not age", schema)


# ---------------------------------------------------------------------------
# Caret + did-you-mean rendering
# ---------------------------------------------------------------------------

def test_caret_pointer_rendered(schema):
    with pytest.raises(FilterFieldError) as exc:
        compile_str("ag > 18", schema)
    msg = str(exc.value)
    # The caret line should highlight 'ag'
    assert "ag > 18" in msg
    assert "^^" in msg  # 2-char span for "ag"


def test_type_error_caret(schema):
    with pytest.raises(FilterTypeError) as exc:
        compile_str("age > 'eighteen'", schema)
    msg = str(exc.value)
    assert "age > 'eighteen'" in msg
    # caret should be on '>' (column 5)
    assert "column 5" in msg


# ---------------------------------------------------------------------------
# Backend selection (Phase F1: always arrow)
# ---------------------------------------------------------------------------

def test_backend_is_arrow(schema):
    c = compile_str("age > 18", schema)
    assert c.backend == "arrow"


def test_backend_arrow_for_complex(schema):
    c = compile_str(
        "age > 18 and category in ['tech', 'news'] or score >= 0.5",
        schema,
    )
    assert c.backend == "arrow"


# ---------------------------------------------------------------------------
# Field collection
# ---------------------------------------------------------------------------

def test_fields_used_collected(schema):
    c = compile_str("age > 18 and category == 'tech'", schema)
    assert set(c.fields.keys()) == {"age", "category"}


def test_fields_used_dedup(schema):
    """A field referenced twice should appear once in the dict."""
    c = compile_str("age > 18 and age < 100", schema)
    assert list(c.fields.keys()) == ["age"]


def test_fields_used_carry_dtype(schema):
    c = compile_str("age > 18", schema)
    assert c.fields["age"].dtype == DataType.INT64
    assert c.fields["age"].sem_type == "int"


# ---------------------------------------------------------------------------
# Phase F2a — arithmetic
# ---------------------------------------------------------------------------

def test_arith_int_plus_int(schema):
    compile_str("age + 1 > 20", schema)


def test_arith_int_plus_float(schema):
    compile_str("age + 0.5 > 20", schema)


def test_arith_float_only(schema):
    compile_str("score * 2.0 > 1.5", schema)


def test_arith_with_string_rejected(schema):
    with pytest.raises(FilterTypeError, match="numeric"):
        compile_str("category + 1 > 0", schema)


def test_arith_with_bool_rejected(schema):
    with pytest.raises(FilterTypeError, match="numeric"):
        compile_str("active + 1 > 0", schema)


def test_arith_unary_minus_field(schema):
    compile_str("-age > -50", schema)


def test_arith_division_promotes_to_float(schema):
    """`/` always returns float, so `int / int >= int` is allowed
    via int→float promotion on the RHS."""
    compile_str("age / 2 >= 10", schema)


def test_arith_in_complex(schema):
    compile_str("age * 2 + 1 < 100 and score - 0.1 > 0", schema)


# ---------------------------------------------------------------------------
# Phase F2a — LIKE
# ---------------------------------------------------------------------------

def test_like_string_field(schema):
    compile_str("title like 'AI%'", schema)


def test_like_int_field_rejected(schema):
    with pytest.raises(FilterTypeError, match="string"):
        compile_str("age like 'pattern'", schema)


def test_like_bool_field_rejected(schema):
    with pytest.raises(FilterTypeError, match="string"):
        compile_str("active like 'x'", schema)


def test_like_in_compound(schema):
    compile_str("title like 'AI%' and age > 18", schema)


# ---------------------------------------------------------------------------
# Phase F2a — IS NULL
# ---------------------------------------------------------------------------

def test_is_null_nullable_field(schema):
    """title is nullable, IS NULL works."""
    compile_str("title is null", schema)


def test_is_null_non_nullable_field(schema):
    """IS NULL is also allowed on non-nullable fields (always False)."""
    compile_str("age is null", schema)


def test_is_not_null(schema):
    compile_str("title is not null", schema)


def test_is_null_in_complex(schema):
    compile_str("title is not null and age > 18 and category in ['tech']", schema)


def test_is_null_unknown_field(schema):
    with pytest.raises(FilterFieldError):
        compile_str("nonexistent is null", schema)


def test_is_null_reserved_field(schema):
    with pytest.raises(FilterFieldError, match="reserved"):
        compile_str("_seq is null", schema)


def test_is_null_vector_field(schema):
    with pytest.raises(FilterTypeError, match="float_vector"):
        compile_str("vec is null", schema)


# ---------------------------------------------------------------------------
# Phase F2b — $meta dynamic field
# ---------------------------------------------------------------------------

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


def test_meta_access_with_dynamic_field(schema_dynamic):
    c = compile_str('$meta["category"] == "tech"', schema_dynamic)
    # Backend should be hybrid because of $meta access (Phase F3+)
    assert c.backend == "hybrid"


def test_meta_without_dynamic_field_rejected(schema):
    """Schema doesn't have enable_dynamic_field=True → reject."""
    with pytest.raises(FilterFieldError, match="enable_dynamic_field"):
        compile_str('$meta["x"] == 1', schema)


def test_meta_compared_with_int(schema_dynamic):
    """$meta has dynamic type, can be compared with int (runtime decides)."""
    c = compile_str('$meta["priority"] > 5', schema_dynamic)
    assert c.backend == "hybrid"


def test_meta_compared_with_float(schema_dynamic):
    c = compile_str('$meta["score"] > 0.5', schema_dynamic)
    assert c.backend == "hybrid"


def test_meta_compared_with_string(schema_dynamic):
    c = compile_str('$meta["category"] == "tech"', schema_dynamic)
    assert c.backend == "hybrid"


def test_meta_in_arithmetic(schema_dynamic):
    """$meta in arithmetic — type is dynamic, allowed."""
    c = compile_str('$meta["score"] * 2 > 1.0', schema_dynamic)
    assert c.backend == "hybrid"


def test_meta_in_complex_expression(schema_dynamic):
    c = compile_str(
        'age > 18 and $meta["category"] == "tech"',
        schema_dynamic,
    )
    # Mixed: age is regular field, $meta is dynamic — backend goes hybrid
    assert c.backend == "hybrid"
    assert "age" in c.fields  # regular field still recorded


def test_normal_expression_still_arrow(schema_dynamic):
    """Even with dynamic-field schema, expressions without $meta stay arrow."""
    c = compile_str("age > 18", schema_dynamic)
    assert c.backend == "arrow"


def test_meta_with_like(schema_dynamic):
    c = compile_str('$meta["title"] like "AI%"', schema_dynamic)
    assert c.backend == "hybrid"
