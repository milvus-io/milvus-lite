"""Tests for search/filter/parser.py — Pratt parser correctness."""

import pytest

from milvus_lite.search.filter.ast import (
    And,
    ArithOp,
    BoolLit,
    CmpOp,
    FieldRef,
    FloatLit,
    InOp,
    IntLit,
    IsNullOp,
    LikeOp,
    ListLit,
    MetaAccess,
    Not,
    Or,
    StringLit,
)
from milvus_lite.search.filter.exceptions import FilterParseError
from milvus_lite.search.filter.parser import parse_expr


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

def test_int_literal():
    e = parse_expr("42")
    assert isinstance(e, IntLit)
    assert e.value == 42


def test_negative_int_literal_folded():
    """Unary minus on int literal is constant-folded into a single IntLit."""
    e = parse_expr("-7")
    assert isinstance(e, IntLit)
    assert e.value == -7


def test_float_literal():
    e = parse_expr("3.14")
    assert isinstance(e, FloatLit)
    assert e.value == pytest.approx(3.14)


def test_negative_float_folded():
    e = parse_expr("-1.5")
    assert isinstance(e, FloatLit)
    assert e.value == pytest.approx(-1.5)


def test_string_literal():
    e = parse_expr("'hello'")
    assert isinstance(e, StringLit)
    assert e.value == "hello"


def test_bool_literal_true():
    e = parse_expr("true")
    assert isinstance(e, BoolLit)
    assert e.value is True


# ---------------------------------------------------------------------------
# Field references
# ---------------------------------------------------------------------------

def test_field_ref():
    e = parse_expr("age")
    assert isinstance(e, FieldRef)
    assert e.name == "age"


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def test_cmp_simple():
    e = parse_expr("age > 18")
    assert isinstance(e, CmpOp)
    assert e.op == ">"
    assert isinstance(e.left, FieldRef)
    assert e.left.name == "age"
    assert isinstance(e.right, IntLit)
    assert e.right.value == 18


@pytest.mark.parametrize("op_text,op_value", [
    ("==", "=="), ("!=", "!="),
    ("<", "<"), ("<=", "<="),
    (">", ">"), (">=", ">="),
])
def test_all_cmp_operators(op_text, op_value):
    e = parse_expr(f"a {op_text} 1")
    assert isinstance(e, CmpOp)
    assert e.op == op_value


def test_cmp_reversed_lhs_literal():
    """Milvus accepts `18 < age` (literal on the left)."""
    e = parse_expr("18 < age")
    assert isinstance(e, CmpOp)
    assert e.op == "<"
    assert isinstance(e.left, IntLit)
    assert isinstance(e.right, FieldRef)


def test_chained_comparison_parses():
    """`a == b == c` is accepted at parse time. semantic.py rejects it
    later (the LHS of the second == becomes bool, which can't compare
    to int)."""
    e = parse_expr("a == b == 1")
    # Outer node should be the second ==
    assert isinstance(e, CmpOp)
    assert e.op == "=="
    assert isinstance(e.left, CmpOp)
    assert e.left.op == "=="


# ---------------------------------------------------------------------------
# Logical AND / OR / NOT
# ---------------------------------------------------------------------------

def test_and_two_operands():
    e = parse_expr("a > 1 and b < 2")
    assert isinstance(e, And)
    assert len(e.operands) == 2


def test_and_chain_flattened():
    """`a and b and c and d` flattens into a single And with 4 operands."""
    e = parse_expr("a and b and c and d")
    assert isinstance(e, And)
    assert len(e.operands) == 4


def test_or_chain_flattened():
    e = parse_expr("a or b or c")
    assert isinstance(e, Or)
    assert len(e.operands) == 3


def test_not_prefix():
    e = parse_expr("not (a > 1)")
    assert isinstance(e, Not)
    assert isinstance(e.operand, CmpOp)


def test_not_double():
    e = parse_expr("not not a")
    assert isinstance(e, Not)
    assert isinstance(e.operand, Not)


def test_bang_alias():
    e = parse_expr("!(a > 1)")
    assert isinstance(e, Not)


def test_logical_symbol_aliases():
    e1 = parse_expr("a > 1 && b < 2")
    e2 = parse_expr("a > 1 and b < 2")
    assert isinstance(e1, And)
    assert isinstance(e2, And)


# ---------------------------------------------------------------------------
# Operator precedence
# ---------------------------------------------------------------------------

def test_and_binds_tighter_than_or():
    """`a or b and c` parses as `a or (b and c)`."""
    e = parse_expr("a or b and c")
    assert isinstance(e, Or)
    assert len(e.operands) == 2
    # Second operand is the And
    assert isinstance(e.operands[1], And)


def test_not_binds_tighter_than_and():
    """`not a and b` parses as `(not a) and b`."""
    e = parse_expr("not a and b")
    assert isinstance(e, And)
    assert isinstance(e.operands[0], Not)


def test_cmp_binds_tighter_than_and():
    """`a > 1 and b < 2` — comparisons evaluated first."""
    e = parse_expr("a > 1 and b < 2")
    assert isinstance(e, And)
    assert all(isinstance(op, CmpOp) for op in e.operands)


def test_parens_override_precedence():
    """`(a or b) and c` — parens force or to be deeper than and."""
    e = parse_expr("(a or b) and c")
    assert isinstance(e, And)
    assert isinstance(e.operands[0], Or)


# ---------------------------------------------------------------------------
# IN / NOT IN
# ---------------------------------------------------------------------------

def test_in_simple():
    e = parse_expr("age in [10, 20, 30]")
    assert isinstance(e, InOp)
    assert e.field.name == "age"
    assert e.negate is False
    assert len(e.values.elements) == 3
    assert all(isinstance(el, IntLit) for el in e.values.elements)


def test_not_in():
    e = parse_expr("category not in ['a', 'b']")
    assert isinstance(e, InOp)
    assert e.negate is True
    assert e.field.name == "category"


def test_in_string_list():
    e = parse_expr("category in ['tech', 'news']")
    assert isinstance(e, InOp)
    assert [el.value for el in e.values.elements] == ["tech", "news"]


def test_in_empty_list():
    e = parse_expr("age in []")
    assert isinstance(e, InOp)
    assert e.values.elements == ()


def test_in_trailing_comma():
    e = parse_expr("age in [1, 2, 3,]")
    assert isinstance(e, InOp)
    assert len(e.values.elements) == 3


def test_in_negative_literal():
    e = parse_expr("temp in [-5, 0, 10]")
    assert [el.value for el in e.values.elements] == [-5, 0, 10]


def test_in_lhs_must_be_field():
    """`'a' in [...]` is rejected — Milvus alignment."""
    with pytest.raises(FilterParseError, match="must be a field"):
        parse_expr("'a' in [1, 2]")


def test_in_rhs_must_be_list():
    with pytest.raises(FilterParseError, match="expected '\\['"):
        parse_expr("age in 5")


# ---------------------------------------------------------------------------
# Realistic combined expressions
# ---------------------------------------------------------------------------

def test_complex_expression():
    e = parse_expr("age > 18 and category in ['tech', 'news'] or score >= 0.5")
    # Should parse as: (age > 18 and category in [...]) or score >= 0.5
    assert isinstance(e, Or)
    assert len(e.operands) == 2
    assert isinstance(e.operands[0], And)
    assert isinstance(e.operands[1], CmpOp)


def test_negated_subexpression():
    e = parse_expr("not (age > 18) and category == 'tech'")
    assert isinstance(e, And)
    assert isinstance(e.operands[0], Not)
    assert isinstance(e.operands[1], CmpOp)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_empty_input():
    with pytest.raises(FilterParseError, match="end of expression"):
        parse_expr("")


def test_unclosed_paren():
    with pytest.raises(FilterParseError, match="expected '\\)'"):
        parse_expr("(a > 1")


def test_dangling_operator():
    with pytest.raises(FilterParseError):
        parse_expr("a > ")


def test_unknown_function_call_rejected():
    with pytest.raises(FilterParseError, match="unknown function"):
        parse_expr("unknown_func(meta, 'x')")


def test_unary_minus_on_field_now_works():
    """Phase F2a: unary minus on field becomes ArithOp(-, 0, field)."""
    e = parse_expr("-age > 0")
    assert isinstance(e, CmpOp)
    assert isinstance(e.left, ArithOp)
    assert e.left.op == "-"
    # left side is the synthetic 0
    assert isinstance(e.left.left, IntLit) and e.left.left.value == 0
    assert isinstance(e.left.right, FieldRef) and e.left.right.name == "age"


def test_extra_token_after_expression():
    with pytest.raises(FilterParseError, match="expected end"):
        parse_expr("a > 1 b")


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

def test_field_ref_pos():
    e = parse_expr("    age > 1")
    assert isinstance(e, CmpOp)
    assert e.left.pos == 4


def test_error_pos_in_message():
    with pytest.raises(FilterParseError) as exc:
        parse_expr("age > > 1")
    # Caret should land on the second '>'
    msg = str(exc.value)
    assert "column 7" in msg


# ---------------------------------------------------------------------------
# Phase F2a — arithmetic
# ---------------------------------------------------------------------------

def test_arith_simple_add():
    e = parse_expr("age + 1 > 20")
    assert isinstance(e, CmpOp)
    assert isinstance(e.left, ArithOp)
    assert e.left.op == "+"


def test_arith_left_assoc():
    """`a + b + c` parses as `(a + b) + c`."""
    e = parse_expr("a + b + c > 0")
    assert isinstance(e, CmpOp)
    add = e.left
    assert isinstance(add, ArithOp) and add.op == "+"
    assert isinstance(add.left, ArithOp) and add.left.op == "+"


def test_mul_binds_tighter_than_add():
    """`a + b * c` parses as `a + (b * c)`."""
    e = parse_expr("a + b * c > 0")
    add = e.left
    assert isinstance(add, ArithOp) and add.op == "+"
    assert isinstance(add.right, ArithOp) and add.right.op == "*"


def test_arith_parens_override():
    """`(a + b) * c` — parens force add to be deeper."""
    e = parse_expr("(a + b) * c > 0")
    mul = e.left
    assert isinstance(mul, ArithOp) and mul.op == "*"
    assert isinstance(mul.left, ArithOp) and mul.left.op == "+"


def test_arith_in_both_sides():
    e = parse_expr("a + 1 > b * 2")
    assert isinstance(e, CmpOp)
    assert isinstance(e.left, ArithOp)
    assert isinstance(e.right, ArithOp)


def test_arith_unary_minus_field():
    """`-age` is now ArithOp(-, 0, age) instead of an error."""
    e = parse_expr("-age + 5 > 0")
    add = e.left
    assert isinstance(add, ArithOp) and add.op == "+"
    assert isinstance(add.left, ArithOp) and add.left.op == "-"


def test_arith_div():
    e = parse_expr("score / 2 > 0.5")
    arith = e.left
    assert isinstance(arith, ArithOp) and arith.op == "/"


# ---------------------------------------------------------------------------
# Phase F2a — LIKE
# ---------------------------------------------------------------------------

def test_like_simple():
    e = parse_expr("title like 'AI%'")
    assert isinstance(e, LikeOp)
    assert isinstance(e.value, FieldRef)
    assert e.value.name == "title"
    assert e.pattern.value == "AI%"


def test_like_uppercase():
    e = parse_expr("title LIKE 'AI%'")
    assert isinstance(e, LikeOp)


def test_like_in_and():
    e = parse_expr("title like 'AI%' and age > 18")
    assert isinstance(e, And)
    assert isinstance(e.operands[0], LikeOp)
    assert isinstance(e.operands[1], CmpOp)


def test_like_pattern_must_be_string():
    with pytest.raises(FilterParseError, match="string literal"):
        parse_expr("title like 42")


def test_like_pattern_must_be_string_not_field():
    with pytest.raises(FilterParseError, match="string literal"):
        parse_expr("title like other_field")


# ---------------------------------------------------------------------------
# Phase F2a — IS NULL / IS NOT NULL
# ---------------------------------------------------------------------------

def test_is_null():
    e = parse_expr("title is null")
    assert isinstance(e, IsNullOp)
    assert e.field.name == "title"
    assert e.negate is False


def test_is_not_null():
    e = parse_expr("title is not null")
    assert isinstance(e, IsNullOp)
    assert e.negate is True


def test_is_null_uppercase():
    e = parse_expr("title IS NULL")
    assert isinstance(e, IsNullOp)
    assert e.negate is False


def test_is_not_null_uppercase():
    e = parse_expr("title IS NOT NULL")
    assert isinstance(e, IsNullOp)
    assert e.negate is True


def test_is_null_in_and():
    e = parse_expr("age > 18 and title is not null")
    assert isinstance(e, And)
    assert isinstance(e.operands[1], IsNullOp)
    assert e.operands[1].negate is True


def test_is_null_lhs_must_be_field():
    with pytest.raises(FilterParseError, match="field reference"):
        parse_expr("'literal' is null")


def test_is_missing_null_keyword():
    with pytest.raises(FilterParseError, match="expected 'null'"):
        parse_expr("title is something")


def test_is_not_missing_null():
    with pytest.raises(FilterParseError, match="expected 'not null'"):
        parse_expr("title is not 5")


# ---------------------------------------------------------------------------
# Phase F2b — $meta access
# ---------------------------------------------------------------------------

def test_meta_access_simple():
    e = parse_expr('$meta["category"] == "tech"')
    assert isinstance(e, CmpOp)
    assert isinstance(e.left, MetaAccess)
    assert e.left.key == "category"


def test_meta_access_int_key_rejected():
    """Phase F2b only supports string keys."""
    with pytest.raises(FilterParseError, match="string literal"):
        parse_expr('$meta[0]')


def test_meta_access_missing_close_bracket():
    with pytest.raises(FilterParseError, match="expected '\\]'"):
        parse_expr('$meta["key"')


def test_meta_access_missing_open_bracket():
    with pytest.raises(FilterParseError, match="expected '\\['"):
        parse_expr('$meta')


def test_meta_in_complex_expr():
    e = parse_expr('$meta["priority"] > 5 and $meta["category"] == "tech"')
    assert isinstance(e, And)
    assert isinstance(e.operands[0], CmpOp)
    assert isinstance(e.operands[0].left, MetaAccess)
    assert isinstance(e.operands[1], CmpOp)
    assert isinstance(e.operands[1].left, MetaAccess)


def test_meta_with_arithmetic():
    e = parse_expr('$meta["score"] * 2 > 1.0')
    assert isinstance(e, CmpOp)
    assert isinstance(e.left, ArithOp)
    assert isinstance(e.left.left, MetaAccess)


def test_meta_in_in_expression():
    """`$meta["category"] in [...]` requires `in`'s LHS to be a field
    ref, but $meta is not a FieldRef. Currently rejected by parser."""
    with pytest.raises(FilterParseError, match="field reference"):
        parse_expr('$meta["category"] in ["tech", "news"]')
