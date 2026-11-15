"""Unit tests for Milvus filter expression template rendering.

Covers the placeholder substitution rules used by pymilvus's
``filter_params`` (``expr_template_values`` on the wire).  End-to-end
coverage through the gRPC adapter lives in ``test_grpc_expr_template.py``.
"""

import pytest
from pymilvus.grpc_gen import schema_pb2

from milvus_lite.adapter.grpc.translators.expr_template import (
    render_expr_template,
)
from milvus_lite.exceptions import SchemaValidationError


def _value(**kwargs):
    return schema_pb2.TemplateValue(**kwargs)


def _array(field, values):
    value = schema_pb2.TemplateValue()
    getattr(value.array_val, field).data.extend(values)
    return value


def test_no_expression_is_unchanged():
    assert render_expr_template(None, {"age": _value(int64_val=1)}) is None
    assert render_expr_template("", {"age": _value(int64_val=1)}) == ""


def test_no_placeholders_is_unchanged():
    expr = "age > 18"
    assert render_expr_template(expr, {"age": _value(int64_val=25)}) == expr


def test_scalar_placeholders():
    values = {
        "age": _value(int64_val=25),
        "score": _value(float_val=1.5),
        "active": _value(bool_val=False),
    }
    assert (
        render_expr_template(
            "age > {age} AND score >= {score} AND active == {active}",
            values,
        )
        == "age > 25 AND score >= 1.5 AND active == false"
    )


def test_float_uses_float32_precision():
    # 0.1 is not exactly representable as float32; do not leak the long
    # binary expansion into the expression.
    rendered = render_expr_template(
        "score < {score}", {"score": _value(float_val=0.1)}
    )
    assert rendered == "score < 0.1"


def test_string_is_quoted_and_escaped():
    value = "O'Reilly \\ \"x\"\n"
    assert (
        render_expr_template("city == {city}", {"city": _value(string_val=value)})
        == "city == 'O\\'Reilly \\\\ \"x\"\\n'"
    )


def test_trailing_backslash_cannot_escape_closing_quote():
    rendered = render_expr_template(
        "city == {city}", {"city": _value(string_val="foo\\")}
    )
    assert rendered == "city == 'foo\\\\'"


def test_regex_value_is_not_rescanned():
    rendered = render_expr_template(
        "message =~ {pattern}",
        {"pattern": _value(string_val="E[0-9]{4}")},
    )
    assert rendered == "message =~ 'E[0-9]{4}'"


def test_string_array():
    rendered = render_expr_template(
        "city IN {city}",
        {"city": _array("string_data", ["北京", "上海"])},
    )
    assert rendered == "city IN ['北京', '上海']"


def test_numeric_and_bool_arrays():
    values = {
        "ids": _array("long_data", [-1, 2, 3]),
        "scores": _array("double_data", [0.5, 1.5]),
        "flags": _array("bool_data", [True, False]),
    }
    assert (
        render_expr_template(
            "id IN {ids} AND score IN {scores} AND active IN {flags}",
            values,
        )
        == "id IN [-1, 2, 3] AND score IN [0.5, 1.5] AND active IN [true, false]"
    )


def test_double_array_preserves_float64_precision():
    value = 1.2345678901234567
    rendered = render_expr_template(
        "score IN {scores}", {"scores": _array("double_data", [value])}
    )
    assert rendered == f"score IN [{value!r}]"


def test_empty_array():
    value = schema_pb2.TemplateValue(
        array_val=schema_pb2.TemplateArrayValue()
    )
    assert render_expr_template("id IN {ids}", {"ids": value}) == "id IN []"


def test_nested_array():
    value = schema_pb2.TemplateValue()
    inner = value.array_val.array_data.data.add()
    inner.long_data.data.extend([1, 2])
    assert render_expr_template("tags IN {tags}", {"tags": value}) == (
        "tags IN [[1, 2]]"
    )


def test_repeated_placeholder():
    assert (
        render_expr_template(
            "age > {age} OR age == {age}", {"age": _value(int64_val=7)}
        )
        == "age > 7 OR age == 7"
    )


def test_all_missing_placeholders_are_reported_together():
    with pytest.raises(SchemaValidationError, match=r"\{age\}, \{city\}"):
        render_expr_template("age > {age} AND city IN {city}", {})


def test_missing_value_with_non_empty_map():
    with pytest.raises(SchemaValidationError, match=r"\{city\}"):
        render_expr_template(
            "age > {age} AND city IN {city}",
            {"age": _value(int64_val=1)},
        )


def test_binary_value_is_rejected():
    with pytest.raises(SchemaValidationError, match="binary"):
        render_expr_template(
            "id == {id}", {"id": _value(bytes_val=b"\x00\x01")}
        )


def test_json_array_is_rejected():
    value = schema_pb2.TemplateValue()
    value.array_val.json_data.data.append(b'{"a": 1}')
    with pytest.raises(SchemaValidationError, match="JSON"):
        render_expr_template("info IN {info}", {"info": value})


def test_unset_value_is_rejected():
    with pytest.raises(SchemaValidationError, match="no value set"):
        render_expr_template("age > {age}", {"age": schema_pb2.TemplateValue()})
