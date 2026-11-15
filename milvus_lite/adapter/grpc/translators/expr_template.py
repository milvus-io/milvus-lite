"""Render Milvus filter expression templates.

Clients may send a filter such as::

    age > {age} AND city IN {city}

together with a map of placeholder names to typed values.  pymilvus exposes
the map as ``filter_params`` and serializes it into the request's
``expr_template_values`` field.  The engine parser never sees the braces:
this module substitutes expression literals before the filter is handed to
the engine.

Only values are templated.  Field names, operators, and function names stay
part of the expression.  Strings are quoted and escaped here, so callers can
pass untrusted text without building expressions by concatenation.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

from milvus_lite.exceptions import SchemaValidationError

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Must match the escape map in search/filter/tokens.py.  All other
# characters are emitted verbatim; the tokenizer rejects raw newlines.
_STRING_ESCAPES = {
    "\\": "\\\\",
    "'": "\\'",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def render_expr_template(
    expr: Optional[str],
    values: Any,
) -> Optional[str]:
    """Replace ``{name}`` placeholders in *expr* with typed *values*.

    Args:
        expr: Filter expression, or ``None``/empty for no filter.
        values: Protobuf map of placeholder name -> ``TemplateValue``.
            ``None`` and an empty map are both treated as "no values".

    Returns:
        The rendered expression, or *expr* unchanged when it is empty.

    Raises:
        SchemaValidationError: A placeholder has no value, a value has no
            type set, or the value type cannot be represented as a filter
            literal (binary / JSON arrays).
    """
    if not expr:
        return expr

    values = values or {}
    missing = sorted(
        {m.group(1) for m in _PLACEHOLDER_RE.finditer(expr)} - set(values)
    )
    if missing:
        names = ", ".join(f"{{{name}}}" for name in missing)
        raise SchemaValidationError(
            f"missing filter template value(s) for {names}"
        )

    return _PLACEHOLDER_RE.sub(
        lambda m: _render_value(values[m.group(1)], m.group(1)),
        expr,
    )


def _render_value(value: Any, name: str) -> str:
    """Render one ``schema.TemplateValue`` as a filter literal."""
    kind = value.WhichOneof("val")
    if kind == "bool_val":
        return "true" if value.bool_val else "false"
    if kind == "int64_val":
        return str(value.int64_val)
    if kind == "float_val":
        return _render_float32(value.float_val, name)
    if kind == "string_val":
        return _quote_string(value.string_val)
    if kind == "array_val":
        return _render_array(value.array_val, name)
    if kind == "bytes_val":
        raise SchemaValidationError(
            f"filter template value {name!r}: binary values are not "
            f"supported in expressions"
        )
    raise SchemaValidationError(
        f"filter template value {name!r} has no value set"
    )


def _render_array(array: Any, name: str) -> str:
    """Render a ``schema.TemplateArrayValue`` as ``[v1, v2, ...]``."""
    kind = array.WhichOneof("data")
    if kind is None:
        # pymilvus emits an empty TemplateArrayValue for an empty list.
        return "[]"
    if kind == "bool_data":
        items = ["true" if v else "false" for v in array.bool_data.data]
    elif kind == "long_data":
        items = [str(v) for v in array.long_data.data]
    elif kind == "double_data":
        items = [_render_double(v, name) for v in array.double_data.data]
    elif kind == "string_data":
        items = [_quote_string(v) for v in array.string_data.data]
    elif kind == "array_data":
        items = [_render_array(v, name) for v in array.array_data.data]
    elif kind == "json_data":
        raise SchemaValidationError(
            f"filter template value {name!r}: JSON arrays are not "
            f"supported in expressions"
        )
    else:  # pragma: no cover - protobuf oneof is closed
        raise SchemaValidationError(
            f"filter template value {name!r}: unsupported array type {kind!r}"
        )
    return "[" + ", ".join(items) + "]"


def _render_float32(value: float, name: str) -> str:
    _reject_non_finite(value, name)
    # TemplateValue.float_val is a 32-bit float.  Nine significant digits
    # round-trip float32 and avoid emitting 0.10000000149011612 for 0.1.
    return format(value, ".9g")


def _render_double(value: float, name: str) -> str:
    _reject_non_finite(value, name)
    # TemplateArrayValue.double_data is 64-bit; repr gives the shortest
    # round-tripping representation.
    return repr(value)


def _reject_non_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise SchemaValidationError(
            f"filter template value {name!r}: non-finite floats are not "
            f"supported"
        )


def _quote_string(value: str) -> str:
    escaped = "".join(_STRING_ESCAPES.get(ch, ch) for ch in value)
    return f"'{escaped}'"
