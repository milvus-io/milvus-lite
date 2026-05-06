"""Filter expression error types.

All exceptions inherit from MilvusLiteError so callers can catch the
public base class. Each carries the source string and the column
position so __str__ can render a caret-style pointer.
"""

from __future__ import annotations

import difflib
from typing import List, Optional

from milvus_lite.exceptions import MilvusLiteError


class FilterError(MilvusLiteError):
    """Base class for all filter expression errors."""


def _render_caret(source: str, pos: int, span: int = 1) -> str:
    """Render the source line with a caret pointer.

    The expression is assumed to be a single line (no newlines in
    source). For multi-line input the caret falls on the relevant line.
    Returns a 2-line string ready to embed in __str__.
    """
    if not source:
        return ""
    if pos < 0:
        pos = 0
    if pos > len(source):
        pos = len(source)
    span = max(1, span)
    line = source.rstrip("\n")
    indicator = " " * pos + "^" * min(span, max(1, len(line) - pos + 1))
    return f"  {line}\n  {indicator}"


class FilterParseError(FilterError):
    """Lexing or parsing failed.

    Raised by tokens.tokenize() and parser.parse_expr().

    Args:
        message: short human-readable message
        source: the original expression string (for caret rendering)
        pos: column in source where the error was detected (0-indexed)
        span: number of source columns the error covers (default 1)
        hint: optional one-line hint appended after the caret
    """

    def __init__(
        self,
        message: str,
        source: str,
        pos: int,
        span: int = 1,
        hint: Optional[str] = None,
    ) -> None:
        self.message = message
        self.source = source
        self.pos = pos
        self.span = span
        self.hint = hint
        super().__init__(self._format())

    def _format(self) -> str:
        out = [f"{self.message} at column {self.pos + 1}"]
        rendered = _render_caret(self.source, self.pos, self.span)
        if rendered:
            out.append(rendered)
        if self.hint:
            out.append(self.hint)
        return "\n".join(out)


class FilterFieldError(FilterError):
    """Reference to a field that does not exist in the schema.

    Includes did-you-mean suggestion via difflib.
    """

    def __init__(
        self,
        message: str,
        source: str,
        pos: int,
        field_name: str,
        available_fields: Optional[List[str]] = None,
    ) -> None:
        self.message = message
        self.source = source
        self.pos = pos
        self.field_name = field_name
        self.available_fields = available_fields or []
        self.suggestion = self._guess()
        super().__init__(self._format())

    def _guess(self) -> Optional[str]:
        if not self.available_fields:
            return None
        matches = difflib.get_close_matches(
            self.field_name, self.available_fields, n=1, cutoff=0.6,
        )
        return matches[0] if matches else None

    def _format(self) -> str:
        out = [f"{self.message} at column {self.pos + 1}"]
        rendered = _render_caret(self.source, self.pos, span=len(self.field_name))
        if rendered:
            out.append(rendered)
        if self.available_fields:
            out.append(f"available fields: {sorted(self.available_fields)}")
        if self.suggestion:
            out.append(f"did you mean {self.suggestion!r}?")
        return "\n".join(out)


class FilterTypeError(FilterError):
    """Type mismatch in expression operands.

    Used by semantic.compile_expr when an operator is applied to
    incompatible operand types.
    """

    def __init__(
        self,
        message: str,
        source: str,
        pos: int,
        span: int = 1,
        left_desc: Optional[str] = None,
        right_desc: Optional[str] = None,
    ) -> None:
        self.message = message
        self.source = source
        self.pos = pos
        self.span = span
        self.left_desc = left_desc
        self.right_desc = right_desc
        super().__init__(self._format())

    def _format(self) -> str:
        out = [f"{self.message} at column {self.pos + 1}"]
        rendered = _render_caret(self.source, self.pos, self.span)
        if rendered:
            out.append(rendered)
        if self.left_desc and self.right_desc:
            out.append(f"left side is {self.left_desc}, right side is {self.right_desc}")
        elif self.left_desc:
            out.append(f"operand is {self.left_desc}")
        return "\n".join(out)
