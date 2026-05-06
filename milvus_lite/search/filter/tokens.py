"""Tokenizer for the filter expression DSL.

Single-pass character scanner. Handles:
    - Whitespace skipping (no comments)
    - Numeric literals (decimal int + float + scientific)
    - String literals (double or single quoted, C-style escapes)
    - Boolean literals (six forms: true/True/TRUE/false/False/FALSE)
    - Identifiers (case-sensitive)
    - Keywords (case-insensitive — and/AND, or/OR, not/NOT, in/IN)
    - Symbol operators (== != < <= > >= && || ! ( ) [ ] , -)

All errors raise FilterParseError with source + pos.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from milvus_lite.search.filter.exceptions import FilterParseError


class TokenKind(Enum):
    # Literals
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOL = "BOOL"

    # Identifier
    IDENT = "IDENT"

    # Punctuation
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    COMMA = ","

    # Comparison
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    # Boolean / unary
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # Membership
    IN = "IN"

    # Arithmetic (Phase F2a)
    ADD = "+"
    SUB = "-"          # also unary minus — parser disambiguates
    MUL = "*"
    DIV = "/"

    # String pattern (Phase F2a)
    LIKE = "LIKE"

    # NULL tests (Phase F2a). Tokenized as separate IS / NULL keywords;
    # the parser combines `IS NULL` and `IS NOT NULL` into IsNullOp.
    IS = "IS"
    NULL = "NULL"

    # Dynamic-field marker (Phase F2b). The lexer emits META on `$meta`,
    # the parser then expects '[' STRING ']' to form a MetaAccess node.
    META = "META"

    # Sentinel
    EOF = "EOF"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    pos: int
    value: Any = None  # parsed literal value (None for non-literals)


# Keywords are case-insensitive (per Milvus Plan.g4 alignment).
# Booleans are special: only the 6 listed forms are accepted.
_KEYWORD_MAP = {
    "and": TokenKind.AND, "AND": TokenKind.AND,
    "or": TokenKind.OR, "OR": TokenKind.OR,
    "not": TokenKind.NOT, "NOT": TokenKind.NOT,
    "in": TokenKind.IN, "IN": TokenKind.IN,
    # Phase F2a additions
    "like": TokenKind.LIKE, "LIKE": TokenKind.LIKE,
    "is": TokenKind.IS, "IS": TokenKind.IS,
    "null": TokenKind.NULL, "NULL": TokenKind.NULL,
}

# Boolean literals — only these 6 forms are accepted as bool.
# Anything else (e.g. "tRuE") is rejected with did-you-mean.
_BOOL_TRUE = {"true", "True", "TRUE"}
_BOOL_FALSE = {"false", "False", "FALSE"}


def tokenize(source: str) -> List[Token]:
    """Single-pass lexer. Returns a list of Tokens ending with EOF."""
    tokens: List[Token] = []
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        # ── Whitespace ──────────────────────────────────────────
        if ch in " \t\r\n":
            i += 1
            continue

        # ── Single-char punctuation ─────────────────────────────
        if ch == "(":
            tokens.append(Token(TokenKind.LPAREN, "(", i))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(TokenKind.RPAREN, ")", i))
            i += 1
            continue
        if ch == "[":
            tokens.append(Token(TokenKind.LBRACKET, "[", i))
            i += 1
            continue
        if ch == "]":
            tokens.append(Token(TokenKind.RBRACKET, "]", i))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token(TokenKind.COMMA, ",", i))
            i += 1
            continue
        if ch == "-":
            tokens.append(Token(TokenKind.SUB, "-", i))
            i += 1
            continue
        if ch == "+":
            tokens.append(Token(TokenKind.ADD, "+", i))
            i += 1
            continue
        if ch == "*":
            tokens.append(Token(TokenKind.MUL, "*", i))
            i += 1
            continue
        if ch == "/":
            tokens.append(Token(TokenKind.DIV, "/", i))
            i += 1
            continue

        # ── Two-char operators ──────────────────────────────────
        if ch == "=":
            if i + 1 < n and source[i + 1] == "=":
                tokens.append(Token(TokenKind.EQ, "==", i))
                i += 2
                continue
            raise FilterParseError(
                "unexpected character '='", source, i,
                hint="did you mean '=='?",
            )
        if ch == "!":
            if i + 1 < n and source[i + 1] == "=":
                tokens.append(Token(TokenKind.NE, "!=", i))
                i += 2
                continue
            # Bare '!' is a NOT alias.
            tokens.append(Token(TokenKind.NOT, "!", i))
            i += 1
            continue
        if ch == "<":
            if i + 1 < n and source[i + 1] == "=":
                tokens.append(Token(TokenKind.LE, "<=", i))
                i += 2
                continue
            if i + 1 < n and source[i + 1] == ">":
                raise FilterParseError(
                    "unexpected token '<>'", source, i, span=2,
                    hint="did you mean '!='?",
                )
            tokens.append(Token(TokenKind.LT, "<", i))
            i += 1
            continue
        if ch == ">":
            if i + 1 < n and source[i + 1] == "=":
                tokens.append(Token(TokenKind.GE, ">=", i))
                i += 2
                continue
            tokens.append(Token(TokenKind.GT, ">", i))
            i += 1
            continue
        if ch == "&":
            if i + 1 < n and source[i + 1] == "&":
                tokens.append(Token(TokenKind.AND, "&&", i))
                i += 2
                continue
            raise FilterParseError(
                "unexpected character '&'", source, i,
                hint="did you mean '&&'?",
            )
        if ch == "|":
            if i + 1 < n and source[i + 1] == "|":
                tokens.append(Token(TokenKind.OR, "||", i))
                i += 2
                continue
            raise FilterParseError(
                "unexpected character '|'", source, i,
                hint="did you mean '||'?",
            )

        # ── String literals ─────────────────────────────────────
        if ch == '"' or ch == "'":
            tok, advance = _read_string(source, i)
            tokens.append(tok)
            i += advance
            continue

        # ── $meta dynamic-field marker (Phase F2b) ──────────────
        if ch == "$":
            if source[i:i + 5] == "$meta":
                tokens.append(Token(TokenKind.META, "$meta", i))
                i += 5
                continue
            raise FilterParseError(
                f"unexpected character '$'", source, i,
                hint="only '$meta' is supported as a dynamic-field marker",
            )

        # ── Numeric literals ────────────────────────────────────
        if ch.isdigit():
            tok, advance = _read_number(source, i)
            tokens.append(tok)
            i += advance
            continue

        # ── Identifiers / keywords / bool ───────────────────────
        if ch.isalpha() or ch == "_":
            tok, advance = _read_ident(source, i)
            tokens.append(tok)
            i += advance
            continue

        # ── Anything else: unknown character ────────────────────
        raise FilterParseError(
            f"unexpected character {ch!r}", source, i,
        )

    tokens.append(Token(TokenKind.EOF, "", n))
    return tokens


# ---------------------------------------------------------------------------
# Per-token sub-readers
# ---------------------------------------------------------------------------

def _read_string(source: str, start: int) -> tuple[Token, int]:
    """Read a string literal starting at *start*. Returns (Token, length)."""
    quote = source[start]
    n = len(source)
    i = start + 1
    out_chars: List[str] = []

    while i < n:
        ch = source[i]
        if ch == quote:
            return (
                Token(TokenKind.STRING, source[start:i + 1], start, value="".join(out_chars)),
                i - start + 1,
            )
        if ch == "\\":
            if i + 1 >= n:
                raise FilterParseError(
                    "unterminated string literal", source, start,
                )
            esc = source[i + 1]
            mapped = _ESCAPE_MAP.get(esc)
            if mapped is None:
                raise FilterParseError(
                    f"unknown escape sequence \\{esc}", source, i, span=2,
                )
            out_chars.append(mapped)
            i += 2
            continue
        if ch in "\r\n":
            raise FilterParseError(
                "unterminated string literal (newline)", source, start,
            )
        out_chars.append(ch)
        i += 1

    raise FilterParseError(
        "unterminated string literal", source, start,
    )


_ESCAPE_MAP = {
    '"': '"', "'": "'", "\\": "\\",
    "n": "\n", "r": "\r", "t": "\t",
}


def _read_number(source: str, start: int) -> tuple[Token, int]:
    """Read an int or float literal starting at *start*. Negative numbers
    are NOT handled here — the parser turns them into Unary(SUB, ...)."""
    n = len(source)
    i = start

    # Integer part
    while i < n and source[i].isdigit():
        i += 1

    is_float = False

    # Fractional part
    if i < n and source[i] == ".":
        # Look ahead — must have at least one digit after the dot for a
        # well-formed float (so we don't tokenize "5.foo" as float).
        if i + 1 < n and source[i + 1].isdigit():
            is_float = True
            i += 1
            while i < n and source[i].isdigit():
                i += 1
        else:
            raise FilterParseError(
                "malformed float literal (digit expected after '.')",
                source, i,
            )

    # Exponent part
    if i < n and source[i] in "eE":
        is_float = True
        i += 1
        if i < n and source[i] in "+-":
            i += 1
        if i >= n or not source[i].isdigit():
            raise FilterParseError(
                "malformed float literal (digit expected in exponent)",
                source, i,
            )
        while i < n and source[i].isdigit():
            i += 1

    text = source[start:i]
    if is_float:
        try:
            value = float(text)
        except ValueError:
            raise FilterParseError(
                f"invalid float literal {text!r}", source, start, span=len(text),
            )
        return (Token(TokenKind.FLOAT, text, start, value=value), i - start)
    else:
        try:
            value = int(text)
        except ValueError:
            raise FilterParseError(
                f"invalid integer literal {text!r}", source, start, span=len(text),
            )
        return (Token(TokenKind.INT, text, start, value=value), i - start)


def _read_ident(source: str, start: int) -> tuple[Token, int]:
    """Read an identifier or keyword. Returns (Token, length)."""
    n = len(source)
    i = start
    while i < n and (source[i].isalnum() or source[i] == "_"):
        i += 1
    text = source[start:i]
    length = i - start

    # Boolean literals (only the 6 accepted forms).
    if text in _BOOL_TRUE:
        return (Token(TokenKind.BOOL, text, start, value=True), length)
    if text in _BOOL_FALSE:
        return (Token(TokenKind.BOOL, text, start, value=False), length)

    # Mixed-case bool? Reject with hint.
    lower = text.lower()
    if lower in ("true", "false"):
        raise FilterParseError(
            f"unexpected identifier {text!r}",
            source, start, span=length,
            hint=f"did you mean {lower!r}, {lower.capitalize()!r}, or {lower.upper()!r}?",
        )

    # Keywords (case-insensitive).
    kw = _KEYWORD_MAP.get(text) or _KEYWORD_MAP.get(lower)
    if kw is not None:
        return (Token(kw, text, start), length)

    # Plain identifier (case-sensitive).
    return (Token(TokenKind.IDENT, text, start), length)
