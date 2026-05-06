"""Pratt parser for filter expressions.

Hand-written recursive descent with precedence climbing. Grammar
matches the F1 subset documented in plan/filter-design.md §3:

    Prec  Operator        Assoc
    1     or / OR / ||    left
    2     and / AND / &&  left
    3     not / NOT / !   right (prefix)
    4     == != < <= > >= left
    4     in [...]        non-assoc
    5     - (unary minus) right (prefix)
    6     literal | ident | ( expr )

Comparison chaining (`a == b == c`) is accepted at parse time and
rejected at semantic time with a type error pointing to the bool/int
mismatch — this matches Milvus's Plan.g4 (Equality is left-assoc).

For F1 we deliberately reject features that exist in Milvus but
require Tier 2/3 work. The reject is in parse_primary, with a
"will be supported in Phase F2/F3" hint.
"""

from __future__ import annotations

from typing import List, Optional

from milvus_lite.search.filter.ast import (
    And,
    ArithOp,
    BoolLit,
    CmpOp,
    Expr,
    FieldRef,
    FloatLit,
    InOp,
    IntLit,
    ArrayAccessOp,
    ArrayContainsOp,
    ArrayLengthOp,
    IsNullOp,
    JsonAccess,
    LikeOp,
    ListLit,
    Literal,
    MetaAccess,
    Not,
    Or,
    StringLit,
    TextMatchOp,
)
from milvus_lite.search.filter.exceptions import FilterParseError
from milvus_lite.search.filter.tokens import Token, TokenKind, tokenize


_CMP_KINDS = (
    TokenKind.EQ, TokenKind.NE,
    TokenKind.LT, TokenKind.LE,
    TokenKind.GT, TokenKind.GE,
)
_CMP_TEXT = {
    TokenKind.EQ: "==", TokenKind.NE: "!=",
    TokenKind.LT: "<",  TokenKind.LE: "<=",
    TokenKind.GT: ">",  TokenKind.GE: ">=",
}


class Parser:
    """Recursive-descent parser; instances are single-use (one parse call)."""

    def __init__(self, tokens: List[Token], source: str) -> None:
        self.tokens = tokens
        self.source = source
        self.pos = 0

    # ── public entry ────────────────────────────────────────────

    def parse(self) -> Expr:
        expr = self.parse_or()
        if self._peek().kind != TokenKind.EOF:
            tok = self._peek()
            raise FilterParseError(
                f"unexpected token {tok.text!r}",
                self.source, tok.pos, span=max(1, len(tok.text)),
                hint="expected end of expression",
            )
        return expr

    # ── precedence levels (low → high) ──────────────────────────

    def parse_or(self) -> Expr:
        """prec 1: a or b or c (left-assoc, flattened)."""
        left = self.parse_and()
        operands: list[Expr] = []
        while self._peek().kind == TokenKind.OR:
            self._consume()
            operands.append(self.parse_and())
        if not operands:
            return left
        return Or(operands=tuple([left, *operands]), pos=left_pos_of(left))

    def parse_and(self) -> Expr:
        """prec 2: a and b and c (left-assoc, flattened)."""
        left = self.parse_not()
        operands: list[Expr] = []
        while self._peek().kind == TokenKind.AND:
            self._consume()
            operands.append(self.parse_not())
        if not operands:
            return left
        return And(operands=tuple([left, *operands]), pos=left_pos_of(left))

    def parse_not(self) -> Expr:
        """prec 3: not a (prefix, right-assoc)."""
        if self._peek().kind == TokenKind.NOT:
            tok = self._consume()
            # Special case: `field not in [...]` is handled in parse_in_or_cmp
            # by looking at the next token. But the parser at this level
            # expects `not <factor>`, so we need to be careful — `not in`
            # is only valid AFTER an identifier. Here we always treat
            # standalone `not` as a logical-not prefix.
            operand = self.parse_not()
            return Not(operand=operand, pos=tok.pos)
        return self.parse_cmp()

    def parse_cmp(self) -> Expr:
        """prec 4: comparisons + IN + LIKE + IS NULL.

        After parsing the LHS additive expression, dispatch on the
        next token:
            - cmp op  → CmpOp (left-assoc, chaining parse-accepted)
            - in      → InOp
            - not in  → InOp(negate=True)
            - LIKE    → LikeOp
            - IS NULL / IS NOT NULL → IsNullOp
        """
        left = self.parse_add()

        while True:
            peek = self._peek()
            if peek.kind in _CMP_KINDS:
                op_tok = self._consume()
                right = self.parse_add()
                left = CmpOp(
                    op=_CMP_TEXT[op_tok.kind],
                    left=left,
                    right=right,
                    pos=op_tok.pos,
                )
                continue
            if peek.kind == TokenKind.IN:
                return self._parse_in_tail(left, negate=False)
            if peek.kind == TokenKind.NOT:
                # `not in` after a comparable LHS — the `not` here is
                # the negation marker on `in`, not a logical-not prefix.
                if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].kind == TokenKind.IN:
                    self._consume()  # NOT
                    return self._parse_in_tail(left, negate=True)
            if peek.kind == TokenKind.LIKE:
                return self._parse_like_tail(left)
            if peek.kind == TokenKind.IS:
                return self._parse_is_null_tail(left)
            break

        return left

    def parse_add(self) -> Expr:
        """prec 5: additive (+, -) — left-associative."""
        left = self.parse_mul()
        while self._peek().kind in (TokenKind.ADD, TokenKind.SUB):
            op_tok = self._consume()
            right = self.parse_mul()
            op = "+" if op_tok.kind == TokenKind.ADD else "-"
            left = ArithOp(op=op, left=left, right=right, pos=op_tok.pos)
        return left

    def parse_mul(self) -> Expr:
        """prec 6: multiplicative (*, /) — left-associative."""
        left = self.parse_unary()
        while self._peek().kind in (TokenKind.MUL, TokenKind.DIV):
            op_tok = self._consume()
            right = self.parse_unary()
            op = "*" if op_tok.kind == TokenKind.MUL else "/"
            left = ArithOp(op=op, left=left, right=right, pos=op_tok.pos)
        return left

    def _parse_like_tail(self, left: Expr) -> Expr:
        """`<expr> LIKE 'pattern'` — parse the LIKE tail."""
        like_tok = self._consume()  # LIKE
        peek = self._peek()
        if peek.kind != TokenKind.STRING:
            raise FilterParseError(
                "LIKE requires a string literal pattern",
                self.source, peek.pos,
                hint="example: title like 'AI%'",
            )
        pattern_tok = self._consume()
        return LikeOp(
            value=left,
            pattern=StringLit(value=pattern_tok.value, pos=pattern_tok.pos),
            pos=like_tok.pos,
        )

    def _parse_meta_access(self) -> Expr:
        """Parse `$meta["key"]`. Caller hasn't consumed the META token.

        Phase F2b only supports string keys (`$meta["category"]`).
        Integer indexing (`$meta[0]`) is deferred to F3 and rejected
        with a hint.
        """
        meta_tok = self._consume()  # META
        if self._peek().kind != TokenKind.LBRACKET:
            tok = self._peek()
            raise FilterParseError(
                f"expected '[' after '$meta', got {tok.text!r}",
                self.source, tok.pos,
                hint="example: $meta[\"category\"]",
            )
        self._consume()  # [

        key_tok = self._peek()
        if key_tok.kind != TokenKind.STRING:
            raise FilterParseError(
                f"$meta key must be a string literal, got {key_tok.text!r}",
                self.source, key_tok.pos,
                hint="example: $meta[\"category\"]",
            )
        self._consume()

        if self._peek().kind != TokenKind.RBRACKET:
            tok = self._peek()
            raise FilterParseError(
                f"expected ']' after $meta key, got {tok.text!r}",
                self.source, tok.pos,
            )
        self._consume()  # ]

        return MetaAccess(key=key_tok.value, pos=meta_tok.pos)

    def _parse_is_null_tail(self, left: Expr) -> Expr:
        """`<field> IS NULL` or `<field> IS NOT NULL`.

        IS NULL only applies to a FieldRef LHS. Anything else is a
        compile-time error in semantic.py — but we already enforce the
        FieldRef constraint here at parse time for a more direct error.
        """
        is_tok = self._consume()  # IS
        if not isinstance(left, FieldRef):
            raise FilterParseError(
                "left side of 'is null' must be a field reference",
                self.source, is_tok.pos,
                hint="example: title is null",
            )
        negate = False
        if self._peek().kind == TokenKind.NOT:
            self._consume()
            negate = True
        if self._peek().kind != TokenKind.NULL:
            tok = self._peek()
            expected = "'not null'" if negate else "'null'"
            raise FilterParseError(
                f"expected {expected} after 'is', got {tok.text!r}",
                self.source, tok.pos,
            )
        self._consume()  # NULL
        return IsNullOp(field=left, negate=negate, pos=is_tok.pos)

    def _parse_text_match(self, func_tok: Token) -> Expr:
        """``text_match(field_name, 'query tokens')``.

        The IDENT "text_match" has already been consumed; we expect:
            '(' IDENT ',' STRING ')'
        """
        self._consume()  # '('
        # Field name
        field_tok = self._peek()
        if field_tok.kind != TokenKind.IDENT:
            raise FilterParseError(
                f"text_match: expected field name, got {field_tok.text!r}",
                self.source, field_tok.pos,
            )
        self._consume()
        field = FieldRef(name=field_tok.text, pos=field_tok.pos)
        # Comma
        if self._peek().kind != TokenKind.COMMA:
            raise FilterParseError(
                "text_match: expected ',' after field name",
                self.source, self._peek().pos,
            )
        self._consume()
        # Query string
        query_tok = self._peek()
        if query_tok.kind != TokenKind.STRING:
            raise FilterParseError(
                f"text_match: expected query string, got {query_tok.text!r}",
                self.source, query_tok.pos,
            )
        self._consume()
        query = StringLit(value=query_tok.value, pos=query_tok.pos)
        # Close paren
        if self._peek().kind != TokenKind.RPAREN:
            raise FilterParseError(
                "text_match: expected ')'",
                self.source, self._peek().pos,
            )
        self._consume()
        return TextMatchOp(field=field, query=query, pos=func_tok.pos)

    def _parse_bracket_access(self, ident_tok: Token) -> Expr:
        """``field["key"]``, ``field["a"]["b"]`` (JSON path) or ``field[N]`` (array index).

        The IDENT has already been consumed; we expect '[' (STRING|INT) ']'.
        For string keys, chained bracket access is supported.
        """
        self._consume()  # '['
        key_tok = self._peek()
        if key_tok.kind == TokenKind.STRING:
            self._consume()
            if self._peek().kind != TokenKind.RBRACKET:
                raise FilterParseError(
                    "field access: expected ']'",
                    self.source, self._peek().pos,
                )
            self._consume()
            # Collect chained keys: field["a"]["b"]["c"]
            keys = [key_tok.value]
            while self._peek().kind == TokenKind.LBRACKET:
                self._consume()  # '['
                next_key = self._peek()
                if next_key.kind != TokenKind.STRING:
                    raise FilterParseError(
                        "chained JSON access: expected string key",
                        self.source, next_key.pos,
                    )
                self._consume()
                if self._peek().kind != TokenKind.RBRACKET:
                    raise FilterParseError(
                        "chained JSON access: expected ']'",
                        self.source, self._peek().pos,
                    )
                self._consume()
                keys.append(next_key.value)
            return JsonAccess(
                field_name=ident_tok.text,
                keys=tuple(keys),
                pos=ident_tok.pos,
            )
        if key_tok.kind == TokenKind.INT:
            self._consume()
            if self._peek().kind != TokenKind.RBRACKET:
                raise FilterParseError(
                    "array index: expected ']'",
                    self.source, self._peek().pos,
                )
            self._consume()
            return ArrayAccessOp(
                field_name=ident_tok.text,
                index=key_tok.value,
                pos=ident_tok.pos,
            )
        raise FilterParseError(
            f"field access: expected string key or integer index, got {key_tok.text!r}",
            self.source, key_tok.pos,
        )

    def _parse_array_contains(self, func_tok: Token, mode: str) -> Expr:
        """``array_contains(field, value)`` or ``json_contains(field["key"], value)``."""
        self._consume()  # '('
        field_tok = self._peek()
        if field_tok.kind != TokenKind.IDENT:
            raise FilterParseError(
                f"{func_tok.text}: expected field name",
                self.source, field_tok.pos,
            )
        self._consume()
        # Support json_contains(info["tags"], val) — parse bracket access
        if self._peek().kind == TokenKind.LBRACKET:
            field = self._parse_bracket_access(field_tok)
        else:
            field = FieldRef(name=field_tok.text, pos=field_tok.pos)
        if self._peek().kind != TokenKind.COMMA:
            raise FilterParseError(
                f"{func_tok.text}: expected ','",
                self.source, self._peek().pos,
            )
        self._consume()
        # Value: single literal or list
        if self._peek().kind == TokenKind.LBRACKET:
            values = self.parse_list_literal()
        else:
            values = self.parse_primary()
        if self._peek().kind != TokenKind.RPAREN:
            raise FilterParseError(
                f"{func_tok.text}: expected ')'",
                self.source, self._peek().pos,
            )
        self._consume()
        return ArrayContainsOp(field=field, values=values, mode=mode, pos=func_tok.pos)

    def _parse_array_length(self, func_tok: Token) -> Expr:
        """``array_length(field)`` — returns int, used in comparisons."""
        self._consume()  # '('
        field_tok = self._peek()
        if field_tok.kind != TokenKind.IDENT:
            raise FilterParseError(
                f"array_length: expected field name",
                self.source, field_tok.pos,
            )
        self._consume()
        if self._peek().kind != TokenKind.RPAREN:
            raise FilterParseError(
                "array_length: expected ')'",
                self.source, self._peek().pos,
            )
        self._consume()
        return ArrayLengthOp(field=FieldRef(name=field_tok.text, pos=field_tok.pos),
                             pos=func_tok.pos)

    def _parse_in_tail(self, left: Expr, negate: bool) -> Expr:
        """Parse the `in [...]` tail. *left* must be a FieldRef."""
        in_tok = self._consume()  # IN
        if not isinstance(left, FieldRef):
            raise FilterParseError(
                "left side of 'in' must be a field reference",
                self.source, in_tok.pos,
                hint="example: field in [1, 2, 3]",
            )
        if self._peek().kind != TokenKind.LBRACKET:
            tok = self._peek()
            raise FilterParseError(
                f"expected '[' after 'in', got {tok.text!r}",
                self.source, tok.pos,
                hint="'in' must be followed by a literal list, e.g. [1, 2, 3]",
            )
        list_lit = self.parse_list_literal()
        return InOp(field=left, values=list_lit, negate=negate, pos=in_tok.pos)

    def parse_unary(self) -> Expr:
        """prec 7: unary minus (prefix)."""
        if self._peek().kind == TokenKind.SUB:
            sub_tok = self._consume()
            operand = self.parse_unary()
            # Constant-fold unary minus on int/float literals to keep
            # the AST minimal and avoid generating ArithOp(-, 0, x) noise.
            if isinstance(operand, IntLit):
                return IntLit(value=-operand.value, pos=sub_tok.pos)
            if isinstance(operand, FloatLit):
                return FloatLit(value=-operand.value, pos=sub_tok.pos)
            # General unary minus: ArithOp(-, 0, operand). The semantic
            # checker accepts this once the operand is numeric.
            return ArithOp(
                op="-",
                left=IntLit(value=0, pos=sub_tok.pos),
                right=operand,
                pos=sub_tok.pos,
            )
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        """prec 6: literal | ident | (expr)."""
        tok = self._peek()

        if tok.kind == TokenKind.INT:
            self._consume()
            return IntLit(value=tok.value, pos=tok.pos)

        if tok.kind == TokenKind.FLOAT:
            self._consume()
            return FloatLit(value=tok.value, pos=tok.pos)

        if tok.kind == TokenKind.STRING:
            self._consume()
            return StringLit(value=tok.value, pos=tok.pos)

        if tok.kind == TokenKind.BOOL:
            self._consume()
            return BoolLit(value=tok.value, pos=tok.pos)

        if tok.kind == TokenKind.IDENT:
            self._consume()
            # Function-call syntax
            if self._peek().kind == TokenKind.LPAREN:
                fn_name = tok.text.lower()
                if fn_name == "text_match":
                    return self._parse_text_match(tok)
                if fn_name in ("array_contains", "json_contains"):
                    return self._parse_array_contains(tok, "any_one")
                if fn_name in ("array_contains_all", "json_contains_all"):
                    return self._parse_array_contains(tok, "all")
                if fn_name in ("array_contains_any", "json_contains_any"):
                    return self._parse_array_contains(tok, "any")
                if fn_name == "array_length":
                    return self._parse_array_length(tok)
                raise FilterParseError(
                    f"unknown function {tok.text!r}",
                    self.source, tok.pos, span=len(tok.text),
                    hint="supported: text_match, array_contains, json_contains, "
                         "array_contains_all, array_contains_any, array_length",
                )
            # field[...] access: JSON path or array index
            if self._peek().kind == TokenKind.LBRACKET:
                return self._parse_bracket_access(tok)
            return FieldRef(name=tok.text, pos=tok.pos)

        if tok.kind == TokenKind.META:
            return self._parse_meta_access()

        if tok.kind == TokenKind.LPAREN:
            self._consume()
            inner = self.parse_or()
            if self._peek().kind != TokenKind.RPAREN:
                close = self._peek()
                raise FilterParseError(
                    f"expected ')' got {close.text!r}",
                    self.source, close.pos,
                )
            self._consume()
            return inner

        if tok.kind == TokenKind.EOF:
            raise FilterParseError(
                "unexpected end of expression",
                self.source, tok.pos,
                hint="expression cannot be empty",
            )

        raise FilterParseError(
            f"unexpected token {tok.text!r}",
            self.source, tok.pos, span=max(1, len(tok.text)),
            hint="expected literal, identifier, or '('",
        )

    def parse_list_literal(self) -> ListLit:
        """Parse `[ literal (',' literal)* (',')? ]`. Caller has not
        yet consumed the '['."""
        lbracket = self._consume()  # [
        elements: list[Literal] = []

        # Empty list: []
        if self._peek().kind == TokenKind.RBRACKET:
            self._consume()
            return ListLit(elements=tuple(), pos=lbracket.pos)

        elements.append(self._parse_list_element())

        while self._peek().kind == TokenKind.COMMA:
            self._consume()  # ,
            # Trailing comma allowed: ',' followed by ']'
            if self._peek().kind == TokenKind.RBRACKET:
                break
            elements.append(self._parse_list_element())

        if self._peek().kind != TokenKind.RBRACKET:
            tok = self._peek()
            raise FilterParseError(
                f"expected ']' or ',' got {tok.text!r}",
                self.source, tok.pos,
            )
        self._consume()  # ]
        return ListLit(elements=tuple(elements), pos=lbracket.pos)

    def _parse_list_element(self) -> Literal:
        """List elements are literals only (no field refs, no nested
        expressions). Negative numeric literals are folded here too."""
        tok = self._peek()
        if tok.kind == TokenKind.SUB:
            sub_tok = self._consume()
            inner = self._peek()
            if inner.kind not in (TokenKind.INT, TokenKind.FLOAT):
                raise FilterParseError(
                    "unary '-' in list must be followed by a numeric literal",
                    self.source, sub_tok.pos,
                )
            self._consume()
            if inner.kind == TokenKind.INT:
                return IntLit(value=-inner.value, pos=sub_tok.pos)
            return FloatLit(value=-inner.value, pos=sub_tok.pos)

        if tok.kind == TokenKind.INT:
            self._consume()
            return IntLit(value=tok.value, pos=tok.pos)
        if tok.kind == TokenKind.FLOAT:
            self._consume()
            return FloatLit(value=tok.value, pos=tok.pos)
        if tok.kind == TokenKind.STRING:
            self._consume()
            return StringLit(value=tok.value, pos=tok.pos)
        if tok.kind == TokenKind.BOOL:
            self._consume()
            return BoolLit(value=tok.value, pos=tok.pos)

        raise FilterParseError(
            f"list elements must be literals, got {tok.text!r}",
            self.source, tok.pos, span=max(1, len(tok.text)),
        )

    # ── token utilities ─────────────────────────────────────────

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def left_pos_of(node: Expr) -> int:
    """Best-effort source position for the leftmost child of *node*.

    Used to anchor And/Or nodes at their leftmost operand's position
    rather than at the operator (which would point to the rightmost
    `and`/`or` in a chain).
    """
    return getattr(node, "pos", 0)


def parse_expr(source: str) -> Expr:
    """Public entry point: lex + parse a single expression.

    Raises:
        FilterParseError: on lex or parse errors. Always carries source + pos.
    """
    tokens = tokenize(source)
    return Parser(tokens, source).parse()
