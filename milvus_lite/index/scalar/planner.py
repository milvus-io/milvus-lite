"""Planner for scalar-index-covered filter expressions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from milvus_lite.search.filter.ast import (
    And,
    BoolLit,
    CmpOp,
    FieldRef,
    FloatLit,
    InOp,
    IntLit,
    IsNullOp,
    ListLit,
    Or,
    StringLit,
    TimestampLit,
)


@dataclass(frozen=True)
class ScalarPredicate:
    op: str
    field_name: str
    value: Any = None
    values: Tuple[Any, ...] = ()


class IndexedFilterPlan:
    required_fields: frozenset[str]

    def evaluate(self, indexes: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class PredicatePlan(IndexedFilterPlan):
    predicate: ScalarPredicate
    required_fields: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_fields", frozenset((self.predicate.field_name,)))

    def evaluate(self, indexes: dict[str, Any]) -> np.ndarray:
        return indexes[self.predicate.field_name].match(self.predicate)


@dataclass(frozen=True)
class AndPlan(IndexedFilterPlan):
    operands: Tuple[IndexedFilterPlan, ...]
    required_fields: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        fields = frozenset()
        for operand in self.operands:
            fields |= operand.required_fields
        object.__setattr__(self, "required_fields", fields)

    def evaluate(self, indexes: dict[str, Any]) -> np.ndarray:
        out = self.operands[0].evaluate(indexes).copy()
        if not out.any():
            return out
        for operand in self.operands[1:]:
            out &= operand.evaluate(indexes)
            if not out.any():
                return out
        return out


@dataclass(frozen=True)
class OrPlan(IndexedFilterPlan):
    operands: Tuple[IndexedFilterPlan, ...]
    required_fields: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        fields = frozenset()
        for operand in self.operands:
            fields |= operand.required_fields
        object.__setattr__(self, "required_fields", fields)

    def evaluate(self, indexes: dict[str, Any]) -> np.ndarray:
        out = self.operands[0].evaluate(indexes).copy()
        if out.all():
            return out
        for operand in self.operands[1:]:
            out |= operand.evaluate(indexes)
            if out.all():
                return out
        return out


_REVERSE_OP = {
    "<": ">",
    "<=": ">=",
    ">": "<",
    ">=": "<=",
    "==": "==",
    "!=": "!=",
}


def plan_indexed_filter(compiled, indexed_fields: Iterable[str]) -> Optional[IndexedFilterPlan]:
    return _plan(compiled.ast, set(indexed_fields))


def _plan(node, indexed_fields: set[str]) -> Optional[IndexedFilterPlan]:
    if isinstance(node, CmpOp):
        return _plan_cmp(node, indexed_fields)
    if isinstance(node, InOp):
        if not isinstance(node.field, FieldRef):
            return None
        if node.field.name not in indexed_fields:
            return None
        values = _literal_list(node.values)
        if values is None:
            return None
        op = "not in" if node.negate else "in"
        return PredicatePlan(ScalarPredicate(op=op, field_name=node.field.name, values=values))
    if isinstance(node, IsNullOp):
        if not isinstance(node.field, FieldRef):
            return None
        if node.field.name not in indexed_fields:
            return None
        op = "is not null" if node.negate else "is null"
        return PredicatePlan(ScalarPredicate(op=op, field_name=node.field.name))
    if isinstance(node, And):
        operands = tuple(_plan(op, indexed_fields) for op in node.operands)
        if any(op is None for op in operands):
            return None
        return AndPlan(operands)  # type: ignore[arg-type]
    if isinstance(node, Or):
        operands = tuple(_plan(op, indexed_fields) for op in node.operands)
        if any(op is None for op in operands):
            return None
        return OrPlan(operands)  # type: ignore[arg-type]
    return None


def _plan_cmp(node: CmpOp, indexed_fields: set[str]) -> Optional[IndexedFilterPlan]:
    if isinstance(node.left, FieldRef):
        field = node.left.name
        value = _literal_value(node.right)
        op = node.op
    elif isinstance(node.right, FieldRef):
        field = node.right.name
        value = _literal_value(node.left)
        op = _REVERSE_OP[node.op]
    else:
        return None
    if field not in indexed_fields or value is _UNSUPPORTED:
        return None
    return PredicatePlan(ScalarPredicate(op=op, field_name=field, value=value))


_UNSUPPORTED = object()


def _literal_value(node) -> Any:
    if isinstance(node, (IntLit, FloatLit, StringLit, BoolLit, TimestampLit)):
        return node.value
    return _UNSUPPORTED


def _literal_list(node: ListLit) -> Optional[Tuple[Any, ...]]:
    values = []
    for element in node.elements:
        value = _literal_value(element)
        if value is _UNSUPPORTED:
            return None
        values.append(value)
    return tuple(values)
