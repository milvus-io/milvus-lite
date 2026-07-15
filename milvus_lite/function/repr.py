"""Protocol-neutral representation of a public Function Chain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias


def freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ColumnArg:
    name: str


@dataclass(frozen=True)
class LiteralArg:
    value: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_value(self.value))


ExprArg: TypeAlias = ColumnArg | LiteralArg


@dataclass(frozen=True)
class ExprRepr:
    name: str
    args: tuple[ExprArg, ...]
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", freeze_value(self.params))


@dataclass(frozen=True)
class OpRepr:
    op: str
    expr: ExprRepr | None
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: Mapping[str, object]
    read_names: tuple[str, ...]
    write_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", freeze_value(self.params))


@dataclass(frozen=True)
class ChainInfo:
    required_inputs: tuple[str, ...]
    written_names: tuple[str, ...]


@dataclass(frozen=True)
class ChainRepr:
    name: str
    stage: str
    ops: tuple[OpRepr, ...]
    info: ChainInfo


def build_chain_info(ops: tuple[OpRepr, ...]) -> ChainInfo:
    available: set[str] = set()
    required: list[str] = []
    required_seen: set[str] = set()
    written: list[str] = []
    written_seen: set[str] = set()

    for op in ops:
        for name in op.read_names:
            if name not in available and name not in required_seen:
                required_seen.add(name)
                required.append(name)
        for name in op.write_names:
            available.add(name)
            if name not in written_seen:
                written_seen.add(name)
                written.append(name)

    return ChainInfo(tuple(required), tuple(written))
