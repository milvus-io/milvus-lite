"""Compile validated public function chains into runtime operators."""

from __future__ import annotations

from collections.abc import Callable

from milvus_lite.function.chain import FuncChain
from milvus_lite.function.expr.decay_expr import DecayExpr
from milvus_lite.function.expr.num_combine import NumCombineExpr
from milvus_lite.function.expr.rerank_model import RerankModelExpr
from milvus_lite.function.expr.round_decimal import RoundDecimalExpr
from milvus_lite.function.ops.limit_op import LimitOp
from milvus_lite.function.ops.map_op import (
    ColumnBinding,
    InputBinding,
    LiteralBinding,
    MapOp,
)
from milvus_lite.function.ops.sort_op import SortOp
from milvus_lite.function.repr import ColumnArg, ExprRepr, LiteralArg, OpRepr
from milvus_lite.function.types import FunctionExpr, STAGE_L2_RERANK
from milvus_lite.function.validator import ValidatedChain


def _bindings(expr: ExprRepr) -> list[InputBinding]:
    bindings: list[InputBinding] = []
    for arg in expr.args:
        if isinstance(arg, ColumnArg):
            bindings.append(ColumnBinding(arg.name))
        elif isinstance(arg, LiteralArg):
            bindings.append(LiteralBinding(arg.value))
        else:
            raise ValueError(
                "unsupported validated function chain expression argument: "
                f"{type(arg).__name__}"
            )
    return bindings


def _build_decay(expr: ExprRepr) -> FunctionExpr:
    return DecayExpr(
        function=expr.params["function"],
        origin=expr.params["origin"],
        scale=expr.params["scale"],
        offset=expr.params.get("offset", 0.0),
        decay=expr.params.get("decay", 0.5),
    )


def _build_num_combine(expr: ExprRepr) -> FunctionExpr:
    return NumCombineExpr(
        mode=expr.params.get("mode", "sum"),
        weights=expr.params.get("weights"),
    )


def _build_round_decimal(expr: ExprRepr) -> FunctionExpr:
    return RoundDecimalExpr(decimal=expr.params["decimal"])


def _build_rerank_model(expr: ExprRepr) -> FunctionExpr:
    from milvus_lite.rerank.factory import create_rerank_provider

    params = dict(expr.params)
    query_texts = list(params.pop("queries"))
    provider = create_rerank_provider(params)
    return RerankModelExpr(provider, query_texts=query_texts)


_EXPR_BUILDERS: dict[str, Callable[[ExprRepr], FunctionExpr]] = {
    "decay": _build_decay,
    "num_combine": _build_num_combine,
    "round_decimal": _build_round_decimal,
    "rerank_model": _build_rerank_model,
}


def _compile_map(op: OpRepr) -> MapOp:
    if op.expr is None:
        raise ValueError(
            "validated function chain map operator is missing an expression"
        )
    builder = _EXPR_BUILDERS.get(op.expr.name)
    if builder is None:
        raise ValueError(
            "unsupported validated function chain expression: "
            f"{op.expr.name}"
        )
    return MapOp(builder(op.expr), _bindings(op.expr), list(op.outputs))


def compile_function_chain(validated: ValidatedChain) -> FuncChain:
    """Compile a validated public chain without adding implicit operators."""

    result = FuncChain(validated.repr.name, STAGE_L2_RERANK)
    for op in validated.repr.ops:
        if op.op == "map":
            result.add(_compile_map(op))
        elif op.op == "sort":
            result.add(
                SortOp(
                    column=op.params["column"],
                    desc=op.params.get("desc", True),
                    tie_break_col=op.params.get("tie_break_col"),
                )
            )
        elif op.op == "limit":
            result.add(
                LimitOp(
                    limit=op.params["limit"],
                    offset=op.params.get("offset", 0),
                )
            )
        else:
            raise ValueError(
                f"unsupported validated function chain operator: {op.op}"
            )
    return result
