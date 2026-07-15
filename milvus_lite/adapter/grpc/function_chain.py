"""Search-specific planning and execution for public Function Chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from milvus_lite.adapter.grpc.translators.function_chain import (
    function_chain_to_repr,
)
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.function.chain import FuncChain
from milvus_lite.function.compiler import compile_function_chain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD
from milvus_lite.function.validator import validate_function_chain


@dataclass(frozen=True)
class SearchFunctionChainPlan:
    chain: FuncChain
    required_fields: tuple[str, ...]
    requested_output_fields: tuple[str, ...] | None


def merge_internal_output_fields(
    requested_output_fields: Optional[list[str]],
    required_fields: tuple[str, ...],
) -> Optional[list[str]]:
    if requested_output_fields is None:
        return None
    return list(dict.fromkeys([*requested_output_fields, *required_fields]))


def prepare_search_function_chain(
    *,
    function_chains,
    has_function_score: bool,
    schema,
    num_queries: int,
    requested_output_fields: Optional[list[str]],
    order_by_fields=None,
) -> SearchFunctionChainPlan | None:
    chains = list(function_chains or ())
    if not chains:
        return None
    if has_function_score:
        raise SchemaValidationError(
            "function_score and function_chains cannot be used together"
        )
    if order_by_fields:
        raise SchemaValidationError(
            "order_by and function rerank cannot be used together"
        )
    if len(chains) != 1:
        raise SchemaValidationError(
            "ordinary search supports exactly one function chain"
        )

    representation = function_chain_to_repr(chains[0])
    validated = validate_function_chain(representation, schema, num_queries)
    return SearchFunctionChainPlan(
        chain=compile_function_chain(validated),
        required_fields=validated.required_schema_fields,
        requested_output_fields=(
            tuple(requested_output_fields)
            if requested_output_fields is not None
            else None
        ),
    )


def hit_score_for_chain(hit: dict, metric_type: str) -> float:
    distance = hit["distance"]
    return -distance if metric_type.upper() == "BM25" else distance


def _project_entity(
    row: dict,
    original_entity: dict,
    schema_names: set[str],
    primary_field_name: str,
    requested_output_fields: tuple[str, ...] | None,
) -> dict:
    if requested_output_fields is None:
        field_names = original_entity
    else:
        field_names = dict.fromkeys(requested_output_fields)

    entity = {}
    for name in field_names:
        if name == primary_field_name or name in {ID_FIELD, SCORE_FIELD}:
            continue
        if name in schema_names:
            if name in row:
                entity[name] = row[name]
        elif name in original_entity:
            entity[name] = original_entity[name]
    return entity


def _runtime_row_value(
    row: dict,
    name: str,
    chunk_index: int,
    row_index: int,
):
    if name not in row:
        raise ValueError(
            "function chain runtime output "
            f"chunk {chunk_index} row {row_index} "
            f"is missing required {name}"
        )
    return row[name]


def execute_search_function_chain(
    plan: SearchFunctionChainPlan,
    results: list[list[dict]],
    *,
    metric_type: str,
    schema,
    primary_field_name: str,
    group_by_field: str | None = None,
) -> list[list[dict]]:
    chunks = []
    original_entities = []
    for query_hits in results:
        chunk = []
        entities_by_id = {}
        for hit in query_hits:
            hit_id = hit["id"]
            if hit_id in entities_by_id:
                raise ValueError(
                    f"function chain input contains duplicate id {hit_id!r} "
                    "within one query chunk"
                )
            entity = dict(hit.get("entity") or {})
            entities_by_id[hit_id] = entity
            row = dict(entity)
            row[ID_FIELD] = hit_id
            row[primary_field_name] = hit_id
            row[SCORE_FIELD] = hit_score_for_chain(hit, metric_type)
            if group_by_field is not None and "_group_by_value" in hit:
                row[group_by_field] = hit["_group_by_value"]
            chunk.append(row)
        chunks.append(chunk)
        original_entities.append(entities_by_id)

    output = plan.chain.execute(DataFrame(chunks))
    schema_names = {field.name for field in schema.fields}
    reranked = []
    for chunk_index in range(output.num_chunks):
        hits = []
        for row_index, row in enumerate(output.chunk(chunk_index)):
            hit_id = _runtime_row_value(
                row, ID_FIELD, chunk_index, row_index
            )
            original_entity = original_entities[chunk_index].get(hit_id)
            if original_entity is None:
                raise ValueError(
                    "function chain runtime output "
                    f"chunk {chunk_index} row {row_index} has unknown "
                    f"{ID_FIELD} value {hit_id!r}"
                )
            hit = {
                "id": hit_id,
                "distance": _runtime_row_value(
                    row, SCORE_FIELD, chunk_index, row_index
                ),
                "entity": _project_entity(
                    row,
                    original_entity,
                    schema_names,
                    primary_field_name,
                    plan.requested_output_fields,
                ),
            }
            if group_by_field is not None and group_by_field in row:
                hit["_group_by_value"] = row[group_by_field]
            hits.append(hit)
        reranked.append(hits)
    return reranked
