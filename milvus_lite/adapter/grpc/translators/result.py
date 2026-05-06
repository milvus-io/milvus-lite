"""Engine search results → Milvus SearchResultData proto.

The engine returns ``List[List[dict]]`` shaped as
``[per_query[per_hit{"id", "distance", "entity"}]]``. Milvus's
SearchResultData is FLATTENED across queries, with a per-query
``topks`` array providing the boundaries:

    SearchResultData
      num_queries: nq
      top_k:       requested k
      primary_field_name: pk_name
      output_fields: list of field names emitted in fields_data
      topks:       [k1, k2, ..., kn]  — actual hits per query
      ids:         IDs proto with int_id.data or str_id.data flat list
      scores:      flat float list, length sum(topks)
      fields_data: columnar FieldData list — each column has length sum(topks)

This file does the flattening. The records → fields_data part reuses
the Phase 10.3 records translator since the encoding is the same.
"""

from __future__ import annotations

from typing import List, Optional

from pymilvus.grpc_gen import schema_pb2

from milvus_lite.adapter.grpc.translators.records import records_to_fields_data
from milvus_lite.schema.types import CollectionSchema, DataType


def build_search_result_data(
    results: List[List[dict]],
    schema: CollectionSchema,
    top_k: int,
    pk_name: str,
    output_fields: Optional[List[str]] = None,
    group_by_field: Optional[str] = None,
) -> schema_pb2.SearchResultData:
    """Flatten the engine's per-query results into one SearchResultData.

    Args:
        results: engine's nested ``[per_query[hit_dict]]`` shape, where
            each hit dict has keys ``"id"``, ``"distance"``, ``"entity"``.
        schema: source CollectionSchema, needed by records_to_fields_data
            for per-field type info.
        top_k: requested top_k (used for the response's top_k field;
            actual returned per-query counts are in ``topks``).
        pk_name: primary key field name. Stored in
            ``primary_field_name`` and used to pick the int_id vs
            str_id slot for the IDs proto.
        output_fields: optional whitelist; passed through to
            records_to_fields_data.
    """
    nq = len(results)

    # Flatten per-query → flat lists of (id, score, entity_dict).
    all_pks: list = []
    all_scores: List[float] = []
    flat_records: List[dict] = []
    topks: List[int] = []

    pk_field = next((f for f in schema.fields if f.is_primary), None)
    pk_dtype = pk_field.dtype if pk_field else None

    for query_hits in results:
        topks.append(len(query_hits))
        for hit in query_hits:
            all_pks.append(hit["id"])
            all_scores.append(float(hit["distance"]))
            # Reconstruct a "row dict" from id + entity so the
            # records translator can re-encode it.
            row = dict(hit.get("entity") or {})
            row[pk_name] = hit["id"]
            flat_records.append(row)

    # Build the IDs proto.
    ids = schema_pb2.IDs()
    if pk_dtype == DataType.VARCHAR:
        ids.str_id.data.extend([str(p) for p in all_pks])
    else:
        ids.int_id.data.extend([int(p) for p in all_pks])

    # Build the columnar fields_data using the Phase 10.3 translator.
    # Note: when there are 0 hits across all queries, the records
    # translator still emits one empty FieldData per emitted field
    # (Phase 10.3 fix), so pymilvus's parser doesn't choke.
    fields_data = records_to_fields_data(
        flat_records, schema, output_fields=output_fields,
    )

    # Determine emitted output_fields list. pymilvus's parser uses
    # this to know which non-pk fields to attach to each hit.
    if output_fields is None:
        emitted = [f.name for f in schema.fields if f.name != pk_name]
    else:
        # Preserve user order; drop pk if user listed it (it's
        # surfaced via "id" anyway)
        emitted = [f for f in output_fields if f != pk_name]

    result = schema_pb2.SearchResultData(
        num_queries=nq,
        top_k=top_k,
        fields_data=fields_data,
        scores=all_scores,
        ids=ids,
        topks=topks,
        output_fields=emitted,
        primary_field_name=pk_name,
    )

    # Build group_by_field_value if group_by was used
    if group_by_field is not None:
        gb_field = next((f for f in schema.fields if f.name == group_by_field), None)
        if gb_field is not None:
            group_values = []
            for query_hits in results:
                for hit in query_hits:
                    gval = hit.get("_group_by_value")
                    if gval is None:
                        gval = (hit.get("entity") or {}).get(group_by_field)
                    group_values.append(gval)
            if group_values:
                from milvus_lite.adapter.grpc.translators.records import _build_field_data
                gb_fd = _build_field_data(group_by_field, gb_field, group_values)
                result.group_by_field_value.CopyFrom(gb_fd)

    return result
