"""SearchRequest decoder.

Pulls the engine-relevant pieces out of pymilvus's SearchRequest:

    - query vectors    — packed inside placeholder_group bytes
    - top_k            — search_params["topk"], JSON-encoded
    - metric_type      — search_params["metric_type"], JSON-encoded string
    - filter expr      — request.dsl
    - partition_names  — request.partition_names
    - output_fields    — request.output_fields

The placeholder_group is itself a serialized PlaceholderGroup proto.
We deserialize it, walk PlaceholderValues, and unpack each
``vector_float_to_bytes`` blob via ``struct.unpack``.

Notes:
    - We accept FloatVector PlaceholderValues only (type 101). Other
      types (binary, sparse, int8, etc.) raise UnsupportedFieldTypeError.
    - Each PlaceholderValue's ``values`` is a repeated bytes; each
      element is one query vector packed as ``"<dim>f"`` little-endian.
    - dim is inferred from len(bytes) // 4 since we don't get it
      explicitly. We accept this — there's no place to validate it
      against the collection schema until we hand off to engine.
"""

from __future__ import annotations

import json
import struct
from typing import List, Optional, Tuple

from pymilvus.grpc_gen import common_pb2

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.rerank.boost import decode_hybrid_function_score


# PlaceholderType enum values from common.proto
_PH_FLOAT_VECTOR = 101
_PH_SPARSE_FLOAT_VECTOR = 104
_PH_VARCHAR = 21


def parse_search_request(request, default_metric_type: str = "COSINE") -> dict:
    """Decode the engine-relevant fields from a SearchRequest.

    Args:
        request: SearchRequest proto
        default_metric_type: fallback metric when search_params doesn't
            contain one. The servicer passes the collection's
            IndexSpec.metric_type so the engine uses the same metric
            the index was built with.

    Returns:
        dict with keys::
            query_vectors:    List[List[float]]   (nq × dim)
            top_k:            int
            metric_type:      str  ("COSINE" / "L2" / "IP")
            expr:             Optional[str]  (None if no filter)
            partition_names:  Optional[List[str]]
            output_fields:    Optional[List[str]]
            search_params:    dict (the engine-side params, e.g. {"ef": 64})

    Raises:
        SchemaValidationError: missing required field, unsupported
            placeholder type, or malformed search_params
    """
    query_vectors = _decode_placeholder_group(request.placeholder_group)
    top_k, metric_type, search_params = _decode_search_params(
        request.search_params, default_metric_type=default_metric_type,
    )

    expr = request.dsl if request.dsl else None
    partition_names = list(request.partition_names) or None
    output_fields = list(request.output_fields) or None

    # anns_field: which vector field to search on.
    # pymilvus puts it in search_params as "anns_field" (JSON string).
    raw_params: dict = {}
    for kv in request.search_params:
        try:
            raw_params[kv.key] = json.loads(kv.value)
        except (json.JSONDecodeError, ValueError):
            raw_params[kv.key] = kv.value
    anns_field = raw_params.get("anns_field")
    if isinstance(anns_field, str) and anns_field:
        pass  # use it
    else:
        anns_field = None  # let engine default

    # range search parameters (inside "params" dict or top-level)
    # pymilvus puts them in params: {"radius": ..., "range_filter": ...}
    inner_params = raw_params.get("params", {})
    if isinstance(inner_params, dict):
        radius = inner_params.get("radius")
        range_filter_val = inner_params.get("range_filter")
    else:
        radius = None
        range_filter_val = None
    # Also check top-level (pymilvus supports both locations)
    if radius is None:
        radius = raw_params.get("radius")
    if range_filter_val is None:
        range_filter_val = raw_params.get("range_filter")
    # Convert to float if present
    if radius is not None:
        radius = float(radius)
    if range_filter_val is not None:
        range_filter_val = float(range_filter_val)

    # group_by parameters
    group_by_field = raw_params.get("group_by_field")
    if isinstance(group_by_field, str) and group_by_field:
        pass
    else:
        group_by_field = None
    group_size = raw_params.get("group_size")
    if group_size is not None:
        group_size = int(group_size)
    strict_group_size = raw_params.get("strict_group_size")
    if strict_group_size is not None:
        strict_group_size = bool(strict_group_size)

    function_score = decode_hybrid_function_score(request.function_score)

    return {
        "query_vectors": query_vectors,
        "top_k": top_k,
        "metric_type": metric_type,
        "expr": expr,
        "partition_names": partition_names,
        "output_fields": output_fields,
        "search_params": search_params,
        "anns_field": anns_field,
        "group_by_field": group_by_field,
        "group_size": group_size,
        "group_size_strict": strict_group_size,
        "radius": radius,
        "range_filter": range_filter_val,
        "offset": int(raw_params.get("offset", 0)),
        "round_decimal": int(raw_params.get("round_decimal", -1)),
        "ranker": function_score.get("boost"),
        "rerank": function_score.get("rerank"),
    }


def _decode_placeholder_group(placeholder_group_bytes: bytes) -> List[List[float]]:
    """Decode the serialized PlaceholderGroup → list of query vectors."""
    if not placeholder_group_bytes:
        raise SchemaValidationError("SearchRequest is missing placeholder_group")

    pg = common_pb2.PlaceholderGroup()
    pg.ParseFromString(placeholder_group_bytes)

    if not pg.placeholders:
        raise SchemaValidationError("PlaceholderGroup is empty")

    # We only handle the first PlaceholderValue (single anns_field).
    pv = pg.placeholders[0]

    if pv.type == _PH_FLOAT_VECTOR:
        # Each value is a packed float buffer for one query vector.
        out: List = []
        for blob in pv.values:
            if not blob:
                raise SchemaValidationError(
                    "PlaceholderValue contains an empty FloatVector blob"
                )
            n_floats = len(blob) // 4
            if n_floats * 4 != len(blob):
                raise SchemaValidationError(
                    f"PlaceholderValue blob is not a multiple of 4 bytes "
                    f"({len(blob)})"
                )
            out.append(list(struct.unpack(f"{n_floats}f", blob)))
        if not out:
            raise SchemaValidationError("PlaceholderGroup has no query vectors")
        return out

    if pv.type == _PH_SPARSE_FLOAT_VECTOR:
        # Sparse vectors: each value is SparseFloatArray-format bytes
        from milvus_lite.analyzer.sparse import bytes_to_sparse
        out = []
        for blob in pv.values:
            out.append(bytes_to_sparse(blob) if blob else {})
        if not out:
            raise SchemaValidationError("PlaceholderGroup has no query vectors")
        return out

    if pv.type == _PH_VARCHAR:
        # Text queries: each value is UTF-8 encoded string
        out = []
        for blob in pv.values:
            out.append(blob.decode("utf-8") if blob else "")
        if not out:
            raise SchemaValidationError("PlaceholderGroup has no query vectors")
        return out

    raise SchemaValidationError(
        f"PlaceholderValue type {pv.type} not supported "
        f"(supported: FloatVector=101, SparseFloatVector=104, VarChar=21)"
    )


def _decode_search_params(
    kv_pairs,
    default_metric_type: str = "COSINE",
) -> Tuple[int, str, dict]:
    """Decode the search_params KeyValuePair list.

    Each value is a JSON-encoded string (pymilvus's utils.dumps).
    We pull out:
        - topk        → int (required)
        - metric_type → str (optional; default_metric_type used if missing.
                              pymilvus's MilvusClient.search doesn't put
                              metric_type in the request, so the caller
                              must supply the collection's IndexSpec
                              metric as the default)
        - params      → dict (engine-side, e.g. {"ef": 64})
    Other keys (round_decimal, anns_field, ignore_growing, etc.) are
    ignored — they have no MilvusLite equivalent.
    """
    raw: dict = {}
    for kv in kv_pairs:
        try:
            raw[kv.key] = json.loads(kv.value)
        except (json.JSONDecodeError, ValueError):
            raw[kv.key] = kv.value

    top_k = raw.get("topk")
    if top_k is None:
        raise SchemaValidationError(
            "SearchRequest.search_params is missing required 'topk'"
        )
    if not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            raise SchemaValidationError(
                f"SearchRequest.search_params['topk'] must be int, got {top_k!r}"
            )

    metric_type = raw.get("metric_type", default_metric_type)
    if not isinstance(metric_type, str):
        metric_type = str(metric_type)

    engine_params = raw.get("params") or {}
    if not isinstance(engine_params, dict):
        engine_params = {}

    return top_k, metric_type, engine_params
