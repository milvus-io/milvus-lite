"""IndexParams ↔ IndexSpec translation.

The Milvus protocol carries index parameters as a list of
``KeyValuePair`` (string key, string value). pymilvus's
``IndexParams.add_index()`` produces this list with the following
well-known keys:

    field_name      → IndexSpec.field_name (also explicit on the request)
    index_type      → IndexSpec.index_type ("HNSW" / "BRUTE_FORCE" / ...)
    metric_type     → IndexSpec.metric_type ("COSINE" / "L2" / "IP")
    params          → JSON-encoded dict, becomes IndexSpec.build_params

We accept any other keys but ignore them — pymilvus may emit
``index_name`` or ``mmap.enabled`` etc. that have no MilvusLite
counterpart. Forward-compat is more useful here than strict rejection.

The reverse path (IndexSpec → IndexDescription proto for the
DescribeIndex response) flattens build_params back into KeyValuePair
form so pymilvus's describe_index can read it back.
"""

from __future__ import annotations

import json
from typing import Iterable, List

from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.index.spec import IndexSpec


def kv_pairs_to_index_params_dict(
    extra_params: Iterable,
    field_name: str = "",
) -> dict:
    """Decode the CreateIndexRequest.extra_params KeyValuePair list
    into a dict suitable for ``Collection.create_index(field, params)``.

    Args:
        extra_params: iterable of common_pb2.KeyValuePair messages
        field_name: fallback field_name from the outer request, used
            if the KV list doesn't carry one

    Returns:
        ``{"index_type": ..., "metric_type": ..., "params": {...},
            "search_params": {...}}``  — the shape Collection.create_index
        consumes.

    Raises:
        SchemaValidationError: required keys (index_type, metric_type)
            missing or malformed
    """
    kv: dict[str, str] = {p.key: p.value for p in extra_params}

    index_type = kv.get("index_type")
    metric_type = kv.get("metric_type")

    if not index_type:
        raise SchemaValidationError(
            "create_index missing required 'index_type' parameter"
        )
    if not metric_type:
        raise SchemaValidationError(
            "create_index missing required 'metric_type' parameter"
        )

    # `params` is a JSON-encoded string in the wire format. pymilvus
    # uses utils.dumps which is plain json.dumps for dicts.
    raw_params = kv.get("params")
    if raw_params:
        try:
            build_params = json.loads(raw_params)
            if not isinstance(build_params, dict):
                build_params = {}
        except (json.JSONDecodeError, ValueError):
            build_params = {}
    else:
        build_params = {}

    raw_search = kv.get("search_params")
    if raw_search:
        try:
            search_params = json.loads(raw_search)
            if not isinstance(search_params, dict):
                search_params = {}
        except (json.JSONDecodeError, ValueError):
            search_params = {}
    else:
        search_params = {}

    return {
        "index_type": index_type,
        "metric_type": metric_type,
        "params": build_params,
        "search_params": search_params,
    }


def index_spec_to_kv_pairs(spec: IndexSpec) -> List:
    """Build the IndexDescription.params KeyValuePair list for a
    DescribeIndex response.

    Mirrors how pymilvus's describe_index parses the response: it
    looks at the ``params`` KeyValuePair list and reconstructs the
    index_type / metric_type / params dict.
    """
    from pymilvus.grpc_gen import common_pb2

    pairs: List = []

    def _add(k: str, v: str) -> None:
        pairs.append(common_pb2.KeyValuePair(key=k, value=v))

    _add("index_type", spec.index_type)
    _add("metric_type", spec.metric_type)
    if spec.build_params:
        _add("params", json.dumps(dict(spec.build_params), sort_keys=True))
    if spec.search_params:
        _add("search_params", json.dumps(dict(spec.search_params), sort_keys=True))

    return pairs
