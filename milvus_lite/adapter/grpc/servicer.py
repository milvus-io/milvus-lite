"""MilvusServicer — gRPC RPC dispatcher for the MilvusLite engine.

Inherits from ``pymilvus.grpc_gen.milvus_pb2_grpc.MilvusServiceServicer``.
The base class auto-generates UNIMPLEMENTED responses for every method
we don't override, so an empty subclass already returns the right
"not supported" status for the 100+ RPCs we don't plan to implement.

Phase 10.1 ships only the bare minimum needed for pymilvus.connect()
to succeed (Connect + GetVersion). Phase 10.2-10.6 fills in:
    10.2 — Collection lifecycle (Create/Drop/Has/Describe/Show)
    10.3 — Insert/Upsert/Delete/Query/Get  (FieldData ↔ records)
    10.4 — Search/CreateIndex/Load/Release  (the search path)
    10.5 — Partition + Flush + Stats  (rounding out the quickstart)
    10.6 — Error code mapping  (MilvusLiteError → grpc Status)

Implementation discipline (from grpc-adapter-design.md §15):
    The servicer ONLY translates protocol; it never adds engine
    capability. Anything we don't have an engine API for must
    return UNIMPLEMENTED with a friendly message — never silent fail.
"""

from __future__ import annotations

import logging
import json
from typing import TYPE_CHECKING

import grpc
from pymilvus.grpc_gen import common_pb2, milvus_pb2, milvus_pb2_grpc, schema_pb2

from milvus_lite.adapter.grpc.errors import (
    SUCCESS as _SUCCESS,
    UNEXPECTED_ERROR as _UNEXPECTED_ERROR,
    success_status_kwargs,
    to_status_kwargs,
)
from milvus_lite.adapter.grpc.translators.index import (
    index_spec_to_kv_pairs,
    kv_pairs_to_index_params_dict,
)
from milvus_lite.adapter.grpc.translators.records import (
    fields_data_to_records,
    records_to_fields_data,
)
from milvus_lite.adapter.grpc.translators.result import build_search_result_data
from milvus_lite.adapter.grpc.translators.schema import (
    milvus_lite_to_milvus_schema,
    milvus_to_milvus_lite_schema,
)
from milvus_lite.adapter.grpc.translators.search import parse_search_request
from milvus_lite._version import get_version
from milvus_lite.exceptions import MilvusLiteError
from milvus_lite.schema.types import DataType

if TYPE_CHECKING:
    from milvus_lite.db import MilvusLite

logger = logging.getLogger(__name__)


def _extract_anns_field(sub_req) -> str | None:
    """Extract anns_field from a sub-SearchRequest's search_params."""
    import json as _json
    for kv in sub_req.search_params:
        if kv.key == "anns_field":
            try:
                v = _json.loads(kv.value)
            except (ValueError, _json.JSONDecodeError):
                v = kv.value
            return v if isinstance(v, str) and v else None
    return None


def _kv_pairs_to_dict(pairs) -> dict[str, str]:
    return {
        str(p.key): str(p.value)
        for p in pairs
        if getattr(p, "key", None)
    }


def _dict_to_kv_pairs(values: dict) -> list[common_pb2.KeyValuePair]:
    return [
        common_pb2.KeyValuePair(key=str(key), value=str(value))
        for key, value in values.items()
    ]


def _decode_kv_value(value: str):
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _extract_timezone(pairs) -> str | None:
    for kv in pairs:
        if kv.key == "timezone":
            value = _decode_kv_value(kv.value)
            return value if isinstance(value, str) and value else None
    return None


def _extract_time_fields(pairs) -> str | None:
    for kv in pairs:
        if kv.key == "time_fields":
            value = _decode_kv_value(kv.value)
            return value if isinstance(value, str) and value else None
    return None


def _hit_score_for_chain(hit: dict, metric_type: str) -> float:
    """Convert Collection.search() hit distance to chain score.

    FuncChain follows Milvus merge semantics: incoming scores keep the
    metric's natural direction, and MergeOp decides whether it needs to
    normalize, direction-convert, and sort ascending or descending.
    """
    distance = hit["distance"]
    metric = metric_type.upper()
    if metric == "COSINE":
        return 1.0 - distance
    if metric == "BM25":
        return -distance
    if metric in {"IP", "L2"}:
        return distance
    return distance


class MilvusServicer(milvus_pb2_grpc.MilvusServiceServicer):
    """Maps Milvus RPCs onto MilvusLite engine calls.

    Phase 10.1 handles connection-level RPCs only. All data-plane
    methods inherited from MilvusServiceServicer return UNIMPLEMENTED
    via the gRPC default implementation.
    """

    def __init__(self, db: "MilvusLite") -> None:
        self._db = db

    # ── Connection-level RPCs ───────────────────────────────────
    #
    # pymilvus.MilvusClient(uri=...) does a Connect call as part of
    # client construction. Without this override the client init
    # would itself raise UNIMPLEMENTED, and users couldn't even open
    # a connection to the server. So this is the absolute minimum
    # surface to ship in Phase 10.1.

    def Connect(self, request, context):
        """Acknowledge the client identity. We don't track sessions
        — every request is processed independently — so this just
        returns a success status with our server identity."""
        return milvus_pb2.ConnectResponse(
            status=common_pb2.Status(code=_SUCCESS, reason=""),
            server_info=common_pb2.ServerInfo(
                build_tags="milvus_lite",
                build_time="",
                git_commit="",
                go_version="",
                deploy_mode="embedded",
            ),
            identifier=0,
        )

    def GetVersion(self, request, context):
        """Return the MilvusLite version string. pymilvus uses this
        as a smoke test for "the server is alive and speaks the
        Milvus protocol"."""
        return milvus_pb2.GetVersionResponse(
            status=common_pb2.Status(code=_SUCCESS, reason=""),
            version=f"milvus_lite-{get_version()}",
        )

    def CheckHealth(self, request, context):
        """Health probe. Always reports healthy — single-process
        embedded servers don't have a partial-failure mode."""
        return milvus_pb2.CheckHealthResponse(
            status=common_pb2.Status(code=_SUCCESS, reason=""),
            isHealthy=True,
            reasons=[],
        )

    # ── Collection lifecycle (Phase 10.2) ───────────────────────

    def CreateCollection(self, request, context):
        """Decode the schema bytes blob, validate, then call
        ``MilvusLite.create_collection``."""
        try:
            proto_schema = schema_pb2.CollectionSchema()
            proto_schema.ParseFromString(request.schema)
            milvus_lite_schema = milvus_to_milvus_lite_schema(proto_schema)
            properties = _kv_pairs_to_dict(getattr(request, "properties", []))
            self._db.create_collection(
                request.collection_name,
                milvus_lite_schema,
                properties=properties,
            )
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("CreateCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DropCollection(self, request, context):
        try:
            self._db.drop_collection(request.collection_name)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("DropCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def HasCollection(self, request, context):
        try:
            exists = self._db.has_collection(request.collection_name)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                value=exists,
            )
        except Exception as e:
            logger.exception("HasCollection failed: %s", e)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
                value=False,
            )

    def DescribeCollection(self, request, context):
        """Return the collection's schema + basic stats. The Phase 9
        Collection.describe() output is rebuilt into Milvus's
        DescribeCollectionResponse shape."""
        try:
            col = self._db.get_collection(request.collection_name)
            proto_schema = milvus_lite_to_milvus_schema(col.name, col.schema)
            return milvus_pb2.DescribeCollectionResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                schema=proto_schema,
                collection_name=col.name,
                shards_num=1,
                num_partitions=len(col.list_partitions()),
                properties=_dict_to_kv_pairs(col.schema.properties),
            )
        except MilvusLiteError as e:
            return milvus_pb2.DescribeCollectionResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("DescribeCollection failed: %s", e)
            return milvus_pb2.DescribeCollectionResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def ShowCollections(self, request, context):
        """Return the list of collection names. Milvus's response also
        carries timestamps and IDs which we don't track — those slots
        stay empty."""
        try:
            names = self._db.list_collections()
            return milvus_pb2.ShowCollectionsResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                collection_names=names,
            )
        except Exception as e:
            logger.exception("ShowCollections failed: %s", e)
            return milvus_pb2.ShowCollectionsResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Data CRUD (Phase 10.3) ──────────────────────────────────

    def Insert(self, request, context):
        """Decode columnar fields_data into records, dispatch to
        ``Collection.insert``. Returns a MutationResult with the
        inserted IDs (which double as the success indicator for
        pymilvus's MilvusClient.insert)."""
        try:
            col = self._db.get_collection(request.collection_name)
            records = fields_data_to_records(
                request.fields_data,
                request.num_rows,
                default_timezone=col._effective_timezone(),  # noqa: SLF001
            )
            partition_name = request.partition_name or "_default"
            inserted_pks = col.insert(records, partition_name=partition_name)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**success_status_kwargs()),
                IDs=self._build_ids_proto(inserted_pks, col),
                insert_cnt=len(inserted_pks),
                succ_index=list(range(len(inserted_pks))),
            )
        except MilvusLiteError as e:
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Insert failed: %s", e)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def Upsert(self, request, context):
        """Upsert with partial update — merges new fields onto existing
        records so callers don't need to provide every field."""
        try:
            col = self._db.get_collection(request.collection_name)
            records = fields_data_to_records(
                request.fields_data,
                request.num_rows,
                default_timezone=col._effective_timezone(),  # noqa: SLF001
            )
            partition_name = request.partition_name or "_default"
            upserted_pks = col.upsert(records, partition_name=partition_name)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**success_status_kwargs()),
                IDs=self._build_ids_proto(upserted_pks, col),
                upsert_cnt=len(upserted_pks),
                succ_index=list(range(len(upserted_pks))),
            )
        except MilvusLiteError as e:
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Upsert failed: %s", e)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def Delete(self, request, context):
        """Two paths:

        1. Filter expression looks like ``id in [1,2,3]`` → extract pks
           and call ``col.delete(pks=[...])`` directly. This is what
           pymilvus emits for ``client.delete(ids=[...])``.

        2. Any other expression → fall back to "query → extract pks →
           delete". The cost is one extra read pass; it's the only way
           to honor delete-by-filter without engine-native support.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            partition_name = request.partition_name or None

            pks = self._extract_pks_from_expr(request.expr, col)
            if pks is None:
                # Fall back: query to find matching pks, then delete.
                # Temporarily ensure loaded state for the query, then
                # restore — delete should not require explicit load().
                original_state = col._load_state
                if original_state != "loaded":
                    col._load_state = "loaded"
                try:
                    hits = col.query(request.expr, output_fields=[col._pk_name])
                finally:
                    if original_state != "loaded":
                        col._load_state = original_state
                pks = [r[col._pk_name] for r in hits]

            count = col.delete(pks, partition_name=partition_name)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**success_status_kwargs()),
                IDs=self._build_ids_proto(pks, col),
                delete_cnt=count,
            )
        except MilvusLiteError as e:
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Delete failed: %s", e)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def Query(self, request, context):
        """Two paths, same dispatch as Delete:

        1. Expression looks like ``id in [...]`` → call ``col.get(pks)``.
           This is what pymilvus emits for ``client.get(ids=[...])``.

        2. Any other expression → call ``col.query(expr, ...)``.

        Both paths return their results encoded as columnar
        fields_data via the records translator.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            partition_names = list(request.partition_names) or None
            output_fields = list(request.output_fields) or None

            # Extract limit and offset from query_params KV list.
            limit = None
            offset = 0
            timezone = _extract_timezone(request.query_params)
            time_fields = _extract_time_fields(request.query_params)
            for kv in request.query_params:
                if kv.key == "limit":
                    try:
                        limit = int(kv.value)
                    except (ValueError, TypeError):
                        pass
                elif kv.key == "offset":
                    try:
                        offset = int(kv.value)
                    except (ValueError, TypeError):
                        pass

            # Expand output_fields=["*"] → all schema field names
            if output_fields and "*" in output_fields:
                output_fields = [f.name for f in col.schema.fields]

            # Handle count(*) aggregation
            if output_fields and "count(*)" in output_fields:
                expr = request.expr if request.expr else None
                if expr:
                    rows = col.query(
                        expr,
                        partition_names=partition_names,
                        timezone=timezone,
                    )
                    count = len(rows)
                else:
                    count = col.num_entities
                # Return count as a single-row result with count field
                count_fd = schema_pb2.FieldData()
                count_fd.field_name = "count(*)"
                count_fd.type = 5  # INT64
                count_fd.scalars.long_data.data.append(count)
                return milvus_pb2.QueryResults(
                    status=common_pb2.Status(**success_status_kwargs()),
                    fields_data=[count_fd],
                    collection_name=col.name,
                    output_fields=["count(*)"],
                )

            expr = request.expr if request.expr else None
            pks = self._extract_pks_from_expr(expr, col) if expr else None
            if pks is not None:
                rows = col.get(pks, partition_names=partition_names,
                               output_fields=output_fields)
            else:
                rows = col.query(
                    expr,
                    output_fields=output_fields,
                    partition_names=partition_names,
                    limit=limit,
                    offset=offset,
                    timezone=timezone,
                )

            return milvus_pb2.QueryResults(
                status=common_pb2.Status(**success_status_kwargs()),
                fields_data=records_to_fields_data(
                    rows,
                    col.schema,
                    output_fields=output_fields,
                    time_fields=time_fields,
                    timezone=col._effective_timezone(timezone),  # noqa: SLF001
                ),
                collection_name=col.name,
                output_fields=output_fields or [],
                primary_field_name=col._pk_name,
            )
        except MilvusLiteError as e:
            return milvus_pb2.QueryResults(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Query failed: %s", e)
            return milvus_pb2.QueryResults(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Vector search (Phase 10.4) ──────────────────────────────

    def Search(self, request, context):
        """Decode the search request, dispatch to ``Collection.search``,
        flatten the result back into a SearchResultData proto.

        The hard part is the request decoding (placeholder_group bytes
        → list of query vectors via PlaceholderGroup proto + struct
        unpack), centralized in translators/search.py.

        Metric resolution: pymilvus's MilvusClient.search doesn't
        include metric_type in search_params by default. We fall back
        to the collection's IndexSpec.metric_type so the engine uses
        the same metric the index was built with.
        """
        try:
            from milvus_lite.function.builder import build_hybrid_function_score_chain
            from milvus_lite.function.dataframe import DataFrame
            from milvus_lite.function.types import ID_FIELD, SCORE_FIELD

            col = self._db.get_collection(request.collection_name)
            # Pull the canonical metric from a vector index if any.
            first_spec = (
                col._index_specs.get(col._vector_name)  # noqa: SLF001
                if col._index_specs and col._vector_name is not None  # noqa: SLF001
                else None
            )
            default_metric = (
                first_spec.metric_type
                if first_spec is not None and first_spec.metric_type != "NONE"
                else "COSINE"
            )
            parsed = parse_search_request(request, default_metric_type=default_metric)

            group_by_field = parsed.get("group_by_field")
            group_size = parsed.get("group_size") or 1
            strict = parsed.get("group_size_strict") or False
            requested_output_fields = parsed["output_fields"]
            l2_func = parsed.get("rerank")

            internal_output_fields = []
            if l2_func is not None:
                if group_by_field is not None:
                    internal_output_fields.append(group_by_field)
                internal_output_fields.extend(
                    list(getattr(l2_func, "input_field_names", []))
                )
            if requested_output_fields is None:
                search_output_fields = None
            else:
                search_output_fields = list(dict.fromkeys(
                    requested_output_fields + internal_output_fields
                ))

            search_top_k = parsed["top_k"]
            search_offset = parsed.get("offset", 0)
            search_group_by_field = group_by_field
            if l2_func is not None:
                search_top_k = parsed["top_k"] + search_offset
                if group_by_field is not None:
                    search_top_k = max(search_top_k, parsed["top_k"] * group_size * 3)
                search_top_k = max(search_top_k, parsed["top_k"] * 10)
                search_offset = 0
                search_group_by_field = None

            results = col.search(
                query_vectors=parsed["query_vectors"],
                top_k=search_top_k,
                metric_type=parsed["metric_type"],
                partition_names=parsed["partition_names"],
                expr=parsed["expr"],
                output_fields=search_output_fields,
                anns_field=parsed.get("anns_field"),
                group_by_field=search_group_by_field,
                group_size=group_size,
                strict_group_size=strict,
                radius=parsed.get("radius"),
                range_filter=parsed.get("range_filter"),
                offset=search_offset,
                ranker=parsed.get("ranker"),
                timezone=parsed.get("timezone"),
            )

            if l2_func is not None:
                chunks = []
                for query_hits in results:
                    chunk = []
                    for hit in query_hits:
                        flat = {
                            ID_FIELD: hit["id"],
                            SCORE_FIELD: _hit_score_for_chain(
                                hit, parsed["metric_type"]
                            ),
                        }
                        flat.update(hit.get("entity", {}))
                        chunk.append(flat)
                    chunks.append(chunk)

                search_params = {
                    "limit": parsed["top_k"],
                    "offset": parsed.get("offset", 0),
                    "group_by_field": group_by_field,
                    "group_size": group_size,
                    "metric_types": [parsed["metric_type"]],
                }
                chain = build_hybrid_function_score_chain(
                    l2_func,
                    search_params=search_params,
                    search_metrics=[parsed["metric_type"]],
                    collection_schema=col.schema,
                )
                result_df = chain.execute(DataFrame(chunks))

                from milvus_lite.function.types import GROUP_SCORE_FIELD
                virtual = {ID_FIELD, SCORE_FIELD, GROUP_SCORE_FIELD}
                reranked = []
                for ci in range(result_df.num_chunks):
                    hits = []
                    for row in result_df.chunk(ci):
                        entity = {
                            k: v for k, v in row.items() if k not in virtual
                        }
                        hit = {
                            "id": row[ID_FIELD],
                            "distance": row[SCORE_FIELD],
                            "entity": entity,
                        }
                        if group_by_field is not None and group_by_field in row:
                            hit["_group_by_value"] = row[group_by_field]
                        hits.append(hit)
                    reranked.append(hits)
                results = reranked

            # Apply round_decimal to distance values
            rd = parsed.get("round_decimal", -1)
            if rd >= 0:
                for query_hits in results:
                    for hit in query_hits:
                        hit["distance"] = round(hit["distance"], rd)

            result_data = build_search_result_data(
                results=results,
                schema=col.schema,
                top_k=parsed["top_k"],
                pk_name=col._pk_name,  # noqa: SLF001
                output_fields=requested_output_fields,
                group_by_field=group_by_field,
                time_fields=parsed.get("time_fields"),
                timezone=col._effective_timezone(parsed.get("timezone")),  # noqa: SLF001
            )

            return milvus_pb2.SearchResults(
                status=common_pb2.Status(**success_status_kwargs()),
                results=result_data,
                collection_name=col.name,
            )
        except MilvusLiteError as e:
            return milvus_pb2.SearchResults(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Search failed: %s", e)
            return milvus_pb2.SearchResults(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Index lifecycle (Phase 10.4) ────────────────────────────

    def CreateIndex(self, request, context):
        """Decode IndexParams and call ``Collection.create_index``.

        pymilvus's MilvusClient.create_index packs ``index_type``,
        ``metric_type``, ``params``, and (optionally) ``search_params``
        into the ``extra_params`` KeyValuePair list. The translator
        unpacks these into the dict shape Collection.create_index
        consumes.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            params = kv_pairs_to_index_params_dict(request.extra_params)
            col.create_index(request.field_name, params)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("CreateIndex failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DropIndex(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            field_name = request.field_name or None
            # Resolve index_name → field_name if field_name not provided
            if field_name is None and request.index_name:
                # Our naming convention: "{field_name}_idx"
                idx_name = request.index_name
                for fn in col._index_specs:  # noqa: SLF001
                    if f"{fn}_idx" == idx_name:
                        field_name = fn
                        break
            col.drop_index(field_name)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("DropIndex failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DescribeIndex(self, request, context):
        """Return the IndexSpec wrapped in a DescribeIndexResponse.

        Returns an INDEX_NOT_FOUND status when there's no matching
        index (pymilvus's describe_index parses this as None rather
        than raising AmbiguousIndexName, which is what would happen
        if we returned SUCCESS + empty list).

        IndexState is always ``Finished`` because Phase 9 builds
        indexes synchronously inside ``load()``.
        """
        try:
            from milvus_lite.exceptions import IndexNotFoundError as _INFE

            col = self._db.get_collection(request.collection_name)
            all_specs = col._index_specs  # noqa: SLF001

            if not all_specs:
                raise _INFE(
                    f"no index on collection {request.collection_name!r}"
                )

            # Resolve field_name: try request.field_name first, then
            # index_name (pymilvus sends field_name as index_name).
            field_name = request.field_name
            if not field_name and request.index_name:
                # index_name might be the field_name directly, or
                # "{field_name}_idx" convention
                idx_name = request.index_name
                if idx_name in all_specs:
                    field_name = idx_name
                elif idx_name.endswith("_idx"):
                    candidate = idx_name[:-4]
                    if candidate in all_specs:
                        field_name = candidate

            if field_name:
                spec = all_specs.get(field_name)
                if spec is None:
                    raise _INFE(
                        f"no index on field {field_name!r} of "
                        f"collection {request.collection_name!r}"
                    )
                specs_to_report = [spec]
            else:
                specs_to_report = list(all_specs.values())

            num_ent = col.num_entities
            descriptions = [
                milvus_pb2.IndexDescription(
                    index_name=s.field_name,
                    field_name=s.field_name,
                    params=index_spec_to_kv_pairs(s),
                    state=common_pb2.IndexState.Finished,
                    indexed_rows=num_ent,
                    total_rows=num_ent,
                )
                for s in specs_to_report
            ]
            return milvus_pb2.DescribeIndexResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                index_descriptions=descriptions,
            )
        except MilvusLiteError as e:
            return milvus_pb2.DescribeIndexResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("DescribeIndex failed: %s", e)
            return milvus_pb2.DescribeIndexResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Load / release (Phase 10.4) ─────────────────────────────

    def LoadCollection(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.load()
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("LoadCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def ReleaseCollection(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.release()
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("ReleaseCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def GetLoadingProgress(self, request, context):
        """Polled by pymilvus's load_collection wrapper.

        Our load is synchronous (Phase 9), so once it returns the
        collection is fully loaded → progress 100. If the collection
        is in 'loading' state we still report 0; if released, 0.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            progress = 100 if col.load_state == "loaded" else 0
            return milvus_pb2.GetLoadingProgressResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                progress=progress,
            )
        except MilvusLiteError as e:
            return milvus_pb2.GetLoadingProgressResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("GetLoadingProgress failed: %s", e)
            return milvus_pb2.GetLoadingProgressResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def GetLoadState(self, request, context):
        """Map Collection._load_state to Milvus's LoadState enum.

            released → LoadStateNotLoad   (1)
            loading  → LoadStateLoading   (2)
            loaded   → LoadStateLoaded    (3)
        """
        try:
            if not self._db.has_collection(request.collection_name):
                return milvus_pb2.GetLoadStateResponse(
                    status=common_pb2.Status(**success_status_kwargs()),
                    state=common_pb2.LoadState.LoadStateNotExist,
                )
            col = self._db.get_collection(request.collection_name)
            mapping = {
                "released": common_pb2.LoadState.LoadStateNotLoad,
                "loading":  common_pb2.LoadState.LoadStateLoading,
                "loaded":   common_pb2.LoadState.LoadStateLoaded,
            }
            return milvus_pb2.GetLoadStateResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                state=mapping.get(col.load_state, common_pb2.LoadState.LoadStateNotLoad),
            )
        except MilvusLiteError as e:
            return milvus_pb2.GetLoadStateResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("GetLoadState failed: %s", e)
            return milvus_pb2.GetLoadStateResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Partition + Flush + Stats (Phase 10.5) ─────────────────

    def CreatePartition(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.create_partition(request.partition_name)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("CreatePartition failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DropPartition(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.drop_partition(request.partition_name)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("DropPartition failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def HasPartition(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            exists = col.has_partition(request.partition_name)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                value=exists,
            )
        except MilvusLiteError as e:
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
                value=False,
            )
        except Exception as e:
            logger.exception("HasPartition failed: %s", e)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
                value=False,
            )

    def ShowPartitions(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            names = col.list_partitions()
            return milvus_pb2.ShowPartitionsResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                partition_names=names,
            )
        except MilvusLiteError as e:
            return milvus_pb2.ShowPartitionsResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("ShowPartitions failed: %s", e)
            return milvus_pb2.ShowPartitionsResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def Flush(self, request, context):
        """Flush all named collections. pymilvus sends the collection
        name(s) in request.collection_names (plural)."""
        try:
            for cname in request.collection_names:
                if self._db.has_collection(cname):
                    col = self._db.get_collection(cname)
                    col.flush()
            return milvus_pb2.FlushResponse(
                status=common_pb2.Status(**success_status_kwargs()),
            )
        except MilvusLiteError as e:
            return milvus_pb2.FlushResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Flush failed: %s", e)
            return milvus_pb2.FlushResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def GetFlushState(self, request, context):
        """pymilvus polls this after Flush. Phase 9 flush is synchronous
        so the answer is always True (flushed)."""
        return milvus_pb2.GetFlushStateResponse(
            status=common_pb2.Status(**success_status_kwargs()),
            flushed=True,
        )

    def GetCollectionStatistics(self, request, context):
        """Return row_count as a KeyValuePair list. pymilvus's
        get_collection_stats parses these pairs into a dict."""
        try:
            stats = self._db.get_collection_stats(request.collection_name)
            kv_pairs = [
                common_pb2.KeyValuePair(key=str(k), value=str(v))
                for k, v in stats.items()
            ]
            return milvus_pb2.GetCollectionStatisticsResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                stats=kv_pairs,
            )
        except MilvusLiteError as e:
            return milvus_pb2.GetCollectionStatisticsResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("GetCollectionStatistics failed: %s", e)
            return milvus_pb2.GetCollectionStatisticsResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def GetPartitionStatistics(self, request, context):
        """Return row_count for one partition."""
        try:
            stats = self._db.get_partition_stats(
                request.collection_name, request.partition_name
            )
            kv_pairs = [
                common_pb2.KeyValuePair(key=str(k), value=str(v))
                for k, v in stats.items()
            ]
            return milvus_pb2.GetPartitionStatisticsResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                stats=kv_pairs,
            )
        except MilvusLiteError as e:
            return milvus_pb2.GetPartitionStatisticsResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("GetPartitionStatistics failed: %s", e)
            return milvus_pb2.GetPartitionStatisticsResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def TruncateCollection(self, request, context):
        """Clear collection data while preserving schema and aliases."""
        try:
            self._db.truncate_collection(request.collection_name)
            return milvus_pb2.TruncateCollectionResponse(
                status=common_pb2.Status(**success_status_kwargs()),
            )
        except MilvusLiteError as e:
            return milvus_pb2.TruncateCollectionResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("TruncateCollection failed: %s", e)
            return milvus_pb2.TruncateCollectionResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def ListDatabases(self, request, context):
        """MilvusLite has no database concept; return a single default."""
        return milvus_pb2.ListDatabasesResponse(
            status=common_pb2.Status(**success_status_kwargs()),
            db_names=["default"],
        )

    def DescribeDatabase(self, request, context):
        try:
            desc = self._db.describe_database(request.db_name or "default")
            return milvus_pb2.DescribeDatabaseResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                db_name=desc["name"],
                properties=_dict_to_kv_pairs(desc["properties"]),
            )
        except Exception as e:
            logger.exception("DescribeDatabase failed: %s", e)
            return milvus_pb2.DescribeDatabaseResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def AlterDatabase(self, request, context):
        try:
            properties = _kv_pairs_to_dict(getattr(request, "properties", []))
            delete_keys = list(getattr(request, "delete_keys", [])) or None
            self._db.alter_database_properties(
                request.db_name or "default",
                properties=properties or None,
                delete_keys=delete_keys,
            )
            return common_pb2.Status(**success_status_kwargs())
        except Exception as e:
            logger.exception("AlterDatabase failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    # ── Explicitly UNIMPLEMENTED stubs ─────────────────────────
    #
    # The base class returns UNIMPLEMENTED for every method we don't
    # override, but those responses carry a generic "Method not
    # implemented!" message. For high-frequency RPCs that pymilvus
    # users might hit, we override with a friendlier explanation so
    # error messages point the user in the right direction instead of
    # being cryptic.

    def HybridSearch(self, request, context):
        """Multi-route ANN search with reranking fusion.

        Parses each sub-SearchRequest independently, dispatches to
        Collection.search(), then merges results via FuncChain
        (MergeOp + Sort/GroupBy + Limit + Select).
        """
        try:
            from milvus_lite.adapter.grpc.reranker import parse_rank_params
            from milvus_lite.function.builder import (
                build_hybrid_function_score_chain,
                build_hybrid_rerank_chain,
            )
            from milvus_lite.function.dataframe import DataFrame
            from milvus_lite.function.types import ID_FIELD, SCORE_FIELD
            from milvus_lite.rerank.boost import (
                decode_hybrid_function_score,
                merge_boost_rankers,
            )

            col = self._db.get_collection(request.collection_name)
            all_specs = col._index_specs or {}  # noqa: SLF001

            # Parse rank_params
            rp = parse_rank_params(request.rank_params)
            function_score = decode_hybrid_function_score(request.function_score)
            top_level_l0_ranker = function_score.get("boost")
            top_level_l2_func = function_score.get("rerank")
            requested_output_fields = list(request.output_fields) or None
            hybrid_timezone = _extract_timezone(request.rank_params)
            hybrid_time_fields = _extract_time_fields(request.rank_params)
            output_timezone = hybrid_timezone
            output_time_fields = hybrid_time_fields

            gb_field = rp.get("group_by_field")
            internal_output_fields = []
            if gb_field is not None:
                internal_output_fields.append(gb_field)
            if top_level_l2_func is not None:
                internal_output_fields.extend(
                    list(getattr(top_level_l2_func, "input_field_names", []))
                )
            if requested_output_fields is None:
                route_output_fields = None
            else:
                route_output_fields = list(dict.fromkeys(
                    requested_output_fields + internal_output_fields
                ))

            # Execute each sub-request independently
            all_results = []
            route_metrics = []
            for sub_req in request.requests:
                sub_anns = _extract_anns_field(sub_req)
                if sub_anns and sub_anns in all_specs:
                    sub_default_metric = all_specs[sub_anns].metric_type
                else:
                    first_spec = next(iter(all_specs.values()), None)
                    sub_default_metric = first_spec.metric_type if first_spec else "COSINE"
                parsed = parse_search_request(sub_req, default_metric_type=sub_default_metric)
                route_timezone = parsed.get("timezone") or hybrid_timezone
                if output_timezone is None and parsed.get("timezone") is not None:
                    output_timezone = parsed.get("timezone")
                if output_time_fields is None and parsed.get("time_fields") is not None:
                    output_time_fields = parsed.get("time_fields")
                sub_ranker = merge_boost_rankers(
                    parsed.get("ranker"),
                    top_level_l0_ranker,
                )
                sub_top_k = parsed["top_k"]
                if gb_field is not None:
                    gb_size = rp.get("group_size") or 1
                    sub_top_k = max(sub_top_k, rp["limit"] * gb_size * 3)
                if sub_ranker is not None:
                    sub_top_k = max(sub_top_k, rp["limit"] * 10)
                results = col.search(
                    query_vectors=parsed["query_vectors"],
                    top_k=sub_top_k,
                    metric_type=parsed["metric_type"],
                    partition_names=parsed.get("partition_names") or (
                        list(request.partition_names) or None
                    ),
                    expr=parsed["expr"],
                    output_fields=route_output_fields,
                    anns_field=parsed.get("anns_field"),
                    ranker=sub_ranker,
                    timezone=route_timezone,
                )
                all_results.append(results)
                route_metrics.append(parsed["metric_type"])

            # Convert each route to DataFrame with $id/$score virtual columns
            dfs = []
            for route_results, metric_type in zip(all_results, route_metrics):
                chunks = []
                for query_hits in route_results:
                    chunk = []
                    for hit in query_hits:
                        flat = {
                            ID_FIELD: hit["id"],
                            SCORE_FIELD: _hit_score_for_chain(hit, metric_type),
                        }
                        flat.update(hit.get("entity", {}))
                        chunk.append(flat)
                    chunks.append(chunk)
                dfs.append(DataFrame(chunks))

            # Build and execute rerank chain
            search_params = {
                "limit": rp["limit"],
                "offset": rp["offset"],
                "group_by_field": gb_field,
                "group_size": rp.get("group_size") or 1,
                "metric_types": route_metrics,
            }
            if top_level_l2_func is not None:
                chain = build_hybrid_function_score_chain(
                    top_level_l2_func,
                    search_params=search_params,
                    search_metrics=route_metrics,
                    collection_schema=col.schema,
                )
            else:
                chain = build_hybrid_rerank_chain(
                    strategy=rp["strategy"],
                    params=rp["params"],
                    search_params=search_params,
                )
            result_df = chain.execute(*dfs)

            # Convert chain rows back to the SearchResultData input shape.
            # For L2 rerank/merge, the value in "distance" is already the
            # final score stored in SearchResultData.scores.
            from milvus_lite.function.types import GROUP_SCORE_FIELD
            _virtual = {ID_FIELD, SCORE_FIELD, GROUP_SCORE_FIELD}
            merged = []
            for ci in range(result_df.num_chunks):
                hits = []
                for row in result_df.chunk(ci):
                    entity = {k: v for k, v in row.items() if k not in _virtual}
                    hit = {
                        "id": row[ID_FIELD],
                        "distance": row[SCORE_FIELD],
                        "entity": entity,
                    }
                    if gb_field is not None and gb_field in row:
                        hit["_group_by_value"] = row[gb_field]
                    hits.append(hit)
                merged.append(hits)

            # Build response
            output_fields = requested_output_fields
            result_data = build_search_result_data(
                results=merged,
                schema=col.schema,
                top_k=rp["limit"],
                pk_name=col._pk_name,  # noqa: SLF001
                output_fields=output_fields,
                group_by_field=gb_field,
                time_fields=output_time_fields,
                timezone=col._effective_timezone(output_timezone),  # noqa: SLF001
            )

            return milvus_pb2.SearchResults(
                status=common_pb2.Status(**success_status_kwargs()),
                results=result_data,
                collection_name=col.name,
            )
        except MilvusLiteError as e:
            return milvus_pb2.SearchResults(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("HybridSearch failed: %s", e)
            return milvus_pb2.SearchResults(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def RenameCollection(self, request, context):
        try:
            self._db.rename_collection(request.oldName, request.newName)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("RenameCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def CreateAlias(self, request, context):
        try:
            self._db.create_alias(request.collection_name, request.alias)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("CreateAlias failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def AlterAlias(self, request, context):
        try:
            self._db.alter_alias(request.collection_name, request.alias)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("AlterAlias failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DropAlias(self, request, context):
        try:
            self._db.drop_alias(request.alias)
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("DropAlias failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DescribeAlias(self, request, context):
        try:
            info = self._db.describe_alias(request.alias)
            return milvus_pb2.DescribeAliasResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                alias=info["alias"],
                collection=info["collection"],
            )
        except MilvusLiteError as e:
            return milvus_pb2.DescribeAliasResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("DescribeAlias failed: %s", e)
            return milvus_pb2.DescribeAliasResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def ListAliases(self, request, context):
        try:
            collection_name = request.collection_name or None
            aliases = self._db.list_aliases(collection_name)
            resolved = (
                self._db.resolve_collection_name(collection_name)
                if collection_name else ""
            )
            return milvus_pb2.ListAliasesResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                collection_name=resolved,
                aliases=aliases,
            )
        except MilvusLiteError as e:
            return milvus_pb2.ListAliasesResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("ListAliases failed: %s", e)
            return milvus_pb2.ListAliasesResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def AlterCollection(self, request, context):
        try:
            properties = _kv_pairs_to_dict(getattr(request, "properties", []))
            delete_keys = list(getattr(request, "delete_keys", [])) or None
            self._db.alter_collection_properties(
                request.collection_name,
                properties=properties or None,
                delete_keys=delete_keys,
            )
            return common_pb2.Status(**success_status_kwargs())
        except MilvusLiteError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("AlterCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def LoadPartitions(self, request, context):
        return self._unimplemented(
            context, "LoadPartitions",
            "partition-level load is not supported; use load_collection instead",
        )

    def ReleasePartitions(self, request, context):
        return self._unimplemented(
            context, "ReleasePartitions",
            "partition-level release is not supported; use release_collection instead",
        )

    def ManualCompaction(self, request, context):
        """Compaction runs automatically after flush. pymilvus
        exposes compact() → ManualCompaction, but the engine doesn't
        have an on-demand trigger via the Collection API yet. Return
        success so pymilvus clients don't crash — the effect is already
        achieved by the automatic post-flush compaction."""
        return milvus_pb2.ManualCompactionResponse(
            status=common_pb2.Status(**success_status_kwargs()),
        )

    def GetCompactionState(self, request, context):
        return milvus_pb2.GetCompactionStateResponse(
            status=common_pb2.Status(**success_status_kwargs()),
            state=common_pb2.CompactionState.Completed,
        )

    # ── Helpers ─────────────────────────────────────────────────

    def _build_ids_proto(self, pks, col):
        """Construct an ``IDs`` proto from a list of pk values.

        Picks the int_id or str_id slot based on the pk field's
        DataType. We don't (yet) handle mixed-type pk lists — Milvus
        doesn't either; pks must all be the same type.
        """
        from pymilvus.grpc_gen import schema_pb2 as _schema_pb2
        ids = _schema_pb2.IDs()
        pk_field = next((f for f in col.schema.fields if f.is_primary), None)
        if pk_field is None or not pks:
            return ids
        if pk_field.dtype == DataType.VARCHAR:
            ids.str_id.data.extend([str(p) for p in pks])
        else:
            ids.int_id.data.extend([int(p) for p in pks])
        return ids

    @staticmethod
    def _extract_pks_from_expr(expr: str, col) -> "list | None":
        """If *expr* is the trivial ``<pk_field> in [v1, v2, ...]`` form,
        return the pk list. Otherwise return None to signal "fall back
        to the general query path".

        This pattern is what pymilvus's ``client.get(ids=[...])`` emits
        — recognizing it lets us route directly to ``col.get`` instead
        of doing a full filter pass.
        """
        if not expr:
            return None
        from milvus_lite.search.filter import parse_expr
        from milvus_lite.search.filter.ast import InOp, FieldRef, IntLit, StringLit
        try:
            ast = parse_expr(expr)
        except Exception:
            return None
        if not isinstance(ast, InOp):
            return None
        if not isinstance(ast.field, FieldRef):
            return None
        if ast.field.name != col._pk_name:
            return None
        if ast.negate:
            return None  # "not in" — too broad, fall through to query
        pks: list = []
        for el in ast.values.elements:
            if isinstance(el, IntLit):
                pks.append(el.value)
            elif isinstance(el, StringLit):
                pks.append(el.value)
            else:
                return None  # mixed/non-literal — fall back
        return pks

    @staticmethod
    def _unimplemented(context, rpc_name: str, reason: str = "") -> common_pb2.Status:
        """Build a friendly UNIMPLEMENTED status. Used by stubs that
        need to return a Status-shaped response (some RPCs return
        Status directly; others wrap it in a richer message). The
        gRPC context is also marked so the client sees the proper
        StatusCode."""
        msg = f"MilvusLite does not support {rpc_name}"
        if reason:
            msg += f": {reason}"
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(msg)
        return common_pb2.Status(
            code=_UNEXPECTED_ERROR,
            reason=msg,
        )
