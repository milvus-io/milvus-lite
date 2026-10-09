"""Public sparse field contracts from docs/sparse-vector-design.md.

They exercise the real Engine, including validation before ingestion/WAL writes.
Low-level BM25 dictionaries remain covered in tests/index/test_sparse_inverted.py.
"""

from contextlib import ExitStack, closing
import copy
import json
import math
from unittest.mock import Mock

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.exceptions import SchemaValidationError
from milvus_lite.index.spec import IndexSpec
from milvus_lite.schema.types import (
    CollectionSchema, DataType, FieldSchema, Function, FunctionType,
)


def _schema():
    return CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("dense", DataType.FLOAT_VECTOR, dim=2),
            FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("label", DataType.VARCHAR, max_length=64),
            FieldSchema("text", DataType.VARCHAR, max_length=256, enable_analyzer=True,
                        analyzer_params={"tokenizer": "standard", "filter": [
                            {"type": "stop", "stop_words": ["alpha"]},
                        ]}),
            FieldSchema("bm25", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("other_text", DataType.VARCHAR, max_length=256,
                        enable_analyzer=True),
            FieldSchema("other_bm25", DataType.SPARSE_FLOAT_VECTOR),
        ],
        functions=[
            Function("first", FunctionType.BM25, ["text"], ["bm25"]),
            Function("second", FunctionType.BM25, ["other_text"], ["other_bm25"]),
        ],
    )


def _row(pk=1, **updates):
    row = {
        "id": pk, "dense": [1.0, 0.0], "sparse": {0: 1.0, 1: 100.0},
        "label": "keep", "text": "alpha beta", "other_text": "alpha gamma",
    }
    row.update(updates)
    return row


def _index(col, field, metric, **params):
    col.create_index(field, {
        "index_type": "SPARSE_INVERTED_INDEX", "metric_type": metric, "params": params,
    })


@pytest.fixture
def make_collection(tmp_path):
    with ExitStack() as stack:
        def make(name="c", schema=None):
            return stack.enter_context(closing(Collection(name, str(tmp_path / name), schema or _schema())))
        yield make


@pytest.fixture
def col(make_collection):
    return make_collection()


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("field,metric,query", [("sparse", "IP", {0: 1.0}), ("bm25", "BM25", "beta")])
def test_search_metric_is_resolved_from_target_field(col, indexed, explicit, field, metric, query):
    col.insert([_row(), _row(2, sparse={0: 0.5}, text="delta")])
    # A dense index must not supply the sparse field's default metric.
    col.create_index("dense", {"index_type": "BRUTE_FORCE", "metric_type": "COSINE"})
    if indexed:
        _index(col, field, metric)
    col.load()

    [hits] = col.search([query], anns_field=field, top_k=2,
                        **({"metric_type": metric} if explicit else {}))

    if metric == "IP":
        assert [hit["id"] for hit in hits] == [1, 2]
        assert [hit["distance"] for hit in hits] == pytest.approx([1.0, 0.5])
    else:
        assert [hit["id"] for hit in hits] == [1]
        assert hits[0]["distance"] < 0  # Preserve the existing BM25 public sign.


@pytest.mark.parametrize("field,metric,queries", [
    ("sparse", "IP", ["beta"]),
    ("bm25", "BM25", [{0: 1.0}]),
    ("sparse", "IP", [{0: 1.0}, "beta"]),
    ("bm25", "BM25", ["beta", {0: 1.0}]),
])
def test_query_input_must_match_field_purpose(col, field, metric, queries):
    col.insert([_row()])
    with pytest.raises(SchemaValidationError, match=field):
        col.search(queries, anns_field=field, metric_type=metric)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("field,correct,wrong,query", [
    ("sparse", "IP", "BM25", {0: 1.0}),
    ("sparse", "IP", "COSINE", {0: 1.0}),
    ("sparse", "IP", "L2", {0: 1.0}),
    ("bm25", "BM25", "IP", "beta"),
])
def test_conflicting_search_metric_is_rejected(col, indexed, field, correct, wrong, query):
    col.insert([_row()])
    if indexed:
        _index(col, field, correct)
    col.load()
    with pytest.raises(SchemaValidationError, match=field):
        col.search([query], anns_field=field, metric_type=wrong)


@pytest.mark.parametrize("field,index_type,metric,params", [
    ("sparse", "SPARSE_INVERTED_INDEX", "BM25", {}),
    ("bm25", "SPARSE_INVERTED_INDEX", "IP", {}),
    ("sparse", "SPARSE_INVERTED_INDEX", "COSINE", {}),
    ("sparse", "SPARSE_INVERTED_INDEX", "L2", {}),
    ("sparse", "SPARSE_INVERTED_INDEX", "NONE", {}),
    ("sparse", "SPARSE_INVERTED_INDEX", "unknown", {}),
    ("sparse", "HNSW", "IP", {}),
    ("sparse", "SPARSE_INVERTED_INDEX", "IP", {"bm25_k1": 1.2}),
    ("sparse", "SPARSE_INVERTED_INDEX", "IP", {"bm25_b": 0.5}),
])
def test_invalid_index_request_does_not_persist_configuration(col, field, index_type, metric, params):
    with pytest.raises(SchemaValidationError, match=field):
        col.create_index(field, {"index_type": index_type, "metric_type": metric, "params": params})
    assert not col.has_index(field)
    assert col._manifest.index_specs == {}


def test_bm25_uses_analyzer_of_requested_output_field(col):
    col.insert([_row()])
    assert col.search(["alpha"], anns_field="bm25", metric_type="BM25") == [[]]
    [hits] = col.search(["alpha"], anns_field="other_bm25", metric_type="BM25")
    assert [hit["id"] for hit in hits] == [1]


@pytest.mark.parametrize("operation", ["insert", "upsert"])
def test_caller_cannot_supply_bm25_outputs_before_ingestion(col, monkeypatch, operation):
    col.insert([_row()])
    wal_write = Mock(wraps=col._wal.write_insert)
    monkeypatch.setattr(col._wal, "write_insert", wal_write)
    with pytest.raises(SchemaValidationError, match="bm25"):
        getattr(col, operation)([_row(2), _row(1, label="tampered", bm25={0: 99.0})])
    wal_write.assert_not_called()
    assert col.get([2]) == []
    assert col.get([1])[0]["label"] == "keep"


def test_partial_upsert_regenerates_internal_bm25_outputs(col):
    col.insert([_row()])
    col.flush()
    col._wait_for_bg()
    col.upsert([{"id": 1, "label": "changed"}])
    [hits] = col.search(["beta"], anns_field="bm25", metric_type="BM25", output_fields=["label"])
    assert [hit["id"] for hit in hits] == [1]
    assert hits[0]["entity"]["label"] == "changed"
    col.upsert([{"id": 1, "text": "omega"}])
    assert col.search(["beta"], anns_field="bm25", metric_type="BM25") == [[]]
    [hits] = col.search(["omega"], anns_field="bm25", metric_type="BM25")
    assert [hit["id"] for hit in hits] == [1]


@pytest.mark.parametrize("vector", [
    {-1: 1.0}, {2**32 - 1: 1.0}, {2**32: 1.0}, {True: 1.0},
    {0: -1.0}, {0: float("nan")}, {0: float("inf")}, {0: 1e40}, {0: True},
])
@pytest.mark.parametrize("operation", ["insert", "search"])
def test_public_sparse_values_are_validated_consistently(col, operation, vector):
    col.insert([_row()])
    with pytest.raises(SchemaValidationError, match="sparse"):
        if operation == "insert":
            col.insert([_row(2, sparse=vector)])
        else:
            col.search([vector], anns_field="sparse", metric_type="IP")


@pytest.mark.parametrize("vector", [{}, {0: 0.0}, {0: 1e-50}])
def test_empty_stored_embedding_is_rejected(col, vector):
    with pytest.raises(SchemaValidationError, match="sparse"):
        col.insert([_row(2, sparse=vector)])


@pytest.mark.parametrize("vector", [{}, {0: 0.0}, {0: 1e-50}])
def test_empty_query_has_no_hits(col, vector):
    col.insert([_row()])
    assert col.search([vector], anns_field="sparse", metric_type="IP") == [[]]


def test_unproduced_function_output_flag_is_invalid(make_collection):
    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("orphan", DataType.SPARSE_FLOAT_VECTOR, is_function_output=True),
    ])
    with pytest.raises(SchemaValidationError, match="orphan"):
        make_collection(schema=schema)


def test_bm25_cache_tracks_index_parameters_and_drop(col):
    col.insert([_row(text="beta beta omega"), _row(2, text="delta")])
    col.flush()
    col._wait_for_bg()

    def score():
        return col.search(["beta"], anns_field="bm25", metric_type="BM25")[0][0]["distance"]

    before = score()  # Warm the no-index cache.
    _index(col, "bm25", "BM25", bm25_k1=3.0, bm25_b=0.0)
    col.load()
    # N=2, df=1, tf=2, dl=3, avgdl=2, k1=3, b=0.
    assert score() == pytest.approx(-math.log(2.0) * 8.0 / 5.0)
    assert score() != pytest.approx(before)
    col.release()
    col.drop_index("bm25")
    col.load()
    assert score() == pytest.approx(before)


@pytest.mark.parametrize("field,legacy_metric,correct", [
    ("sparse", "BM25", "IP"), ("bm25", "IP", "BM25"),
])
def test_legacy_invalid_spec_is_inspectable_and_repairable(make_collection, field, legacy_metric, correct):
    col = make_collection()
    # Simulate old persisted configuration without relying on new create_index
    # accepting an invalid combination. Opening must remain possible for repair.
    col._manifest.set_index_spec(IndexSpec(field, "SPARSE_INVERTED_INDEX", legacy_metric, {}))
    col._manifest.save()
    col.close()
    col = make_collection()
    assert col.has_index(field)
    assert col.get_index_info(field)["metric_type"] == legacy_metric
    with pytest.raises(SchemaValidationError, match=field) as error:
        col.load()
    assert legacy_metric in str(error.value)
    assert correct in str(error.value)
    assert "drop" in str(error.value).lower()
    col.release()
    col.drop_index(field)
    _index(col, field, correct)
    col.load()
    assert col.get_index_info(field)["metric_type"] == correct


@pytest.mark.parametrize("state", ["memtable", "flushed", "reloaded", "reopened", "compacted"])
def test_ip_scores_survive_storage_lifecycle(make_collection, state):
    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
    ])
    col = make_collection(schema=schema)
    _index(col, "sparse", "IP")
    for i in range(4):
        col.insert([{"id": i, "sparse": {0: float(i + 1), 100 + i: 20.0}}])
        if state == "compacted":
            col.flush()
    if state != "memtable":
        col.flush()
        col._wait_for_bg()
    if state == "compacted":
        assert len(col._manifest.get_data_files("_default")) == 1
    if state == "reopened":
        col.close()
        col = make_collection(schema=schema)
    col.load()
    if state == "reloaded":
        col.search([{0: 0.5}], anns_field="sparse", metric_type="IP")
        col.release()
        col.load()
    [hits] = col.search([{0: 0.5}], anns_field="sparse", metric_type="IP", top_k=4)
    assert [hit["id"] for hit in hits] == [3, 2, 1, 0]
    assert [hit["distance"] for hit in hits] == pytest.approx([2.0, 1.5, 1.0, 0.5])


def test_ip_visibility_precedes_topk_across_partitions_and_versions(col):
    col.create_partition("other")
    col.insert([_row(1, sparse={0: 100.0}), _row(2, sparse={0: 90.0}), _row(3, sparse={0: 3.0})])
    col.flush()
    col._wait_for_bg()
    col.insert([_row(1, sparse={0: 0.5})])
    col.delete([2])
    col.insert([_row(4, sparse={0: 200.0})], partition_name="other")

    [hits] = col.search([{0: 1.0}], anns_field="sparse", metric_type="IP",
                        partition_names=["_default"], expr="label == 'keep'", top_k=2)
    assert [hit["id"] for hit in hits] == [3, 1]
    assert [hit["distance"] for hit in hits] == pytest.approx([3.0, 0.5])


@pytest.mark.parametrize("state", ["memtable", "flushed", "reloaded", "reopened"])
def test_ip_and_bm25_indexes_remain_independent(make_collection, state):
    """Alternate real searches; do not couple the assertion to cache internals."""
    col = make_collection()
    col.insert([_row(text="beta beta omega"), _row(2, sparse={0: 0.5}, text="delta")])
    _index(col, "sparse", "IP")
    _index(col, "bm25", "BM25")
    col.load()

    def search_both(collection):
        # Reversing order on the second round catches shared mutable scorer state.
        results = {}
        for field in ["sparse", "bm25", "bm25", "sparse"]:
            query, metric = ({0: 1.0}, "IP") if field == "sparse" else ("beta", "BM25")
            [hits] = collection.search([query], anns_field=field, metric_type=metric, top_k=2)
            results.setdefault(field, []).append(hits)
        return results

    search_both(col)  # Warm both caches before any lifecycle transition.
    if state != "memtable":
        col.flush()
        col._wait_for_bg()
        search_both(col)  # Warm the immutable-segment caches too.
    if state == "reloaded":
        col.release()
        col.load()
    elif state == "reopened":
        col.close()
        col = make_collection()
        col.load()

    results = search_both(col)
    for hits in results["sparse"]:
        assert [hit["id"] for hit in hits] == [1, 2]
        assert [hit["distance"] for hit in hits] == pytest.approx([1.0, 0.5])
    # Equal corpus before/after a single flush: N=2, df=1, tf=2, dl=3, avgdl=2.
    bm25_score = math.log(2.0) * 5.0 / (2.0 + 1.5 * (0.25 + 0.75 * 3.0 / 2.0))
    for hits in results["bm25"]:
        assert [hit["id"] for hit in hits] == [1]
        assert hits[0]["distance"] == pytest.approx(-bm25_score)


def test_sparse_nullable_and_defaults_survive_flush_and_reopen(make_collection):
    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("optional", DataType.SPARSE_FLOAT_VECTOR, nullable=True),
        FieldSchema("defaulted", DataType.SPARSE_FLOAT_VECTOR, default_value={0: 0.5}),
    ])
    col = make_collection(schema=schema)
    col.insert([{"id": 1}, {"id": 2, "optional": {0: 2.0}}])
    col.close()
    col = make_collection(schema=schema)
    assert col.get([1])[0]["optional"] is None
    assert col.search([{0: 1.0}], anns_field="optional")[0][0]["distance"] == pytest.approx(2.0)
    hits = col.search([{0: 1.0}], anns_field="defaulted")[0]
    assert {hit["id"] for hit in hits} == {1, 2}
    assert [hit["distance"] for hit in hits] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("default", [{}, {0: -1.0}, {0: float("inf")}, {0: 1e40}])
def test_sparse_invalid_defaults_rejected_at_schema_creation(make_collection, default):
    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR, default_value=default),
    ])
    with pytest.raises(SchemaValidationError, match="sparse"):
        make_collection(schema=schema)


@pytest.mark.parametrize("params", [
    {"drop_ratio_search": 0.2}, {"inverted_index_algo": "DAAT_WAND"}, {"bm25_b": 0.5},
])
def test_sparse_search_rejects_unsupported_tuning(col, params):
    col.insert([_row()])
    with pytest.raises(SchemaValidationError, match="sparse"):
        col.search([{0: 1.0}], anns_field="sparse", search_params=params)


def test_sparse_lowercase_metrics_are_normalized(col):
    col.insert([_row(sparse={0: 2.0})])
    _index(col, "sparse", "ip")
    col.load()
    assert col.get_index_info("sparse")["metric_type"] == "IP"
    assert col.search([{0: 0.5}], anns_field="sparse", metric_type="ip")[0][0]["distance"] == 1.0


def test_sparse_failed_batch_does_not_write_wal_or_mutate_caller(col, monkeypatch):
    records = [_row(sparse={0: 0.1, 9: 0.0}), _row(2, sparse={0: -1.0})]
    original = copy.deepcopy(records)
    wal_write = Mock(wraps=col._wal.write_insert)
    monkeypatch.setattr(col._wal, "write_insert", wal_write)
    with pytest.raises(SchemaValidationError, match="sparse"):
        col.insert(records)
    wal_write.assert_not_called()
    assert col.get([1, 2]) == []
    assert all("bm25" not in record for record in records)
    assert records == original


def test_release_one_sparse_field_keeps_other_cached_index(col):
    col.insert([_row()])
    col.flush()
    col._wait_for_bg()
    col.search([{0: 1.0}], anns_field="sparse")
    col.search(["beta"], anns_field="bm25")
    [segment] = col._segments_snapshot()
    bm25_index = segment.sparse_indexes["bm25"]
    segment.release_index("sparse")
    assert "sparse" not in segment.sparse_indexes
    assert segment.sparse_indexes["bm25"] is bm25_index
    segment.release_index()
    assert segment.sparse_indexes == {}


@pytest.mark.parametrize("flush_new_version", [False, True])
def test_partition_filter_does_not_resurrect_an_older_sparse_version(col, flush_new_version):
    col.create_partition("other")
    col.insert([_row()])
    col.flush()
    col._wait_for_bg()
    col.insert([_row(sparse={0: 0.5})], partition_name="other")
    if flush_new_version:
        col.flush()
        col._wait_for_bg()
    assert col.search([{0: 1.0}], anns_field="sparse", partition_names=["_default"]) == [[]]
    hits = col.search([{0: 1.0}], anns_field="sparse", partition_names=["other"])[0]
    assert [hit["id"] for hit in hits] == [1]
    assert hits[0]["distance"] == pytest.approx(0.5)


@pytest.mark.parametrize("segment_count", [0, 3])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_ip_queries_are_normalized_once_across_sources(make_collection, monkeypatch, segment_count, warm_cache):
    """A cache miss may normalize documents, but must not re-prepare each query."""
    import milvus_lite.index.sparse_ip as ip_module
    import milvus_lite.schema.sparse as sparse_module

    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
    ])
    col = make_collection(schema=schema)
    _index(col, "sparse", "IP")
    for i in range(segment_count):
        col.insert([{"id": i, "sparse": {0: float(i + 1)}}])
        col.flush()
        col._wait_for_bg()
    col.insert([{"id": segment_count, "sparse": {0: float(segment_count + 1)}}])
    col.load()
    assert len(col._segments_snapshot()) == segment_count
    queries = [{i: weight for i in range(256)} for weight in (0.1, 0.2)]
    original_queries = copy.deepcopy(queries)
    if warm_cache:
        col.search(queries, anns_field="sparse", top_k=1)

    normalization = Mock(wraps=sparse_module.normalize_sparse_vector)
    monkeypatch.setattr(sparse_module, "normalize_sparse_vector", normalization)
    monkeypatch.setattr(ip_module, "normalize_sparse_vector", normalization)

    results = col.search(queries, anns_field="sparse", top_k=1)

    # Documents have one coordinate; queries have 256, so building cold caches
    # or the temporary MemTable index does not inflate this query-only count.
    query_calls = [call for call in normalization.call_args_list if len(call.args[0]) == 256]
    assert len(query_calls) == len(queries)
    assert queries == original_queries
    for hits, weight in zip(results, (0.1, 0.2)):
        assert [hit["id"] for hit in hits] == [segment_count]
        assert hits[0]["distance"] == pytest.approx((segment_count + 1) * weight)


def test_bm25_insert_isolates_caller_records_and_nested_payloads(make_collection):
    schema = _schema()
    schema.fields[0].auto_id = True
    schema.fields.extend([
        FieldSchema("metadata", DataType.JSON),
        FieldSchema("tags", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=8),
        FieldSchema("fallback", DataType.JSON, default_value={"values": []}),
    ])
    col = make_collection(schema=schema)
    record = _row(
        dense=[0.1, 0.2], sparse={0: 0.1, 9: 0.0},
        metadata={"values": [1, 2]}, tags=["original"],
    )
    record.pop("id")
    original = copy.deepcopy(record)
    pks = col.insert([record])
    assert record == original  # No auto ID, defaults, generated output, or normalization leaks.

    # Arrow/WAL must own the inserted values after insert returns, even though
    # the ingestion pipeline can reuse read-only nested payloads while executing.
    record["dense"][0] = 99.0
    record["sparse"][0] = 99.0
    record["metadata"]["values"].append(99)
    record["tags"].append("changed")
    col.flush()
    col._wait_for_bg()
    [stored] = col.get(pks)
    assert stored["dense"] == pytest.approx([0.1, 0.2])
    assert stored["tags"] == ["original"]
    assert json.loads(stored["metadata"]) == {"values": [1, 2]}
    assert json.loads(stored["fallback"]) == {"values": []}
    assert schema.fields[-1].default_value == {"values": []}
    [hits] = col.search([{0: 1.0}], anns_field="sparse")
    assert hits[0]["distance"] == pytest.approx(0.1)
    assert col.search(["beta"], anns_field="bm25")[0][0]["id"] == pks[0]
