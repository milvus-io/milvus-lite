"""Ordinary sparse IP scenarios adapted from upstream Milvus client tests.

Source: milvus/tests/python_client/milvus_client at
a11a375e05356420fdeed0e2cc39994312433b69. Individual tests identify the source
method; see docs/sparse-upstream-test-migration.md for the coverage mapping.

Adaptations: use this repository's real milvus_client fixture instead of upstream
Base/ResponseChecker; seeded 640-row data instead of 3000/4000 random rows;
SPARSE_INVERTED_INDEX with default/TAAT_NAIVE and zero drop ratios instead of the
upstream default drop_ratio_search=0.2. Algorithm-specific SINDI/WAND/codec/mmap
claims are not covered. Generic scenarios taken from SINDI fixtures are renamed.

The independent dot-product oracle strengthens upstream ordering-only checks.
The upstream error-code/message contract is retained in explicit strict xfails,
separate from checks of Lite's documented validation errors.
"""

import math

import numpy as np
import pytest
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException


PK = "int64"
VECTOR = "sparse_vector"
ROW_COUNT = 640
DIM = 1000
LIMIT = 10
TIMEOUT = 20


def _vectors(count, *, dim=DIM, seed=19530):
    """Keep common dimensions 0/1 like upstream, with reproducible float32 data."""
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(count):
        dimensions = sorted({0, 1, *map(int, rng.choice(dim, size=24, replace=False))})
        weights = rng.uniform(0.01, 1.0, len(dimensions)).astype(np.float32)
        vectors.append(dict(zip(dimensions, map(float, weights))))
    return vectors


def _rows(vectors):
    return [
        {PK: i, "float": i / 4.0, "varchar": f"row_{i}", VECTOR: vector}
        for i, vector in enumerate(vectors)
    ]


def _ordered_rows():
    # Preserve the upstream inverted-index fixture's distinct IP score pattern.
    return _rows([
        {1: float(np.float32(1.003 ** (ROW_COUNT - i))), 2: 1.0, 3: 0.5, 4: 0.25}
        for i in range(ROW_COUNT)
    ])


def _schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(PK, DataType.INT64, is_primary=True)
    schema.add_field("float", DataType.FLOAT)
    schema.add_field("varchar", DataType.VARCHAR, max_length=65535)
    schema.add_field(VECTOR, DataType.SPARSE_FLOAT_VECTOR)
    return schema


def _create(client, rows, *, params=None, insert=True):
    name = "upstream_sparse_ip"
    client.create_collection(name, schema=_schema(), timeout=TIMEOUT)
    if insert:
        client.insert(name, rows, timeout=TIMEOUT)
        client.flush(name, timeout=TIMEOUT)
    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name=VECTOR, index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
        params={} if params is None else params,
    )
    client.create_index(name, indexes, timeout=TIMEOUT)
    client.load_collection(name, timeout=TIMEOUT)
    return name


@pytest.fixture
def sparse_collection(milvus_client):
    rows = _rows(_vectors(ROW_COUNT))
    return _create(milvus_client, rows), rows


def _search(client, name, queries, *, limit=LIMIT, **kwargs):
    kwargs.setdefault("search_params", {"metric_type": "IP", "params": {}})
    kwargs.setdefault("output_fields", [PK])
    return client.search(
        name, data=queries, anns_field=VECTOR, limit=limit, timeout=TIMEOUT, **kwargs,
    )


def _reference(rows, query):
    # A per-document scan, independent of the production posting-list scorer.
    query = {int(k): float(np.float32(v)) for k, v in query.items()}
    scores = {}
    for row in rows:
        score = math.fsum(
            float(np.float32(weight)) * query.get(dimension, 0.0)
            for dimension, weight in row[VECTOR].items()
        )
        if score > 0:
            scores[row[PK]] = float(np.float32(score))
    return scores


def _assert_exact_ip(results, rows, queries, limit):
    assert len(results) == len(queries)
    for hits, query in zip(results, queries):
        reference = _reference(rows, query)
        expected_scores = sorted(reference.values(), reverse=True)[:limit]
        assert len(hits) == len(expected_scores)
        ids = [hit[PK] for hit in hits]
        assert len(set(ids)) == len(ids)
        scores = [hit["distance"] for hit in hits]
        assert scores == sorted(scores, reverse=True)
        assert scores == pytest.approx(expected_scores, rel=1e-6, abs=1e-7)
        for hit in hits:
            assert hit[PK] in reference
            assert hit["distance"] == pytest.approx(reference[hit[PK]], rel=1e-6, abs=1e-7)
        if expected_scores:
            # Ties at the cutoff may choose any matching ID, but no better row
            # may be omitted. Do not prescribe an incidental tie order.
            cutoff = expected_scores[-1]
            assert {pk for pk, score in reference.items() if score > cutoff} <= set(ids)
            assert all(reference[pk] >= cutoff for pk in ids)


def test_sparse_search_default(milvus_client, sparse_collection):
    """Source: sparse_search.py::test_sparse_search_default."""
    name, rows = sparse_collection
    queries = _vectors(2, seed=19)
    hits = _search(milvus_client, name, queries, output_fields=[VECTOR])
    _assert_exact_ip(hits, rows, queries, LIMIT)
    for group in hits:
        for hit in group:
            assert set(hit["entity"]) == {VECTOR}
            assert hit["entity"][VECTOR] == pytest.approx(rows[hit[PK]][VECTOR])


def test_sparse_search_with_filter(milvus_client, sparse_collection):
    """Source: sparse_search.py::test_sparse_search_with_filter."""
    name, rows = sparse_collection
    queries = _vectors(2, seed=20)
    hits = _search(milvus_client, name, queries, filter=f"{PK} < 100")
    _assert_exact_ip(hits, rows[:100], queries, LIMIT)
    assert all(hit[PK] < 100 for group in hits for hit in group)


def test_sparse_search_output_field(milvus_client, sparse_collection):
    """Source: sparse_search.py::test_sparse_search_output_field."""
    name, rows = sparse_collection
    queries = _vectors(2, seed=21)
    hits = _search(milvus_client, name, queries, output_fields=["float", VECTOR])
    _assert_exact_ip(hits, rows, queries, LIMIT)
    for group in hits:
        for hit in group:
            assert set(hit["entity"]) == {"float", VECTOR}
            assert hit["entity"]["float"] == rows[hit[PK]]["float"]
            assert hit["entity"][VECTOR] == pytest.approx(rows[hit[PK]][VECTOR])


@pytest.mark.parametrize("nq", [1, 100])
def test_sparse_search_different_nq(milvus_client, sparse_collection, nq):
    """Source: sparse_search.py::test_sparse_search_different_nq."""
    name, rows = sparse_collection
    queries = _vectors(nq, seed=22)
    _assert_exact_ip(_search(milvus_client, name, queries), rows, queries, LIMIT)


@pytest.mark.parametrize("metric", ["L2", "COSINE"])
def test_sparse_search_invalid_metric_type(milvus_client, sparse_collection, metric):
    """Source: sparse_search.py::test_sparse_search_invalid_metric_type; Lite error contract."""
    name, _ = sparse_collection
    with pytest.raises(MilvusException) as error:
        _search(milvus_client, name, [{0: 1.0}], search_params={"metric_type": metric})
    assert error.value.code == 6  # Lite's IllegalArgument; upstream requires 1100.
    assert VECTOR in str(error.value)
    assert "IP" in str(error.value) and metric in str(error.value)


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="Known protocol gap: Lite uses IllegalArgument=6 and field-aware text; upstream uses 1100.",
)
@pytest.mark.parametrize("metric", ["L2", "COSINE"])
def test_sparse_invalid_metric_upstream_error_contract(milvus_client, sparse_collection, metric):
    """Retain the upstream error assertions separately rather than silently weakening them."""
    name, _ = sparse_collection
    with pytest.raises(MilvusException) as error:
        _search(milvus_client, name, [{0: 1.0}], search_params={"metric_type": metric})
    assert error.value.code == 1100
    assert f"metric type not match: invalid parameter[expected=IP][actual={metric}]" in str(error.value)


def _invalid_index_error(client, metric):
    name = "upstream_sparse_bad_metric"
    client.create_collection(name, schema=_schema(), timeout=TIMEOUT)
    indexes = client.prepare_index_params()
    indexes.add_index(field_name=VECTOR, index_type="SPARSE_INVERTED_INDEX", metric_type=metric)
    with pytest.raises(MilvusException) as error:
        client.create_index(name, indexes, timeout=TIMEOUT)
    assert client.list_indexes(name, timeout=TIMEOUT) == []
    return error.value


@pytest.mark.parametrize("metric", ["L2", "COSINE"])
def test_sparse_index_rejects_unsupported_metric(milvus_client, metric):
    """Source: sparse_inverted_index.py::test_sparse_index_rejects_unsupported_metric."""
    error = _invalid_index_error(milvus_client, metric)
    assert error.code == 6
    assert VECTOR in str(error)
    assert "IP" in str(error) and metric in str(error)


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="Known CreateIndex protocol gap: Lite returns code 6 instead of upstream 1100.",
)
@pytest.mark.parametrize("metric", ["L2", "COSINE"])
def test_sparse_invalid_index_metric_upstream_error_contract(milvus_client, metric):
    """Retain the exact upstream CreateIndex error contract and no-index side-effect check."""
    error = _invalid_index_error(milvus_client, metric)
    assert error.code == 1100
    assert (
        "only IP&BM25 is the supported metric type for sparse index: "
        "invalid parameter[expected=valid index params][actual=invalid index params]"
    ) in str(error)


@pytest.mark.parametrize("max_dimension_id", [32768, 2**32 - 2])
def test_sparse_index_dim(milvus_client, max_dimension_id):
    """Source: sparse_search.py::test_sparse_index_dim; explicitly exercise the high coordinate."""
    rows = _rows([{0: 0.125, max_dimension_id: (i + 1) / 16.0} for i in range(100)])
    name = _create(milvus_client, rows)
    queries = [{max_dimension_id: 0.5}]
    _assert_exact_ip(_search(milvus_client, name, queries), rows, queries, LIMIT)


@pytest.mark.parametrize("flush_delete", [False, True])
def test_sparse_search_after_delete(milvus_client, sparse_collection, flush_delete):
    """Source: sparse_search.py::test_sparse_search_after_delete; also verify after delete flush."""
    name, rows = sparse_collection
    halfway = len(rows) // 2
    milvus_client.delete(name, filter=f"{PK} < {halfway}", timeout=TIMEOUT)
    if flush_delete:
        milvus_client.flush(name, timeout=TIMEOUT)
    queries = _vectors(2, seed=23)
    hits = _search(milvus_client, name, queries)
    _assert_exact_ip(hits, rows[halfway:], queries, LIMIT)


@pytest.mark.parametrize("limit", [1, 10, 100])
def test_ip_topk_and_score_order(milvus_client, limit):
    """Source: sparse_inverted_index.py::test_ip_sindi_search_topk_and_score_order; exact backend."""
    rows = _ordered_rows()
    name = _create(milvus_client, rows)
    queries = [{1: 1.0}]
    hits = _search(milvus_client, name, queries, limit=limit)
    _assert_exact_ip(hits, rows, queries, limit)
    assert [hit[PK] for hit in hits[0]] == list(range(limit))


@pytest.mark.parametrize("query", [{}, {9999: 1.0}], ids=["empty", "no_overlap"])
def test_sparse_query_without_matches(milvus_client, query):
    """Sources: sparse_inverted_index.py::test_ip_sindi_search_empty_query / no_overlap_query."""
    name = _create(milvus_client, _ordered_rows())
    assert _search(milvus_client, name, [query]) == [[]]


@pytest.mark.parametrize("params", [{}, {"inverted_index_algo": "TAAT_NAIVE"}], ids=["default", "taat_naive"])
def test_sparse_index_search_after_reload(milvus_client, params):
    """Sources: inverted-index default-params and ordinary-algorithm reload tests; supported subset."""
    rows = _ordered_rows()
    name = _create(milvus_client, rows, params=params)
    queries = [{1: 1.0}]
    before = _search(milvus_client, name, queries)
    milvus_client.release_collection(name, timeout=TIMEOUT)
    milvus_client.load_collection(name, timeout=TIMEOUT)
    after = _search(milvus_client, name, queries)
    _assert_exact_ip(before, rows, queries, LIMIT)
    _assert_exact_ip(after, rows, queries, LIMIT)
    assert [hit[PK] for hit in after[0]] == [hit[PK] for hit in before[0]]
    info = milvus_client.describe_index(name, index_name=VECTOR, timeout=TIMEOUT)
    assert info["index_type"] == "SPARSE_INVERTED_INDEX"
    assert info["metric_type"] == "IP"


@pytest.mark.parametrize("filtered_ratio", [0.4, 0.9])
def test_sparse_search_high_filter_ratio(milvus_client, filtered_ratio):
    """Source: sparse_inverted_index.py::test_ip_sindi_search_with_high_filter_ratio; exact backend."""
    rows = _ordered_rows()
    name = _create(milvus_client, rows)
    minimum_id = int(ROW_COUNT * filtered_ratio)
    queries = [{1: 1.0}]
    hits = _search(milvus_client, name, queries, filter=f"{PK} >= {minimum_id}")
    _assert_exact_ip(hits, rows[minimum_id:], queries, LIMIT)
    assert [hit[PK] for hit in hits[0]] == list(range(minimum_id, minimum_id + LIMIT))


@pytest.mark.parametrize("batch_size", [10, 100, 500])
def test_sparse_search_iterator(milvus_client, batch_size):
    """Sources: sparse_search.py::test_sparse_search_iterator and ordered iterator checks."""
    rows = _ordered_rows()
    name = _create(milvus_client, rows)
    query = {1: 1.0}
    iterator = milvus_client.search_iterator(
        name, data=[query], anns_field=VECTOR, batch_size=batch_size, limit=500,
        search_params={"metric_type": "IP", "params": {}}, output_fields=[PK], timeout=TIMEOUT,
    )
    hits = []
    try:
        # Bounded consumption detects failure to terminate without hanging CI.
        for _ in range(math.ceil(500 / batch_size) + 1):
            batch = iterator.next()
            if not batch:
                break
            assert len(batch) <= batch_size
            hits.extend(batch)
        else:
            pytest.fail("sparse iterator did not terminate at its limit")
    finally:
        iterator.close()
    _assert_exact_ip([hits], rows, [query], 500)


def test_sparse_range_search(milvus_client):
    """Source: range_search.py::test_range_search_sparse; bounded deterministic score window."""
    rows = _ordered_rows()
    name = _create(milvus_client, rows)
    query = {1: 1.0}
    radius = rows[100][VECTOR][1]
    range_filter = rows[20][VECTOR][1]
    eligible = [row for row in rows if radius < row[VECTOR][1] <= range_filter]
    hits = _search(
        milvus_client, name, [query], limit=100,
        search_params={"metric_type": "IP", "params": {"radius": radius, "range_filter": range_filter}},
    )
    _assert_exact_ip(hits, eligible, [query], 100)


def test_sparse_upsert_preserves_count_and_updates_scores(milvus_client):
    """Source: upsert.py::test_milvus_client_upsert_sparse_data; additionally change weights."""
    rows = _rows(_vectors(128, seed=24))
    name = _create(milvus_client, rows, insert=False)
    for iteration in range(5):
        updated = [dict(row, **{VECTOR: {0: (iteration + 1) * (i + 1) / 16.0}}) for i, row in enumerate(rows)]
        milvus_client.upsert(name, updated, timeout=TIMEOUT)
        milvus_client.flush(name, timeout=TIMEOUT)
        assert milvus_client.get_collection_stats(name, timeout=TIMEOUT)["row_count"] == len(rows)
        counts = milvus_client.query(name, filter=f"{PK} >= 0", output_fields=["count(*)"], timeout=TIMEOUT)
        assert counts == [{"count(*)": len(rows)}]
        queries = [{0: 1.0}]
        _assert_exact_ip(_search(milvus_client, name, queries), updated, queries, LIMIT)


@pytest.mark.parametrize("sparse_format", [
    "csr_matrix", "csr_array", "csc_matrix", "coo_matrix", "dok_matrix", "lil_matrix", "coo_array",
])
def test_sparse_scipy_insert_and_search(milvus_client, sparse_format):
    """Sources: insert.py::test_milvus_client_insert_sparse_vector_scipy / scipy_to_csr."""
    from scipy import sparse as sp

    vectors = _vectors(128, dim=10000, seed=25)
    rows = _rows(vectors)

    def as_csr(vector):
        indices = sorted(vector)
        weights = np.array([vector[i] for i in indices], dtype=np.float32)
        csr = sp.csr_matrix((weights, indices, [0, len(indices)]), shape=(1, 10000))
        converted = getattr(sp, sparse_format)(csr)
        # Upstream converts non-CSR formats before passing them to pymilvus.
        return converted if sparse_format.startswith("csr_") else converted.tocsr()

    wire_rows = [dict(row, **{VECTOR: as_csr(row[VECTOR])}) for row in rows]
    name = _create(milvus_client, wire_rows)
    queries = _vectors(2, dim=10000, seed=26)
    results = _search(milvus_client, name, [as_csr(query) for query in queries])
    _assert_exact_ip(results, rows, queries, LIMIT)
