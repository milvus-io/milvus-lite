"""Executable contract for the independent sparse inner-product index.

SparseIpIndex is a dedicated inner-product implementation. It sums
query_weight * document_weight without normalization or corpus statistics.
BM25 has a separate implementation and tests in test_sparse_inverted.py.

Weights must be finite and non-negative, matching Milvus's
ValidateSparseFloatRows contract. Explicit zeros are ignored. Only positive
scores are hits: empty/all-zero queries and non-overlapping documents do not
produce zero-score fillers. This follows Milvus sparse retrieval, rather than
dense exhaustive top-k semantics. Empty rows are allowed at this internal layer.

Results are (local_ids, distances), shaped (nq, top_k), with distance=-score and
ascending distances. Missing slots are (-1, +inf); ties have no specified order.
Build/search masks intersect, preserving original row IDs. IP scores must be
independent of segment boundaries and unrelated rows.

Public BM25 Function fields accept text only;
tests/engine/test_sparse_field_contract.py covers that separate boundary.
"""

import json

import numpy as np
import pytest

from milvus_lite.index.sparse_ip import SparseIpIndex


def _assert_rows(result, expected, top_k):
    """Assert scores, ordering, shapes, and padding using explicit expected hits."""
    ids, distances = result
    assert ids.shape == distances.shape == (len(expected), top_k)
    assert np.issubdtype(ids.dtype, np.signedinteger)
    assert np.issubdtype(distances.dtype, np.floating)
    for row, hits in enumerate(expected):
        count = len(hits)
        np.testing.assert_array_equal(ids[row, :count], [i for i, _ in hits])
        np.testing.assert_allclose(
            distances[row, :count], [-score for _, score in hits],
            rtol=1e-6, atol=1e-7,
        )
        assert np.all(np.diff(distances[row, :count]) >= 0)
        assert len(set(ids[row, :count])) == count
        assert np.all(ids[row, count:] == -1)
        assert np.all(np.isposinf(distances[row, count:]))


def test_ip_sums_products_for_each_query():
    index = SparseIpIndex()
    index.build([{1: 2.0, 7: 3.0}, {1: 4.0}, {99: 5.0}])

    result = index.search([{1: 0.5, 7: 2.0}, {99: 0.25}], top_k=4)

    _assert_rows(result, [[(0, 7.0), (1, 2.0)], [(2, 1.25)]], top_k=4)


def test_ip_does_not_use_bm25_document_length_ranking():
    """Regression: BM25 incorrectly ranks row 1 above row 0 for these inputs."""
    index = SparseIpIndex()
    index.build([{0: 1.0, 1: 100.0}, {0: 0.5}])

    _assert_rows(index.search([{0: 1.0}], 2), [[(0, 1.0), (1, 0.5)]], 2)


def test_ip_handles_large_noncontiguous_dimension_ids():
    index = SparseIpIndex()
    index.build([{0: 2.0, 2**32 - 2: 3.0}, {2**31: 5.0}])

    _assert_rows(
        index.search([{0: 0.5, 2**32 - 2: 2.0}, {2**31: 0.25}], 3),
        [[(0, 7.0)], [(1, 1.25)]], 3,
    )


@pytest.mark.parametrize("vector,score", [({1: 0.6, 9: 0.8}, 1.0), ({1: 3.0, 9: 4.0}, 25.0)])
def test_ip_self_match_is_squared_norm_not_implicit_cosine(vector, score):
    index = SparseIpIndex()
    index.build([vector])

    _assert_rows(index.search([vector], 1), [[(0, score)]], 1)


def test_ip_scales_linearly_with_query_and_document_weights():
    index = SparseIpIndex()
    index.build([{1: 2.0}, {1: 6.0}])

    _assert_rows(
        index.search([{1: 0.5}, {1: 2.0}], 2),
        [[(1, 3.0), (0, 1.0)], [(1, 12.0), (0, 4.0)]], 2,
    )


def test_ip_score_is_independent_of_document_frequency_and_corpus_size():
    for extra_rows in [[], [{1: 0.25}] * 10, [{99: 100.0}] * 10]:
        index = SparseIpIndex()
        index.build([{1: 2.0}] + extra_rows)
        _assert_rows(index.search([{1: 3.0}], 1), [[(0, 6.0)]], 1)


def test_ip_segment_topk_merge_matches_single_index():
    """Segment-local scores can be merged directly, without IDF calibration."""
    rows = [{1: 2.0, 7: 100.0}, {1: 1.0}, {1: 4.0}, {9: 3.0}]
    whole = SparseIpIndex()
    whole.build(rows)
    expected = [[(2, 2.0), (0, 1.0)]]
    _assert_rows(whole.search([{1: 0.5}], 2), expected, 2)

    for boundaries in [(0, 1, 4), (0, 2, 4), (0, 1, 2, 3, 4)]:
        candidates = []
        for start, end in zip(boundaries, boundaries[1:]):
            segment = SparseIpIndex()
            segment.build(rows[start:end])
            ids, distances = segment.search([{1: 0.5}], 2)
            candidates.extend(
                (start + int(local_id), -float(distance))
                for local_id, distance in zip(ids[0], distances[0]) if local_id >= 0
            )
        assert sorted(candidates, key=lambda hit: -hit[1])[:2] == expected[0]


def test_ip_masks_intersect_before_topk_and_preserve_row_ids():
    index = SparseIpIndex()
    index.build(
        [{1: 100.0}, {1: 10.0}, {1: 3.0}, {1: 1.0}],
        valid_mask=np.array([False, True, True, True]),
    )

    _assert_rows(
        index.search([{1: 2.0}], 1, valid_mask=np.array([True, False, True, True])),
        [[(2, 6.0)]], 1,
    )
    # A per-query filter must not permanently mutate the index.
    _assert_rows(index.search([{1: 2.0}], 4), [[(1, 20.0), (2, 6.0), (3, 2.0)]], 4)


@pytest.mark.parametrize("mask_stage", ["build", "search"])
def test_ip_all_masked_returns_only_padding(mask_stage):
    index = SparseIpIndex()
    mask = np.array([False, False])
    index.build([{1: 1.0}, {1: 2.0}], valid_mask=mask if mask_stage == "build" else None)
    result = index.search([{1: 1.0}], 3, valid_mask=mask if mask_stage == "search" else None)

    _assert_rows(result, [[]], 3)


@pytest.mark.parametrize("query", [{}, {1: 0.0}, {999: 1.0}])
def test_ip_empty_zero_and_nonoverlapping_queries_have_no_hits(query):
    index = SparseIpIndex()
    index.build([{1: 2.0}, {}])

    _assert_rows(index.search([query], 3), [[]], 3)


def test_ip_zero_weights_and_nonoverlapping_rows_do_not_fill_topk():
    index = SparseIpIndex()
    index.build([{1: 0.0}, {1: 2.0, 3: 0.0}, {3: 9.0}, {99: 1.0}, {}])

    _assert_rows(index.search([{1: 0.5, 3: 0.0}], 7), [[(1, 1.0)]], 7)


def test_ip_empty_index_returns_shaped_padding():
    index = SparseIpIndex()
    index.build([])

    _assert_rows(index.search([{1: 1.0}, {}], 3), [[], []], 3)


def test_ip_empty_query_batch_and_zero_topk_keep_result_shape():
    index = SparseIpIndex()
    index.build([{1: 1.0}])

    _assert_rows(index.search([], 3), [], 3)
    _assert_rows(index.search([{1: 1.0}, {}], 0), [[], []], 0)


def test_ip_tied_scores_do_not_require_a_specific_order():
    index = SparseIpIndex()
    index.build([{1: 2.0}, {1: 2.0}, {1: 2.0}, {1: 1.0}])

    ids, distances = index.search([{1: 0.5}], 2)

    assert ids.shape == distances.shape == (1, 2)
    assert len(set(ids[0])) == 2
    assert set(ids[0]) <= {0, 1, 2}
    np.testing.assert_allclose(distances, [[-1.0, -1.0]])


def test_ip_rebuild_discards_old_postings_and_masks():
    index = SparseIpIndex()
    index.build([{1: 100.0}, {2: 2.0}], valid_mask=np.array([True, False]))
    index.build([{2: 3.0}, {2: 4.0}])

    _assert_rows(index.search([{1: 1.0}, {2: 2.0}], 3), [[], [(1, 8.0), (0, 6.0)]], 3)


@pytest.mark.parametrize("top_k", [1, 7, 100])
def test_ip_matches_independent_dense_dot_product(top_k):
    """An independent NumPy oracle catches formula, mask, ranking and padding bugs."""
    rng = np.random.default_rng(20261008)
    documents = rng.uniform(0.1, 3.0, size=(41, 29))
    documents[rng.random(documents.shape) < 0.85] = 0.0
    queries = rng.uniform(0.1, 2.0, size=(5, 29))
    queries[rng.random(queries.shape) < 0.8] = 0.0
    build_mask = rng.random(len(documents)) > 0.2
    search_mask = rng.random(len(documents)) > 0.3

    def sparse_rows(matrix):
        return [{int(i): float(row[i]) for i in np.flatnonzero(row)} for row in matrix]

    index = SparseIpIndex()
    index.build(sparse_rows(documents), valid_mask=build_mask)
    result = index.search(sparse_rows(queries), top_k, valid_mask=search_mask)

    expected = []
    for scores in queries @ documents.T:
        eligible = np.flatnonzero(build_mask & search_mask & (scores > 0))
        # Continuous seeded weights avoid ambiguous ties in this oracle.
        assert len(np.unique(scores[eligible])) == len(eligible)
        ranked = eligible[np.argsort(-scores[eligible])][:top_k]
        expected.append([(int(i), float(scores[i])) for i in ranked])
    _assert_rows(result, expected, top_k)


@pytest.mark.parametrize("weight", [-0.5, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("stage", ["build", "search"])
def test_ip_rejects_negative_or_nonfinite_weights(weight, stage):
    index = SparseIpIndex()
    with pytest.raises(ValueError, match="weight|value|finite|non-negative"):
        if stage == "build":
            index.build([{1: weight}])
        else:
            index.build([{1: 2.0}])
            index.search([{1: weight}], 1)


@pytest.mark.parametrize("stage", ["build", "search"])
def test_ip_rejects_mask_length_mismatch(stage):
    index = SparseIpIndex()
    with pytest.raises(ValueError, match="mask"):
        if stage == "build":
            index.build([{1: 2.0}, {1: 3.0}], valid_mask=np.array([True]))
        else:
            index.build([{1: 2.0}, {1: 3.0}])
            index.search([{1: 1.0}], 1, valid_mask=np.array([True]))


def test_ip_save_load_preserves_type_scores_and_build_mask(tmp_path):
    index = SparseIpIndex()
    index.build([{1: 100.0}, {1: 2.0}, {9: 4.0}], valid_mask=np.array([False, True, True]))
    path = str(tmp_path / "sparse.json")
    index.save(path)

    loaded = type(index).load(path)

    assert isinstance(loaded, type(index))
    assert loaded.index_type == "SPARSE_INVERTED_INDEX"
    _assert_rows(loaded.search([{1: 3.0}, {9: 0.5}], 4), [[(1, 6.0)], [(2, 2.0)]], 4)


@pytest.mark.parametrize("metric", ["BM25", "COSINE", None])
def test_ip_loader_rejects_non_ip_payload(tmp_path, metric):
    index = SparseIpIndex()
    index.build([{1: 1.0}])
    path = tmp_path / "invalid_metric.json"
    index.save(str(path))
    payload = json.loads(path.read_text())
    if metric is None:
        payload.pop("metric_type", None)
    else:
        payload["metric_type"] = metric
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="metric"):
        type(index).load(str(path))
