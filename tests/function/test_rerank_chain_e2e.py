"""End-to-end tests for rerank chains."""

import pytest

from milvus_lite.function.builder import build_hybrid_rerank_chain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD


def _hit(pk, score, **extra):
    h = {ID_FIELD: pk, SCORE_FIELD: score}
    h.update(extra)
    return h


def test_rrf_chain_e2e():
    chain = build_hybrid_rerank_chain("rrf", {"k": 60}, {"limit": 2})
    path0 = DataFrame([[_hit(1, 0.9), _hit(2, 0.8), _hit(3, 0.7)]])
    path1 = DataFrame([[_hit(2, 0.6), _hit(4, 0.5)]])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    assert len(chunk) == 2  # limit=2
    # pk=2 appears in both routes → highest RRF score
    pks = [h[ID_FIELD] for h in chunk]
    assert 2 in pks
    # Only $id and $score columns remain (Select)
    assert set(chunk[0].keys()) == {ID_FIELD, SCORE_FIELD}


def test_weighted_chain_e2e():
    chain = build_hybrid_rerank_chain(
        "weighted", {"weights": [0.7, 0.3]}, {"limit": 3}
    )
    path0 = DataFrame([[_hit(1, 0.9), _hit(2, 0.1)]])
    path1 = DataFrame([[_hit(1, 0.1), _hit(3, 0.9)]])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    assert len(chunk) == 3
    # Results sorted by score desc
    scores = [h[SCORE_FIELD] for h in chunk]
    assert scores == sorted(scores, reverse=True)


def test_weighted_conflicting_rankings_weight_change_flips_top1():
    path0 = DataFrame([[
        _hit("A", 0.95),
        _hit("B", 0.30),
    ]])
    path1 = DataFrame([[
        _hit("B", 0.90),
        _hit("A", 0.20),
    ]])

    dense_heavy = build_hybrid_rerank_chain(
        "weighted",
        {"weights": [0.8, 0.2]},
        {"limit": 2, "metric_types": ["IP", "IP"]},
    )
    sparse_heavy = build_hybrid_rerank_chain(
        "weighted",
        {"weights": [0.2, 0.8]},
        {"limit": 2, "metric_types": ["IP", "IP"]},
    )

    dense_chunk = dense_heavy.execute(path0, path1).chunk(0)
    sparse_chunk = sparse_heavy.execute(path0, path1).chunk(0)

    assert dense_chunk[0][ID_FIELD] == "A"
    assert dense_chunk[0][SCORE_FIELD] == pytest.approx(0.8 * 0.95 + 0.2 * 0.20)
    assert dense_chunk[1][SCORE_FIELD] == pytest.approx(0.8 * 0.30 + 0.2 * 0.90)
    assert sparse_chunk[0][ID_FIELD] == "B"
    assert sparse_chunk[0][SCORE_FIELD] == pytest.approx(0.2 * 0.30 + 0.8 * 0.90)
    assert sparse_chunk[1][SCORE_FIELD] == pytest.approx(0.2 * 0.95 + 0.8 * 0.20)


def test_rrf_conflicting_rankings_duplicate_pk_rises_to_top():
    chain = build_hybrid_rerank_chain("rrf", {"k": 60}, {"limit": 3})
    path0 = DataFrame([[
        _hit("A", 0.95),
        _hit("X", 0.90),
    ]])
    path1 = DataFrame([[
        _hit("B", 0.95),
        _hit("A", 0.90),
    ]])

    chunk = chain.execute(path0, path1).chunk(0)
    scores = {hit[ID_FIELD]: hit[SCORE_FIELD] for hit in chunk}

    assert [hit[ID_FIELD] for hit in chunk] == ["A", "B", "X"]
    assert scores["A"] == pytest.approx(1.0 / 61.0 + 1.0 / 62.0)
    assert scores["B"] == pytest.approx(1.0 / 61.0)
    assert scores["X"] == pytest.approx(1.0 / 62.0)


def test_rrf_single_route_preserves_route_order_with_explainable_scores():
    chain = build_hybrid_rerank_chain("rrf", {"k": 60}, {"limit": 3})
    path0 = DataFrame([[
        _hit("A", 0.95),
        _hit("B", 0.90),
        _hit("C", 0.85),
    ]])

    chunk = chain.execute(path0).chunk(0)

    assert [hit[ID_FIELD] for hit in chunk] == ["A", "B", "C"]
    assert [hit[SCORE_FIELD] for hit in chunk] == pytest.approx([
        1.0 / 61.0,
        1.0 / 62.0,
        1.0 / 63.0,
    ])


def test_weighted_multi_route_dedup_score_is_weighted_sum():
    chain = build_hybrid_rerank_chain(
        "weighted",
        {"weights": [0.25, 0.75]},
        {"limit": 3, "metric_types": ["IP", "IP"]},
    )
    path0 = DataFrame([[
        _hit("A", 0.80),
        _hit("B", 0.60),
    ]])
    path1 = DataFrame([[
        _hit("A", 0.40),
        _hit("C", 0.70),
    ]])

    chunk = chain.execute(path0, path1).chunk(0)
    scores = {hit[ID_FIELD]: hit[SCORE_FIELD] for hit in chunk}

    assert [hit[ID_FIELD] for hit in chunk] == ["C", "A", "B"]
    assert scores["A"] == pytest.approx(0.25 * 0.80 + 0.75 * 0.40)
    assert scores["B"] == pytest.approx(0.25 * 0.60)
    assert scores["C"] == pytest.approx(0.75 * 0.70)


def test_weighted_l2_no_norm_chain_sorts_ascending():
    chain = build_hybrid_rerank_chain(
        "weighted",
        {"weights": [1.0], "norm_score": False},
        {"limit": 2, "metric_types": ["L2"]},
    )
    path0 = DataFrame([[_hit(1, 0.2), _hit(2, 1.0), _hit(3, 3.0)]])
    result = chain.execute(path0)
    chunk = result.chunk(0)
    assert [h[ID_FIELD] for h in chunk] == [1, 2]
    assert [h[SCORE_FIELD] for h in chunk] == [0.2, 1.0]


def test_weighted_l2_norm_chain_sorts_descending():
    chain = build_hybrid_rerank_chain(
        "weighted",
        {"weights": [1.0], "norm_score": True},
        {"limit": 2, "metric_types": ["L2"]},
    )
    path0 = DataFrame([[_hit(1, 0.2), _hit(2, 1.0), _hit(3, 3.0)]])
    result = chain.execute(path0)
    chunk = result.chunk(0)
    assert [h[ID_FIELD] for h in chunk] == [1, 2]
    assert chunk[0][SCORE_FIELD] > chunk[1][SCORE_FIELD]


def test_rrf_multi_query():
    chain = build_hybrid_rerank_chain("rrf", {}, {"limit": 10})
    path0 = DataFrame([
        [_hit(1, 0.9)],
        [_hit(2, 0.8)],
    ])
    path1 = DataFrame([
        [_hit(3, 0.7)],
        [_hit(4, 0.6)],
    ])
    result = chain.execute(path0, path1)
    assert result.num_chunks == 2
    assert len(result.chunk(0)) == 2
    assert len(result.chunk(1)) == 2


def test_rrf_with_offset():
    chain = build_hybrid_rerank_chain("rrf", {}, {"limit": 1, "offset": 1})
    path0 = DataFrame([[_hit(1, 0.9), _hit(2, 0.8), _hit(3, 0.7)]])
    path1 = DataFrame([[]])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    assert len(chunk) == 1
    # Should be the 2nd result (offset=1)
    assert chunk[0][ID_FIELD] != 1  # not the top result


def test_chain_with_group_by_e2e():
    chain = build_hybrid_rerank_chain(
        "rrf", {},
        {"limit": 2, "group_by_field": "cat", "group_size": 1},
    )
    path0 = DataFrame([[
        _hit(1, 0.9, cat="A"),
        _hit(2, 0.8, cat="A"),
        _hit(3, 0.7, cat="B"),
    ]])
    path1 = DataFrame([[]])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    # 2 groups × 1 per group = 2 results
    assert len(chunk) == 2
    cats = {h["cat"] for h in chunk}
    assert cats == {"A", "B"}
