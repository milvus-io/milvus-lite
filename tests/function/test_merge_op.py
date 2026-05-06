"""Tests for MergeOp."""

import math
import pytest

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.merge_op import MergeOp
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD, STAGE_L2_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_L2_RERANK)


def _df(hits_per_query):
    """Build a DataFrame from per-query hit lists."""
    return DataFrame(hits_per_query)


def _hit(pk, score, **extra):
    h = {ID_FIELD: pk, SCORE_FIELD: score}
    h.update(extra)
    return h


# ── RRF ──────────────────────────────────────────────────────


def test_merge_rrf_basic():
    path0 = _df([[_hit(1, 0.9), _hit(2, 0.8)]])
    path1 = _df([[_hit(3, 0.7), _hit(1, 0.6)]])
    op = MergeOp("rrf", rrf_k=60.0)
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    pks = {h[ID_FIELD] for h in chunk}
    assert pks == {1, 2, 3}
    # pk=1 appears in both paths: rank 0 in path0 + rank 1 in path1
    pk1 = next(h for h in chunk if h[ID_FIELD] == 1)
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert abs(pk1[SCORE_FIELD] - expected) < 1e-9


def test_merge_rrf_dedup():
    """Same pk in multiple paths should appear once in output."""
    path0 = _df([[_hit(1, 0.9)]])
    path1 = _df([[_hit(1, 0.5)]])
    op = MergeOp("rrf")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert len(result.chunk(0)) == 1


# ── Weighted ─────────────────────────────────────────────────


def test_merge_weighted_basic():
    path0 = _df([[_hit(1, 0.8), _hit(2, 0.4)]])
    path1 = _df([[_hit(1, 0.6), _hit(3, 0.9)]])
    op = MergeOp("weighted", weights=[0.7, 0.3])
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    assert len(chunk) == 3  # pk 1, 2, 3


def test_merge_weighted_normalization():
    """With norm_score=true, Milvus uses metric-aware atan normalization."""
    path0 = _df([[_hit(1, 10.0), _hit(2, 20.0)]])
    path1 = _df([[_hit(1, 5.0)]])
    op = MergeOp(
        "weighted",
        weights=[0.5, 0.5],
        normalize=True,
        metric_types=["IP", "IP"],
    )
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    pk1 = next(h for h in chunk if h[ID_FIELD] == 1)
    pk2 = next(h for h in chunk if h[ID_FIELD] == 2)
    norm10 = 0.5 + math.atan(10.0) / math.pi
    norm5 = 0.5 + math.atan(5.0) / math.pi
    norm20 = 0.5 + math.atan(20.0) / math.pi
    assert abs(pk1[SCORE_FIELD] - (0.5 * norm10 + 0.5 * norm5)) < 1e-9
    assert abs(pk2[SCORE_FIELD] - (0.5 * norm20)) < 1e-9
    assert pk1[SCORE_FIELD] > pk2[SCORE_FIELD]


# ── Simple strategies ────────────────────────────────────────


def test_merge_max():
    path0 = _df([[_hit(1, 0.3)]])
    path1 = _df([[_hit(1, 0.9)]])
    op = MergeOp("max")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert result.chunk(0)[0][SCORE_FIELD] == 0.9


def test_merge_sum():
    path0 = _df([[_hit(1, 0.3)]])
    path1 = _df([[_hit(1, 0.7)]])
    op = MergeOp("sum")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert abs(result.chunk(0)[0][SCORE_FIELD] - 1.0) < 1e-9


def test_merge_avg():
    path0 = _df([[_hit(1, 0.2)]])
    path1 = _df([[_hit(1, 0.8)]])
    op = MergeOp("avg")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert abs(result.chunk(0)[0][SCORE_FIELD] - 0.5) < 1e-9


# ── Edge cases ───────────────────────────────────────────────


def test_merge_single_input_passthrough():
    df = _df([[_hit(1, 0.5), _hit(2, 0.4)]])
    op = MergeOp("rrf")
    result = op.execute_multi(_ctx(), [df])
    assert result is not df
    assert [h[ID_FIELD] for h in result.chunk(0)] == [1, 2]
    assert result.chunk(0)[0][SCORE_FIELD] == pytest.approx(1.0 / 61.0)


def test_merge_multi_query():
    path0 = _df([[_hit(1, 0.9)], [_hit(2, 0.8)], [_hit(3, 0.7)]])
    path1 = _df([[_hit(4, 0.6)], [_hit(5, 0.5)], [_hit(6, 0.4)]])
    op = MergeOp("rrf")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert result.num_chunks == 3
    assert len(result.chunk(0)) == 2
    assert len(result.chunk(1)) == 2
    assert len(result.chunk(2)) == 2


def test_merge_execute_raises():
    op = MergeOp("rrf")
    with pytest.raises(RuntimeError):
        op.execute(_ctx(), _df([[_hit(1, 0.5)]]))


# ── Weighted: all scores identical (range=0) ─────────────────


def test_merge_weighted_identical_scores():
    path0 = _df([[_hit(1, 5.0), _hit(2, 5.0)]])
    path1 = _df([[_hit(1, 3.0)]])
    op = MergeOp("weighted", weights=[0.6, 0.4], normalize=False)
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    pk1 = next(h for h in chunk if h[ID_FIELD] == 1)
    pk2 = next(h for h in chunk if h[ID_FIELD] == 2)
    assert abs(pk1[SCORE_FIELD] - 4.2) < 1e-9
    assert abs(pk2[SCORE_FIELD] - 3.0) < 1e-9


def test_merge_weighted_without_normalization():
    path0 = _df([[_hit(1, 10.0), _hit(2, 1.0)]])
    path1 = _df([[_hit(1, 3.0), _hit(2, 5.0)]])
    op = MergeOp("weighted", weights=[0.25, 0.75], normalize=False)
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    pk1 = next(h for h in chunk if h[ID_FIELD] == 1)
    pk2 = next(h for h in chunk if h[ID_FIELD] == 2)
    assert abs(pk1[SCORE_FIELD] - 4.75) < 1e-9
    assert abs(pk2[SCORE_FIELD] - 4.0) < 1e-9


def test_merge_weighted_l2_without_normalization_sorts_ascending():
    path0 = _df([[_hit(1, 1.0), _hit(2, 4.0)]])
    path1 = _df([[_hit(1, 2.0), _hit(2, 3.0)]])
    op = MergeOp(
        "weighted",
        weights=[0.5, 0.5],
        normalize=False,
        metric_types=["L2", "L2"],
    )
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    assert op.sort_descending is False
    assert next(h for h in chunk if h[ID_FIELD] == 1)[SCORE_FIELD] == 1.5
    assert next(h for h in chunk if h[ID_FIELD] == 2)[SCORE_FIELD] == 3.5


def test_merge_weighted_mixed_metrics_converts_distance_direction():
    path0 = _df([[_hit(1, 0.0), _hit(2, 10.0)]])  # L2
    path1 = _df([[_hit(1, 0.1), _hit(2, 0.1)]])  # IP
    op = MergeOp(
        "weighted",
        weights=[1.0, 0.0],
        normalize=False,
        metric_types=["L2", "IP"],
    )
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    assert op.sort_descending is True
    assert next(h for h in chunk if h[ID_FIELD] == 1)[SCORE_FIELD] == 1.0
    assert next(h for h in chunk if h[ID_FIELD] == 2)[SCORE_FIELD] < 0.1


# ── Inputs with mismatched num_chunks ────────────────────────


def test_merge_mismatched_num_chunks():
    """All routes must have the same query chunk count."""
    path0 = _df([[_hit(1, 0.9)], [_hit(2, 0.8)]])
    path1 = _df([[_hit(3, 0.7)]])  # only 1 query
    op = MergeOp("rrf")
    with pytest.raises(ValueError, match="same number of chunks"):
        op.execute_multi(_ctx(), [path0, path1])


# ── Weights longer than routes ───────────────────────────────


def test_merge_weighted_extra_weights_ignored():
    """Weighted merge requires exactly one weight per route."""
    path0 = _df([[_hit(1, 0.8)]])
    path1 = _df([[_hit(1, 0.4)]])
    op = MergeOp("weighted", weights=[0.7, 0.3, 0.5])  # 3 weights, 2 routes
    with pytest.raises(ValueError, match="requires 2 weights"):
        op.execute_multi(_ctx(), [path0, path1])
