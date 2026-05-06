"""Tests for search/distance.py — manual computation cross-checks."""

import math

import numpy as np
import pytest

from milvus_lite.search.distance import (
    METRIC_COSINE,
    METRIC_IP,
    METRIC_L2,
    compute_distances,
    cosine_distance,
    ip_distance,
    l2_distance,
)


# ---------------------------------------------------------------------------
# Cosine — single query
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors_zero():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    d = cosine_distance(q, c)
    assert d.shape == (1,)
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_orthogonal_one():
    q = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([[0.0, 1.0]], dtype=np.float32)
    assert cosine_distance(q, c)[0] == pytest.approx(1.0, abs=1e-6)


def test_cosine_opposite_two():
    q = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([[-1.0, 0.0]], dtype=np.float32)
    assert cosine_distance(q, c)[0] == pytest.approx(2.0, abs=1e-6)


def test_cosine_multiple_candidates_order():
    """Closer vector → smaller distance."""
    q = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([
        [1.0, 0.0],     # identical    → 0
        [1.0, 1.0],     # 45°          → 1 - cos(45°) ≈ 0.293
        [0.0, 1.0],     # orthogonal   → 1
    ], dtype=np.float32)
    d = cosine_distance(q, c)
    assert d[0] < d[1] < d[2]


def test_cosine_zero_vector_no_nan():
    q = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([[0.0, 0.0]], dtype=np.float32)
    d = cosine_distance(q, c)
    assert not np.isnan(d).any()


# ---------------------------------------------------------------------------
# Cosine — batch query
# ---------------------------------------------------------------------------

def test_cosine_batch_shape():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    d = cosine_distance(q, c)
    assert d.shape == (2, 3)
    assert d[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert d[1, 1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------

def test_l2_identical_zero():
    q = np.array([1.0, 2.0], dtype=np.float32)
    c = np.array([[1.0, 2.0]], dtype=np.float32)
    assert l2_distance(q, c)[0] == pytest.approx(0.0, abs=1e-6)


def test_l2_manual_3_4_5():
    q = np.array([0.0, 0.0], dtype=np.float32)
    c = np.array([[3.0, 4.0]], dtype=np.float32)
    assert l2_distance(q, c)[0] == pytest.approx(5.0, abs=1e-6)


def test_l2_batch():
    q = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    c = np.array([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32)
    d = l2_distance(q, c)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(5.0, abs=1e-6)
    assert d[1, 0] == pytest.approx(math.sqrt(4 + 9), abs=1e-6)  # diff=[2,3]


# ---------------------------------------------------------------------------
# IP
# ---------------------------------------------------------------------------

def test_ip_negative_dot():
    q = np.array([1.0, 2.0], dtype=np.float32)
    c = np.array([[3.0, 4.0]], dtype=np.float32)
    # dot = 3 + 8 = 11; distance = -11
    assert ip_distance(q, c)[0] == pytest.approx(-11.0, abs=1e-6)


def test_ip_smaller_means_more_similar():
    """Vectors with larger dot products should have smaller distances."""
    q = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([
        [10.0, 0.0],    # dot=10  → -10
        [1.0, 0.0],     # dot=1   → -1
        [0.0, 1.0],     # dot=0   → 0
    ], dtype=np.float32)
    d = ip_distance(q, c)
    assert d[0] < d[1] < d[2]


def test_ip_batch():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    d = ip_distance(q, c)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(-2.0, abs=1e-6)
    assert d[1, 1] == pytest.approx(-5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_distances dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", [METRIC_COSINE, METRIC_L2, METRIC_IP])
def test_compute_distances_dispatch(metric):
    q = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = compute_distances(q, c, metric)
    assert d.shape == (2,)


def test_compute_distances_unsupported_raises():
    q = np.array([1.0], dtype=np.float32)
    c = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported metric_type"):
        compute_distances(q, c, "TANIMOTO")


# ---------------------------------------------------------------------------
# Empty candidates
# ---------------------------------------------------------------------------

def test_cosine_empty_candidates():
    q = np.array([1.0, 0.0], dtype=np.float32)
    c = np.zeros((0, 2), dtype=np.float32)
    d = cosine_distance(q, c)
    assert d.shape == (0,)


def test_l2_batch_empty_candidates():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c = np.zeros((0, 2), dtype=np.float32)
    d = l2_distance(q, c)
    assert d.shape == (2, 0)
