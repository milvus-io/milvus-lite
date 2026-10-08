"""Sparse dispatch selects independent scorers and validates cache identity."""

import pytest

from milvus_lite.index.sparse_factory import create_sparse_index, sparse_index_matches
from milvus_lite.index.sparse_inverted import SparseInvertedIndex
from milvus_lite.index.sparse_ip import SparseIpIndex


def test_sparse_dispatch_keeps_scorers_and_state_separate():
    ip = create_sparse_index("IP", {})
    bm25 = create_sparse_index("BM25", {})
    assert isinstance(ip, SparseIpIndex)
    assert isinstance(bm25, SparseInvertedIndex)
    rows = [{1: 2.0, 2: 100.0}, {1: 1.0}]
    ip.build(rows)
    bm25.build(rows)
    assert ip.search([{1: 1.0}], 2)[0].tolist() == [[0, 1]]
    assert bm25.search([{1: 1.0}], 2)[0].tolist() == [[1, 0]]
    bm25.build([{9: 1.0}])
    assert ip.search([{1: 1.0}], 2)[1].tolist() == [[-2.0, -1.0]]


def test_sparse_cache_matches_implementation_and_bm25_parameters():
    ip = create_sparse_index("IP", {})
    bm25 = create_sparse_index("BM25", {"bm25_k1": 2.0, "bm25_b": 0.25})
    assert sparse_index_matches(ip, "IP", {})
    assert not sparse_index_matches(ip, "BM25", {})
    assert not sparse_index_matches(bm25, "IP", {})
    assert not sparse_index_matches(bm25, "BM25", {})
    assert sparse_index_matches(bm25, "BM25", {"bm25_k1": 2, "bm25_b": 0.25})
    assert not sparse_index_matches(None, "IP", {})


@pytest.mark.parametrize("metric", ["COSINE", "L2", "NONE"])
def test_sparse_factory_rejects_other_metrics(metric):
    with pytest.raises(ValueError, match="metric"):
        create_sparse_index(metric, {})
