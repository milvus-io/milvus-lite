"""Dispatch sparse dictionaries to their independent scoring implementations."""

from milvus_lite.index.sparse_inverted import SparseInvertedIndex
from milvus_lite.index.sparse_ip import SparseIpIndex


def create_sparse_index(metric_type: str, params: dict):
    if metric_type == "IP":
        return SparseIpIndex()
    if metric_type == "BM25":
        return SparseInvertedIndex(
            k1=float(params.get("bm25_k1", 1.5)),
            b=float(params.get("bm25_b", 0.75)),
        )
    raise ValueError(f"unsupported sparse metric {metric_type!r}")


def sparse_index_matches(index, metric_type: str, params: dict) -> bool:
    if metric_type == "IP":
        return isinstance(index, SparseIpIndex)
    return (
        metric_type == "BM25" and isinstance(index, SparseInvertedIndex)
        and index.k1 == float(params.get("bm25_k1", 1.5))
        and index.b == float(params.get("bm25_b", 0.75))
    )


def _search_prepared_sparse_index(index, queries, top_k, valid_mask=None):
    """Engine-only dispatch for queries produced by Collection._prepare_sparse_queries.

    IP dictionaries are already validated/canonicalized; BM25 dictionaries are
    already analyzed TFs. Neither scorer mutates the shared query dictionaries.
    """
    search = index._search_prepared if isinstance(index, SparseIpIndex) else index.search
    return search(queries, top_k, valid_mask=valid_mask)
