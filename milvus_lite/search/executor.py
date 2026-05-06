"""Vector search executor — bitmap pipeline + distance + top-k.

Receives pre-assembled candidate arrays (from search/assembler.py)
and a delta_index, runs the bitmap dedup/tombstone filter, computes
distances per query, and returns top-k results in result-dict shape.

Result format (per query):
    {"id": pk_value, "distance": float, "entity": {field: value, ...}}

The "entity" dict is the full record minus the pk and vector fields
(those are surfaced as "id" and absent respectively — the vector is
typically not echoed back to the caller in vector search).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from milvus_lite.search.bitmap import build_valid_mask
from milvus_lite.search.distance import compute_distances

if TYPE_CHECKING:
    from milvus_lite.storage.delta_index import DeltaIndex


def execute_search(
    query_vectors: np.ndarray,
    all_pks: List[Any],
    all_seqs: np.ndarray,
    all_vectors: np.ndarray,
    all_records: List[dict],
    delta_index: "DeltaIndex",
    top_k: int,
    metric_type: str,
    pk_field: str,
    vector_field: str,
    filter_mask: Optional[np.ndarray] = None,
    output_fields: Optional[List[str]] = None,
    memtable=None,
) -> List[List[dict]]:
    """Run search and return ``nq`` lists of top-k result dicts.

    Args:
        query_vectors: np.ndarray[float32], shape (nq, dim)
        all_pks: candidate pk values, length N
        all_seqs: np.ndarray[uint64], shape (N,)
        all_vectors: np.ndarray[float32], shape (N, dim)
        all_records: list of N dicts (full entity fields)
        delta_index: tombstone source for the bitmap pipeline
        top_k: requested k
        metric_type: "COSINE" / "L2" / "IP"
        pk_field: name of the primary-key column (used to extract "id"
            from each record)
        vector_field: name of the vector column (stripped from "entity"
            unless explicitly listed in output_fields)
        filter_mask: optional Phase-8 scalar filter result, length N.
            AND'd into the bitmap mask before top-k selection.
        output_fields: optional whitelist of fields to include in
            ``entity``. Phase 9.1 semantics:
              - None  → all fields except pk and vector (legacy default)
              - []    → empty entity (only id + distance)
              - list  → exactly those fields (vector included if listed,
                        pk excluded since it's already in "id")

    Returns:
        List of length nq. Each inner list has up to top_k dicts, sorted
        by ascending distance.
    """
    # Pre-compute the entity-projection function once per call. The two
    # branches let us keep the legacy "strip pk + vector" path zero-cost.
    if output_fields is None:
        def project_entity(record: dict) -> dict:
            return {
                k: v for k, v in record.items()
                if k != pk_field and k != vector_field
            }
    else:
        keep = set(output_fields)
        keep.discard(pk_field)  # pk is always surfaced as "id"

        def project_entity(record: dict) -> dict:
            return {k: record[k] for k in keep if k in record}
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)
    nq = query_vectors.shape[0]

    n_total = len(all_pks)

    # Empty candidate set → empty results.
    if n_total == 0:
        return [[] for _ in range(nq)]

    # ── 1. bitmap ───────────────────────────────────────────────
    mask = build_valid_mask(
        all_pks, all_seqs, delta_index, filter_mask=filter_mask,
        memtable=memtable,
    )
    valid_indices = np.flatnonzero(mask)

    if valid_indices.size == 0:
        return [[] for _ in range(nq)]

    valid_vectors = all_vectors[valid_indices]

    # ── 2. distances ────────────────────────────────────────────
    # Compute one query at a time so result-dict assembly is simple
    # and memory bounded. Vectorising over nq is a Phase-6 optimization.
    results: List[List[dict]] = []
    effective_k = min(top_k, valid_indices.size)

    for q_idx in range(nq):
        q = query_vectors[q_idx]
        dists = compute_distances(q, valid_vectors, metric_type)

        # ── 3. top-k via argpartition then sort ─────────────────
        if effective_k < valid_indices.size:
            # argpartition is O(N), then sort the k items: O(k log k)
            part = np.argpartition(dists, effective_k - 1)[:effective_k]
            order = part[np.argsort(dists[part])]
        else:
            order = np.argsort(dists)

        # ── 4. assemble result dicts ────────────────────────────
        from milvus_lite.search.assembler import materialize_record
        per_query: List[dict] = []
        for local_idx in order:
            global_idx = int(valid_indices[local_idx])
            record = all_records[global_idx]
            # Deferred materialization: record may be a RecordSource tuple
            if isinstance(record, tuple):
                record = materialize_record(record)
            pk = record.get(pk_field)
            entity = project_entity(record)
            per_query.append({
                "id": pk,
                "distance": float(dists[local_idx]),
                "entity": entity,
            })
        results.append(per_query)

    return results
