"""Phase 9.2 — index-aware search executor.

The new search path: per-segment recall via VectorIndex (or ad-hoc
BruteForceIndex if the segment has no attached index yet) + memtable
brute force + global merge.

Differs from `execute_search` in two architectural ways:

1. **Per-source recall, then global merge**. The old path concatenates
   all sources into one big numpy array and runs distance + top-k
   once. The new path runs top-k *per source* (each segment, then
   the memtable) and merges results across sources at the end. This
   is the structure we need for FAISS HNSW: each segment's index
   knows nothing about the other segments, so we have to call them
   independently and stitch results.

2. **Cross-source dedup is computed once up front**. Upserts can
   leave the same pk in multiple segments + the memtable. The old
   path's `bitmap.build_valid_mask` does dedup across the unified
   candidate set; here we compute the global max-seq map first,
   then derive a per-source local mask that excludes any row whose
   seq is not the global winner.

Result-equivalence with the old path is guaranteed by the
differential test in tests/search/test_executor_with_index.py.

Phase 9.2 always builds an ad-hoc BruteForceIndex if `segment.index`
is None — this exercises the new architecture from day one even
before Collection.create_index lands in 9.3. Building a BruteForceIndex
is essentially free (just stores a numpy reference).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from milvus_lite.index.brute_force import BruteForceIndex

if TYPE_CHECKING:
    from milvus_lite.index.scalar import IndexedFilterPlan
    from milvus_lite.search.filter.semantic import CompiledExpr
    from milvus_lite.storage.delta_index import DeltaIndex
    from milvus_lite.storage.memtable import MemTable
    from milvus_lite.storage.segment import Segment


def execute_search_with_index(
    query_vectors: np.ndarray,
    segments: Iterable["Segment"],
    memtable: "MemTable",
    delta_index: "DeltaIndex",
    top_k: int,
    metric_type: str,
    pk_field: str,
    vector_field: str,
    partition_names: Optional[List[str]] = None,
    compiled_filter: Optional["CompiledExpr"] = None,
    output_fields: Optional[List[str]] = None,
    indexed_filter_plan: Optional["IndexedFilterPlan"] = None,
    project_record_fn: Optional[Callable[[dict], dict]] = None,
) -> List[List[dict]]:
    """Per-source recall + global merge search path.

    Args:
        query_vectors: (nq, dim) float32
        segments: iterable of Segment (typically Collection._segment_cache.values())
        memtable: live MemTable
        delta_index: tombstone source for the bitmap pipeline
        top_k: requested k
        metric_type: "COSINE" / "L2" / "IP"
        pk_field: primary key column name
        vector_field: vector column name (used to extract memtable arrays
            and to strip from default entity projection)
        partition_names: optional partition filter
        compiled_filter: optional Phase-8 CompiledExpr
        output_fields: optional Phase-9.1 entity projection list. See
            Collection.search for the three semantics.

    Returns:
        nq lists of up to top_k result dicts of shape
        ``{"id": pk, "distance": float, "entity": {...}}``,
        sorted by ascending distance per query.
    """
    # ── 0. shape + scope ────────────────────────────────────────
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32, copy=False)
    nq = int(query_vectors.shape[0])

    if nq == 0 or top_k <= 0:
        return [[] for _ in range(nq)]

    partition_filter: Optional[set] = (
        set(partition_names) if partition_names else None
    )

    # In-scope segments (skip empty + partition-filtered).
    in_scope_segments: List["Segment"] = [
        s for s in segments
        if s.num_rows > 0
        and (partition_filter is None or s.partition in partition_filter)
    ]

    # Memtable arrays (records deferred — only refs stored).
    mt_pks, mt_seqs, mt_vecs, mt_row_refs, mt_table = memtable.to_search_snapshot(
        vector_field=vector_field,
        partition_names=partition_names,
        include_table=compiled_filter is not None,
    )

    # If we have nothing in scope at all, return empty.
    if not in_scope_segments and not mt_pks:
        return [[] for _ in range(nq)]

    # ── 1. cross-source dedup map (global max-seq per pk) ───────
    global_max_seqs: Dict[Any, int] = {}

    def _update_max_seqs(pks: List[Any], seqs: np.ndarray) -> None:
        seqs_list = seqs.tolist() if hasattr(seqs, "tolist") else list(seqs)
        for pk, seq in zip(pks, seqs_list):
            seq_int = int(seq)
            existing = global_max_seqs.get(pk)
            if existing is None or seq_int > existing:
                global_max_seqs[pk] = seq_int

    for seg in in_scope_segments:
        _update_max_seqs(list(seg.pks), seg.seqs)
    if mt_pks:
        _update_max_seqs(mt_pks, mt_seqs)

    # ── 2. optional per-source filter masks (Phase 8) ───────────
    seg_filter_masks: Dict[int, np.ndarray] = {}
    mt_filter_mask: Optional[np.ndarray] = None
    if compiled_filter is not None:
        from milvus_lite.search.filter import evaluate_mask
        from milvus_lite.search.indexed_filter import evaluate_segment_filter
        for seg in in_scope_segments:
            seg_filter_masks[id(seg)] = evaluate_segment_filter(
                seg, compiled_filter, indexed_filter_plan,
            )
        if mt_pks and mt_table is not None:
            mt_filter_mask = evaluate_mask(compiled_filter, mt_table)

    # ── 3. per-source recall ────────────────────────────────────
    # Per query, accumulate (distance, source_idx, local_id) candidate
    # tuples. We materialize records lazily after the global sort so
    # we only pay row_to_dict cost for rows that actually win a slot.
    Candidate = Tuple[float, int, int]
    per_query_candidates: List[List[Candidate]] = [[] for _ in range(nq)]

    # Sources are numbered 0..N-1 = segments, then N = memtable.
    sources: List[Tuple[List[Any], np.ndarray, np.ndarray]] = []

    # MemTable-local tombstones for pks already flushed to segments —
    # not yet in delta_index (folded at next flush). See issue #21.
    mt_deletes = memtable.delete_index_snapshot()  # {pk: (seq, partition)}

    def _build_local_mask(
        pks: List[Any],
        seqs: np.ndarray,
        filter_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        n = len(pks)
        local_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            pk = pks[i]
            seq = int(seqs[i])
            # Cross-source max-seq dedup.
            if global_max_seqs.get(pk) != seq:
                continue
            # Tombstone: a strictly larger delete_seq shadows this row.
            if delta_index.is_deleted(pk, seq):
                continue
            mt_entry = mt_deletes.get(pk)
            if mt_entry is not None and mt_entry[0] > seq:
                continue
            local_mask[i] = True
        if filter_mask is not None:
            local_mask = local_mask & filter_mask.astype(bool)
        return local_mask

    def _recall_source(
        source_idx: int,
        pks: List[Any],
        seqs: np.ndarray,
        vectors: np.ndarray,
        filter_mask: Optional[np.ndarray],
        index,  # Optional[VectorIndex]
    ) -> None:
        n = len(pks)
        if n == 0:
            return
        local_mask = _build_local_mask(pks, seqs, filter_mask)
        if not local_mask.any():
            return
        if index is None:
            index = BruteForceIndex.build(vectors, metric_type)
        local_ids, dists = index.search(
            query_vectors, top_k, valid_mask=local_mask
        )
        for q in range(nq):
            for j in range(top_k):
                lid = int(local_ids[q, j])
                if lid < 0:
                    break  # padding from here on (sorted ascending; -1 is sentinel)
                per_query_candidates[q].append(
                    (float(dists[q, j]), source_idx, lid)
                )

    # Segments first.
    for seg in in_scope_segments:
        sources.append((list(seg.pks), seg.seqs, seg.vectors))
        # Combine scalar filter mask with vector null mask
        combined_mask = seg_filter_masks.get(id(seg))
        if seg.vector_null_mask is not None:
            if combined_mask is not None:
                combined_mask = combined_mask & seg.vector_null_mask
            else:
                combined_mask = seg.vector_null_mask
        _recall_source(
            source_idx=len(sources) - 1,
            pks=sources[-1][0],
            seqs=sources[-1][1],
            vectors=sources[-1][2],
            filter_mask=combined_mask,
            index=seg.index,
        )

    # Memtable last (always brute-force; never has an attached index).
    if mt_pks:
        sources.append((mt_pks, mt_seqs, mt_vecs))
        _recall_source(
            source_idx=len(sources) - 1,
            pks=mt_pks,
            seqs=mt_seqs,
            vectors=mt_vecs,
            filter_mask=mt_filter_mask,
            index=None,
        )

    # Track which (segment, local_id) maps to which source so we can
    # also pull records out at materialization time. The mapping is
    # implicit via source_idx into `in_scope_segments + [memtable]`.
    n_segments = len(in_scope_segments)

    # ── 4. global merge + entity projection ─────────────────────
    if project_record_fn is not None:
        project_entity = project_record_fn
    elif output_fields is None:
        def project_entity(record: dict) -> dict:
            return {
                k: v for k, v in record.items()
                if k != pk_field and k != vector_field
            }
    else:
        keep = set(output_fields)
        keep.discard(pk_field)

        def project_entity(record: dict) -> dict:
            return {k: record[k] for k in keep if k in record}

    def _materialize_record(source_idx: int, local_id: int) -> dict:
        if source_idx < n_segments:
            return in_scope_segments[source_idx].row_to_dict(local_id)
        # Memtable source — deferred materialization via row refs.
        batch_idx, row_idx = mt_row_refs[local_id]
        return memtable.materialize_row(batch_idx, row_idx)

    results: List[List[dict]] = []
    for q in range(nq):
        # Sort by (distance, source_idx, local_id) for determinism on ties.
        candidates = per_query_candidates[q]
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        per_query: List[dict] = []
        for dist, source_idx, local_id in candidates[:top_k]:
            record = _materialize_record(source_idx, local_id)
            pk = record.get(pk_field) if pk_field in record else sources[source_idx][0][local_id]
            per_query.append({
                "id": pk,
                "distance": dist,
                "entity": project_entity(record),
            })
        results.append(per_query)

    return results
