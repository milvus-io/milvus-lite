"""Search candidate assembler.

Merges segments (on-disk Parquet caches) and the live MemTable into a
single candidate set ready for the bitmap pipeline + distance computation.

This is the only module in the search/ package that knows about both
storage layer types (Segment, MemTable) AND the filter subsystem —
keeping the rest of the search package storage-agnostic AND filter-
agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from milvus_lite.search.filter.semantic import CompiledExpr
    from milvus_lite.storage.memtable import MemTable
    from milvus_lite.storage.segment import Segment


"""Record source for deferred materialization."""
RecordSource = Tuple[Any, int]  # (segment_or_memtable, row_index)


def assemble_candidates(
    segments: Iterable["Segment"],
    memtable: "MemTable",
    vector_field: Optional[str],
    partition_names: Optional[List[str]] = None,
    filter_compiled: Optional["CompiledExpr"] = None,
) -> Tuple[List[Any], np.ndarray, np.ndarray, List[RecordSource], Optional[np.ndarray]]:
    """Build the unified candidate arrays.

    Returns:
        all_pks:        list of pk values (length N total)
        all_seqs:       np.ndarray[uint64], shape (N,)
        all_vectors:    np.ndarray[float32], shape (N, dim)
        all_rec_sources: list of (source_obj, row_idx) for deferred
                        materialization. source_obj is a Segment or
                        the MemTable. Call source_obj.row_to_dict(row_idx)
                        or memtable.materialize_row(*ref) to get the dict.
        filter_mask:    np.ndarray[bool] (length N) or None
    """
    from milvus_lite.search.filter.eval import evaluate as filter_evaluate

    partition_filter = set(partition_names) if partition_names else None

    pk_chunks: List[List[Any]] = []
    seq_chunks: List[np.ndarray] = []
    vec_chunks: List[np.ndarray] = []
    rec_source_chunks: List[List[RecordSource]] = []
    filter_chunks: List[np.ndarray] = []

    # ── segments (deferred materialization) ────────────────────
    for seg in segments:
        if partition_filter is not None and seg.partition not in partition_filter:
            continue
        if seg.num_rows == 0:
            continue
        pk_chunks.append(list(seg.pks))
        seq_chunks.append(seg.seqs)
        vec_chunks.append(seg.vectors)
        # Store (segment, row_idx) refs — NO row_to_dict call here
        rec_source_chunks.append([(seg, i) for i in range(seg.num_rows)])
        if filter_compiled is not None:
            mask = filter_evaluate(filter_compiled, seg.table)
            filter_chunks.append(mask.to_numpy(zero_copy_only=False))

    # ── memtable (deferred materialization) ─────────────────────
    mt_pks, mt_seqs, mt_vecs, mt_row_refs, mt_table = memtable.to_search_snapshot(
        vector_field=vector_field,
        partition_names=partition_names,
        include_table=filter_compiled is not None,
    )
    if mt_pks:
        pk_chunks.append(mt_pks)
        seq_chunks.append(mt_seqs)
        vec_chunks.append(mt_vecs)
        # Store (memtable, (batch_idx, row_idx)) refs
        rec_source_chunks.append([(memtable, ref) for ref in mt_row_refs])
        if filter_compiled is not None and mt_table is not None:
            mask = filter_evaluate(filter_compiled, mt_table)
            filter_chunks.append(mask.to_numpy(zero_copy_only=False))

    # ── concatenate ─────────────────────────────────────────────
    if not pk_chunks:
        empty_filter = np.zeros(0, dtype=bool) if filter_compiled is not None else None
        return ([], np.zeros(0, dtype=np.uint64),
                np.zeros((0, 0), dtype=np.float32), [], empty_filter)

    all_pks: List[Any] = []
    for chunk in pk_chunks:
        all_pks.extend(chunk)

    all_seqs = np.concatenate(seq_chunks)

    dim = next((v.shape[1] for v in vec_chunks if v.shape[1] > 0), 0)
    if dim == 0:
        all_vectors = np.zeros((len(all_pks), 0), dtype=np.float32)
    else:
        normalized_chunks = []
        for v in vec_chunks:
            if v.size == 0:
                continue
            if v.shape[1] != dim:
                raise ValueError(
                    f"vector dim mismatch across candidate sources: {v.shape[1]} vs {dim}"
                )
            normalized_chunks.append(v.astype(np.float32, copy=False))
        all_vectors = (np.concatenate(normalized_chunks, axis=0)
                       if normalized_chunks
                       else np.zeros((0, dim), dtype=np.float32))

    all_rec_sources: List[RecordSource] = []
    for chunk in rec_source_chunks:
        all_rec_sources.extend(chunk)

    if filter_compiled is not None:
        filter_mask = (np.concatenate(filter_chunks)
                       if filter_chunks else np.zeros(0, dtype=bool))
    else:
        filter_mask = None

    return all_pks, all_seqs, all_vectors, all_rec_sources, filter_mask


def materialize_record(source: RecordSource) -> dict:
    """Materialize a single record from a RecordSource reference.

    Works for both Segment sources (source_obj.row_to_dict(row_idx))
    and MemTable sources (memtable.materialize_row(batch_idx, row_idx)).
    """
    source_obj, ref = source
    # Segment: ref is an int (row index)
    if isinstance(ref, int):
        return source_obj.row_to_dict(ref)
    # MemTable: ref is (batch_idx, row_idx)
    batch_idx, row_idx = ref
    return source_obj.materialize_row(batch_idx, row_idx)
