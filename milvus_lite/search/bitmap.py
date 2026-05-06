"""Bitmap pipeline — produce a boolean mask of "valid" search candidates.

Three filters (in order):
    1. Dedup by max-seq: when the same pk appears multiple times in the
       candidate set (e.g. across multiple data files due to upsert),
       only the row with the largest _seq is kept.
    2. Tombstone filter: rows whose pk has a delete entry in delta_index
       with a strictly larger seq are dropped.
    3. Scalar filter (optional, Phase 8): caller-provided boolean mask
       from search/filter evaluator. AND'd into the final result.

Performance note: dedup + tombstone are O(N) Python with dict lookups.
For 1M candidates it's ~1s, which is fine for an embedded MVP.
Vectorizing with numpy would help only if pks are uniformly typed (all
int or all str); for the current API the dict approach is simplest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

if TYPE_CHECKING:
    from milvus_lite.storage.delta_index import DeltaIndex
    from milvus_lite.storage.memtable import MemTable


def build_valid_mask(
    all_pks: List[Any],
    all_seqs: np.ndarray,
    delta_index: "DeltaIndex",
    filter_mask: Optional[np.ndarray] = None,
    memtable: Optional["MemTable"] = None,
) -> np.ndarray:
    """Return a boolean mask over the candidate rows.

    Args:
        all_pks: list of pk values, length N. Python list (not numpy)
            because pk dtype may be string or int.
        all_seqs: np.ndarray[uint64], shape (N,)
        delta_index: tombstone source (already-flushed deletes).
        filter_mask: optional Phase-8 scalar filter result, length N.
            If provided, AND'd into the final mask.
        memtable: optional live MemTable. Its _delete_index is consulted
            for tombstones applied but not yet flushed to delta_index —
            without this, deletes targeting already-flushed segment data
            would remain invisible to search/query until the next flush
            (issue #21).

    Returns:
        np.ndarray[bool] shape (N,). True means "this row is the latest
        live version of its pk, is not deleted, and (if filter_mask is
        provided) satisfies the scalar filter expression".
    """
    n = len(all_pks)
    if n == 0:
        return np.zeros(0, dtype=bool)

    if all_seqs.shape[0] != n:
        raise ValueError(
            f"all_pks ({n}) and all_seqs ({all_seqs.shape[0]}) must have the same length"
        )

    if filter_mask is not None and filter_mask.shape[0] != n:
        raise ValueError(
            f"filter_mask length ({filter_mask.shape[0]}) does not match "
            f"all_pks length ({n})"
        )

    # Step 1: per-pk max seq
    max_seqs: dict[Any, int] = {}
    for i in range(n):
        pk = all_pks[i]
        seq = int(all_seqs[i])
        existing = max_seqs.get(pk)
        if existing is None or seq > existing:
            max_seqs[pk] = seq

    # Step 2: build mask — keep rows whose seq is the max for their pk AND
    # whose pk is not tombstoned with a larger seq.
    # MemTable's local _delete_index holds tombstones applied but not
    # yet folded into delta_index (which happens at flush time). We
    # must also consult it so recent deletes are visible to readers.
    mt_deletes = memtable.delete_index_snapshot() if memtable is not None else {}
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        pk = all_pks[i]
        seq = int(all_seqs[i])
        if seq != max_seqs[pk]:
            continue  # an older version of this pk exists later in the array
        if delta_index.is_deleted(pk, seq):
            continue
        mt_entry = mt_deletes.get(pk)
        if mt_entry is not None and mt_entry[0] > seq:
            continue  # MemTable-local tombstone shadows this row
        mask[i] = True

    # Step 3: AND in the scalar filter mask, if provided.
    if filter_mask is not None:
        mask = mask & filter_mask.astype(bool)

    return mask
