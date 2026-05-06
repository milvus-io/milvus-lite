"""In-memory cache for one immutable data Parquet file.

Loaded once and never invalidated (the underlying Parquet is immutable —
modules.md architectural invariant §4). Compaction (Phase 6) drops the
old segment from the cache and adds the new merged one.

Why pre-extract numpy arrays at load time:
    - Search hot path needs (N, dim) float32 vectors for distance
      computation. PyArrow's FixedSizeListArray needs reshaping every
      access — doing it once at load amortizes the cost across all
      future searches.
    - pks and seqs are used by the bitmap pipeline (dedup + tombstone
      check), also per-search.

The original pa.Table is retained so that returning entity fields
(non-vector columns) for top-k results doesn't require a second read.

Phase 9.2: Each Segment may carry an attached VectorIndex. The index
is bound 1:1 to the segment and shares its lifetime — when the segment
is evicted (compaction, drop_partition), the index is dropped with it.
The index is None until Collection.load() (or a flush/compaction hook)
attaches one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pyarrow as pa

from milvus_lite.storage.data_file import read_data_file

if TYPE_CHECKING:
    from milvus_lite.index.protocol import VectorIndex
    from milvus_lite.index.spec import IndexSpec


class Segment:
    """A single data Parquet file, loaded into memory.

    Public state (read-only):
        file_path: absolute path of the source Parquet
        partition: name of the partition this segment belongs to
        pks:       list of pk values (Python list, not numpy — pk dtype
                   may be string or int and a Python list is the simplest
                   uniform handling)
        seqs:      np.ndarray[uint64], shape (N,)
        vectors:   np.ndarray[float32], shape (N, dim)
        table:     original pa.Table for entity-field extraction
        pk_to_row: {pk_value: row_index} for O(1) point reads
    """

    __slots__ = (
        "file_path",
        "partition",
        "pks",
        "seqs",
        "vectors",
        "vector_null_mask",
        "table",
        "pk_to_row",
        "index",
        "indexes",
        "_pk_field",
        "_vector_field",
    )

    def __init__(
        self,
        file_path: str,
        partition: str,
        pk_field: str,
        vector_field: str,
        pks: List[Any],
        seqs: np.ndarray,
        vectors: np.ndarray,
        table: pa.Table,
        vector_null_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.file_path = file_path
        self.partition = partition
        self._pk_field = pk_field
        self._vector_field = vector_field
        self.pks = pks
        self.seqs = seqs
        self.vectors = vectors
        self.table = table
        # Boolean mask: True = vector is valid, False = null vector.
        # None means all vectors are valid (no nullable vector field).
        self.vector_null_mask = vector_null_mask
        self.pk_to_row: Dict[Any, int] = {pk: i for i, pk in enumerate(pks)}
        # Phase 9.2 / Phase 18: per-field attached indexes.
        # self.index is backward-compat shortcut (first/only index).
        # self.indexes maps field_name → VectorIndex.
        self.index: Optional["VectorIndex"] = None
        self.indexes: Dict[str, "VectorIndex"] = {}

    # ── factory ─────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        file_path: str,
        partition: str,
        pk_field: str,
        vector_field: Optional[str],
    ) -> "Segment":
        """Load a Parquet file from *file_path* into a Segment.

        *vector_field* may be ``None`` for sparse-only collections that
        have no dense FLOAT_VECTOR column.  In that case ``vectors`` is
        set to an empty (0, 0) array.
        """
        table = read_data_file(file_path)
        pks = table.column(pk_field).to_pylist()
        seqs = np.asarray(table.column("_seq").to_pylist(), dtype=np.uint64)
        if vector_field is not None:
            vectors, null_mask = _extract_vector_array(table.column(vector_field))
        else:
            vectors = np.zeros((len(pks), 0), dtype=np.float32)
            null_mask = None
        return cls(
            file_path=file_path,
            partition=partition,
            pk_field=pk_field,
            vector_field=vector_field or "",
            pks=pks,
            seqs=seqs,
            vectors=vectors,
            table=table,
            vector_null_mask=null_mask,
        )

    # ── point read ──────────────────────────────────────────────

    def find_row(self, pk_value: Any) -> Optional[int]:
        """Return the row index for *pk_value*, or None."""
        return self.pk_to_row.get(pk_value)

    def row_to_dict(self, row_idx: int) -> dict:
        """Materialize a row as a dict (excluding _seq).

        Used by Collection.get() and search executor's result builder.
        Dynamic fields stored in `$meta` are unpacked into the result.
        """
        result: dict = {}
        meta_raw = None
        for name in self.table.schema.names:
            if name == "_seq":
                continue
            val = self.table.column(name)[row_idx].as_py()
            if name == "$meta":
                meta_raw = val
            result[name] = val
        # Unpack $meta JSON into top-level keys (keeps $meta for filters)
        if meta_raw is not None:
            import json
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                    if isinstance(meta, dict):
                        result.update(meta)
                except (json.JSONDecodeError, ValueError):
                    pass
            elif isinstance(meta_raw, dict):
                result.update(meta_raw)
        return result

    # ── index lifecycle (Phase 9.2 / 9.4) ───────────────────────

    def attach_index(self, index: "VectorIndex", field_name: Optional[str] = None) -> None:
        """Attach a built or loaded VectorIndex to this segment.

        Args:
            index: the VectorIndex instance.
            field_name: vector field this index covers. If None,
                uses the segment's default _vector_field (backward compat).
        """
        if field_name is None:
            field_name = self._vector_field
        self.indexes[field_name] = index
        # Backward compat: self.index points to the default vector field's index
        if field_name == self._vector_field:
            self.index = index

    def release_index(self, field_name: Optional[str] = None) -> None:
        """Drop index reference(s). Memory freed when GC collects.

        Args:
            field_name: specific field to release. None = release ALL.
        """
        if field_name is None:
            self.index = None
            self.indexes.clear()
        else:
            self.indexes.pop(field_name, None)
            if field_name == self._vector_field:
                self.index = None

    def index_file_path(
        self, index_dir: str, index_type: str, field_name: str,
    ) -> str:
        """Return the canonical .idx path for (segment, field, index_type).

        Naming convention (architectural invariant §11):
            ``<index_dir>/<segment_stem>.<field_name>.<index_type>.idx``

        Example: data file ``data_000001_000500.parquet`` with an HNSW
        index on the ``dense_vec`` field →
        ``indexes/data_000001_000500.dense_vec.hnsw.idx``.

        field_name is part of the filename so a single segment can hold
        multiple indexes (e.g. hybrid-search collections with multiple
        FLOAT_VECTOR fields, each with its own HNSW).

        Orphan cleanup reverses the parse: strip the .idx suffix, then
        rpartition twice to recover (stem, field, index_type).
        """
        import os
        stem = os.path.splitext(os.path.basename(self.file_path))[0]
        return os.path.join(
            index_dir,
            f"{stem}.{field_name}.{index_type.lower()}.idx",
        )

    def build_or_load_index(
        self,
        spec: "IndexSpec",  # type: ignore[name-defined]  # forward
        index_dir: str,
    ) -> None:
        """Ensure this segment has an attached index for *spec.field_name*.

        - If a matching .idx file exists in *index_dir*, load it.
        - Else, build a fresh VectorIndex from the segment's vectors
          and persist it for next time.

        Idempotent: if the segment already has an index for this field,
        this is a no-op.
        """
        field_name = spec.field_name
        if field_name in self.indexes:
            return
        if self.num_rows == 0:
            return

        # Sparse inverted indexes are built on-the-fly during search,
        # not per-segment via the VectorIndex factory.
        if spec.index_type == "SPARSE_INVERTED_INDEX":
            return

        import os
        from milvus_lite.index.factory import (
            build_index_from_spec,
            load_index_from_spec,
        )

        path = self.index_file_path(index_dir, spec.index_type, field_name)

        if os.path.exists(path):
            idx = load_index_from_spec(spec, path, self.vector_dim)
        else:
            # Use the correct vector data for the requested field
            if field_name == self._vector_field:
                vectors = self.vectors
            else:
                vectors, _ = _extract_vector_array(self.table.column(field_name))
            idx = build_index_from_spec(spec, vectors)
            os.makedirs(index_dir, exist_ok=True)
            idx.save(path)

        self.attach_index(idx, field_name=field_name)

    # ── introspection ───────────────────────────────────────────

    @property
    def num_rows(self) -> int:
        return len(self.pks)

    @property
    def vector_dim(self) -> int:
        return self.vectors.shape[1] if self.vectors.size > 0 else 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_vector_array(
    arr: pa.ChunkedArray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert a FixedSizeList<float32, dim> column to (N, dim) numpy array.

    Returns:
        (vectors, null_mask) where null_mask is a bool array (True=valid)
        or None if no nulls. Null vectors are replaced with zero vectors
        in the numpy array so the shape is always (N, dim).
    """
    if isinstance(arr, pa.ChunkedArray):
        if arr.num_chunks == 0:
            return np.zeros((0, 0), dtype=np.float32), None
        if arr.num_chunks == 1:
            arr = arr.chunk(0)
        else:
            arr = arr.combine_chunks()

    if not isinstance(arr.type, pa.FixedSizeListType):
        raise ValueError(
            f"vector column must be FixedSizeList, got {arr.type}"
        )

    n = len(arr)
    dim = arr.type.list_size
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32), None

    # arr.values is the underlying flat float32 array (length n * dim)
    flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    vectors = flat.reshape(n, dim)

    # Check for Arrow-level nulls (from in-memory Arrow tables, e.g. MemTable)
    if arr.null_count > 0:
        null_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not arr[i].is_valid:
                null_mask[i] = False
                vectors[i] = 0.0  # zero-fill for numpy consistency
        return vectors, null_mask

    return vectors, None
