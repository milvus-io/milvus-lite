"""Lightweight columnar data container for the function chain.

Internal storage is ``List[List[dict]]`` where each inner list is a
*chunk*.

* **Ingestion stage** -- single chunk: ``chunks = [records]``
* **Rerank stage** -- *nq* chunks: ``chunks[i]`` = search results for the
  *i*-th query

Corresponds to Milvus: internal/util/function/chain/dataframe.go
"""

from __future__ import annotations

from typing import List


class DataFrame:
    """Lightweight chunked data container."""

    __slots__ = ("_chunks",)

    def __init__(self, chunks: List[List[dict]]) -> None:
        self._chunks = chunks

    # ── Factory methods ───────────���───────────────────────────

    @classmethod
    def from_records(cls, records: List[dict]) -> DataFrame:
        """Create from insert records (single chunk)."""
        return cls([records])

    @classmethod
    def from_search_results(cls, results: List[List[dict]]) -> DataFrame:
        """Create from per-query search results."""
        return cls(results)

    # ── Export ─────────��──────────────────────────────────────

    def to_records(self) -> List[dict]:
        """Export as flat records (single-chunk only)."""
        if len(self._chunks) != 1:
            raise ValueError(
                f"to_records() requires single chunk, got {len(self._chunks)}"
            )
        return self._chunks[0]

    def to_search_results(self) -> List[List[dict]]:
        """Export as per-query search results."""
        return self._chunks

    # ── Accessors ─────────────────────────────────────────────

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def chunk(self, idx: int) -> List[dict]:
        return self._chunks[idx]

    def column(self, name: str, chunk_idx: int) -> list:
        """Read all values of *name* from the specified chunk."""
        return [r.get(name) for r in self._chunks[chunk_idx]]

    def set_column(self, name: str, chunk_idx: int, values: list) -> None:
        """Write a column of values back to the specified chunk (in-place)."""
        chunk = self._chunks[chunk_idx]
        if len(values) != len(chunk):
            raise ValueError(
                f"set_column({name!r}): values length {len(values)} "
                f"!= chunk length {len(chunk)}"
            )
        for r, v in zip(chunk, values):
            r[name] = v

    def column_names(self, chunk_idx: int = 0) -> List[str]:
        """Column names derived from the first record's keys."""
        chunk = self._chunks[chunk_idx]
        return list(chunk[0].keys()) if chunk else []
