"""Write-orchestration abstraction layer.

InsertOp / DeleteOp are frozen dataclasses wrapping a pa.RecordBatch with
its target partition. They are pure descriptions — no behaviour.

Layering note: this module lives in engine/ and is consumed by Collection,
flush, recovery, and compaction. storage/wal.py and storage/memtable.py do
NOT import this module — they accept raw RecordBatches. Dispatch from
Operation to the right WAL/MemTable method happens in Collection._apply.
This keeps the storage layer free of engine-layer types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import pyarrow as pa


@dataclass(frozen=True)
class InsertOp:
    """A single insert call's transactional intent.

    ``batch`` has the wal_data schema (``_seq + _partition + user fields + [$meta]``).
    Each row carries an independent ``_seq`` allocated by Collection.
    """

    partition: str
    batch: pa.RecordBatch

    @property
    def num_rows(self) -> int:
        return self.batch.num_rows

    @property
    def seq_min(self) -> int:
        if self.batch.num_rows == 0:
            raise ValueError("InsertOp.seq_min on empty batch")
        col = self.batch.column("_seq")
        return col[0].as_py()

    @property
    def seq_max(self) -> int:
        if self.batch.num_rows == 0:
            raise ValueError("InsertOp.seq_max on empty batch")
        col = self.batch.column("_seq")
        return col[self.batch.num_rows - 1].as_py()


@dataclass(frozen=True)
class DeleteOp:
    """A single delete call's transactional intent.

    ``batch`` has the wal_delta schema (``{pk} + _seq + _partition``).
    All rows in the batch share the same ``_seq``.
    ``partition`` may be the cross-partition sentinel ``"_all"``.
    """

    partition: str
    batch: pa.RecordBatch

    @property
    def num_rows(self) -> int:
        return self.batch.num_rows

    @property
    def seq(self) -> int:
        if self.batch.num_rows == 0:
            raise ValueError("DeleteOp.seq on empty batch")
        return self.batch.column("_seq")[0].as_py()


Operation = Union[InsertOp, DeleteOp]
