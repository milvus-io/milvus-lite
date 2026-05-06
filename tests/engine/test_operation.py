"""Tests for engine/operation.py"""

import pytest
import pyarrow as pa

from milvus_lite.engine.operation import DeleteOp, InsertOp, Operation


def _insert_batch(seqs):
    return pa.RecordBatch.from_pydict(
        {
            "_seq": seqs,
            "_partition": ["_default"] * len(seqs),
            "id": list(range(100, 100 + len(seqs))),
        },
        schema=pa.schema([
            pa.field("_seq", pa.uint64(), nullable=False),
            pa.field("_partition", pa.string(), nullable=False),
            pa.field("id", pa.int64(), nullable=False),
        ]),
    )


def _delete_batch(pks, seq):
    return pa.RecordBatch.from_pydict(
        {
            "id": pks,
            "_seq": [seq] * len(pks),
            "_partition": ["_default"] * len(pks),
        },
        schema=pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("_seq", pa.uint64(), nullable=False),
            pa.field("_partition", pa.string(), nullable=False),
        ]),
    )


def test_insert_op_basic():
    op = InsertOp(partition="_default", batch=_insert_batch([1, 2, 3]))
    assert op.partition == "_default"
    assert op.num_rows == 3
    assert op.seq_min == 1
    assert op.seq_max == 3


def test_insert_op_single_row():
    op = InsertOp(partition="p1", batch=_insert_batch([42]))
    assert op.num_rows == 1
    assert op.seq_min == 42
    assert op.seq_max == 42


def test_insert_op_frozen():
    op = InsertOp(partition="_default", batch=_insert_batch([1]))
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        op.partition = "other"


def test_insert_op_empty_seq_raises():
    empty = pa.RecordBatch.from_pydict(
        {"_seq": [], "_partition": [], "id": []},
        schema=pa.schema([
            pa.field("_seq", pa.uint64()),
            pa.field("_partition", pa.string()),
            pa.field("id", pa.int64()),
        ]),
    )
    op = InsertOp(partition="_default", batch=empty)
    assert op.num_rows == 0
    with pytest.raises(ValueError, match="empty batch"):
        op.seq_min
    with pytest.raises(ValueError, match="empty batch"):
        op.seq_max


def test_delete_op_basic():
    op = DeleteOp(partition="_default", batch=_delete_batch([100, 200, 300], seq=7))
    assert op.partition == "_default"
    assert op.num_rows == 3
    assert op.seq == 7  # all rows share the same seq


def test_delete_op_all_partitions_sentinel():
    op = DeleteOp(partition="_all", batch=_delete_batch([100], seq=5))
    assert op.partition == "_all"


def test_delete_op_frozen():
    op = DeleteOp(partition="_default", batch=_delete_batch([1], seq=1))
    with pytest.raises(Exception):
        op.partition = "other"


def test_delete_op_empty_seq_raises():
    empty = pa.RecordBatch.from_pydict(
        {"id": [], "_seq": [], "_partition": []},
        schema=pa.schema([
            pa.field("id", pa.int64()),
            pa.field("_seq", pa.uint64()),
            pa.field("_partition", pa.string()),
        ]),
    )
    op = DeleteOp(partition="_default", batch=empty)
    assert op.num_rows == 0
    with pytest.raises(ValueError, match="empty batch"):
        op.seq


def test_operation_union_isinstance():
    """Operation = Union[InsertOp, DeleteOp] — isinstance dispatch must work."""
    insert = InsertOp(partition="_default", batch=_insert_batch([1]))
    delete = DeleteOp(partition="_default", batch=_delete_batch([1], seq=2))

    def classify(op: Operation) -> str:
        if isinstance(op, InsertOp):
            return "insert"
        elif isinstance(op, DeleteOp):
            return "delete"
        return "unknown"

    assert classify(insert) == "insert"
    assert classify(delete) == "delete"


def test_insert_and_delete_are_distinct_types():
    """They must NOT share a common base class — Union dispatch only."""
    assert not issubclass(InsertOp, DeleteOp)
    assert not issubclass(DeleteOp, InsertOp)
