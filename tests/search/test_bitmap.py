"""Tests for search/bitmap.py"""

import numpy as np
import pyarrow as pa
import pytest

from milvus_lite.schema.arrow_builder import build_delta_schema
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.search.bitmap import build_valid_mask
from milvus_lite.storage.delta_index import DeltaIndex


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ])


def _empty_index() -> DeltaIndex:
    return DeltaIndex("id")


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    mask = build_valid_mask([], np.zeros(0, dtype=np.uint64), _empty_index())
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        build_valid_mask(["a", "b"], np.array([1], dtype=np.uint64), _empty_index())


# ---------------------------------------------------------------------------
# Dedup by max seq
# ---------------------------------------------------------------------------

def test_no_duplicates_all_true():
    mask = build_valid_mask(
        ["a", "b", "c"],
        np.array([1, 2, 3], dtype=np.uint64),
        _empty_index(),
    )
    assert mask.tolist() == [True, True, True]


def test_dedup_keeps_max_seq():
    """pk 'a' appears at seq=1 and seq=5; only seq=5 row stays."""
    mask = build_valid_mask(
        ["a", "b", "a"],
        np.array([1, 2, 5], dtype=np.uint64),
        _empty_index(),
    )
    assert mask.tolist() == [False, True, True]


def test_dedup_three_versions_same_pk():
    mask = build_valid_mask(
        ["x", "x", "x"],
        np.array([3, 1, 2], dtype=np.uint64),
        _empty_index(),
    )
    # Only the row with seq=3 stays (which is index 0).
    assert mask.tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# Tombstone filtering
# ---------------------------------------------------------------------------

def test_tombstone_filters_out_row(schema):
    idx = DeltaIndex("id")
    idx.add_batch(pa.RecordBatch.from_pydict(
        {"id": ["a"], "_seq": [10]}, schema=build_delta_schema(schema),
    ))
    mask = build_valid_mask(
        ["a", "b"],
        np.array([5, 5], dtype=np.uint64),
        idx,
    )
    # 'a' was deleted with seq=10 > data_seq=5 → False
    # 'b' has no tombstone → True
    assert mask.tolist() == [False, True]


def test_tombstone_strictly_greater(schema):
    """delta_seq == data_seq does NOT count as deleted (strict >)."""
    idx = DeltaIndex("id")
    idx.add_batch(pa.RecordBatch.from_pydict(
        {"id": ["a"], "_seq": [5]}, schema=build_delta_schema(schema),
    ))
    mask = build_valid_mask(["a"], np.array([5], dtype=np.uint64), idx)
    assert mask[0] == True


def test_tombstone_does_not_apply_to_newer_data(schema):
    """A delete at seq=5 cannot delete an insert at seq=10."""
    idx = DeltaIndex("id")
    idx.add_batch(pa.RecordBatch.from_pydict(
        {"id": ["a"], "_seq": [5]}, schema=build_delta_schema(schema),
    ))
    mask = build_valid_mask(["a"], np.array([10], dtype=np.uint64), idx)
    assert mask[0] == True


# ---------------------------------------------------------------------------
# Combined: dedup + tombstone
# ---------------------------------------------------------------------------

def test_dedup_then_tombstone(schema):
    """pk 'x' has rows at seq 1, 7. tombstone at seq=5 affects neither
    (5 < 7, the surviving max). Mask should keep the row with seq=7."""
    idx = DeltaIndex("id")
    idx.add_batch(pa.RecordBatch.from_pydict(
        {"id": ["x"], "_seq": [5]}, schema=build_delta_schema(schema),
    ))
    mask = build_valid_mask(
        ["x", "x"],
        np.array([1, 7], dtype=np.uint64),
        idx,
    )
    # row 0 (seq=1) is shadowed by row 1 (seq=7); row 1 has seq=7 > tombstone=5
    assert mask.tolist() == [False, True]


def test_dedup_then_tombstone_kills_max(schema):
    """pk 'x' has rows at seq 1, 7. tombstone at seq=10 kills the max."""
    idx = DeltaIndex("id")
    idx.add_batch(pa.RecordBatch.from_pydict(
        {"id": ["x"], "_seq": [10]}, schema=build_delta_schema(schema),
    ))
    mask = build_valid_mask(
        ["x", "x"],
        np.array([1, 7], dtype=np.uint64),
        idx,
    )
    assert mask.tolist() == [False, False]


# ---------------------------------------------------------------------------
# Larger arrays
# ---------------------------------------------------------------------------

def test_large_dedup():
    """100 rows, 50 unique pks, each appearing twice with consecutive seqs.
    Expect 50 rows kept (the second occurrence of each)."""
    pks = []
    seqs = []
    for i in range(50):
        pks.extend([f"pk_{i}", f"pk_{i}"])
        seqs.extend([2 * i, 2 * i + 1])
    mask = build_valid_mask(pks, np.array(seqs, dtype=np.uint64), _empty_index())
    assert int(mask.sum()) == 50
    # Each kept row should be the second occurrence
    for i in range(50):
        assert mask[2 * i] == False  # first (smaller seq)
        assert mask[2 * i + 1] == True  # second (larger seq)
