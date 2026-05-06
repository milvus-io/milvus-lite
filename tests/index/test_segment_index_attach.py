"""Phase 9.2.2 — Segment.index slot + attach_index / release_index.

These are very small tests because the slot itself is trivial. The
real heavy lifting (build_or_load_index, persistence, lifecycle hooks
into flush / compaction / recovery) happens in Phase 9.3 and 9.4.
"""

import numpy as np
import pyarrow as pa
import pytest

from milvus_lite.index.brute_force import BruteForceIndex
from milvus_lite.storage.segment import Segment


@pytest.fixture
def segment(tmp_path):
    """Build a Segment by calling its constructor directly with a
    minimal in-memory pa.Table — avoids the full data_file write/read
    round-trip and keeps the test self-contained."""
    pks = ["a", "b", "c"]
    seqs = np.array([1, 2, 3], dtype=np.uint64)
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
    )
    table = pa.table({
        "id": pks,
        "_seq": seqs,
        "vec": pa.array(
            [list(v) for v in vectors],
            type=pa.list_(pa.float32(), 2),
        ),
    })
    return Segment(
        file_path=str(tmp_path / "fake.parquet"),
        partition="_default",
        pk_field="id",
        vector_field="vec",
        pks=pks,
        seqs=seqs,
        vectors=vectors,
        table=table,
    )


def test_segment_index_starts_as_none(segment):
    assert segment.index is None


def test_attach_index_sets_field(segment):
    idx = BruteForceIndex.build(segment.vectors, "L2")
    segment.attach_index(idx)
    assert segment.index is idx


def test_release_index_clears_field(segment):
    segment.attach_index(BruteForceIndex.build(segment.vectors, "L2"))
    segment.release_index()
    assert segment.index is None


def test_release_when_no_index_is_noop(segment):
    segment.release_index()  # should not raise
    assert segment.index is None


def test_attach_replaces_existing_index(segment):
    a = BruteForceIndex.build(segment.vectors, "L2")
    b = BruteForceIndex.build(segment.vectors, "COSINE")
    segment.attach_index(a)
    segment.attach_index(b)
    assert segment.index is b
