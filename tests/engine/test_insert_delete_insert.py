"""End-to-end Insert/Delete/Insert tests.

This file is the safety net for the architectural invariant §1 fix from
Phase 0 — that all override/discard decisions are keyed on _seq, not on
call order or file order. The MemTable already has unit-level coverage
for the property (`test_seq_5_7_6_recovery_replay_gives_seq_7` and the
6-permutation property test). Here we exercise the SAME invariant
through the full Collection stack, including:

    - in-memory only (no flushes)
    - across a flush boundary (data hits a Parquet segment between ops)
    - across a process restart (delta_index rebuilt from disk)
    - random orderings of mixed insert/delete sequences

If any of these regress, the LSM correctness story is broken.
"""

import itertools
import random

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


def _rec(label: str) -> dict:
    return {"id": "X", "vec": [0.5, 0.25], "title": label}


# ---------------------------------------------------------------------------
# In-memory only (no flush)
# ---------------------------------------------------------------------------

def test_insert_delete_insert_in_memory(tmp_path, schema):
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([_rec("v1")])
    col.delete(["X"])
    col.insert([_rec("v2")])
    [rec] = col.get(["X"])
    assert rec["title"] == "v2"
    col.close()


def test_insert_delete_insert_delete_in_memory(tmp_path, schema):
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([_rec("v1")])
    col.delete(["X"])
    col.insert([_rec("v2")])
    col.delete(["X"])
    assert col.get(["X"]) == []
    col.close()


# ---------------------------------------------------------------------------
# Across flush boundaries
# ---------------------------------------------------------------------------

def test_insert_flush_delete(tmp_path, schema):
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([_rec("v1")])
    col.flush()
    col.delete(["X"])
    assert col.get(["X"]) == []
    col.close()


def test_insert_flush_delete_flush_insert(tmp_path, schema):
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([_rec("v1")])
    col.flush()  # X@1 in segment
    col.delete(["X"])
    col.flush()  # delete@2 in delta segment, delta_index has X→2
    col.insert([_rec("v2")])  # X@3 in MemTable
    [rec] = col.get(["X"])
    assert rec["title"] == "v2"
    col.close()


def test_each_op_flushed_separately(tmp_path, schema):
    """Hardest case: every single op gets flushed before the next.
    Verifies cross-segment dedup + tombstone routing all work."""
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([_rec("v1")])
    col.flush()
    col.delete(["X"])
    col.flush()
    col.insert([_rec("v2")])
    col.flush()
    col.delete(["X"])
    col.flush()
    col.insert([_rec("v3")])
    col.flush()
    [rec] = col.get(["X"])
    assert rec["title"] == "v3"
    col.close()


def test_search_after_each_op_flushed(tmp_path, schema):
    """Same as above but verify via search instead of get."""
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([_rec("v1")])
    col.flush()
    col.delete(["X"])
    col.flush()
    col.insert([_rec("v2")])
    col.flush()

    results = col.search([[0.5, 0.25]], top_k=10, metric_type="L2")
    [hits] = results
    assert len(hits) == 1
    assert hits[0]["entity"]["title"] == "v2"
    col.close()


# ---------------------------------------------------------------------------
# Across process restarts
# ---------------------------------------------------------------------------

def test_insert_delete_insert_across_restart(tmp_path, schema):
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([_rec("v1")])
    col1.delete(["X"])
    col1.insert([_rec("v2")])
    col1.close()  # full flush

    col2 = Collection("c", data_dir, schema)
    [rec] = col2.get(["X"])
    assert rec["title"] == "v2"
    col2.close()


def test_alternating_restarts(tmp_path, schema):
    """Restart the Collection between every operation. Each restart
    reloads manifest + delta_index from disk; the final state must
    still be correct."""
    data_dir = str(tmp_path / "d")

    col = Collection("c", data_dir, schema)
    col.insert([_rec("v1")])
    col.close()

    col = Collection("c", data_dir, schema)
    col.delete(["X"])
    col.close()

    col = Collection("c", data_dir, schema)
    col.insert([_rec("v2")])
    col.close()

    col = Collection("c", data_dir, schema)
    [rec] = col.get(["X"])
    assert rec["title"] == "v2"
    col.close()


# ---------------------------------------------------------------------------
# 50 random orderings property test
# ---------------------------------------------------------------------------

def _expected_state(ops):
    """Compute the expected final visibility for X given a sequence of
    ('insert', label) / ('delete',) ops applied in order."""
    visible = None
    for op in ops:
        if op[0] == "insert":
            visible = op[1]
        else:
            visible = None
    return visible


@pytest.mark.parametrize("seed", list(range(50)))
def test_random_insert_delete_orderings(tmp_path, schema, seed):
    """Generate a random sequence of inserts and deletes for the same
    pk X, run it through the Collection (with occasional flushes), and
    assert the final state matches a trivial Python simulation."""
    rng = random.Random(seed)
    n_ops = rng.randint(3, 10)
    ops = []
    for i in range(n_ops):
        kind = rng.choice(["insert", "insert", "delete"])  # bias toward insert
        if kind == "insert":
            ops.append(("insert", f"v{i}"))
        else:
            ops.append(("delete",))

    expected = _expected_state(ops)

    data_dir = str(tmp_path / f"d{seed}")
    col = Collection("c", data_dir, schema)
    for op in ops:
        if op[0] == "insert":
            col.insert([_rec(op[1])])
        else:
            col.delete(["X"])

        # Random flush ~30% of the time to exercise cross-segment paths.
        if rng.random() < 0.3:
            col.flush()

    got = col.get(["X"])
    if expected is None:
        assert got == [], f"seed={seed} ops={ops}: expected absent, got {got}"
    else:
        assert len(got) == 1, f"seed={seed} ops={ops}: expected v={expected}"
        assert got[0]["title"] == expected, f"seed={seed} ops={ops}"

    col.close()

    # Also verify across restart.
    col2 = Collection("c", data_dir, schema)
    got2 = col2.get(["X"])
    if expected is None:
        assert got2 == [], f"seed={seed} after restart: expected absent"
    else:
        assert len(got2) == 1
        assert got2[0]["title"] == expected, f"seed={seed} after restart"
    col2.close()


# ---------------------------------------------------------------------------
# Multi-pk interleaved
# ---------------------------------------------------------------------------

def test_multi_pk_interleaved(tmp_path, schema):
    """Two pks (A, B), interleaved insert/delete operations.
    Each pk's final state must be independently correct."""
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([{"id": "A", "vec": [0.5, 0.25], "title": "a1"}])
    col.insert([{"id": "B", "vec": [0.75, 0.125], "title": "b1"}])
    col.flush()
    col.delete(["A"])
    col.insert([{"id": "B", "vec": [0.0625, 1.5], "title": "b2"}])
    col.flush()
    col.insert([{"id": "A", "vec": [2.0, 0.375], "title": "a2"}])

    a = col.get(["A"])
    b = col.get(["B"])
    assert len(a) == 1 and a[0]["title"] == "a2"
    assert len(b) == 1 and b[0]["title"] == "b2"
    col.close()
