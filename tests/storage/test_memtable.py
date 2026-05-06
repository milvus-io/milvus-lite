"""Tests for storage/memtable.py — Phase 2.

Critical: includes the seq-aware out-of-order reordering tests that
solidify architectural invariant §1-2. These are the safety net for the
Phase 0 design fix.
"""

import pyarrow as pa
import pytest

from milvus_lite.schema.arrow_builder import build_wal_data_schema, build_wal_delta_schema
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.storage.memtable import MemTable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


@pytest.fixture
def wal_data_schema(schema):
    return build_wal_data_schema(schema)


@pytest.fixture
def wal_delta_schema(schema):
    return build_wal_delta_schema(schema)


@pytest.fixture
def mt(schema):
    return MemTable(schema)


def insert_batch(wal_data_schema, rows):
    """rows = [(seq, partition, id, vec, title), ...]"""
    return pa.RecordBatch.from_pydict(
        {
            "_seq": [r[0] for r in rows],
            "_partition": [r[1] for r in rows],
            "id": [r[2] for r in rows],
            "vec": [r[3] for r in rows],
            "title": [r[4] for r in rows],
        },
        schema=wal_data_schema,
    )


def delete_batch(wal_delta_schema, pks, seq, partition="_default"):
    return pa.RecordBatch.from_pydict(
        {
            "id": pks,
            "_seq": [seq] * len(pks),
            "_partition": [partition] * len(pks),
        },
        schema=wal_delta_schema,
    )


# ---------------------------------------------------------------------------
# Basic apply_insert / get
# ---------------------------------------------------------------------------

def test_empty_memtable(mt):
    assert mt.size() == 0
    assert mt.get("anything") is None
    assert mt.get_active_records() == []


def test_insert_single_record(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "doc1", [0.5, 0.25], "hello"),
    ]))
    rec = mt.get("doc1")
    assert rec == {"id": "doc1", "vec": [0.5, 0.25], "title": "hello"}
    assert mt.size() == 1


def test_insert_multiple_records(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "a", [0.5, 0.25], "x"),
        (2, "_default", "b", [0.75, 0.125], "y"),
        (3, "_default", "c", [0.0625, 1.5], "z"),
    ]))
    assert mt.size() == 3
    assert mt.get("a")["id"] == "a"
    assert mt.get("b")["id"] == "b"
    assert mt.get("c")["id"] == "c"


def test_insert_returns_clean_dict_no_seq_no_partition(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (5, "_default", "doc1", [0.5, 0.25], "hi"),
    ]))
    rec = mt.get("doc1")
    assert "_seq" not in rec
    assert "_partition" not in rec


def test_insert_empty_batch_noop(mt, wal_data_schema):
    empty = pa.RecordBatch.from_pydict(
        {"_seq": [], "_partition": [], "id": [], "vec": [], "title": []},
        schema=wal_data_schema,
    )
    mt.apply_insert(empty)
    assert mt.size() == 0
    assert mt.num_physical_rows() == 0


# ---------------------------------------------------------------------------
# Upsert (same pk newer seq overrides)
# ---------------------------------------------------------------------------

def test_upsert_in_separate_batches(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "doc1", [0.5, 0.25], "old"),
    ]))
    mt.apply_insert(insert_batch(wal_data_schema, [
        (5, "_default", "doc1", [2.0, 0.375], "new"),
    ]))
    assert mt.get("doc1")["title"] == "new"
    assert mt.size() == 1  # still one logical pk


def test_upsert_in_same_batch(mt, wal_data_schema):
    """Same pk in one batch — later row (with larger seq) wins."""
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "doc1", [0.5, 0.25], "first"),
        (2, "_default", "doc1", [0.75, 0.125], "second"),
    ]))
    assert mt.get("doc1")["title"] == "second"


# ---------------------------------------------------------------------------
# Basic delete
# ---------------------------------------------------------------------------

def test_delete_after_insert(mt, wal_data_schema, wal_delta_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "doc1", [0.5, 0.25], "x"),
    ]))
    mt.apply_delete(delete_batch(wal_delta_schema, ["doc1"], seq=2))
    assert mt.get("doc1") is None
    assert mt.size() == 1  # delete tombstone


def test_delete_unknown_pk(mt, wal_delta_schema):
    """Deleting a pk that was never inserted should still record the tombstone."""
    mt.apply_delete(delete_batch(wal_delta_schema, ["ghost"], seq=5))
    assert mt.get("ghost") is None
    assert mt.size() == 1


def test_delete_empty_batch_noop(mt, wal_delta_schema):
    empty = pa.RecordBatch.from_pydict(
        {"id": [], "_seq": [], "_partition": []},
        schema=wal_delta_schema,
    )
    mt.apply_delete(empty)
    assert mt.size() == 0


# ---------------------------------------------------------------------------
# THE KEY TEST: seq-aware reordering (out-of-order apply)
# ---------------------------------------------------------------------------

def test_out_of_order_apply_insert_then_smaller_insert(mt, wal_data_schema):
    """
    Apply seq=7 first, then seq=5 for the same pk.
    The seq=5 insert MUST be skipped — we already have a newer version.
    """
    mt.apply_insert(insert_batch(wal_data_schema, [
        (7, "_default", "X", [0.875, 1.25], "newer"),
    ]))
    mt.apply_insert(insert_batch(wal_data_schema, [
        (5, "_default", "X", [0.5, 0.5], "older"),
    ]))
    rec = mt.get("X")
    assert rec["title"] == "newer"
    assert rec["vec"] == [0.875, 1.25]


def test_out_of_order_apply_insert_then_smaller_delete(mt, wal_data_schema, wal_delta_schema):
    """
    Apply insert seq=10 first, then delete seq=5 for the same pk.
    The delete must be discarded (already a newer insert).
    """
    mt.apply_insert(insert_batch(wal_data_schema, [
        (10, "_default", "X", [0.5, 0.25], "alive"),
    ]))
    mt.apply_delete(delete_batch(wal_delta_schema, ["X"], seq=5))
    assert mt.get("X")["title"] == "alive"


def test_out_of_order_apply_delete_then_smaller_insert(mt, wal_data_schema, wal_delta_schema):
    """
    Apply delete seq=10 first, then insert seq=5 for the same pk.
    The insert must be discarded (already a newer delete).
    """
    mt.apply_delete(delete_batch(wal_delta_schema, ["X"], seq=10))
    mt.apply_insert(insert_batch(wal_data_schema, [
        (5, "_default", "X", [0.5, 0.25], "should_not_appear"),
    ]))
    assert mt.get("X") is None


def test_seq_5_7_6_recovery_replay_gives_seq_7(mt, wal_data_schema, wal_delta_schema):
    """
    THE motivating reordering reproducer.

    Logical history (in seq order):
        seq=5: insert(X, "old")
        seq=6: delete(X)
        seq=7: insert(X, "new")

    Final state should be: X = "new" (last insert wins).

    But if we apply in WAL physical order — all wal_data first then all
    wal_delta — we'd see (5, 7, 6) and the original buggy MemTable would
    end up with X deleted. The seq-aware fix must give X = "new" regardless
    of apply order.
    """
    # Physical order: insert seq=5, insert seq=7, delete seq=6
    mt.apply_insert(insert_batch(wal_data_schema, [
        (5, "_default", "X", [0.5, 0.5], "old"),
    ]))
    mt.apply_insert(insert_batch(wal_data_schema, [
        (7, "_default", "X", [0.875, 1.25], "new"),
    ]))
    mt.apply_delete(delete_batch(wal_delta_schema, ["X"], seq=6))

    # X must still be alive with the seq=7 value.
    rec = mt.get("X")
    assert rec is not None, "seq=7 insert was wrongly shadowed by seq=6 delete"
    assert rec["title"] == "new"
    assert rec["vec"] == [0.875, 1.25]


def test_all_six_orderings_give_same_state(mt, wal_data_schema, wal_delta_schema):
    """
    Three operations on pk=X with seqs 5, 6, 7. Try every permutation
    and assert the final state is the same: X alive with title="new".
    """
    from itertools import permutations
    from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema

    def make_mt():
        return MemTable(CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
            FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        ]))

    ops = [
        ("insert", 5, "old"),
        ("delete", 6, None),
        ("insert", 7, "new"),
    ]

    for perm in permutations(ops):
        mt2 = make_mt()
        for kind, seq, title in perm:
            if kind == "insert":
                mt2.apply_insert(insert_batch(wal_data_schema, [
                    (seq, "_default", "X", [0.1, 0.1], title),
                ]))
            else:
                mt2.apply_delete(delete_batch(wal_delta_schema, ["X"], seq=seq))
        rec = mt2.get("X")
        assert rec is not None, f"perm {perm}: X disappeared"
        assert rec["title"] == "new", f"perm {perm}: title={rec['title']}"


# ---------------------------------------------------------------------------
# Re-insert after delete (then delete again — typical churn pattern)
# ---------------------------------------------------------------------------

def test_insert_delete_insert_delete_pattern(mt, wal_data_schema, wal_delta_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "X", [0.1, 0.1], "v1"),
    ]))
    assert mt.get("X")["title"] == "v1"

    mt.apply_delete(delete_batch(wal_delta_schema, ["X"], seq=2))
    assert mt.get("X") is None

    mt.apply_insert(insert_batch(wal_data_schema, [
        (3, "_default", "X", [0.3, 0.3], "v2"),
    ]))
    assert mt.get("X")["title"] == "v2"

    mt.apply_delete(delete_batch(wal_delta_schema, ["X"], seq=4))
    assert mt.get("X") is None


# ---------------------------------------------------------------------------
# Partition filtering on get_active_records
# ---------------------------------------------------------------------------

def test_get_active_records_all(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "p1", "a", [0.1, 0.1], "from_p1"),
        (2, "p2", "b", [0.2, 0.2], "from_p2"),
        (3, "_default", "c", [0.3, 0.3], "from_default"),
    ]))
    recs = mt.get_active_records()
    assert len(recs) == 3
    assert {r["id"] for r in recs} == {"a", "b", "c"}


def test_get_active_records_partition_filter(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "p1", "a", [0.1, 0.1], "from_p1"),
        (2, "p2", "b", [0.2, 0.2], "from_p2"),
        (3, "_default", "c", [0.3, 0.3], "from_default"),
    ]))
    recs = mt.get_active_records(partition_names=["p1"])
    assert len(recs) == 1
    assert recs[0]["id"] == "a"

    recs = mt.get_active_records(partition_names=["p1", "p2"])
    assert {r["id"] for r in recs} == {"a", "b"}


# ---------------------------------------------------------------------------
# Search snapshots
# ---------------------------------------------------------------------------

def test_search_snapshot_arrays_and_filter_table_are_aligned(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "a", [0.1, 0.1], "from_default"),
        (2, "p1", "b", [0.2, 0.2], "from_p1"),
    ]))

    pks, seqs, vectors, row_refs, table = mt.to_search_snapshot(
        vector_field="vec",
        partition_names=None,
        include_table=True,
    )

    assert pks == ["a", "b"]
    assert seqs.tolist() == [1, 2]
    assert vectors.shape == (2, 2)
    assert vectors.ravel().tolist() == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert row_refs == [(0, 0), (0, 1)]
    assert table is not None
    assert table.num_rows == len(pks)
    assert table.column("id").to_pylist() == pks
    assert "_partition" not in table.column_names

    # Later writes must not change the previously captured filter table.
    mt.apply_insert(insert_batch(wal_data_schema, [
        (3, "_default", "c", [0.3, 0.3], "later"),
    ]))
    assert table.num_rows == 2
    assert table.column("id").to_pylist() == ["a", "b"]


def test_search_snapshot_partition_filter_alignment(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "a", [0.1, 0.1], "from_default"),
        (2, "p1", "b", [0.2, 0.2], "from_p1"),
        (3, "p2", "c", [0.3, 0.3], "from_p2"),
    ]))

    pks, seqs, vectors, row_refs, table = mt.to_search_snapshot(
        vector_field="vec",
        partition_names=["p1", "p2"],
        include_table=True,
    )

    assert pks == ["b", "c"]
    assert seqs.tolist() == [2, 3]
    assert vectors.shape == (2, 2)
    assert vectors.ravel().tolist() == pytest.approx([0.2, 0.2, 0.3, 0.3])
    assert row_refs == [(0, 1), (0, 2)]
    assert table is not None
    assert table.num_rows == len(pks)
    assert table.column("id").to_pylist() == pks


# ---------------------------------------------------------------------------
# size() correctness (active pk count, not physical row count)
# ---------------------------------------------------------------------------

def test_size_counts_active_pks_not_physical_rows(mt, wal_data_schema):
    """Repeatedly upsert the same pk — size stays at 1, physical grows."""
    for i in range(5):
        mt.apply_insert(insert_batch(wal_data_schema, [
            (i + 1, "_default", "X", [float(i), float(i)], f"v{i}"),
        ]))
    assert mt.size() == 1  # one active pk
    assert mt.num_physical_rows() == 5  # five physical rows accumulated


def test_size_includes_tombstones(mt, wal_data_schema, wal_delta_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "a", [0.1, 0.1], "x"),
        (2, "_default", "b", [0.2, 0.2], "y"),
    ]))
    mt.apply_delete(delete_batch(wal_delta_schema, ["a"], seq=3))
    # a is now in delete_index, b is still in pk_index → size = 2
    assert mt.size() == 2


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_apply_insert_bad_schema_raises(mt):
    bad = pa.RecordBatch.from_pydict(
        {"foo": [1], "bar": [2]},
        schema=pa.schema([pa.field("foo", pa.int64()), pa.field("bar", pa.int64())]),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        mt.apply_insert(bad)


def test_apply_delete_bad_schema_raises(mt):
    bad = pa.RecordBatch.from_pydict(
        {"_seq": [1]},
        schema=pa.schema([pa.field("_seq", pa.uint64())]),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        mt.apply_delete(bad)


# ---------------------------------------------------------------------------
# flush() — Phase 3
# ---------------------------------------------------------------------------

def test_flush_empty(mt):
    result = mt.flush()
    assert result == {}


def test_flush_inserts_only_single_partition(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "a", [0.5, 0.25], "x"),
        (2, "_default", "b", [0.75, 0.125], "y"),
    ]))
    result = mt.flush()
    assert set(result.keys()) == {"_default"}
    data_table, delta_table = result["_default"]
    assert data_table is not None
    assert delta_table is None
    assert data_table.num_rows == 2

    # data_table uses data_schema (no _partition)
    assert "_partition" not in data_table.schema.names
    assert "_seq" in data_table.schema.names
    assert {pk for pk in data_table.column("id").to_pylist()} == {"a", "b"}


def test_flush_inserts_multi_partition(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "p1", "a", [0.5, 0.25], "x"),
        (2, "p2", "b", [0.75, 0.125], "y"),
        (3, "p1", "c", [0.0625, 1.5], "z"),
    ]))
    result = mt.flush()
    assert set(result.keys()) == {"p1", "p2"}

    p1_data, p1_delta = result["p1"]
    assert p1_data is not None
    assert p1_data.num_rows == 2  # a, c
    assert {x for x in p1_data.column("id").to_pylist()} == {"a", "c"}
    assert p1_delta is None

    p2_data, _ = result["p2"]
    assert p2_data.num_rows == 1
    assert p2_data.column("id").to_pylist() == ["b"]


def test_flush_dedup_via_pk_index(mt, wal_data_schema):
    """Repeatedly upsert the same pk — flush emits one row, not all 5."""
    for i in range(5):
        mt.apply_insert(insert_batch(wal_data_schema, [
            (i + 1, "_default", "X", [float(i), 0.5], f"v{i}"),
        ]))
    result = mt.flush()
    data_table, _ = result["_default"]
    assert data_table.num_rows == 1
    # Latest version (seq=5)
    [row] = data_table.to_pylist()
    assert row["title"] == "v4"
    assert row["_seq"] == 5


def test_flush_deletes_only(mt, wal_delta_schema):
    mt.apply_delete(delete_batch(wal_delta_schema, ["a", "b", "c"], seq=10, partition="p1"))
    result = mt.flush()
    assert set(result.keys()) == {"p1"}
    data_table, delta_table = result["p1"]
    assert data_table is None
    assert delta_table is not None
    assert delta_table.num_rows == 3
    assert "_partition" not in delta_table.schema.names
    assert set(delta_table.column("id").to_pylist()) == {"a", "b", "c"}
    assert set(delta_table.column("_seq").to_pylist()) == {10}


def test_flush_inserts_and_deletes_same_partition(mt, wal_data_schema, wal_delta_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "a", [0.5, 0.25], "x"),
    ]))
    mt.apply_delete(delete_batch(wal_delta_schema, ["b"], seq=2, partition="_default"))

    result = mt.flush()
    assert set(result.keys()) == {"_default"}
    data_table, delta_table = result["_default"]
    assert data_table is not None and data_table.num_rows == 1
    assert delta_table is not None and delta_table.num_rows == 1


def test_flush_cross_partition_delete_no_known_partitions(mt, wal_delta_schema):
    """Without known_partitions arg, _all deletes stay under _all bucket."""
    mt.apply_delete(delete_batch(wal_delta_schema, ["X"], seq=5, partition="_all"))
    result = mt.flush()
    assert "_all" in result
    _, delta = result["_all"]
    assert delta is not None
    assert delta.num_rows == 1


def test_flush_cross_partition_delete_replicated(mt, wal_delta_schema):
    """With known_partitions, _all deletes are replicated into each."""
    mt.apply_delete(delete_batch(wal_delta_schema, ["X"], seq=5, partition="_all"))
    result = mt.flush(known_partitions=["_default", "p1", "p2"])
    assert set(result.keys()) == {"_default", "p1", "p2"}
    for part in ("_default", "p1", "p2"):
        _, delta = result[part]
        assert delta is not None
        assert delta.column("id").to_pylist() == ["X"]
        assert delta.column("_seq").to_pylist() == [5]
    assert "_all" not in result


def test_flush_does_not_clear_internal_state(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "a", [0.5, 0.25], "x"),
    ]))
    mt.flush()
    # Caller is expected to drop the MemTable, but the contract says
    # internal state is not auto-cleared.
    assert mt.size() == 1
    assert mt.get("a") is not None


def test_flush_schema_matches_data_schema(mt, wal_data_schema, schema):
    """The data_table emitted by flush must match build_data_schema(schema)."""
    from milvus_lite.schema.arrow_builder import build_data_schema

    mt.apply_insert(insert_batch(wal_data_schema, [
        (1, "_default", "a", [0.5, 0.25], "x"),
    ]))
    result = mt.flush()
    data_table, _ = result["_default"]
    expected = build_data_schema(schema)
    assert data_table.schema == expected


def test_flush_schema_matches_delta_schema(mt, wal_delta_schema, schema):
    from milvus_lite.schema.arrow_builder import build_delta_schema

    mt.apply_delete(delete_batch(wal_delta_schema, ["x"], seq=1, partition="_default"))
    result = mt.flush()
    _, delta_table = result["_default"]
    expected = build_delta_schema(schema)
    assert delta_table.schema == expected


# ---------------------------------------------------------------------------
# max_seq
# ---------------------------------------------------------------------------

def test_max_seq_empty(mt):
    assert mt.max_seq == -1


def test_max_seq_inserts(mt, wal_data_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (3, "_default", "a", [0.5, 0.25], "x"),
        (5, "_default", "b", [0.75, 0.125], "y"),
        (7, "_default", "c", [0.0625, 1.5], "z"),
    ]))
    assert mt.max_seq == 7


def test_max_seq_mixed_insert_delete(mt, wal_data_schema, wal_delta_schema):
    mt.apply_insert(insert_batch(wal_data_schema, [
        (3, "_default", "a", [0.5, 0.25], "x"),
    ]))
    mt.apply_delete(delete_batch(wal_delta_schema, ["b"], seq=10))
    assert mt.max_seq == 10
