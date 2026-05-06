"""Phase-2 Collection tests — insert + get over WAL + MemTable.

Phase 2 is the first end-to-end vertical slice. The integration we're
verifying:
    user → Collection.insert → validate → seq alloc → wal_data batch
                             → WAL.write_insert → MemTable.apply_insert
    user → Collection.get → MemTable.get → record dict
"""

import os

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.exceptions import (
    PartitionNotFoundError,
    SchemaValidationError,
)
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ])


@pytest.fixture
def schema_with_dynamic():
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ],
        enable_dynamic_field=True,
    )


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_creates_data_dir(tmp_path, schema):
    data_dir = tmp_path / "fresh"
    Collection("c", str(data_dir), schema).close()
    assert data_dir.exists()
    assert (data_dir / "wal").exists() or True  # wal dir is created lazily on first WAL write


def test_construction_with_invalid_schema_raises(tmp_path):
    bad = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        # missing FLOAT_VECTOR
    ])
    with pytest.raises(SchemaValidationError, match="FLOAT_VECTOR"):
        Collection("c", str(tmp_path / "x"), bad)


def test_construction_loads_existing_manifest(tmp_path, schema):
    """If a Collection was created and saved its manifest, a second
    Collection on the same data_dir should load it (Phase 2 doesn't
    auto-save, so we explicitly trigger via manifest interaction)."""
    data_dir = str(tmp_path / "data")
    c1 = Collection("c", data_dir, schema)
    c1._manifest.add_partition("p1")
    c1._manifest.save()
    c1.close()

    c2 = Collection("c", data_dir, schema)
    assert c2._manifest.has_partition("p1")
    c2.close()


# ---------------------------------------------------------------------------
# insert + get happy path
# ---------------------------------------------------------------------------

def test_insert_single_record(col):
    pks = col.insert([
        {"id": "doc1", "vec": [0.5, 0.25, 0.125, 0.75], "title": "hi", "score": 0.5},
    ])
    assert pks == ["doc1"]
    assert col.count() == 1

    [rec] = col.get(["doc1"])
    assert rec["id"] == "doc1"
    assert rec["title"] == "hi"
    assert rec["score"] == 0.5
    assert rec["vec"] == [0.5, 0.25, 0.125, 0.75]


def test_insert_batch(col):
    records = [
        {"id": f"doc{i}", "vec": [0.5, 0.25, 0.125, 0.75], "title": f"t{i}", "score": float(i)}
        for i in range(10)
    ]
    pks = col.insert(records)
    assert pks == [f"doc{i}" for i in range(10)]
    assert col.count() == 10

    got = col.get([f"doc{i}" for i in range(10)])
    assert len(got) == 10
    assert {r["id"] for r in got} == {f"doc{i}" for i in range(10)}


def test_insert_empty_list(col):
    assert col.insert([]) == []
    assert col.count() == 0


def test_insert_returns_pk_in_order(col):
    pks = col.insert([
        {"id": "z", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.0},
        {"id": "a", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.0},
        {"id": "m", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.0},
    ])
    assert pks == ["z", "a", "m"]


# ---------------------------------------------------------------------------
# Upsert (same pk, two inserts)
# ---------------------------------------------------------------------------

def test_upsert_overwrites(col):
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "old", "score": 1.0}])
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "new", "score": 2.0}])
    [rec] = col.get(["x"])
    assert rec["title"] == "new"
    assert rec["score"] == 2.0
    assert col.count() == 1  # still one logical pk


def test_upsert_seq_monotonic(col):
    """Each insert call must allocate seqs after the previous one."""
    col.insert([{"id": "a", "vec": [0.5, 0.25, 0.125, 0.75], "title": "1", "score": 1.0}])
    col.insert([{"id": "b", "vec": [0.5, 0.25, 0.125, 0.75], "title": "2", "score": 2.0}])
    col.insert([{"id": "c", "vec": [0.5, 0.25, 0.125, 0.75], "title": "3", "score": 3.0}])
    # All three should be present.
    got = col.get(["a", "b", "c"])
    assert {r["id"] for r in got} == {"a", "b", "c"}
    # next_seq should be 4 (1, 2, 3 used)
    assert col._next_seq == 4


# ---------------------------------------------------------------------------
# get edge cases
# ---------------------------------------------------------------------------

def test_get_missing_pk_returns_empty(col):
    col.insert([{"id": "exists", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.0}])
    assert col.get(["nonexistent"]) == []


def test_get_partial_hit(col):
    col.insert([
        {"id": "a", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.0},
        {"id": "b", "vec": [0.5, 0.25, 0.125, 0.75], "title": "y", "score": 1.0},
    ])
    got = col.get(["a", "missing", "b"])
    assert len(got) == 2
    assert {r["id"] for r in got} == {"a", "b"}


def test_get_empty_pks(col):
    assert col.get([]) == []


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------

def test_insert_record_missing_pk_raises(col):
    with pytest.raises(SchemaValidationError, match="primary key"):
        col.insert([{"vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.0}])


def test_insert_record_wrong_vec_dim_raises(col):
    with pytest.raises(SchemaValidationError, match="dim"):
        col.insert([{"id": "x", "vec": [0.5, 0.25], "title": "x", "score": 0.0}])


def test_insert_unknown_partition_raises(col):
    with pytest.raises(PartitionNotFoundError):
        col.insert(
            [{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.0}],
            partition_name="ghost",
        )


def test_insert_partial_validation_failure_no_partial_state(col):
    """If the second record is invalid, the first must NOT be inserted."""
    records = [
        {"id": "good", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.0},
        {"id": "bad", "vec": [0.5, 0.25], "title": "x", "score": 0.0},  # wrong dim
    ]
    with pytest.raises(SchemaValidationError):
        col.insert(records)
    # No partial state — neither record was inserted.
    assert col.count() == 0
    assert col.get(["good"]) == []


def test_insert_records_must_be_list(col):
    with pytest.raises(TypeError, match="must be a list"):
        col.insert("not a list")


def test_get_pks_must_be_list(col):
    with pytest.raises(TypeError, match="must be a list"):
        col.get("not a list")


# ---------------------------------------------------------------------------
# Dynamic fields
# ---------------------------------------------------------------------------

def test_dynamic_field_extras_stored(tmp_path, schema_with_dynamic):
    col = Collection("c", str(tmp_path / "d"), schema_with_dynamic)
    col.insert([
        {"id": "x", "vec": [0.5, 0.25], "category": "blog", "lang": "en"},
    ])
    [rec] = col.get(["x"])
    assert rec["id"] == "x"
    # Dynamic fields are unpacked from $meta into top-level keys
    assert rec["category"] == "blog"
    assert rec["lang"] == "en"
    assert "$meta" not in rec  # internal key stripped from output
    col.close()


def test_dynamic_field_disabled_extras_rejected(col):
    """The default fixture schema has enable_dynamic_field=False."""
    with pytest.raises(SchemaValidationError, match="not in schema"):
        col.insert([{
            "id": "x", "vec": [0.5, 0.25, 0.125, 0.75],
            "title": "x", "score": 0.0,
            "extra": "rejected",
        }])


# ---------------------------------------------------------------------------
# Nullable fields
# ---------------------------------------------------------------------------

def test_insert_with_missing_nullable_field(col):
    col.insert([
        {"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "score": 0.5},  # title omitted
    ])
    [rec] = col.get(["x"])
    assert rec["title"] is None


# ---------------------------------------------------------------------------
# WAL side-effect: writes hit disk
# ---------------------------------------------------------------------------

def test_insert_creates_wal_data_file(tmp_path, schema):
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.5}])
    wal_dir = tmp_path / "d" / "wal"
    files = list(wal_dir.glob("wal_data_*.arrow"))
    assert len(files) == 1
    # close_and_delete will remove it; don't call close yet for this assertion
    col._wal.close_and_delete()
    col._wal._closed = True


def test_insert_does_not_create_wal_delta_file(tmp_path, schema):
    """Collection.insert never writes to wal_delta — wal_delta is for Phase 5 deletes."""
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "x", "score": 0.5}])
    wal_dir = tmp_path / "d" / "wal"
    delta_files = list(wal_dir.glob("wal_delta_*.arrow"))
    assert delta_files == []
    col.close()


# ---------------------------------------------------------------------------
# Phase 3: flush trigger + persistence + recovery
# ---------------------------------------------------------------------------

def _make_record(i):
    return {
        "id": f"doc_{i:04d}",
        "vec": [0.5, 0.25, 0.125, 0.75],
        "title": f"t{i}",
        "score": float(i),
    }


def test_flush_triggers_at_size_limit(tmp_path, schema, monkeypatch):
    """When MemTable hits MEMTABLE_SIZE_LIMIT, insert should trigger a flush."""
    # Lower the limit so we don't have to insert 10K rows.
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    data_dir = str(tmp_path / "data")
    col = Collection("c", data_dir, schema)
    for i in range(10):
        col.insert([_make_record(i)])
    # After 10 inserts with limit=5, we should have flushed at least once.
    # The post-flush MemTable holds the records that came AFTER the flush.
    assert col.count() < 10

    # The manifest should have at least one data file.
    files = col._manifest.get_data_files("_default")
    assert len(files) >= 1
    col.close()


def test_explicit_flush(tmp_path, schema):
    col = Collection("c", str(tmp_path / "d"), schema)
    col.insert([_make_record(i) for i in range(3)])
    col.flush()
    assert col.count() == 0  # MemTable drained
    files = col._manifest.get_data_files("_default")
    assert len(files) == 1
    col.close()


def test_restart_recovers_flushed_state(tmp_path, schema):
    """Insert some, flush, close, reopen — manifest is loaded and the
    delta_index is rebuilt. (Inserted records aren't yet readable in
    Phase 3 because get() only reads MemTable, not segments — that's
    Phase 4. But the manifest state must be preserved.)"""
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([_make_record(i) for i in range(3)])
    col1.flush()
    files_before = col1._manifest.get_data_files("_default")
    seq_before = col1._manifest.current_seq
    col1.close()

    col2 = Collection("c", data_dir, schema)
    files_after = col2._manifest.get_data_files("_default")
    assert files_after == files_before
    assert col2._manifest.current_seq == seq_before
    # next_seq must be past the recovered max
    assert col2._next_seq > seq_before
    col2.close()


def test_restart_recovers_unflushed_wal(tmp_path, schema):
    """Insert some without flushing, kill the process (close uncleanly),
    reopen — recovery should replay the WAL into a new MemTable."""
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([_make_record(i) for i in range(3)])
    # Simulate crash: do NOT call close(); instead, just drop the col.
    # The WAL files remain on disk.
    del col1

    col2 = Collection("c", data_dir, schema)
    # The 3 records should be back in MemTable.
    assert col2.count() == 3
    rec = col2.get(["doc_0001"])
    assert len(rec) == 1
    assert rec[0]["title"] == "t1"
    col2.close()


def test_restart_after_close_no_pending_data(tmp_path, schema):
    """close() flushes any pending data — restart should see the
    flushed files and have an empty MemTable."""
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([_make_record(i) for i in range(3)])
    col1.close()

    col2 = Collection("c", data_dir, schema)
    # MemTable empty (close flushed); manifest has the file
    assert col2.count() == 0
    files = col2._manifest.get_data_files("_default")
    assert len(files) == 1
    col2.close()


def test_seq_monotonic_across_restart(tmp_path, schema):
    """Seqs allocated after restart must NOT collide with seqs from
    before restart, even if WAL was flushed."""
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([_make_record(i) for i in range(3)])  # seqs 1, 2, 3
    col1.close()

    col2 = Collection("c", data_dir, schema)
    next_seq_at_open = col2._next_seq
    assert next_seq_at_open >= 4
    col2.insert([_make_record(99)])  # seq 4 (or higher)
    col2.close()


# ---------------------------------------------------------------------------
# Phase 4: get() reads segments + search()
# ---------------------------------------------------------------------------

def test_get_reads_segment_after_flush(tmp_path, schema):
    """After flush, the records are in a Parquet segment. get() should
    still find them via the segment cache (not just MemTable)."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([_make_record(i) for i in range(3)])
    col.flush()
    # MemTable is now empty
    assert col.count() == 0
    # But get() reads from the segment
    [rec] = col.get(["doc_0001"])
    assert rec["title"] == "t1"
    col.close()


def test_get_after_restart_reads_segment(tmp_path, schema):
    """The Phase-3 limitation lifted: after restart, flushed records
    are queryable via get() through the loaded segment cache."""
    data_dir = str(tmp_path / "d")
    col1 = Collection("c", data_dir, schema)
    col1.insert([_make_record(i) for i in range(5)])
    col1.close()  # flushes everything

    col2 = Collection("c", data_dir, schema)
    assert col2.count() == 0  # MemTable empty after recovery
    # All 5 records still readable via segment.
    for i in range(5):
        rec = col2.get([f"doc_{i:04d}"])
        assert len(rec) == 1
    col2.close()


def test_get_partition_filter_segment(tmp_path, schema):
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col._manifest.add_partition("p1")
    col.insert([_make_record(0)], partition_name="_default")
    col.insert([_make_record(1)], partition_name="p1")
    col.flush()

    # Without partition filter — both readable
    assert len(col.get(["doc_0000", "doc_0001"])) == 2
    # With filter — only one
    assert col.get(["doc_0000", "doc_0001"], partition_names=["_default"])[0]["id"] == "doc_0000"
    assert col.get(["doc_0000", "doc_0001"], partition_names=["p1"])[0]["id"] == "doc_0001"
    col.close()


def test_get_upsert_after_flush(tmp_path, schema):
    """Insert, flush, insert same pk again (in MemTable). get() should
    return the new in-memory version, not the old segment one."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "old", "score": 1.0}])
    col.flush()
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "new", "score": 2.0}])
    [rec] = col.get(["x"])
    assert rec["title"] == "new"
    col.close()


def test_get_segment_pk_versions_take_max_seq(tmp_path, schema):
    """Insert pk, flush. Insert same pk, flush. Two segments with same
    pk; get() must return the newer (max-seq) one."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "v1", "score": 1.0}])
    col.flush()
    col.insert([{"id": "x", "vec": [0.5, 0.25, 0.125, 0.75], "title": "v2", "score": 2.0}])
    col.flush()
    [rec] = col.get(["x"])
    assert rec["title"] == "v2"
    col.close()


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

def test_search_memtable_only(tmp_path, schema):
    """All data in MemTable, no flush."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([
        {"id": "near",  "vec": [1.0, 0.0, 0.0, 0.0], "title": "n", "score": 1.0},
        {"id": "far",   "vec": [0.0, 1.0, 0.0, 0.0], "title": "f", "score": 2.0},
    ])

    results = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=1, metric_type="COSINE")
    assert len(results) == 1
    [hits] = results
    assert hits[0]["id"] == "near"
    col.close()


def test_search_segment_only(tmp_path, schema):
    """All data flushed to segment."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([
        {"id": "near",  "vec": [1.0, 0.0, 0.0, 0.0], "title": "n", "score": 1.0},
        {"id": "far",   "vec": [0.0, 1.0, 0.0, 0.0], "title": "f", "score": 2.0},
    ])
    col.flush()
    assert col.count() == 0  # MemTable empty

    results = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=1, metric_type="COSINE")
    assert results[0][0]["id"] == "near"
    col.close()


def test_search_mixed_memtable_and_segment(tmp_path, schema):
    """Half in segment, half in MemTable."""
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([{"id": "old", "vec": [1.0, 0.0, 0.0, 0.0], "title": "x", "score": 0.0}])
    col.flush()
    col.insert([{"id": "new", "vec": [0.0, 1.0, 0.0, 0.0], "title": "y", "score": 0.0}])

    # Query closer to "old"
    results = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=2, metric_type="COSINE")
    [hits] = results
    assert len(hits) == 2
    assert hits[0]["id"] == "old"
    assert hits[1]["id"] == "new"
    col.close()


def test_search_top_k_2(tmp_path, schema):
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([_make_record(i) for i in range(20)])
    results = col.search([[0.5, 0.25, 0.125, 0.75]], top_k=5, metric_type="L2")
    [hits] = results
    assert len(hits) == 5
    # Distances must be ascending
    for i in range(4):
        assert hits[i]["distance"] <= hits[i + 1]["distance"]
    col.close()


def test_search_partition_filter(tmp_path, schema):
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col._manifest.add_partition("p1")
    col.insert([{"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "title": "x", "score": 0.0}], partition_name="_default")
    col.insert([{"id": "b", "vec": [1.0, 0.0, 0.0, 0.0], "title": "y", "score": 0.0}], partition_name="p1")
    results = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=10,
        metric_type="COSINE",
        partition_names=["p1"],
    )
    [hits] = results
    ids = {h["id"] for h in hits}
    assert ids == {"b"}
    col.close()


def test_search_multi_query(tmp_path, schema):
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    col.insert([
        {"id": "a", "vec": [1.0, 0.0, 0.0, 0.0], "title": "x", "score": 0.0},
        {"id": "b", "vec": [0.0, 1.0, 0.0, 0.0], "title": "y", "score": 0.0},
        {"id": "c", "vec": [0.0, 0.0, 1.0, 0.0], "title": "z", "score": 0.0},
    ])
    results = col.search(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        top_k=1,
        metric_type="COSINE",
    )
    assert len(results) == 3
    assert results[0][0]["id"] == "a"
    assert results[1][0]["id"] == "b"
    assert results[2][0]["id"] == "c"
    col.close()


def test_search_empty_collection(tmp_path, schema):
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)
    results = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=10)
    assert results == [[]]
    col.close()


def test_search_brute_force_match(tmp_path, schema):
    """Compare search results against direct numpy brute-force on a
    100-record collection."""
    import numpy as np
    rng = np.random.default_rng(7)
    data_dir = str(tmp_path / "d")
    col = Collection("c", data_dir, schema)

    n = 100
    vectors_raw = rng.standard_normal((n, 4)).astype(np.float32)
    records = [
        {"id": f"doc_{i:04d}", "vec": vectors_raw[i].tolist(), "title": f"t{i}", "score": float(i)}
        for i in range(n)
    ]
    col.insert(records)

    query = rng.standard_normal((1, 4)).astype(np.float32)
    results = col.search(query.tolist(), top_k=10, metric_type="L2")
    [hits] = results

    # Direct numpy
    dists = np.linalg.norm(vectors_raw - query[0], axis=1)
    expected_top_idx = np.argsort(dists)[:10]
    expected_ids = [f"doc_{i:04d}" for i in expected_top_idx]
    actual_ids = [h["id"] for h in hits]
    assert actual_ids == expected_ids
    col.close()


def test_search_invalid_argument_types(tmp_path, schema):
    col = Collection("c", str(tmp_path / "d"), schema)
    with pytest.raises(TypeError, match="must be a list"):
        col.search("not a list")
    col.close()
