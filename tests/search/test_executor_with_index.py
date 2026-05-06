"""Phase 9.2.5 — Differential test: execute_search_with_index ≡ execute_search.

Both paths must produce equivalent top-k for any combination of:
    - segments + memtable layout (memtable-only, segment-only, mixed)
    - upserts (same pk in multiple sources)
    - tombstones (delete + re-insert)
    - partition filters
    - scalar filter expressions
    - all three metrics
    - output_fields projection (None / [] / list)

The new path uses BruteForceIndex per source + global merge; the old
path concatenates everything and runs one big numpy distance call.
Result equivalence is validated by exact id-set match (vectors are
chosen so there are no distance ties).
"""

from typing import List

import numpy as np
import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.search.assembler import assemble_candidates
from milvus_lite.search.executor import execute_search
from milvus_lite.search.executor_indexed import execute_search_with_index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="active", dtype=DataType.BOOL),
    ])


def _vec(seed: int) -> List[float]:
    """Deterministic 8-dim vector. Different seeds yield different vectors
    so distance ties are extremely unlikely."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(8).astype(np.float32).tolist()


@pytest.fixture
def col_mixed(tmp_path, schema):
    """A Collection with rows in segments AND in the live memtable,
    plus some upserts and a few tombstones."""
    c = Collection("diff", str(tmp_path / "data"), schema)
    # First batch → flush → segment
    c.insert([
        {"id": i, "vec": _vec(i), "title": f"t{i}", "score": float(i) / 10, "active": (i % 2 == 0)}
        for i in range(20)
    ])
    c.flush()

    # Upsert two existing pks (will land in memtable; segment versions are stale)
    c.insert([
        {"id": 5, "vec": _vec(500), "title": "upsert5", "score": 9.99, "active": False},
        {"id": 12, "vec": _vec(120), "title": "upsert12", "score": 8.88, "active": True},
    ])

    # Delete two pks via tombstone
    c.delete(pks=[3, 7])

    # Add a few brand-new memtable rows
    c.insert([
        {"id": i, "vec": _vec(i), "title": f"t{i}", "score": float(i) / 10, "active": (i % 2 == 0)}
        for i in range(20, 25)
    ])

    yield c
    c.close()


@pytest.fixture
def col_two_partitions(tmp_path, schema):
    c = Collection("diff_p", str(tmp_path / "data"), schema)
    c.create_partition("p1")
    c.create_partition("p2")
    c.insert(
        [{"id": i, "vec": _vec(i), "title": "p1", "score": 0.5, "active": True} for i in range(10)],
        partition_name="p1",
    )
    c.insert(
        [{"id": 100 + i, "vec": _vec(100 + i), "title": "p2", "score": 0.5, "active": True} for i in range(10)],
        partition_name="p2",
    )
    c.flush()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Differential helper
# ---------------------------------------------------------------------------

def _run_old_path(
    col: Collection,
    queries,
    top_k,
    metric_type="COSINE",
    partition_names=None,
    expr=None,
    output_fields=None,
):
    q_arr = np.asarray(queries, dtype=np.float32)
    if q_arr.ndim != 2:
        raise ValueError("queries must be 2-D")

    compiled_filter = col._compile_filter(expr) if expr else None  # noqa: SLF001
    all_pks, all_seqs, all_vectors, all_records, filter_mask = assemble_candidates(
        segments=col._segment_cache.values(),  # noqa: SLF001
        memtable=col._memtable,                # noqa: SLF001
        vector_field=col._vector_name,         # noqa: SLF001
        partition_names=partition_names,
        filter_compiled=compiled_filter,
    )
    return execute_search(
        query_vectors=q_arr,
        all_pks=all_pks,
        all_seqs=all_seqs,
        all_vectors=all_vectors,
        all_records=all_records,
        delta_index=col._delta_index,          # noqa: SLF001
        top_k=top_k,
        metric_type=metric_type,
        pk_field=col._pk_name,                  # noqa: SLF001
        vector_field=col._vector_name,         # noqa: SLF001
        filter_mask=filter_mask,
        output_fields=output_fields,
        memtable=col._memtable,                # noqa: SLF001
    )


def _run_new_path(
    col: Collection,
    queries,
    top_k,
    metric_type="COSINE",
    partition_names=None,
    expr=None,
    output_fields=None,
):
    q_arr = np.asarray(queries, dtype=np.float32)
    compiled_filter = col._compile_filter(expr) if expr else None  # noqa: SLF001
    return execute_search_with_index(
        query_vectors=q_arr,
        segments=col._segment_cache.values(),  # noqa: SLF001
        memtable=col._memtable,                # noqa: SLF001
        delta_index=col._delta_index,          # noqa: SLF001
        top_k=top_k,
        metric_type=metric_type,
        pk_field=col._pk_name,                  # noqa: SLF001
        vector_field=col._vector_name,         # noqa: SLF001
        partition_names=partition_names,
        compiled_filter=compiled_filter,
        output_fields=output_fields,
    )


def _assert_results_equal(old, new):
    """Assert two result lists (List[List[dict]]) are equivalent.

    Distance ties can produce different orderings between old and new
    paths, so we compare ID sets and per-id distances rather than
    strict positional equality.
    """
    assert len(old) == len(new), f"nq mismatch: {len(old)} vs {len(new)}"
    for q in range(len(old)):
        old_q = old[q]
        new_q = new[q]
        assert len(old_q) == len(new_q), (
            f"top-k size mismatch on query {q}: old={len(old_q)} new={len(new_q)}"
        )
        old_by_id = {r["id"]: r["distance"] for r in old_q}
        new_by_id = {r["id"]: r["distance"] for r in new_q}
        assert set(old_by_id.keys()) == set(new_by_id.keys()), (
            f"id-set mismatch on query {q}:\n"
            f"  old: {sorted(old_by_id.keys())}\n"
            f"  new: {sorted(new_by_id.keys())}"
        )
        for pk in old_by_id:
            assert abs(old_by_id[pk] - new_by_id[pk]) < 1e-4, (
                f"distance mismatch for pk={pk}: old={old_by_id[pk]} new={new_by_id[pk]}"
            )


# ---------------------------------------------------------------------------
# Differential tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
def test_diff_basic(col_mixed, metric):
    queries = [_vec(900), _vec(901)]
    old = _run_old_path(col_mixed, queries, top_k=5, metric_type=metric)
    new = _run_new_path(col_mixed, queries, top_k=5, metric_type=metric)
    _assert_results_equal(old, new)


def test_diff_top_k_larger_than_collection(col_mixed):
    queries = [_vec(900)]
    old = _run_old_path(col_mixed, queries, top_k=200, metric_type="L2")
    new = _run_new_path(col_mixed, queries, top_k=200, metric_type="L2")
    _assert_results_equal(old, new)


def test_diff_with_filter_expr(col_mixed):
    queries = [_vec(900)]
    old = _run_old_path(
        col_mixed, queries, top_k=10, expr="active == true", metric_type="COSINE"
    )
    new = _run_new_path(
        col_mixed, queries, top_k=10, expr="active == true", metric_type="COSINE"
    )
    _assert_results_equal(old, new)


def test_diff_with_complex_filter(col_mixed):
    queries = [_vec(900)]
    expr = "score > 0.5 and id < 20"
    old = _run_old_path(col_mixed, queries, top_k=5, expr=expr, metric_type="L2")
    new = _run_new_path(col_mixed, queries, top_k=5, expr=expr, metric_type="L2")
    _assert_results_equal(old, new)


def test_diff_partition_filter(col_two_partitions):
    queries = [_vec(900)]
    old = _run_old_path(
        col_two_partitions, queries, top_k=5, partition_names=["p1"]
    )
    new = _run_new_path(
        col_two_partitions, queries, top_k=5, partition_names=["p1"]
    )
    _assert_results_equal(old, new)
    # Sanity: only p1 ids (0..9) should be present.
    assert all(r["id"] < 100 for r in new[0])


def test_diff_partition_filter_multiple(col_two_partitions):
    queries = [_vec(900)]
    old = _run_old_path(
        col_two_partitions, queries, top_k=20, partition_names=["p1", "p2"]
    )
    new = _run_new_path(
        col_two_partitions, queries, top_k=20, partition_names=["p1", "p2"]
    )
    _assert_results_equal(old, new)


def test_diff_output_fields_none(col_mixed):
    queries = [_vec(900)]
    old = _run_old_path(col_mixed, queries, top_k=5, output_fields=None)
    new = _run_new_path(col_mixed, queries, top_k=5, output_fields=None)
    _assert_results_equal(old, new)
    # Both should have the same entity field set.
    for o, n in zip(old[0], new[0]):
        assert set(o["entity"].keys()) == set(n["entity"].keys())


def test_diff_output_fields_subset(col_mixed):
    queries = [_vec(900)]
    of = ["title", "score"]
    old = _run_old_path(col_mixed, queries, top_k=5, output_fields=of)
    new = _run_new_path(col_mixed, queries, top_k=5, output_fields=of)
    _assert_results_equal(old, new)
    for r in new[0]:
        assert set(r["entity"].keys()) == {"title", "score"}


def test_diff_output_fields_empty(col_mixed):
    queries = [_vec(900)]
    old = _run_old_path(col_mixed, queries, top_k=5, output_fields=[])
    new = _run_new_path(col_mixed, queries, top_k=5, output_fields=[])
    _assert_results_equal(old, new)
    for r in new[0]:
        assert r["entity"] == {}


def test_diff_multi_query_topk(col_mixed):
    queries = [_vec(800), _vec(801), _vec(802)]
    old = _run_old_path(col_mixed, queries, top_k=7, metric_type="COSINE")
    new = _run_new_path(col_mixed, queries, top_k=7, metric_type="COSINE")
    _assert_results_equal(old, new)


def test_diff_after_compaction(tmp_path, schema):
    """Trigger multiple flushes to create multiple segments, then run
    the diff. This exercises cross-segment dedup + bitmap pipeline."""
    c = Collection("diff_compact", str(tmp_path / "data"), schema)
    try:
        for batch in range(4):
            c.insert([
                {"id": batch * 10 + i, "vec": _vec(batch * 10 + i),
                 "title": f"b{batch}", "score": 0.5, "active": True}
                for i in range(10)
            ])
            c.flush()
        # Upsert across segments
        c.insert([
            {"id": 5, "vec": _vec(5005), "title": "upsert5", "score": 9.0, "active": False},
            {"id": 25, "vec": _vec(2525), "title": "upsert25", "score": 9.0, "active": False},
        ])
        c.flush()

        queries = [_vec(900), _vec(901)]
        old = _run_old_path(c, queries, top_k=8, metric_type="L2")
        new = _run_new_path(c, queries, top_k=8, metric_type="L2")
        _assert_results_equal(old, new)
    finally:
        c.close()


def test_diff_memtable_only(tmp_path, schema):
    """Edge case: no flushed segments, only live memtable rows."""
    c = Collection("diff_mt", str(tmp_path / "data"), schema)
    try:
        c.insert([
            {"id": i, "vec": _vec(i), "title": "x", "score": 0.5, "active": True}
            for i in range(15)
        ])
        # No flush
        queries = [_vec(900)]
        old = _run_old_path(c, queries, top_k=5, metric_type="COSINE")
        new = _run_new_path(c, queries, top_k=5, metric_type="COSINE")
        _assert_results_equal(old, new)
    finally:
        c.close()


def test_diff_segment_only(tmp_path, schema):
    """Edge case: all data flushed, memtable empty."""
    c = Collection("diff_seg", str(tmp_path / "data"), schema)
    try:
        c.insert([
            {"id": i, "vec": _vec(i), "title": "x", "score": 0.5, "active": True}
            for i in range(15)
        ])
        c.flush()
        queries = [_vec(900)]
        old = _run_old_path(c, queries, top_k=5, metric_type="L2")
        new = _run_new_path(c, queries, top_k=5, metric_type="L2")
        _assert_results_equal(old, new)
    finally:
        c.close()


def test_diff_empty_collection(tmp_path, schema):
    c = Collection("diff_empty", str(tmp_path / "data"), schema)
    try:
        old = _run_old_path(c, [_vec(900)], top_k=5)
        new = _run_new_path(c, [_vec(900)], top_k=5)
        assert old == [[]]
        assert new == [[]]
    finally:
        c.close()


def test_diff_all_filtered_out(col_mixed):
    """Filter that excludes every row → both paths return empty."""
    queries = [_vec(900)]
    old = _run_old_path(col_mixed, queries, top_k=5, expr="score < -1000")
    new = _run_new_path(col_mixed, queries, top_k=5, expr="score < -1000")
    assert old == [[]]
    assert new == [[]]


# ---------------------------------------------------------------------------
# Direct test that the new path actually uses BruteForceIndex on segments
# ---------------------------------------------------------------------------

def test_new_path_uses_attached_segment_index(tmp_path, schema):
    """If a segment has a pre-attached BruteForceIndex, the new path
    should use it instead of building a fresh one. We can't easily
    instrument that, but we can at least confirm the result is still
    correct when an index is attached."""
    from milvus_lite.index.brute_force import BruteForceIndex

    c = Collection("attached", str(tmp_path / "data"), schema)
    try:
        c.insert([
            {"id": i, "vec": _vec(i), "title": "x", "score": 0.5, "active": True}
            for i in range(10)
        ])
        c.flush()

        # Attach a fresh BruteForceIndex to every segment.
        for seg in c._segment_cache.values():  # noqa: SLF001
            seg.attach_index(BruteForceIndex.build(seg.vectors, "COSINE"))

        queries = [_vec(900)]
        new = _run_new_path(c, queries, top_k=5, metric_type="COSINE")
        old = _run_old_path(c, queries, top_k=5, metric_type="COSINE")
        _assert_results_equal(old, new)
    finally:
        c.close()
