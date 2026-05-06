"""Tests for function.dataframe — DataFrame."""

import pytest

from milvus_lite.function.dataframe import DataFrame


# ── Factory methods ──────────────────────────────────────────


def test_from_records_single_chunk():
    records = [{"a": 1}, {"a": 2}]
    df = DataFrame.from_records(records)
    assert df.num_chunks == 1
    assert df.chunk(0) is records


def test_from_search_results_multi_chunk():
    results = [[{"id": 1}], [{"id": 2}], [{"id": 3}]]
    df = DataFrame.from_search_results(results)
    assert df.num_chunks == 3


# ── Export ───────────────────────────────────────────────────


def test_to_records_roundtrip():
    records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    df = DataFrame.from_records(records)
    assert df.to_records() is records


def test_to_search_results_roundtrip():
    results = [[{"id": 1}], [{"id": 2}]]
    df = DataFrame.from_search_results(results)
    assert df.to_search_results() is results


def test_to_records_rejects_multi_chunk():
    df = DataFrame([[{"a": 1}], [{"a": 2}]])
    with pytest.raises(ValueError, match="single chunk"):
        df.to_records()


# ── Column access ────────────────────────────────────────────


def test_column_read():
    df = DataFrame.from_records([{"x": 10, "y": 20}, {"x": 30, "y": 40}])
    assert df.column("x", 0) == [10, 30]
    assert df.column("y", 0) == [20, 40]


def test_column_read_missing_key():
    df = DataFrame.from_records([{"x": 1}, {"x": 2}])
    assert df.column("missing", 0) == [None, None]


def test_set_column_write():
    records = [{"x": 1}, {"x": 2}]
    df = DataFrame.from_records(records)
    df.set_column("y", 0, [10, 20])
    assert records[0]["y"] == 10
    assert records[1]["y"] == 20


def test_set_column_overwrite():
    records = [{"x": 1}, {"x": 2}]
    df = DataFrame.from_records(records)
    df.set_column("x", 0, [100, 200])
    assert records[0]["x"] == 100
    assert records[1]["x"] == 200


# ── Column names ─────────────────────────────────────────────


def test_column_names():
    df = DataFrame.from_records([{"a": 1, "b": 2, "c": 3}])
    assert df.column_names() == ["a", "b", "c"]


def test_column_names_empty_chunk():
    df = DataFrame([[]])
    assert df.column_names() == []


# ── Multi-chunk ──────────────────────────────────────────────


def test_multi_chunk_column_access():
    df = DataFrame([
        [{"s": 0.9}, {"s": 0.8}],
        [{"s": 0.7}],
    ])
    assert df.column("s", 0) == [0.9, 0.8]
    assert df.column("s", 1) == [0.7]


def test_multi_chunk_set_column():
    df = DataFrame([
        [{"s": 0.9}, {"s": 0.8}],
        [{"s": 0.7}],
    ])
    df.set_column("s", 1, [0.99])
    assert df.chunk(1)[0]["s"] == 0.99
    # chunk 0 unchanged
    assert df.chunk(0)[0]["s"] == 0.9
