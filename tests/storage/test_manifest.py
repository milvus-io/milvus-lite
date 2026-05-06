"""Tests for storage/manifest.py — Phase-2 minimal subset."""

import json
import os

import pytest

from milvus_lite.constants import DEFAULT_PARTITION
from milvus_lite.exceptions import (
    DefaultPartitionError,
    ManifestCorruptedError,
    PartitionAlreadyExistsError,
    PartitionNotFoundError,
)
from milvus_lite.storage.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_PREV_FILENAME,
    Manifest,
)


# ---------------------------------------------------------------------------
# Initial / fresh state
# ---------------------------------------------------------------------------

def test_load_nonexistent_returns_fresh(tmp_path):
    m = Manifest.load(str(tmp_path))
    assert m.version == 0
    assert m.current_seq == 0
    assert m.active_wal_number is None
    assert m.list_partitions() == [DEFAULT_PARTITION]
    assert m.has_partition(DEFAULT_PARTITION)


def test_fresh_default_partition_present():
    m = Manifest(data_dir="/dummy")
    assert m.has_partition(DEFAULT_PARTITION)


# ---------------------------------------------------------------------------
# save → load round trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_partition("p1")
    m.add_partition("p2")
    m.current_seq = 42
    m.active_wal_number = 5
    m.save()
    assert m.version == 1

    m2 = Manifest.load(str(tmp_path))
    assert m2.version == 1
    assert m2.current_seq == 42
    assert m2.active_wal_number == 5
    assert set(m2.list_partitions()) == {DEFAULT_PARTITION, "p1", "p2"}


def test_save_creates_files_atomically(tmp_path):
    m = Manifest(str(tmp_path))
    m.save()
    assert os.path.exists(tmp_path / MANIFEST_FILENAME)
    # No tmp file left behind.
    assert not os.path.exists(tmp_path / "manifest.json.tmp")
    # No prev on first save (nothing to back up).
    assert not os.path.exists(tmp_path / MANIFEST_PREV_FILENAME)


def test_save_creates_prev_on_second_save(tmp_path):
    m = Manifest(str(tmp_path))
    m.save()  # version=1, no prev
    m.add_partition("p1")
    m.save()  # version=2, prev=v1

    assert os.path.exists(tmp_path / MANIFEST_PREV_FILENAME)
    # The current file is the new version.
    m2 = Manifest.load(str(tmp_path))
    assert m2.version == 2
    assert m2.has_partition("p1")


def test_version_increments_per_save(tmp_path):
    m = Manifest(str(tmp_path))
    for expected in (1, 2, 3, 4):
        m.save()
        assert m.version == expected


# ---------------------------------------------------------------------------
# .prev fallback
# ---------------------------------------------------------------------------

def test_load_falls_back_to_prev_on_corrupted_current(tmp_path):
    """If manifest.json is corrupted but .prev is good, load .prev."""
    m = Manifest(str(tmp_path))
    m.add_partition("good_partition")
    m.save()  # v1
    m.add_partition("p2")
    m.save()  # v2, prev=v1

    # Corrupt the current manifest.
    (tmp_path / MANIFEST_FILENAME).write_text("{ broken json")

    # Load should fall back to prev (v1, which has good_partition but not p2).
    m2 = Manifest.load(str(tmp_path))
    assert m2.version == 1
    assert m2.has_partition("good_partition")
    assert not m2.has_partition("p2")


def test_load_raises_when_both_corrupted(tmp_path):
    m = Manifest(str(tmp_path))
    m.save()
    m.save()
    # Corrupt both files.
    (tmp_path / MANIFEST_FILENAME).write_text("{ broken")
    (tmp_path / MANIFEST_PREV_FILENAME).write_text("{ also broken")

    with pytest.raises(ManifestCorruptedError):
        Manifest.load(str(tmp_path))


def test_load_raises_when_current_corrupted_no_prev(tmp_path):
    """If only manifest.json exists and it's corrupted, no fallback available."""
    (tmp_path / MANIFEST_FILENAME).write_text("{ broken")
    with pytest.raises(ManifestCorruptedError):
        Manifest.load(str(tmp_path))


# ---------------------------------------------------------------------------
# partition CRUD
# ---------------------------------------------------------------------------

def test_add_partition(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_partition("p1")
    assert m.has_partition("p1")
    assert "p1" in m.list_partitions()


def test_add_duplicate_partition_raises(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_partition("p1")
    with pytest.raises(PartitionAlreadyExistsError):
        m.add_partition("p1")


def test_remove_partition(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_partition("p1")
    m.remove_partition("p1")
    assert not m.has_partition("p1")


def test_remove_nonexistent_partition_raises(tmp_path):
    m = Manifest(str(tmp_path))
    with pytest.raises(PartitionNotFoundError):
        m.remove_partition("nope")


def test_remove_default_partition_raises(tmp_path):
    m = Manifest(str(tmp_path))
    with pytest.raises(DefaultPartitionError):
        m.remove_partition(DEFAULT_PARTITION)


def test_list_partitions_sorted(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_partition("zebra")
    m.add_partition("alpha")
    parts = m.list_partitions()
    assert parts == sorted(parts)


# ---------------------------------------------------------------------------
# current_seq monotonic
# ---------------------------------------------------------------------------

def test_current_seq_monotonic(tmp_path):
    m = Manifest(str(tmp_path))
    m.current_seq = 10
    m.current_seq = 20
    with pytest.raises(ValueError, match="monotonic"):
        m.current_seq = 15


def test_current_seq_same_value_ok(tmp_path):
    m = Manifest(str(tmp_path))
    m.current_seq = 10
    m.current_seq = 10  # equal — allowed (no-op)
    assert m.current_seq == 10


# ---------------------------------------------------------------------------
# JSON shape sanity check
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# data / delta file CRUD
# ---------------------------------------------------------------------------

def test_add_data_file(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_data_file(DEFAULT_PARTITION, "data/data_000001_000010.parquet")
    files = m.get_data_files(DEFAULT_PARTITION)
    assert files == ["data/data_000001_000010.parquet"]


def test_add_data_file_unknown_partition_raises(tmp_path):
    m = Manifest(str(tmp_path))
    with pytest.raises(PartitionNotFoundError):
        m.add_data_file("ghost", "data/foo.parquet")


def test_remove_data_files(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_data_file(DEFAULT_PARTITION, "data/a.parquet")
    m.add_data_file(DEFAULT_PARTITION, "data/b.parquet")
    m.add_data_file(DEFAULT_PARTITION, "data/c.parquet")
    m.remove_data_files(DEFAULT_PARTITION, ["a.parquet", "data/b.parquet"])
    # "a.parquet" doesn't match exactly — idempotent miss
    assert m.get_data_files(DEFAULT_PARTITION) == ["data/a.parquet", "data/c.parquet"]


def test_get_all_data_files(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_partition("p1")
    m.add_data_file(DEFAULT_PARTITION, "data/x.parquet")
    m.add_data_file("p1", "data/y.parquet")
    all_files = m.get_all_data_files()
    assert all_files == {
        DEFAULT_PARTITION: ["data/x.parquet"],
        "p1": ["data/y.parquet"],
    }


def test_add_delta_file(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_delta_file(DEFAULT_PARTITION, "delta/delta_000005_000005.parquet")
    assert m.get_delta_files(DEFAULT_PARTITION) == [
        "delta/delta_000005_000005.parquet"
    ]


def test_get_all_delta_files_includes_empty_partitions(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_partition("p1")
    m.add_delta_file("p1", "delta/x.parquet")
    all_delta = m.get_all_delta_files()
    assert all_delta == {
        DEFAULT_PARTITION: [],
        "p1": ["delta/x.parquet"],
    }


def test_data_files_persist_through_save_load(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_data_file(DEFAULT_PARTITION, "data/a.parquet")
    m.add_delta_file(DEFAULT_PARTITION, "delta/b.parquet")
    m.save()

    m2 = Manifest.load(str(tmp_path))
    assert m2.get_data_files(DEFAULT_PARTITION) == ["data/a.parquet"]
    assert m2.get_delta_files(DEFAULT_PARTITION) == ["delta/b.parquet"]


def test_saved_json_has_expected_keys(tmp_path):
    m = Manifest(str(tmp_path))
    m.add_partition("p1")
    m.current_seq = 7
    m.active_wal_number = 3
    m.save()

    payload = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert "manifest_format_version" in payload
    assert payload["version"] == 1
    assert payload["current_seq"] == 7
    assert payload["active_wal_number"] == 3
    assert DEFAULT_PARTITION in payload["partitions"]
    assert "p1" in payload["partitions"]
    assert payload["partitions"]["p1"] == {"data_files": [], "delta_files": []}
