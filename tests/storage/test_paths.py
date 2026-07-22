"""Tests for cross-platform persisted relative paths."""

from milvus_lite.storage.paths import normalize_rel_path, persisted_rel_path


def test_persisted_rel_path_uses_forward_slashes():
    assert persisted_rel_path("data", "data_000001_000002.parquet") == (
        "data/data_000001_000002.parquet"
    )


def test_normalize_rel_path_converts_windows_separators():
    assert normalize_rel_path(r"delta\delta_000001_000002.parquet") == (
        "delta/delta_000001_000002.parquet"
    )
