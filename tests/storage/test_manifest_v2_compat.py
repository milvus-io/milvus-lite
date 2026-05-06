"""Manifest index_specs persistence tests."""

import json
import os

import pytest

from milvus_lite.index.spec import IndexSpec
from milvus_lite.storage.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT_VERSION,
    Manifest,
)


def test_format_version_is_2():
    assert MANIFEST_FORMAT_VERSION == 2


def test_fresh_manifest_has_no_index_specs(tmp_path):
    m = Manifest(str(tmp_path))
    assert m.index_specs == {}
    assert m.format_version == 2


def test_set_index_spec_then_save_load(tmp_path):
    m = Manifest(str(tmp_path))
    spec = IndexSpec(
        field_name="vec",
        index_type="HNSW",
        metric_type="COSINE",
        build_params={"M": 16, "efConstruction": 200},
        search_params={"ef": 64},
    )
    m.set_index_spec(spec)
    m.save()

    # On-disk JSON should now contain index_specs dict.
    with open(os.path.join(str(tmp_path), MANIFEST_FILENAME)) as f:
        payload = json.load(f)
    assert payload["manifest_format_version"] == 2
    assert "vec" in payload["index_specs"]
    assert payload["index_specs"]["vec"]["index_type"] == "HNSW"
    assert payload["index_specs"]["vec"]["metric_type"] == "COSINE"

    # Reload and verify.
    m2 = Manifest.load(str(tmp_path))
    assert m2.index_specs["vec"] == spec


def test_clear_index_spec_round_trip(tmp_path):
    m = Manifest(str(tmp_path))
    m.set_index_spec(IndexSpec(
        field_name="v", index_type="HNSW", metric_type="L2", build_params={},
    ))
    m.save()
    m.set_index_spec(None)
    m.save()

    m2 = Manifest.load(str(tmp_path))
    assert m2.index_specs == {}


def test_set_index_spec_does_not_save_until_explicit(tmp_path):
    """set_index_spec is in-memory only — caller must save() to persist."""
    m = Manifest(str(tmp_path))
    m.save()

    spec = IndexSpec(
        field_name="vec", index_type="HNSW", metric_type="COSINE", build_params={},
    )
    m.set_index_spec(spec)
    # Don't save.

    m2 = Manifest.load(str(tmp_path))
    # Disk version should still have no specs.
    assert m2.index_specs == {}
