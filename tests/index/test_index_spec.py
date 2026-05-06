"""IndexSpec unit tests — frozen dataclass + validation + round-trip."""

import pytest

from milvus_lite.index.spec import IndexSpec


def test_basic_construction():
    spec = IndexSpec(
        field_name="vec",
        index_type="HNSW",
        metric_type="COSINE",
        build_params={"M": 16, "efConstruction": 200},
        search_params={"ef": 64},
    )
    assert spec.field_name == "vec"
    assert spec.index_type == "HNSW"
    assert spec.metric_type == "COSINE"
    assert spec.build_params == {"M": 16, "efConstruction": 200}
    assert spec.search_params == {"ef": 64}


def test_search_params_default_empty():
    spec = IndexSpec(
        field_name="vec",
        index_type="BRUTE_FORCE",
        metric_type="L2",
        build_params={},
    )
    assert spec.search_params == {}


def test_frozen_blocks_mutation():
    spec = IndexSpec(
        field_name="vec",
        index_type="HNSW",
        metric_type="L2",
        build_params={},
    )
    with pytest.raises((AttributeError, TypeError)):
        spec.field_name = "new"  # frozen dataclass


def test_to_from_dict_round_trip():
    original = IndexSpec(
        field_name="vec",
        index_type="HNSW",
        metric_type="COSINE",
        build_params={"M": 16, "efConstruction": 200},
        search_params={"ef": 64},
    )
    d = original.to_dict()
    assert d == {
        "field_name": "vec",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "build_params": {"M": 16, "efConstruction": 200},
        "search_params": {"ef": 64},
    }
    restored = IndexSpec.from_dict(d)
    assert restored == original


def test_from_dict_missing_search_params_defaults_empty():
    d = {
        "field_name": "vec",
        "index_type": "BRUTE_FORCE",
        "metric_type": "L2",
        "build_params": {},
    }
    spec = IndexSpec.from_dict(d)
    assert spec.search_params == {}


def test_from_dict_missing_build_params_defaults_empty():
    d = {
        "field_name": "vec",
        "index_type": "BRUTE_FORCE",
        "metric_type": "L2",
    }
    spec = IndexSpec.from_dict(d)
    assert spec.build_params == {}


# ── Validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_metric", ["cosine", "l2", "ip", "FOO", ""])
def test_invalid_metric_type_raises(bad_metric):
    with pytest.raises(ValueError, match="metric_type"):
        IndexSpec(
            field_name="vec",
            index_type="HNSW",
            metric_type=bad_metric,
            build_params={},
        )


def test_empty_field_name_raises():
    with pytest.raises(ValueError, match="field_name"):
        IndexSpec(
            field_name="",
            index_type="HNSW",
            metric_type="COSINE",
            build_params={},
        )


def test_empty_index_type_raises():
    with pytest.raises(ValueError, match="index_type"):
        IndexSpec(
            field_name="vec",
            index_type="",
            metric_type="COSINE",
            build_params={},
        )


def test_non_dict_build_params_raises():
    with pytest.raises(TypeError, match="build_params"):
        IndexSpec(
            field_name="vec",
            index_type="HNSW",
            metric_type="L2",
            build_params="not a dict",  # type: ignore[arg-type]
        )


def test_equality_and_hashable():
    a = IndexSpec(field_name="vec", index_type="HNSW", metric_type="L2", build_params={"M": 16})
    b = IndexSpec(field_name="vec", index_type="HNSW", metric_type="L2", build_params={"M": 16})
    assert a == b
    # Both should be hashable since they're frozen — but dict fields
    # make hash impossible. Frozen dataclasses with dict fields are
    # NOT hashable; that's expected. Only equality should hold.
