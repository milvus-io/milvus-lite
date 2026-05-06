"""Phase 10.4 — Index lifecycle RPC tests via pymilvus.

Covers CreateIndex / DropIndex / DescribeIndex / LoadCollection /
ReleaseCollection / GetLoadState / GetLoadingProgress.

Skipped automatically when pymilvus / grpcio is not installed.
"""

import pytest
from pymilvus import DataType, MilvusClient


def _make_schema():
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("title", DataType.VARCHAR, max_length=64)
    return schema


def _hnsw_params(client):
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="vec",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    return idx


def _brute_params(client):
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="vec",
        index_type="BRUTE_FORCE",
        metric_type="L2",
        params={},
    )
    return idx


@pytest.fixture
def populated(milvus_client):
    """A collection with 10 records and a HNSW index; state=loaded."""
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.insert("demo", [
        {"id": i, "vec": [float(i), float(i+1), 0.0, 0.0], "title": f"t{i}"}
        for i in range(10)
    ])
    milvus_client.create_index("demo", _hnsw_params(milvus_client))
    return milvus_client


# ---------------------------------------------------------------------------
# CreateIndex
# ---------------------------------------------------------------------------

def test_create_index_hnsw(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.create_index("demo", _hnsw_params(milvus_client))
    desc = milvus_client.describe_index("demo", "vec")
    assert desc["index_type"] == "HNSW"


def test_create_index_brute_force(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.create_index("demo", _brute_params(milvus_client))
    desc = milvus_client.describe_index("demo", "vec")
    assert desc["index_type"] == "BRUTE_FORCE"


def test_create_index_duplicate_raises(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.create_index("demo", _hnsw_params(milvus_client))
    with pytest.raises(Exception):
        milvus_client.create_index("demo", _hnsw_params(milvus_client))


def test_create_index_unknown_collection_raises(milvus_client):
    with pytest.raises(Exception):
        milvus_client.create_index("ghost", _hnsw_params(milvus_client))


def test_create_index_metric_round_trip(populated):
    desc = populated.describe_index("demo", "vec")
    assert desc["metric_type"] == "COSINE"


# ---------------------------------------------------------------------------
# DropIndex
# ---------------------------------------------------------------------------

def test_drop_index(populated):
    # Milvus semantics: drop_index requires released state.
    populated.release_collection("demo")
    populated.drop_index("demo", "vec")
    # After drop, describe_index returns None (pymilvus parses
    # INDEX_NOT_FOUND status as None, not as a raise)
    desc = populated.describe_index("demo", "vec")
    assert desc is None


def test_drop_index_unknown_raises(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    # No index created → drop should raise
    with pytest.raises(Exception):
        milvus_client.drop_index("demo", "vec")


# ---------------------------------------------------------------------------
# LoadCollection / ReleaseCollection / GetLoadState
# ---------------------------------------------------------------------------

def test_load_state_before_create_index(milvus_client):
    """Without an index, the engine auto-loads on construction →
    state should be Loaded immediately."""
    milvus_client.create_collection("demo", schema=_make_schema())
    state = milvus_client.get_load_state("demo")
    # state is a dict {"state": <LoadState: Loaded>}
    assert "Loaded" in str(state["state"])


def test_load_state_after_create_index_is_loaded(populated):
    """Milvus semantics: create_index preserves load state."""
    state = populated.get_load_state("demo")
    assert "Loaded" in str(state["state"])


def test_load_collection(populated):
    populated.load_collection("demo")
    state = populated.get_load_state("demo")
    assert "Loaded" in str(state["state"])


def test_release_collection(populated):
    populated.load_collection("demo")
    populated.release_collection("demo")
    state = populated.get_load_state("demo")
    assert "NotLoad" in str(state["state"])


def test_load_release_cycle(populated):
    for _ in range(3):
        populated.load_collection("demo")
        assert "Loaded" in str(populated.get_load_state("demo")["state"])
        populated.release_collection("demo")
        assert "NotLoad" in str(populated.get_load_state("demo")["state"])


def test_get_load_state_unknown_collection(milvus_client):
    """pymilvus's get_load_state returns LoadStateNotExist (0) for
    unknown collections, NOT raises."""
    state = milvus_client.get_load_state("ghost")
    assert "NotExist" in str(state["state"])


# ---------------------------------------------------------------------------
# DescribeIndex
# ---------------------------------------------------------------------------

def test_describe_index_returns_params(populated):
    desc = populated.describe_index("demo", "vec")
    assert desc["field_name"] == "vec"
    assert desc["index_type"] == "HNSW"
    assert desc["metric_type"] == "COSINE"


def test_describe_index_state_is_finished(populated):
    """Phase 9 builds indexes synchronously, so state is always
    'Finished' from pymilvus's perspective."""
    desc = populated.describe_index("demo", "vec")
    assert "Finished" in str(desc.get("state", ""))


def test_describe_index_name_matches_field_name(populated):
    """Issue #16: pymilvus search_iterator v1 matches index_name against
    anns_field to extract metric_type. index_name must equal field_name."""
    desc = populated.describe_index("demo", "vec")
    assert desc["index_name"] == "vec"


def test_describe_index_empty_when_no_index(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    desc = milvus_client.describe_index("demo", "vec")
    assert desc is None
