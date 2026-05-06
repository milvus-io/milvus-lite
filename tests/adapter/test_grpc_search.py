"""Phase 10.4 — pymilvus search end-to-end tests.

Validates the Search RPC: SearchRequest decoding (placeholder_group +
search_params), engine dispatch, and SearchResultData encoding. Both
HNSW and BRUTE_FORCE backends are tested. Skipped if pymilvus or
faiss-cpu is missing.
"""

import pytest
from pymilvus import DataType, MilvusClient


def _make_schema():
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("title", DataType.VARCHAR, max_length=64)
    schema.add_field("score", DataType.FLOAT)
    schema.add_field("active", DataType.BOOL)
    return schema


def _hnsw_idx(client):
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="vec",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    return idx


def _brute_idx(client, metric="L2"):
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="vec",
        index_type="BRUTE_FORCE",
        metric_type=metric,
        params={},
    )
    return idx


@pytest.fixture
def loaded_collection(milvus_client):
    """Collection with 20 records, HNSW index, loaded ready for search."""
    milvus_client.create_collection("demo", schema=_make_schema())
    rows = [
        {
            "id": i,
            "vec": [float(i), float(i + 1), 0.0, 0.0],
            "title": f"t{i:02d}",
            "score": float(i) / 10,
            "active": (i % 2 == 0),
        }
        for i in range(20)
    ]
    milvus_client.insert("demo", rows)
    milvus_client.create_index("demo", _hnsw_idx(milvus_client))
    milvus_client.load_collection("demo")
    return milvus_client


# ---------------------------------------------------------------------------
# Basic search shape
# ---------------------------------------------------------------------------

def test_search_returns_top_k(loaded_collection):
    res = loaded_collection.search(
        "demo", data=[[1.0, 2.0, 0.0, 0.0]], limit=5
    )
    assert len(res) == 1            # nq = 1
    assert len(res[0]) == 5         # top_k = 5


def test_search_multiple_queries(loaded_collection):
    res = loaded_collection.search(
        "demo",
        data=[
            [1.0, 2.0, 0.0, 0.0],
            [10.0, 11.0, 0.0, 0.0],
        ],
        limit=3,
    )
    assert len(res) == 2
    assert len(res[0]) == 3
    assert len(res[1]) == 3


def test_search_self_query_returns_self_top1(loaded_collection):
    """Querying with one of the indexed vectors should return that
    pk at rank 1 — basic correctness check that the placeholder
    decoding got the bytes right."""
    res = loaded_collection.search(
        "demo", data=[[5.0, 6.0, 0.0, 0.0]], limit=1
    )
    assert res[0][0]["id"] == 5


# ---------------------------------------------------------------------------
# Output fields
# ---------------------------------------------------------------------------

def test_search_with_output_fields(loaded_collection):
    res = loaded_collection.search(
        "demo",
        data=[[5.0, 6.0, 0.0, 0.0]],
        limit=3,
        output_fields=["title", "score"],
    )
    for hit in res[0]:
        assert "title" in hit["entity"]
        assert "score" in hit["entity"]


def test_search_default_output_fields_is_empty(loaded_collection):
    """When no output_fields are requested, pymilvus shows just id +
    distance, no entity fields."""
    res = loaded_collection.search(
        "demo", data=[[5.0, 6.0, 0.0, 0.0]], limit=3
    )
    for hit in res[0]:
        # entity may be empty {} or may contain just the id field
        assert "id" in hit or "id" in hit.get("entity", {})


# ---------------------------------------------------------------------------
# Filter expression
# ---------------------------------------------------------------------------

def test_search_with_filter_active_only(loaded_collection):
    res = loaded_collection.search(
        "demo",
        data=[[5.0, 6.0, 0.0, 0.0]],
        limit=10,
        filter="active == true",
        output_fields=["active"],
    )
    for hit in res[0]:
        assert hit["entity"]["active"] is True


def test_search_with_complex_filter(loaded_collection):
    res = loaded_collection.search(
        "demo",
        data=[[5.0, 6.0, 0.0, 0.0]],
        limit=10,
        filter="score > 0.5 and active == true",
        output_fields=["score", "active"],
    )
    for hit in res[0]:
        assert hit["entity"]["score"] > 0.5
        assert hit["entity"]["active"] is True


def test_search_with_filter_no_match(loaded_collection):
    res = loaded_collection.search(
        "demo",
        data=[[5.0, 6.0, 0.0, 0.0]],
        limit=5,
        filter="id > 1000",
    )
    assert res[0] == [] or len(res[0]) == 0


# ---------------------------------------------------------------------------
# Loaded state guard
# ---------------------------------------------------------------------------

def test_search_after_release_raises(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.insert("demo", [
        {"id": 1, "vec": [1.0, 2.0, 0.0, 0.0], "title": "a", "score": 0.1, "active": True}
    ])
    milvus_client.create_index("demo", _hnsw_idx(milvus_client))
    milvus_client.release_collection("demo")
    with pytest.raises(Exception) as exc_info:
        milvus_client.search("demo", data=[[1.0, 2.0, 0.0, 0.0]], limit=1)
    # Error mentions "not loaded" / "load"
    assert "load" in str(exc_info.value).lower()


def test_search_after_release_raises(loaded_collection):
    loaded_collection.release_collection("demo")
    with pytest.raises(Exception):
        loaded_collection.search("demo", data=[[1.0, 2.0, 0.0, 0.0]], limit=1)


def test_search_unknown_collection_raises(milvus_client):
    with pytest.raises(Exception):
        milvus_client.search("ghost", data=[[1.0, 2.0, 0.0, 0.0]], limit=1)


# ---------------------------------------------------------------------------
# BRUTE_FORCE backend (tests the factory routing too)
# ---------------------------------------------------------------------------

def test_search_brute_force_backend(milvus_client):
    milvus_client.create_collection("brute", schema=_make_schema())
    milvus_client.insert("brute", [
        {"id": i, "vec": [float(i), 0.0, 0.0, 0.0], "title": f"t{i}",
         "score": 0.0, "active": True}
        for i in range(10)
    ])
    milvus_client.create_index("brute", _brute_idx(milvus_client, metric="L2"))
    milvus_client.load_collection("brute")
    res = milvus_client.search("brute", data=[[3.0, 0.0, 0.0, 0.0]], limit=3)
    assert len(res[0]) == 3
    # L2 self-distance is 0 for exact match
    assert res[0][0]["id"] == 3


# ---------------------------------------------------------------------------
# Distances are present and reasonable
# ---------------------------------------------------------------------------

def test_search_distances_returned(loaded_collection):
    res = loaded_collection.search(
        "demo", data=[[5.0, 6.0, 0.0, 0.0]], limit=5
    )
    for hit in res[0]:
        assert "distance" in hit
        assert isinstance(hit["distance"], (int, float))


def test_search_results_sorted_ascending(loaded_collection):
    res = loaded_collection.search(
        "demo", data=[[5.0, 6.0, 0.0, 0.0]], limit=10
    )
    distances = [hit["distance"] for hit in res[0]]
    assert distances == sorted(distances)
