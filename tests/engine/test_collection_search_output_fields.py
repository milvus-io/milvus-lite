"""Phase 9.1.3 — Collection.search(output_fields=...) tests.

Three semantics to validate:
    output_fields=None  → entity = all fields except pk + vector (legacy)
    output_fields=[]    → entity = {} (only id + distance)
    output_fields=[..]  → entity = exactly those fields (vector included
                          only if listed; pk excluded since it's "id")
"""

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="active", dtype=DataType.BOOL),
    ])


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    c.insert([
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "title": "alpha", "score": 0.9, "active": True},
        {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "title": "beta",  "score": 0.5, "active": False},
        {"id": 3, "vec": [0.0, 0.0, 1.0, 0.0], "title": "gamma", "score": 0.1, "active": True},
    ])
    yield c
    c.close()


def test_default_output_fields_none_keeps_legacy_behavior(col):
    """output_fields=None → entity = all fields except pk and vector."""
    res = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=1)
    hit = res[0][0]
    assert hit["id"] == 1
    assert set(hit["entity"].keys()) == {"title", "score", "active"}
    assert hit["entity"]["title"] == "alpha"
    assert "vec" not in hit["entity"]
    assert "id" not in hit["entity"]


def test_empty_output_fields_returns_empty_entity(col):
    res = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=1, output_fields=[])
    hit = res[0][0]
    assert hit["id"] == 1
    assert hit["entity"] == {}


def test_output_fields_subset(col):
    res = col.search([[1.0, 0.0, 0.0, 0.0]], top_k=1, output_fields=["title"])
    hit = res[0][0]
    assert hit["entity"] == {"title": "alpha"}


def test_output_fields_multiple(col):
    res = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=1,
        output_fields=["title", "score"],
    )
    hit = res[0][0]
    assert hit["entity"] == {"title": "alpha", "score": pytest.approx(0.9)}


def test_output_fields_includes_vector_when_listed(col):
    res = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=1,
        output_fields=["title", "vec"],
    )
    hit = res[0][0]
    assert "vec" in hit["entity"]
    assert hit["entity"]["title"] == "alpha"
    # The vector value is whatever the segment/memtable stored.
    assert len(hit["entity"]["vec"]) == 4


def test_boost_ranker_filter_uses_hidden_fields(col):
    ranker = {
        "functions": [{
            "name": "boost_active",
            "params": {
                "reranker": "boost",
                "filter": "active == true",
                "weight": 0.1,
            },
        }],
        "params": {"boost_mode": "multiply", "function_mode": "multiply"},
    }

    res = col.search(
        [[0.0, 0.8, 0.2, 0.0]],
        top_k=3,
        metric_type="L2",
        output_fields=["title"],
        ranker=ranker,
    )

    assert [hit["id"] for hit in res[0]] == [3, 1, 2]
    assert all(set(hit["entity"]) == {"title"} for hit in res[0])


def test_boost_ranker_preserves_requested_vector_field(col):
    ranker = {
        "functions": [{
            "name": "boost_all",
            "params": {"reranker": "boost", "weight": 1.0},
        }],
        "params": {"boost_mode": "multiply", "function_mode": "multiply"},
    }

    res = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=1,
        output_fields=["vec"],
        ranker=ranker,
    )

    assert res[0][0]["id"] == 1
    assert res[0][0]["entity"] == {"vec": [1.0, 0.0, 0.0, 0.0]}


def test_group_by_uses_hidden_field_without_projecting_it(col):
    res = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=2,
        group_by_field="active",
        group_size=1,
        output_fields=["title"],
    )

    assert len(res[0]) == 2
    assert [hit["_group_by_value"] for hit in res[0]] == [True, False]
    assert all(set(hit["entity"]) == {"title"} for hit in res[0])


def test_output_fields_pk_in_list_is_dropped_silently(col):
    """The pk is always surfaced as 'id'; listing it in output_fields
    is a no-op (not an error)."""
    res = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=1,
        output_fields=["id", "title"],
    )
    hit = res[0][0]
    assert hit["id"] == 1
    # 'id' should NOT appear inside entity.
    assert "id" not in hit["entity"]
    assert hit["entity"] == {"title": "alpha"}


def test_output_fields_unknown_field_silently_skipped(col):
    """Unknown field names are skipped, not raised. This matches Milvus
    leniency and lets clients send wildcard-style projections."""
    res = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=1,
        output_fields=["title", "ghost_field"],
    )
    hit = res[0][0]
    assert hit["entity"] == {"title": "alpha"}


def test_output_fields_multi_query_topk(col):
    res = col.search(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        top_k=2,
        output_fields=["title"],
    )
    assert len(res) == 2
    assert all(set(hit["entity"].keys()) == {"title"} for query in res for hit in query)
    # Top-1 of the first query should be id=1, of the second query id=2.
    assert res[0][0]["id"] == 1
    assert res[1][0]["id"] == 2


def test_output_fields_with_filter_expr(col):
    res = col.search(
        [[1.0, 0.0, 0.0, 0.0]],
        top_k=10,
        expr="active == true",
        output_fields=["title"],
    )
    ids = [h["id"] for h in res[0]]
    # Only id=1 and id=3 are active.
    assert sorted(ids) == [1, 3]
    assert all(set(h["entity"].keys()) == {"title"} for h in res[0])
