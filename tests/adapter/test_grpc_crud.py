"""Phase 10.3 — pymilvus CRUD end-to-end tests.

The 4 RPCs added in Phase 10.3 (Insert / Upsert / Delete / Query)
drive the following pymilvus client API surface:

    client.insert(name, data)
    client.upsert(name, data)
    client.delete(name, ids=[...])
    client.delete(name, filter="...")
    client.query(name, filter="...", output_fields=[...])
    client.get(name, ids=[...], output_fields=[...])

Skipped automatically when pymilvus / grpcio is not installed.
"""

import pytest
from pymilvus import DataType, MilvusClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema():
    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=False
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("title", DataType.VARCHAR, max_length=64)
    schema.add_field("score", DataType.FLOAT)
    schema.add_field("active", DataType.BOOL)
    return schema


def _row(i, active=True):
    return {
        "id": i,
        "vec": [float(i), float(i + 1), float(i + 2), float(i + 3)],
        "title": f"t{i}",
        "score": float(i) / 10,
        "active": active,
    }


@pytest.fixture
def populated_client(milvus_client):
    """A client with a 'demo' collection holding 5 records."""
    milvus_client.create_collection("demo", schema=_make_schema())
    rows = [_row(i, active=(i % 2 == 0)) for i in range(5)]
    milvus_client.insert("demo", rows)
    return milvus_client


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def test_insert_returns_inserted_ids(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    rows = [_row(i) for i in range(5)]
    res = milvus_client.insert("demo", rows)
    assert res["insert_count"] == 5
    assert sorted(res["ids"]) == [0, 1, 2, 3, 4]


def test_insert_multiple_batches_accumulates(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    milvus_client.insert("demo", [_row(i) for i in range(3)])
    milvus_client.insert("demo", [_row(i) for i in range(3, 7)])
    rows = milvus_client.query("demo", filter="id >= 0")
    assert len(rows) == 7


def test_insert_into_unknown_collection_raises(milvus_client):
    with pytest.raises(Exception):
        milvus_client.insert("ghost", [_row(0)])


def test_insert_round_trip_preserves_field_values(populated_client):
    rows = populated_client.query(
        "demo", filter="id == 2", output_fields=["title", "score", "active"]
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["title"] == "t2"
    assert abs(r["score"] - 0.2) < 1e-6
    assert r["active"] is True


# ---------------------------------------------------------------------------
# Upsert (engine treats it as insert; pymilvus surfaces upsert_count)
# ---------------------------------------------------------------------------

def test_upsert_returns_upsert_count(milvus_client):
    milvus_client.create_collection("demo", schema=_make_schema())
    rows = [_row(i) for i in range(3)]
    res = milvus_client.upsert("demo", rows)
    assert res["upsert_count"] == 3


def test_upsert_overwrites_existing(populated_client):
    """Upserting an existing pk replaces the row (engine semantics)."""
    populated_client.upsert("demo", [{
        "id": 1,
        "vec": [9.0, 9.0, 9.0, 9.0],
        "title": "OVERWRITTEN",
        "score": 9.99,
        "active": False,
    }])
    rows = populated_client.query("demo", filter="id == 1", output_fields=["title", "score"])
    assert rows[0]["title"] == "OVERWRITTEN"
    assert abs(rows[0]["score"] - 9.99) < 1e-3


# ---------------------------------------------------------------------------
# Query — id-based path → routed to col.get
# ---------------------------------------------------------------------------

def test_query_by_id_in_list(populated_client):
    rows = populated_client.query(
        "demo", filter="id in [0, 2, 4]", output_fields=["title"]
    )
    titles = sorted(r["title"] for r in rows)
    assert titles == ["t0", "t2", "t4"]


def test_query_by_id_with_output_fields(populated_client):
    rows = populated_client.query(
        "demo", filter="id in [1]", output_fields=["score", "active"]
    )
    assert len(rows) == 1
    # pk is always returned + the requested fields
    assert "id" in rows[0]
    assert "score" in rows[0]
    assert "active" in rows[0]


# ---------------------------------------------------------------------------
# Query — general expression path → col.query
# ---------------------------------------------------------------------------

def test_query_by_filter_expression(populated_client):
    rows = populated_client.query(
        "demo", filter="active == true", output_fields=["id", "title"]
    )
    ids = sorted(r["id"] for r in rows)
    assert ids == [0, 2, 4]


def test_query_with_complex_expression(populated_client):
    rows = populated_client.query(
        "demo",
        filter="score > 0.1 and active == true",
        output_fields=["id"],
    )
    ids = sorted(r["id"] for r in rows)
    # active=True for ids 0, 2, 4. score > 0.1 → score > 0.1, so ids {2, 4}
    assert ids == [2, 4]


def test_query_with_like_pattern(populated_client):
    rows = populated_client.query(
        "demo", filter="title like 't%'", output_fields=["id", "title"]
    )
    assert len(rows) == 5


def test_query_with_no_matches(populated_client):
    rows = populated_client.query(
        "demo", filter="id > 1000", output_fields=["id"]
    )
    assert rows == []


def test_query_unknown_collection_raises(milvus_client):
    with pytest.raises(Exception):
        milvus_client.query("ghost", filter="id > 0")


# ---------------------------------------------------------------------------
# Get (pymilvus client.get → routed through Query RPC)
# ---------------------------------------------------------------------------

def test_get_by_ids(populated_client):
    rows = populated_client.get("demo", ids=[1, 3])
    ids = sorted(r["id"] for r in rows)
    assert ids == [1, 3]


def test_get_by_ids_with_output_fields(populated_client):
    rows = populated_client.get("demo", ids=[2], output_fields=["title"])
    assert len(rows) == 1
    assert rows[0]["id"] == 2
    assert rows[0]["title"] == "t2"


def test_get_missing_ids_returns_empty(populated_client):
    rows = populated_client.get("demo", ids=[999])
    assert rows == []


# ---------------------------------------------------------------------------
# Delete by ids (the pymilvus quickstart path)
# ---------------------------------------------------------------------------

def test_delete_by_ids(populated_client):
    populated_client.delete("demo", ids=[0, 2])
    remaining = populated_client.query("demo", filter="id >= 0", output_fields=["id"])
    ids = sorted(r["id"] for r in remaining)
    assert ids == [1, 3, 4]


def test_delete_idempotent(populated_client):
    """Deleting a non-existent pk is not an error in Milvus or
    MilvusLite — it just produces a no-op tombstone."""
    populated_client.delete("demo", ids=[999])
    rows = populated_client.query("demo", filter="id >= 0", output_fields=["id"])
    assert len(rows) == 5  # nothing was actually removed


# ---------------------------------------------------------------------------
# Delete by filter (the harder path — query then delete)
# ---------------------------------------------------------------------------

def test_delete_by_filter_active_true(populated_client):
    populated_client.delete("demo", filter="active == true")
    remaining = populated_client.query("demo", filter="id >= 0", output_fields=["id", "active"])
    # Only inactive rows should remain (originally active for ids 0, 2, 4)
    for r in remaining:
        assert r["active"] is False
    assert sorted(r["id"] for r in remaining) == [1, 3]


def test_delete_by_filter_no_match(populated_client):
    populated_client.delete("demo", filter="id > 1000")
    rows = populated_client.query("demo", filter="id >= 0", output_fields=["id"])
    assert len(rows) == 5
