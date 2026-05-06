"""Phase 10.5 — pymilvus quickstart full end-to-end smoke test.

This is the Phase 10 completion marker. If this test passes, the
MilvusLite gRPC adapter supports the Milvus "hello world" workflow
that every pymilvus user expects on first contact:

    connect → create_collection → insert → flush →
    create_index → load → search → query → get → delete →
    release → drop_collection

The test exercises EVERY RPC layer implemented across 10.1-10.5.

Skipped if pymilvus / grpcio / faiss-cpu is not installed.
"""

import pytest
from pymilvus import DataType, MilvusClient

from milvus_lite.index.factory import is_faiss_available

pytestmark = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss-cpu is not installed"
)


def test_pymilvus_quickstart(grpc_server):
    """The definitive smoke test."""
    port, _db = grpc_server
    client = MilvusClient(uri=f"http://127.0.0.1:{port}")

    try:
        # ── 1. Create collection ─────────────────────────────────
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        schema.add_field("title", DataType.VARCHAR, max_length=128)
        schema.add_field("score", DataType.FLOAT)

        client.create_collection("quickstart", schema=schema)
        assert client.has_collection("quickstart")
        assert "quickstart" in client.list_collections()

        # ── 2. Insert ────────────────────────────────────────────
        data = [
            {"id": i, "vec": [float(i), float(i + 1), 0.0, 0.0],
             "title": f"doc_{i}", "score": float(i) / 10}
            for i in range(100)
        ]
        res = client.insert("quickstart", data)
        assert res["insert_count"] == 100

        # ── 3. Flush ─────────────────────────────────────────────
        client.flush("quickstart")

        # ── 4. Create index ──────────────────────────────────────
        idx = client.prepare_index_params()
        idx.add_index(
            field_name="vec",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        client.create_index("quickstart", idx)

        desc = client.describe_index("quickstart", "vec")
        assert desc["index_type"] == "HNSW"

        # ── 5. Load ──────────────────────────────────────────────
        client.load_collection("quickstart")
        state = client.get_load_state("quickstart")
        assert "Loaded" in str(state["state"])

        # ── 6. Search ────────────────────────────────────────────
        results = client.search(
            "quickstart",
            data=[[50.0, 51.0, 0.0, 0.0]],
            limit=10,
            output_fields=["title", "score"],
        )
        assert len(results) == 1              # nq = 1
        assert len(results[0]) == 10          # limit = 10
        # HNSW recall isn't 100% on near-collinear vectors with COSINE;
        # check top-1 is at least close to the query origin.
        top_id = results[0][0]["id"]
        assert 45 <= top_id <= 55, f"top-1 id {top_id} too far from 50"
        assert "title" in results[0][0]["entity"]

        # Search with filter
        filtered = client.search(
            "quickstart",
            data=[[50.0, 51.0, 0.0, 0.0]],
            limit=5,
            filter="score > 0.5",
            output_fields=["score"],
        )
        for hit in filtered[0]:
            assert hit["entity"]["score"] > 0.5

        # ── 7. Query (general filter) ────────────────────────────
        rows = client.query(
            "quickstart",
            filter="id >= 95",
            output_fields=["title"],
        )
        assert len(rows) == 5  # ids 95, 96, 97, 98, 99

        # ── 8. Get (by ID list) ──────────────────────────────────
        rows = client.get("quickstart", ids=[10, 20, 30])
        assert len(rows) == 3
        assert sorted(r["id"] for r in rows) == [10, 20, 30]

        # ── 9. Delete by ID ──────────────────────────────────────
        client.delete("quickstart", ids=[0, 1, 2])
        rows = client.get("quickstart", ids=[0, 1, 2])
        assert len(rows) == 0

        # ── 10. Describe collection ──────────────────────────────
        desc = client.describe_collection("quickstart")
        field_names = sorted(f["name"] for f in desc["fields"])
        assert field_names == sorted(["id", "vec", "title", "score"])
        assert desc["num_partitions"] == 1

        # ── 11. Partition CRUD ───────────────────────────────────
        client.create_partition("quickstart", "archive")
        partitions = client.list_partitions("quickstart")
        assert sorted(partitions) == ["_default", "archive"]
        assert client.has_partition("quickstart", "archive")

        client.drop_partition("quickstart", "archive")
        assert not client.has_partition("quickstart", "archive")

        # ── 12. Release ──────────────────────────────────────────
        client.release_collection("quickstart")
        state = client.get_load_state("quickstart")
        assert "NotLoad" in str(state["state"])

        # ── 13. Drop collection ──────────────────────────────────
        client.drop_collection("quickstart")
        assert not client.has_collection("quickstart")
        assert "quickstart" not in client.list_collections()

    finally:
        client.close()


def test_quickstart_multi_query_search(grpc_server):
    """Multi-query search with nq > 1 — covers the per-query topks
    flattening in build_search_result_data."""
    port, _db = grpc_server
    client = MilvusClient(uri=f"http://127.0.0.1:{port}")

    try:
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)

        client.create_collection("multi", schema=schema)
        client.insert("multi", [
            {"id": i, "vec": [float(i), 0.0, 0.0, 0.0]}
            for i in range(50)
        ])
        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW",
                      metric_type="L2", params={"M": 16})
        client.create_index("multi", idx)
        client.load_collection("multi")

        results = client.search(
            "multi",
            data=[
                [0.0, 0.0, 0.0, 0.0],
                [25.0, 0.0, 0.0, 0.0],
                [49.0, 0.0, 0.0, 0.0],
            ],
            limit=3,
        )
        assert len(results) == 3
        assert results[0][0]["id"] == 0   # closest to [0,0,0,0]
        assert results[1][0]["id"] == 25  # closest to [25,0,0,0]
        assert results[2][0]["id"] == 49  # closest to [49,0,0,0]

        client.drop_collection("multi")
    finally:
        client.close()
