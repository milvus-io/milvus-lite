"""M10 demo — pymilvus quickstart against MilvusLite gRPC server.

The full Phase 10 lifecycle through the eyes of a pymilvus user:

    connect → create_collection → insert → flush →
    create_index(HNSW) → load → search(filter+output_fields) →
    query → get → delete → partition CRUD → describe →
    release → drop

This is the exact script that would work against a real Milvus
deployment. The only difference is the URI points to our local
MilvusLite gRPC server instead of a Milvus standalone.

Run:
    # Terminal 1: start the server
    python -m milvus_lite server --data-dir /tmp/milvus_lite_m10 --port 19530

    # Terminal 2: run this demo
    python examples/m10_demo.py

    # Or run both in one process:
    python examples/m10_demo.py --embedded
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedded", action="store_true",
                        help="Start a local gRPC server in-process")
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    args = parser.parse_args()

    try:
        from pymilvus import MilvusClient, DataType
    except ImportError:
        print("pymilvus is required: pip install pymilvus")
        sys.exit(1)

    server = db = None
    data_dir = None

    if args.embedded:
        try:
            from milvus_lite.adapter.grpc.server import start_server_in_thread
        except ImportError:
            print("milvus_lite gRPC support is required: pip install -e .")
            sys.exit(1)
        data_dir = tempfile.mkdtemp(prefix="milvus_lite_m10_")
        server, db, port = start_server_in_thread(data_dir)
        uri = f"http://127.0.0.1:{port}"
        print(f"embedded server on port {port} (data_dir={data_dir})")
    else:
        uri = args.uri
        print(f"connecting to {uri}")

    try:
        client = MilvusClient(uri=uri)
        print("connected\n")

        # ── 1. Create collection ─────────────────────────────────
        print("[1] create_collection")
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=16)
        schema.add_field("title", DataType.VARCHAR, max_length=128)
        schema.add_field("score", DataType.FLOAT)
        schema.add_field("active", DataType.BOOL)
        client.create_collection("demo", schema=schema)
        print(f"   collections: {client.list_collections()}")

        # ── 2. Insert ────────────────────────────────────────────
        print("\n[2] insert 500 records")
        rng = np.random.default_rng(10)
        data = [
            {"id": i, "vec": rng.standard_normal(16).astype(np.float32).tolist(),
             "title": f"doc_{i:04d}", "score": float(rng.uniform(0, 1)),
             "active": bool(rng.integers(0, 2))}
            for i in range(500)
        ]
        t0 = time.time()
        res = client.insert("demo", data)
        print(f"   inserted {res['insert_count']} in {time.time()-t0:.2f}s")

        # ── 3. Flush ─────────────────────────────────────────────
        print("\n[3] flush")
        client.flush("demo")

        # ── 4. Create index ──────────────────────────────────────
        print("\n[4] create_index HNSW COSINE")
        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW",
                      metric_type="COSINE",
                      params={"M": 16, "efConstruction": 200})
        client.create_index("demo", idx)
        desc = client.describe_index("demo", "vec")
        print(f"   index: {desc['index_type']} {desc['metric_type']}")

        # ── 5. Load ──────────────────────────────────────────────
        print("\n[5] load_collection")
        t0 = time.time()
        client.load_collection("demo")
        print(f"   loaded in {time.time()-t0:.2f}s")
        print(f"   state: {client.get_load_state('demo')}")

        # ── 6. Search ────────────────────────────────────────────
        print("\n[6] search top-5 with filter")
        q = rng.standard_normal((1, 16)).astype(np.float32).tolist()
        results = client.search(
            "demo", data=q, limit=5,
            filter="active == true and score > 0.3",
            output_fields=["title", "score", "active"],
        )
        for hit in results[0]:
            e = hit["entity"]
            print(f"   id={hit['id']:4d}  dist={hit['distance']:.4f}"
                  f"  title={e['title']}  score={e['score']:.2f}")

        # ── 7. Multi-query search ────────────────────────────────
        print("\n[7] multi-query search (nq=3, top_k=3)")
        qs = rng.standard_normal((3, 16)).astype(np.float32).tolist()
        results = client.search("demo", data=qs, limit=3)
        for i, hits in enumerate(results):
            ids = [h["id"] for h in hits]
            print(f"   query {i}: top-3 ids = {ids}")

        # ── 8. Query ─────────────────────────────────────────────
        print("\n[8] query(id >= 495)")
        rows = client.query("demo", filter="id >= 495", output_fields=["title"])
        for r in rows:
            print(f"   id={r['id']}  title={r['title']}")
        assert len(rows) == 5

        # ── 9. Get ───────────────────────────────────────────────
        print("\n[9] get(ids=[0, 100, 200])")
        rows = client.get("demo", ids=[0, 100, 200], output_fields=["title", "score"])
        for r in rows:
            print(f"   id={r['id']}  title={r['title']}  score={r['score']:.2f}")
        assert len(rows) == 3

        # ── 10. Delete ───────────────────────────────────────────
        print("\n[10] delete(ids=[0, 1, 2])")
        client.delete("demo", ids=[0, 1, 2])
        rows = client.get("demo", ids=[0, 1, 2])
        assert len(rows) == 0
        print("   deleted, get confirms gone")

        # ── 11. Partitions ───────────────────────────────────────
        print("\n[11] partition CRUD")
        client.create_partition("demo", "archive")
        print(f"   partitions: {client.list_partitions('demo')}")
        assert client.has_partition("demo", "archive")
        client.drop_partition("demo", "archive")
        print(f"   after drop: {client.list_partitions('demo')}")

        # ── 12. Describe ─────────────────────────────────────────
        print("\n[12] describe_collection")
        d = client.describe_collection("demo")
        print(f"   name: {d.get('collection_name', d.get('name'))}")
        print(f"   fields: {[f['name'] for f in d['fields']]}")
        print(f"   partitions: {d.get('num_partitions')}")

        # ── 13. Release + Drop ───────────────────────────────────
        print("\n[13] release + drop")
        client.release_collection("demo")
        print(f"   state after release: {client.get_load_state('demo')}")
        client.drop_collection("demo")
        print(f"   collections after drop: {client.list_collections()}")

        client.close()
        print("\nOK — M10 demo passed: pymilvus quickstart works end-to-end")

    finally:
        if server:
            server.stop(grace=2)
        if db:
            db.close()
        if data_dir:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
