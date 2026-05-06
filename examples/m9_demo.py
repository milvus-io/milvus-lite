"""M9 demo — vector index lifecycle (Phase 9).

Walks through every public API introduced by Phase 9:

    create_index → load → search → release → search-raises → load → search

Plus the FAISS HNSW vs BruteForceIndex backend switch, partition CRUD,
output_fields projection, get_collection_stats, and combined
scalar+vector filtering. This is the long-form smoke test for the
whole Phase 9 stack — if it runs clean, the engine is ready for the
gRPC adapter layer in Phase 10.

Run:
    cd milvus-lite-v2
    python examples/m9_demo.py

Requires faiss-cpu for the HNSW path; falls back to BRUTE_FORCE if
faiss-cpu is not installed (and prints a notice).
"""

from __future__ import annotations

import shutil
import tempfile
import time

import numpy as np

from milvus_lite import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusLite,
)
from milvus_lite.exceptions import (
    CollectionNotLoadedError,
    IndexAlreadyExistsError,
    IndexNotFoundError,
)
from milvus_lite.index import is_faiss_available


def main() -> None:
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=16),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="active", dtype=DataType.BOOL),
        FieldSchema(name="category", dtype=DataType.VARCHAR),
    ])

    rng = np.random.default_rng(9)
    n = 10_000
    print(f"generating {n} records (16-dim vectors)…")
    rows = []
    categories = ["tech", "news", "blog", "ai"]
    for i in range(n):
        rows.append({
            "id": i,
            "vec": rng.standard_normal(16).astype(np.float32).tolist(),
            "title": f"doc_{i:05d}",
            "score": float(rng.uniform(0, 1)),
            "active": bool(rng.integers(0, 2)),
            "category": categories[i % 4],
        })

    use_hnsw = is_faiss_available()
    if use_hnsw:
        index_type = "HNSW"
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
            "search_params": {"ef": 128},
        }
    else:
        print("(notice) faiss-cpu is not installed; falling back to BRUTE_FORCE.")
        print("(notice) install with: pip install faiss-cpu")
        index_type = "BRUTE_FORCE"
        index_params = {
            "index_type": "BRUTE_FORCE",
            "metric_type": "COSINE",
            "params": {},
        }

    data_dir = tempfile.mkdtemp(prefix="milvus_lite_m9_")
    print(f"data_dir = {data_dir}")

    try:
        with MilvusLite(data_dir) as db:
            col = db.create_collection("docs", schema)

            # ── 1. partitions ──────────────────────────────────────
            print("\n[1] create + list partitions")
            col.create_partition("archive")
            print(f"   partitions = {col.list_partitions()}")
            assert col.has_partition("archive")
            assert col.has_partition("_default")

            # ── 2. insert into _default + archive ──────────────────
            print(f"\n[2] insert {n} records into _default, 100 into archive")
            t0 = time.time()
            col.insert(rows)
            col.insert(
                [{"id": 100_000 + i, "vec": rng.standard_normal(16).astype(np.float32).tolist(),
                  "title": "old", "score": 0.1, "active": False, "category": "tech"}
                 for i in range(100)],
                partition_name="archive",
            )
            col.flush()
            print(f"   wrote in {time.time() - t0:.2f}s")
            print(f"   num_entities = {col.num_entities}")
            print(f"   stats = {db.get_collection_stats('docs')}")

            # ── 3. describe (load_state should be 'loaded' — no index yet) ──
            print("\n[3] describe (no index yet)")
            d = col.describe()
            print(f"   load_state = {d['load_state']!r}")
            print(f"   index_spec = {d['index_spec']}")
            print(f"   num_entities = {d['num_entities']}")
            assert d['load_state'] == "loaded"
            assert d['index_spec'] is None

            # ── 4. create_index → released ─────────────────────────
            print(f"\n[4] create_index ({index_type}, COSINE)")
            col.create_index("vec", index_params)
            assert col.has_index() is True
            assert col.load_state == "released"
            print(f"   load_state = {col.load_state!r}")
            print(f"   index_spec = {col.get_index_info()}")

            # ── 5. search before load → CollectionNotLoadedError ───
            print("\n[5] search before load → CollectionNotLoadedError")
            q = rng.standard_normal((1, 16)).astype(np.float32).tolist()
            try:
                col.search(q, top_k=5)
            except CollectionNotLoadedError as e:
                print(f"   raised: {e}")

            # ── 6. load → loaded ───────────────────────────────────
            print("\n[6] load")
            t0 = time.time()
            col.load()
            print(f"   load_state = {col.load_state!r}")
            print(f"   loaded in {time.time() - t0:.2f}s")

            # ── 7. search now works ────────────────────────────────
            print("\n[7] search top-5 + scalar filter")
            results = col.search(
                q, top_k=5, expr="active == true and category == 'tech'",
            )
            for h in results[0]:
                e = h['entity']
                print(f"   id={h['id']:5d}  dist={h['distance']:.4f}"
                      f"  cat={e['category']}  active={e['active']}")
                assert e['category'] == 'tech'
                assert e['active'] is True

            # ── 8. partition-scoped search ─────────────────────────
            print("\n[8] search restricted to archive partition")
            results = col.search(q, top_k=3, partition_names=["archive"])
            for h in results[0]:
                print(f"   id={h['id']:6d}  dist={h['distance']:.4f}")
                assert h['id'] >= 100_000

            # ── 9. output_fields projection ────────────────────────
            print("\n[9] search with output_fields=['title','score']")
            results = col.search(q, top_k=3, output_fields=["title", "score"])
            for h in results[0]:
                assert set(h['entity'].keys()) == {"title", "score"}
                print(f"   id={h['id']}  entity={h['entity']}")

            # ── 10. duplicate create_index → IndexAlreadyExistsError ──
            print("\n[10] duplicate create_index → IndexAlreadyExistsError")
            try:
                col.create_index("vec", index_params)
            except IndexAlreadyExistsError as e:
                print(f"   raised: {e}")

            # ── 11. release → search again raises ──────────────────
            print("\n[11] release → search raises")
            col.release()
            print(f"   load_state = {col.load_state!r}")
            try:
                col.search(q, top_k=5)
            except CollectionNotLoadedError as e:
                print(f"   raised: {e}")

            # ── 12. load again — fast path (reads .idx from disk) ──
            print("\n[12] load again (should be fast — reads existing .idx)")
            t0 = time.time()
            col.load()
            print(f"   load_state = {col.load_state!r}")
            print(f"   reloaded in {time.time() - t0:.3f}s")

            # ── 13. drop_index → loaded (no index) ─────────────────
            print("\n[13] drop_index → state goes back to 'loaded'")
            col.drop_index("vec")
            assert col.has_index() is False
            assert col.load_state == "loaded"
            # Search still works (no index now → brute-force on segments)
            results = col.search(q, top_k=3)
            print(f"   search after drop_index → {len(results[0])} hits")

            # ── 14. drop_index when none → IndexNotFoundError ──────
            print("\n[14] drop_index when none → IndexNotFoundError")
            try:
                col.drop_index()
            except IndexNotFoundError as e:
                print(f"   raised: {e}")

        print("\nOK — M9 demo passed: full Phase 9 lifecycle works end-to-end")
        print(f"     index backend used: {index_type}")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
