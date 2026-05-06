"""M5 demo — delete + restart.

End-to-end:
    insert N records  →  search returns them
                      →  delete half
                      →  search no longer returns the deleted ones
                      →  flush + restart
                      →  search still doesn't return them (delta_index
                         rebuilt from disk)
    insert(X) → delete(X) → insert(X)
                      →  search returns the new X (architectural §1
                         _seq-ordering invariant working end-to-end)

Run:
    cd milvus-lite-v2
    python examples/m5_demo.py
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


N = 100
DIM = 8


def main() -> None:
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])

    rng = np.random.default_rng(2026)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    records = [
        {"id": f"doc_{i:03d}", "vec": vectors[i].tolist(), "title": f"t{i}"}
        for i in range(N)
    ]

    data_dir = tempfile.mkdtemp(prefix="milvus_lite_m5_")
    print(f"data_dir = {data_dir}")

    try:
        # ── 1. insert N records ─────────────────────────────────
        col = Collection("demo", data_dir, schema)
        col.insert(records)
        print(f"inserted {col.count()} records into MemTable")

        query = vectors[0].tolist()  # query closest to doc_000
        results = col.search([query], top_k=5, metric_type="L2")
        print(f"top-5 before delete: {[h['id'] for h in results[0]]}")
        assert results[0][0]["id"] == "doc_000"

        # ── 2. delete the first half ────────────────────────────
        to_delete = [f"doc_{i:03d}" for i in range(N // 2)]
        col.delete(to_delete)
        print(f"deleted {len(to_delete)} pks (in MemTable)")

        results = col.search([query], top_k=5, metric_type="L2")
        top5 = [h["id"] for h in results[0]]
        print(f"top-5 after delete: {top5}")
        # None of the deleted ids should appear
        for pk in top5:
            i = int(pk.split("_")[1])
            assert i >= N // 2, f"deleted pk {pk} still appearing in search"

        # ── 3. flush + restart ──────────────────────────────────
        col.close()
        print("flushed + closed")

        col = Collection("demo", data_dir, schema)
        print(f"after restart: memtable size = {col.count()}, "
              f"data files = {len(col._manifest.get_data_files('_default'))}, "
              f"delta files = {len(col._manifest.get_delta_files('_default'))}")

        # Confirm deletes survived restart.
        results = col.search([query], top_k=5, metric_type="L2")
        top5_after_restart = [h["id"] for h in results[0]]
        print(f"top-5 after restart: {top5_after_restart}")
        for pk in top5_after_restart:
            i = int(pk.split("_")[1])
            assert i >= N // 2, f"deleted pk {pk} resurrected after restart"

        # ── 4. insert(X) → delete(X) → insert(X) ────────────────
        col.insert([{"id": "doc_000", "vec": query, "title": "resurrected_v1"}])
        col.delete(["doc_000"])
        col.insert([{"id": "doc_000", "vec": query, "title": "resurrected_v2"}])

        [hit_recs] = [col.get(["doc_000"])]
        print(f"after IDI sequence: get doc_000 → {hit_recs}")
        assert len(hit_recs) == 1
        assert hit_recs[0]["title"] == "resurrected_v2"

        # And search should now return doc_000 again.
        results = col.search([query], top_k=1, metric_type="L2")
        assert results[0][0]["id"] == "doc_000"
        assert results[0][0]["entity"]["title"] == "resurrected_v2"
        print("doc_000 correctly resurrected with v2")

        col.close()
        print("\nOK — M5 demo passed: delete + restart + insert/delete/insert")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
