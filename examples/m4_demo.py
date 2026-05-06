"""M4 demo — 1000 random vectors, top-10 KNN search.

Verifies that the MilvusLite search result matches a direct numpy
brute-force computation. Crosses the flush boundary so half the data
lives in a Parquet segment and half in the live MemTable.

Run:
    cd milvus-lite-v2
    python examples/m4_demo.py
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


N = 1000
DIM = 16
TOP_K = 10
SEED = 42


def main() -> None:
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])

    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    records = [
        {"id": f"doc_{i:04d}", "vec": vectors[i].tolist(), "title": f"t{i}"}
        for i in range(N)
    ]

    data_dir = tempfile.mkdtemp(prefix="milvus_lite_m4_")
    print(f"data_dir = {data_dir}")

    try:
        col = Collection("demo", data_dir, schema)

        # Insert first half + flush — half the data ends up in a segment.
        col.insert(records[: N // 2])
        col.flush()
        print(f"after flush: memtable size = {col.count()}, manifest data files = "
              f"{len(col._manifest.get_data_files('_default'))}")

        # Insert the rest — these stay in MemTable.
        col.insert(records[N // 2:])
        print(f"after second insert: memtable size = {col.count()}")

        # Random query.
        query = rng.standard_normal((1, DIM)).astype(np.float32)

        # ── MilvusLite search ────────────────────────────────────
        results = col.search(query.tolist(), top_k=TOP_K, metric_type="L2")
        [hits] = results
        actual_ids = [h["id"] for h in hits]
        actual_dists = [h["distance"] for h in hits]
        print(f"MilvusLite top-{TOP_K} ids: {actual_ids}")

        # ── Direct numpy brute-force ────────────────────────────
        dists = np.linalg.norm(vectors - query[0], axis=1)
        expected_top_idx = np.argsort(dists)[:TOP_K]
        expected_ids = [f"doc_{i:04d}" for i in expected_top_idx]
        expected_dists = [float(dists[i]) for i in expected_top_idx]
        print(f"NumPy   top-{TOP_K} ids: {expected_ids}")

        # ── Verify ──────────────────────────────────────────────
        assert actual_ids == expected_ids, "ID mismatch vs brute force"
        for a, e in zip(actual_dists, expected_dists):
            assert abs(a - e) < 1e-4, f"distance mismatch: {a} vs {e}"
        print("\nOK — MilvusLite top-10 matches numpy brute force exactly")

        col.close()
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
