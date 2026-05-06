"""M7 demo — multi-Collection lifecycle through the public API.

This is the README quickstart in executable form. If you change the
public API, this demo (and the matching test_smoke_e2e.py) is the
canonical reference.

Run:
    cd milvus-lite-v2
    python examples/m7_demo.py
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

from milvus_lite import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusLite,
)


def main() -> None:
    data_dir = tempfile.mkdtemp(prefix="milvus_lite_m7_")
    print(f"data_dir = {data_dir}")

    schema_docs = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])
    schema_images = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="caption", dtype=DataType.VARCHAR, nullable=True),
    ])

    rng = np.random.default_rng(2026)

    try:
        # ── Phase A: create + populate two collections ─────────
        with MilvusLite(data_dir) as db:
            print("\n[A] creating two collections under one DB")
            docs = db.create_collection("docs", schema_docs)
            images = db.create_collection("images", schema_images)
            print(f"    list_collections: {db.list_collections()}")

            # Different schemas, different pk types, different dims —
            # they share nothing but the data_dir.
            doc_records = [
                {
                    "id": f"doc_{i:03d}",
                    "vec": rng.standard_normal(8).astype(np.float32).tolist(),
                    "title": f"document {i}",
                }
                for i in range(50)
            ]
            image_records = [
                {
                    "id": i,
                    "embedding": rng.standard_normal(4).astype(np.float32).tolist(),
                    "caption": f"image {i}",
                }
                for i in range(20)
            ]
            docs.insert(doc_records)
            images.insert(image_records)
            print(f"    docs.count = {docs.count()}, images.count = {images.count()}")

        # ── Phase B: reopen, query both ─────────────────────────
        print("\n[B] reopen DB, query each collection")
        with MilvusLite(data_dir) as db:
            assert set(db.list_collections()) == {"docs", "images"}

            docs = db.get_collection("docs")
            images = db.get_collection("images")

            # Query docs
            q_doc = rng.standard_normal(8).astype(np.float32)
            doc_hits = docs.search([q_doc.tolist()], top_k=3, metric_type="L2")[0]
            print(f"    docs top-3: {[h['id'] for h in doc_hits]}")

            # Query images (different pk type — int64)
            q_img = rng.standard_normal(4).astype(np.float32)
            img_hits = images.search([q_img.tolist()], top_k=3, metric_type="COSINE")[0]
            print(f"    images top-3: {[h['id'] for h in img_hits]}")
            assert all(isinstance(h["id"], int) for h in img_hits)

        # ── Phase C: drop one, keep the other ───────────────────
        print("\n[C] drop 'images', keep 'docs'")
        with MilvusLite(data_dir) as db:
            db.drop_collection("images")
            assert db.list_collections() == ["docs"]
            docs = db.get_collection("docs")
            # Verify docs is still intact by doing a search.
            q = rng.standard_normal(8).astype(np.float32)
            hits = docs.search([q.tolist()], top_k=5, metric_type="L2")[0]
            assert len(hits) == 5
            # And every doc is still readable via get
            for i in range(50):
                assert docs.get([f"doc_{i:03d}"]), f"missing doc_{i:03d}"
            print(f"    docs still has 50 records")

        # ── Phase D: LOCK demo ──────────────────────────────────
        print("\n[D] LOCK demo: a second open should be rejected")
        from milvus_lite import DataDirLockedError
        held = MilvusLite(data_dir)
        try:
            try:
                MilvusLite(data_dir)
                print("    ERROR: second open succeeded (LOCK not working)")
            except DataDirLockedError as e:
                print(f"    second open correctly raised DataDirLockedError")
        finally:
            held.close()

        # After release, third open works
        with MilvusLite(data_dir) as db:
            assert "docs" in db.list_collections()
            print("    after release, third open works")

        print("\nOK — M7 demo: multi-collection lifecycle through public API")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
