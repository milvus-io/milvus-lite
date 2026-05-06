"""M8 demo — scalar filter expressions on search / get / query.

Demonstrates the Phase F1 grammar (Tier 1 — comparisons, IN, AND/OR/NOT)
through all three Collection read APIs.

Run:
    cd milvus-lite-v2
    python examples/m8_demo.py
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

from milvus_lite import (
    CollectionSchema,
    DataType,
    FieldSchema,
    FilterFieldError,
    FilterParseError,
    FilterTypeError,
    MilvusLite,
)


def main() -> None:
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="age", dtype=DataType.INT64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="active", dtype=DataType.BOOL),
        FieldSchema(name="category", dtype=DataType.VARCHAR),
    ])

    rng = np.random.default_rng(8)
    n = 100
    rows = []
    categories = ["tech", "news", "blog", "ai"]
    for i in range(n):
        rows.append({
            "id": f"doc_{i:03d}",
            "vec": rng.standard_normal(4).astype(np.float32).tolist(),
            "age": int(rng.integers(15, 80)),
            "title": f"title_{i}",
            "score": float(rng.uniform(0, 1)),
            "active": bool(rng.integers(0, 2)),
            "category": categories[i % 4],
        })

    data_dir = tempfile.mkdtemp(prefix="milvus_lite_m8_")
    print(f"data_dir = {data_dir}")

    try:
        with MilvusLite(data_dir) as db:
            col = db.create_collection("docs", schema)
            col.insert(rows)
            print(f"\ninserted {n} records")

            # ── 1. search() with int filter ─────────────────────
            print("\n[1] search top-5 with age > 50")
            q = rng.standard_normal((1, 4)).astype(np.float32).tolist()
            results = col.search(q, top_k=5, metric_type="L2", expr="age > 50")
            for h in results[0]:
                print(f"   id={h['id']} age={h['entity']['age']} dist={h['distance']:.3f}")
            for h in results[0]:
                assert h['entity']['age'] > 50

            # ── 2. search() with string + IN filter ─────────────
            print("\n[2] search with category in ['tech', 'ai']")
            results = col.search(q, top_k=5, metric_type="L2",
                                 expr="category in ['tech', 'ai']")
            for h in results[0]:
                print(f"   id={h['id']} category={h['entity']['category']}")
            for h in results[0]:
                assert h['entity']['category'] in ('tech', 'ai')

            # ── 3. search() with complex filter ─────────────────
            print("\n[3] search with age >= 30 and (category == 'tech' or score > 0.8)")
            results = col.search(
                q, top_k=5, metric_type="L2",
                expr="age >= 30 and (category == 'tech' or score > 0.8)",
            )
            for h in results[0]:
                e = h['entity']
                print(f"   id={h['id']} age={e['age']} cat={e['category']} score={e['score']:.2f}")

            # ── 4. get() with filter ────────────────────────────
            print("\n[4] get(['doc_000', 'doc_001', 'doc_002'], expr='active')")
            out = col.get(["doc_000", "doc_001", "doc_002"], expr="active")
            for r in out:
                print(f"   id={r['id']} active={r['active']}")
                assert r['active'] is True

            # ── 5. query() — pure scalar ────────────────────────
            print("\n[5] query(age > 70 and score > 0.5) — no vector")
            out = col.query("age > 70 and score > 0.5")
            print(f"   {len(out)} matches:")
            for r in out[:5]:
                print(f"     id={r['id']} age={r['age']} score={r['score']:.2f}")

            # ── 6. query() with output_fields ───────────────────
            print("\n[6] query(category in ['tech']) with output_fields=['title','age']")
            out = col.query("category in ['tech']",
                            output_fields=["title", "age"], limit=3)
            for r in out:
                print(f"   {r}")
                # pk is always kept
                assert set(r.keys()) == {"id", "title", "age"}

            # ── 7. NOT IN filter ────────────────────────────────
            print("\n[7] query(category not in ['tech', 'news']) limit=5")
            out = col.query("category not in ['tech', 'news']", limit=5)
            for r in out:
                assert r['category'] not in ('tech', 'news')
                print(f"   id={r['id']} category={r['category']}")

            # ── 8. error handling — friendly messages ───────────
            print("\n[8] error handling")
            try:
                col.search(q, expr="agg > 18")  # typo
            except FilterFieldError as e:
                print("   FilterFieldError (typo):")
                for line in str(e).splitlines():
                    print(f"     {line}")

            try:
                col.search(q, expr="age > 'eighteen'")  # type mismatch
            except FilterTypeError as e:
                print("   FilterTypeError (str vs int):")
                for line in str(e).splitlines():
                    print(f"     {line}")

            try:
                col.search(q, expr="age >> 18")  # bad syntax
            except FilterParseError as e:
                print("   FilterParseError (bad syntax):")
                for line in str(e).splitlines():
                    print(f"     {line}")

        print("\nOK — M8 demo passed: scalar filter expressions work end-to-end")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
