"""M11 demo — scalar INVERTED index.

Shows scalar indexes on query() and search() filters:

    insert → flush → create_index(age/category) → query/search with filters
    → inspect .sidx sidecars → reopen → load → query again

Run:
    python examples/m11_scalar_index_demo.py
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from milvus_lite import CollectionSchema, DataType, FieldSchema, MilvusLite


def _schema() -> CollectionSchema:
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="age", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="category", dtype=DataType.VARCHAR, nullable=True, max_length=64),
        FieldSchema(name="active", dtype=DataType.BOOL, nullable=True),
    ])


def _rows() -> list[dict]:
    categories = ["tech", "news", "blog", "ai", None]
    rows = []
    for i in range(20):
        rows.append({
            "id": i,
            "vec": [float(i == j) for j in range(4)],
            "age": None if i % 9 == 0 else 18 + i,
            "category": categories[i % len(categories)],
            "active": None if i % 7 == 0 else i % 2 == 0,
        })
    return rows


def _ids(rows: list[dict]) -> list[int]:
    return [int(row["id"]) for row in rows]


def _index_files(data_dir: str) -> list[str]:
    index_dir = os.path.join(data_dir, "collections", "docs", "partitions", "_default", "indexes")
    if not os.path.isdir(index_dir):
        return []
    return sorted(os.listdir(index_dir))


def main() -> None:
    data_dir = tempfile.mkdtemp(prefix="milvus_lite_m11_scalar_index_")
    print(f"data_dir = {data_dir}")

    try:
        with MilvusLite(data_dir) as db:
            col = db.create_collection("docs", _schema())
            col.insert(_rows())
            col.flush()
            print("\n[1] inserted and flushed 20 rows")

            print("\n[2] create scalar INVERTED indexes")
            col.create_index("age", {"index_type": "INVERTED"})
            col.create_index("category", {"index_type": "INVERTED"})
            print(f"   indexes = {col.list_indexes()}")
            print(f"   age index info = {col.get_index_info('age')}")
            print(f"   category index info = {col.get_index_info('category')}")

            print("\n[3] query() uses indexed scalar predicates")
            adults = col.query("age >= 25 and age < 32", output_fields=["age", "category"])
            print(f"   age >= 25 and age < 32 ids = {_ids(adults)}")
            assert _ids(adults) == [7, 8, 10, 11, 12, 13]

            tech_or_ai = col.query(
                "category in ['tech', 'ai']",
                output_fields=["category"],
                limit=6,
            )
            print(f"   category in ['tech', 'ai'] first ids = {_ids(tech_or_ai)}")
            assert all(row["category"] in {"tech", "ai"} for row in tech_or_ai)

            print("\n[4] search() combines vector ranking with indexed scalar filter")
            query = np.asarray([[1, 0, 0, 0]], dtype=np.float32).tolist()
            results = col.search(
                query,
                top_k=5,
                metric_type="L2",
                expr="category == 'tech' and age is not null",
                output_fields=["age", "category"],
            )
            for hit in results[0]:
                entity = hit["entity"]
                print(
                    f"   id={hit['id']:2d} dist={hit['distance']:.1f} "
                    f"age={entity['age']} category={entity['category']}"
                )
                assert entity["category"] == "tech"
                assert entity["age"] is not None

            print("\n[5] .sidx sidecar files were written")
            sidecars = _index_files(data_dir)
            for filename in sidecars:
                print(f"   {filename}")
            assert any(filename.endswith(".age.inverted.sidx") for filename in sidecars)
            assert any(filename.endswith(".category.inverted.sidx") for filename in sidecars)

        print("\n[6] reopen collection and load existing scalar index sidecars")
        with MilvusLite(data_dir) as db:
            reopened = db.get_collection("docs")
            reopened.load()
            rows = reopened.query("category == 'news'", output_fields=["category"])
            print(f"   category == 'news' ids after reopen = {_ids(rows)}")
            assert _ids(rows) == [1, 6, 11, 16]

        print("\nOK — M11 demo passed: scalar INVERTED indexes work end-to-end")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
