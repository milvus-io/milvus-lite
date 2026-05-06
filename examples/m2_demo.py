"""M2 demo — first vertical slice.

Insert a few records, get them back. No flush, no recovery, no search,
no delete. The point of M2 is to prove the write path
(Collection → WAL → MemTable) and the in-memory read path.

Run:
    cd milvus-lite-v2
    pip install -e ".[dev]"
    python examples/m2_demo.py
"""

from __future__ import annotations

import os
import shutil
import tempfile

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def main() -> None:
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
            FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
            FieldSchema(name="score", dtype=DataType.FLOAT),
        ],
    )

    data_dir = tempfile.mkdtemp(prefix="milvus_lite_m2_")
    print(f"data_dir = {data_dir}")

    try:
        col = Collection("demo", data_dir, schema)

        # Insert.
        records = [
            {"id": "doc_1", "vec": [0.5, 0.25, 0.125, 0.75], "title": "first",  "score": 1.0},
            {"id": "doc_2", "vec": [0.0625, 1.5, 0.5, 0.25], "title": "second", "score": 2.0},
            {"id": "doc_3", "vec": [2.0, 0.375, 0.5, 0.25],  "title": "third",  "score": 3.0},
        ]
        pks = col.insert(records)
        print(f"inserted pks: {pks}")
        print(f"collection size: {col.count()}")

        # Get back.
        got = col.get(["doc_1", "doc_2", "doc_3", "doc_missing"])
        print(f"got {len(got)} records:")
        for r in got:
            print(f"  {r}")

        # Upsert.
        col.insert([
            {"id": "doc_2", "vec": [0.0, 0.0, 0.0, 0.0], "title": "second_v2", "score": 99.0},
        ])
        [updated] = col.get(["doc_2"])
        print(f"after upsert: {updated}")
        print(f"collection size still: {col.count()}")

        col.close()
        print("OK")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
