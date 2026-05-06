"""M3 demo — persistence + crash recovery.

Two-phase demo:
    python examples/m3_demo.py write   # insert records, then exit uncleanly
    python examples/m3_demo.py read    # reopen, recover, verify

The 'write' phase inserts records and uses os._exit(0) to skip all
Python finalizers — the strongest form of "process death" we can simulate
in-process. Recovery in the 'read' phase must still find every record.

Usage:
    cd milvus-lite-v2
    python examples/m3_demo.py write
    python examples/m3_demo.py read
"""

from __future__ import annotations

import os
import shutil
import sys

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


DATA_DIR = "/tmp/milvus_lite_m3_demo"
N_RECORDS = 50


def make_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
            FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        ],
    )


def write_phase() -> None:
    # Fresh start.
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    col = Collection("demo", DATA_DIR, make_schema())
    print(f"[write] inserting {N_RECORDS} records...")
    for i in range(N_RECORDS):
        col.insert([{
            "id": f"doc_{i:04d}",
            "vec": [0.5, 0.25, 0.125, 0.75],
            "title": f"title_{i}",
        }])
    print(f"[write] memtable size: {col.count()}")
    print(f"[write] data_dir: {DATA_DIR}")
    print("[write] simulating crash with os._exit(0) — no finalizers run")
    os._exit(0)  # noqa


def read_phase() -> None:
    if not os.path.exists(DATA_DIR):
        print(f"[read] {DATA_DIR} does not exist — run 'write' first")
        sys.exit(1)

    print(f"[read] reopening Collection at {DATA_DIR}")
    col = Collection("demo", DATA_DIR, make_schema())
    print(f"[read] post-recovery memtable size: {col.count()}")

    if col.count() != N_RECORDS:
        print(f"[read] FAIL: expected {N_RECORDS}, got {col.count()}")
        sys.exit(1)

    # Spot-check a few.
    for i in (0, N_RECORDS // 2, N_RECORDS - 1):
        pk = f"doc_{i:04d}"
        [rec] = col.get([pk])
        print(f"[read]   {pk}: title={rec['title']!r}, vec={rec['vec']}")

    col.close()
    print("[read] OK — all records recovered through WAL replay")
    shutil.rmtree(DATA_DIR)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("write", "read"):
        print("usage: python examples/m3_demo.py [write|read]")
        sys.exit(2)
    if sys.argv[1] == "write":
        write_phase()
    else:
        read_phase()


if __name__ == "__main__":
    main()
