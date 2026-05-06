"""M6 demo — long-run with compaction + tombstone GC.

Inserts a moderate number of records with frequent flushes triggered
by a small MEMTABLE_SIZE_LIMIT, periodically deletes some, and prints
the data file count and delta_index size as the run progresses.

Verifies:
    - data file count stays bounded by MAX_DATA_FILES
    - delta_index size never exceeds the number of distinct deleted pks
    - search results remain consistent with the live state
    - all surviving records are still readable

Run:
    cd milvus-lite-v2
    python examples/m6_demo.py
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

import milvus_lite.engine.collection as collection_mod
from milvus_lite.constants import DEFAULT_PARTITION, MAX_DATA_FILES
from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


N = 500
DIM = 8
SEED = 6


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

    # Force frequent flushes so compaction is exercised hard.
    original_limit = collection_mod.MEMTABLE_SIZE_LIMIT
    collection_mod.MEMTABLE_SIZE_LIMIT = 8

    data_dir = tempfile.mkdtemp(prefix="milvus_lite_m6_")
    print(f"data_dir = {data_dir}")
    print(f"MEMTABLE_SIZE_LIMIT = {collection_mod.MEMTABLE_SIZE_LIMIT}, "
          f"MAX_DATA_FILES = {MAX_DATA_FILES}")

    try:
        col = Collection("demo", data_dir, schema)
        deleted_pks = set()
        max_files_seen = 0

        for i in range(N):
            col.insert([records[i]])
            # Delete every 5th record we've inserted so far.
            if i > 0 and i % 5 == 0:
                pk_to_delete = f"doc_{(i // 2):04d}"
                col.delete([pk_to_delete])
                deleted_pks.add(pk_to_delete)
            files_now = len(col._manifest.get_data_files(DEFAULT_PARTITION))
            if files_now > max_files_seen:
                max_files_seen = files_now
            if i % 100 == 0:
                print(
                    f"  i={i:4d}: data_files={files_now:2d}, "
                    f"delta_index={len(col._delta_index):3d}, "
                    f"deleted={len(deleted_pks):3d}"
                )

        files_final = len(col._manifest.get_data_files(DEFAULT_PARTITION))
        print(f"\n--- post-insert ---")
        print(f"max data files seen: {max_files_seen}")
        print(f"final data files:    {files_final}")
        print(f"final delta_index:   {len(col._delta_index)}")
        print(f"distinct deleted:    {len(deleted_pks)}")

        assert max_files_seen <= MAX_DATA_FILES, (
            f"max_files_seen={max_files_seen} > MAX_DATA_FILES={MAX_DATA_FILES}"
        )
        assert len(col._delta_index) <= len(deleted_pks)

        # All deleted pks must be gone.
        for pk in deleted_pks:
            assert col.get([pk]) == [], f"deleted {pk} still queryable"

        # Surviving records must be readable.
        n_alive_check = 0
        for i in range(N):
            pk = f"doc_{i:04d}"
            if pk in deleted_pks:
                continue
            rec = col.get([pk])
            if rec:
                n_alive_check += 1
        print(f"alive records found: {n_alive_check}")
        assert n_alive_check == N - len(deleted_pks)

        # Search must be consistent.
        live_idx = [i for i in range(N) if f"doc_{i:04d}" not in deleted_pks]
        live_vectors = vectors[live_idx]
        q = rng.standard_normal((1, DIM)).astype(np.float32)
        results = col.search(q.tolist(), top_k=5, metric_type="L2")
        actual = [h["id"] for h in results[0]]
        dists = np.linalg.norm(live_vectors - q[0], axis=1)
        expected = [f"doc_{live_idx[i]:04d}" for i in np.argsort(dists)[:5]]
        assert actual == expected, f"search mismatch: {actual} vs {expected}"
        print(f"top-5 search OK: {actual}")

        col.close()
        print("\nOK — M6 demo: file count bounded, deletes effective, search consistent")
    finally:
        collection_mod.MEMTABLE_SIZE_LIMIT = original_limit
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
