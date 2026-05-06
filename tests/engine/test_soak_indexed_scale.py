"""Large-scale indexed soak tests.

These are not part of regular or slow CI. They are intended for explicit
capacity/stability runs:

    MILVUS_LITE_RUN_SOAK=1 pytest -m soak tests/engine/test_soak_indexed_scale.py

The workloads use IVF_SQ8 scalar-quantized indexes to keep memory lower than
float32 indexes while still exercising FAISS index build/load/search paths.
"""

from __future__ import annotations

import math
import os
import random
from typing import Iterable, List

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.index.factory import is_faiss_available
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


pytestmark = [
    pytest.mark.soak,
    pytest.mark.skipif(
        os.environ.get("MILVUS_LITE_RUN_SOAK") != "1",
        reason="set MILVUS_LITE_RUN_SOAK=1 to run large soak tests",
    ),
    pytest.mark.skipif(not is_faiss_available(), reason="faiss-cpu is not installed"),
]


def _schema(dim: int) -> CollectionSchema:
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="bucket", dtype=DataType.INT64),
        FieldSchema(name="version", dtype=DataType.INT64),
    ])


def _vec(pk: int, version: int, dim: int) -> List[float]:
    base = pk * 0.00031 + version * 0.013
    return [
        math.sin(base * (i + 1)) + 0.25 * math.cos(base * (i + 5) * 0.19)
        for i in range(dim)
    ]


def _record(pk: int, version: int, dim: int) -> dict:
    return {
        "id": pk,
        "vec": _vec(pk, version, dim),
        "bucket": pk % 97,
        "version": version,
    }


def _records(start: int, stop: int, version: int, dim: int) -> Iterable[dict]:
    for pk in range(start, stop):
        yield _record(pk, version, dim)


def _parquet_count(data_dir: str) -> int:
    data_path = os.path.join(data_dir, "partitions", "_default", "data")
    if not os.path.exists(data_path):
        return 0
    return len([name for name in os.listdir(data_path) if name.endswith(".parquet")])


def _insert_range(
    col: Collection,
    *,
    start: int,
    stop: int,
    version: int,
    dim: int,
    batch_size: int,
) -> None:
    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        col.insert(list(_records(batch_start, batch_stop, version, dim)))


def _assert_sample_present(
    col: Collection,
    *,
    pks: Iterable[int],
    version: int,
    dim: int,
) -> None:
    for pk in pks:
        got = col.get([pk], output_fields=["version", "bucket", "vec"])
        assert len(got) == 1, f"missing pk={pk}"
        assert got[0]["version"] == version
        assert got[0]["bucket"] == pk % 97
        assert got[0]["vec"] == pytest.approx(_vec(pk, version, dim))


def _assert_sample_present_versions(
    col: Collection,
    *,
    versions: dict[int, int],
    dim: int,
) -> None:
    for pk, version in versions.items():
        got = col.get([pk], output_fields=["version", "bucket", "vec"])
        assert len(got) == 1, f"missing pk={pk}"
        assert got[0]["version"] == version
        assert got[0]["bucket"] == pk % 97
        assert got[0]["vec"] == pytest.approx(_vec(pk, version, dim))


def _assert_sample_deleted(col: Collection, pks: Iterable[int]) -> None:
    for pk in pks:
        assert col.get([pk]) == [], f"deleted pk={pk} is visible"


def _assert_search_returns_live_ids(
    col: Collection,
    *,
    queries: Iterable[int],
    version: int,
    dim: int,
    deleted: set[int],
    max_pk: int,
) -> None:
    for pk in queries:
        hits = col.search([_vec(pk, version, dim)], top_k=10, metric_type="L2")[0]
        assert hits, f"empty search result for pk={pk}"
        for hit in hits:
            assert 0 <= hit["id"] < max_pk
            assert hit["id"] not in deleted


def _expected_version(pk: int, upserted: set[int], changed_versions: dict[int, int]) -> int:
    if pk in changed_versions:
        return changed_versions[pk]
    if pk in upserted:
        return 2
    return 1


def _run_crud_churn(
    *,
    col: Collection,
    name: str,
    data_dir: str,
    dim: int,
    n_rows: int,
    deleted: set[int],
    upserted: set[int],
    changed_versions: dict[int, int],
    rounds: int,
    upsert_batch: int,
    delete_batch: int,
    query_every: int,
    flush_every: int,
    reopen_every: int,
) -> Collection:
    rng = random.Random(910_2026 + dim + n_rows)

    for round_idx in range(rounds):
        version = 10_000 + round_idx

        upsert_pks = [
            (round_idx * 37 + i * 101) % n_rows
            for i in range(upsert_batch)
        ]
        col.insert([_record(pk, version, dim) for pk in upsert_pks])
        for pk in upsert_pks:
            deleted.discard(pk)
            changed_versions[pk] = version

        delete_pks = [
            (round_idx * 53 + i * 89 + 17) % n_rows
            for i in range(delete_batch)
        ]
        col.delete(delete_pks)
        for pk in delete_pks:
            deleted.add(pk)
            changed_versions.pop(pk, None)

        live_probe_pks = [pk for pk in upsert_pks if pk not in deleted][:5]
        _assert_sample_present_versions(
            col,
            versions={
                pk: _expected_version(pk, upserted, changed_versions)
                for pk in live_probe_pks
            },
            dim=dim,
        )
        _assert_sample_deleted(col, delete_pks[:5])

        if round_idx % query_every == 0:
            bucket = rng.randrange(97)
            rows = col.query(f"bucket == {bucket}", output_fields=["version"], limit=50)
            for row in rows:
                assert row["id"] % 97 == bucket
                assert row["id"] not in deleted

            if live_probe_pks:
                _assert_search_returns_live_ids(
                    col,
                    queries=live_probe_pks[:2],
                    version=version,
                    dim=dim,
                    deleted=deleted,
                    max_pk=n_rows,
                )

        if flush_every and round_idx > 0 and round_idx % flush_every == 0:
            col.flush()
            col._wait_for_bg(timeout=600)

        if reopen_every and round_idx > 0 and round_idx % reopen_every == 0:
            col.close()
            col = Collection(name, data_dir, _schema(dim))
            col.load()

    return col


def _run_scale_soak(
    *,
    tmp_path,
    name: str,
    dim: int,
    n_rows: int,
    insert_batch_size: int,
    memtable_limit: int,
    nlist: int,
    nprobe: int,
    upsert_every: int,
    delete_every: int,
    file_count_limit: int,
    sample_queries: int = 8,
    churn_rounds: int = 0,
    churn_upsert_batch: int = 20,
    churn_delete_batch: int = 15,
) -> None:
    import milvus_lite.engine.collection as collection_mod

    old_limit = collection_mod.MEMTABLE_SIZE_LIMIT
    collection_mod.MEMTABLE_SIZE_LIMIT = memtable_limit
    data_dir = str(tmp_path / name)
    col = Collection(name, data_dir, _schema(dim))
    deleted = set(range(0, n_rows, delete_every))
    upserted = set(range(1, n_rows, upsert_every))
    final_deleted = deleted - upserted
    changed_versions: dict[int, int] = {}

    try:
        col.create_index("vec", {
            "index_type": "IVF_SQ8",
            "metric_type": "L2",
            "params": {"nlist": nlist},
            "search_params": {"nprobe": nprobe},
        })
        col.load()

        _insert_range(
            col,
            start=0,
            stop=n_rows,
            version=1,
            dim=dim,
            batch_size=insert_batch_size,
        )
        col.flush()
        col._wait_for_bg(timeout=600)

        col.delete(sorted(deleted))
        for batch_start in range(1, n_rows, upsert_every * insert_batch_size):
            batch = []
            for pk in range(batch_start, min(n_rows, batch_start + upsert_every * insert_batch_size), upsert_every):
                batch.append(_record(pk, 2, dim))
            if batch:
                col.insert(batch)
        col.flush()
        col._wait_for_bg(timeout=600)

        expected_live = n_rows - len(final_deleted)
        assert col.num_entities == expected_live
        assert _parquet_count(data_dir) <= file_count_limit

        sample_candidates = [
            0, 1, 7, 97, n_rows // 4, n_rows // 3, n_rows // 2,
            (n_rows * 3) // 4, n_rows - 1,
        ]
        live_samples = [
            pk for pk in sample_candidates
            if pk not in final_deleted and pk not in upserted
        ]
        upsert_samples = [pk for pk in sorted(upserted - deleted)[:5]]
        _assert_sample_present(col, pks=live_samples, version=1, dim=dim)
        _assert_sample_present(col, pks=upsert_samples, version=2, dim=dim)
        _assert_sample_deleted(col, sorted(final_deleted)[:10])
        _assert_search_returns_live_ids(
            col,
            queries=(live_samples + upsert_samples)[:sample_queries],
            version=1,
            dim=dim,
            deleted=final_deleted,
            max_pk=n_rows,
        )

        if churn_rounds:
            col = _run_crud_churn(
                col=col,
                name=name,
                data_dir=data_dir,
                dim=dim,
                n_rows=n_rows,
                deleted=final_deleted,
                upserted=upserted,
                changed_versions=changed_versions,
                rounds=churn_rounds,
                upsert_batch=churn_upsert_batch,
                delete_batch=churn_delete_batch,
                query_every=5,
                flush_every=50,
                reopen_every=200,
            )
            col.flush()
            col._wait_for_bg(timeout=600)
            expected_live = n_rows - len(final_deleted)
            assert col.num_entities == expected_live
            assert _parquet_count(data_dir) <= file_count_limit
            _assert_sample_present_versions(
                col,
                versions=dict(list(changed_versions.items())[:10]),
                dim=dim,
            )
            _assert_sample_deleted(col, sorted(final_deleted)[:10])

        col.close()
        col = Collection(name, data_dir, _schema(dim))
        col.load()
        assert col.num_entities == expected_live
        live_samples_after_churn = [
            pk for pk in live_samples
            if pk not in final_deleted and pk not in changed_versions
        ]
        upsert_samples_after_churn = [
            pk for pk in upsert_samples
            if pk not in final_deleted and pk not in changed_versions
        ]
        _assert_sample_present(col, pks=live_samples_after_churn, version=1, dim=dim)
        _assert_sample_present(col, pks=upsert_samples_after_churn, version=2, dim=dim)
        _assert_sample_present_versions(
            col,
            versions=dict(list(changed_versions.items())[:10]),
            dim=dim,
        )
        _assert_sample_deleted(col, sorted(final_deleted)[:10])
    finally:
        col.close()
        collection_mod.MEMTABLE_SIZE_LIMIT = old_limit


def test_soak_ivf_sq8_32d_500k_many_segments(tmp_path):
    _run_scale_soak(
        tmp_path=tmp_path,
        name="ivf_sq8_32d_500k_many_segments",
        dim=32,
        n_rows=500_000,
        insert_batch_size=5_000,
        memtable_limit=5_000,
        nlist=128,
        nprobe=32,
        upsert_every=101,
        delete_every=89,
        file_count_limit=160,
        churn_rounds=700,
        churn_upsert_batch=30,
        churn_delete_batch=20,
    )


def test_soak_ivf_sq8_32d_500k_large_segments(tmp_path):
    _run_scale_soak(
        tmp_path=tmp_path,
        name="ivf_sq8_32d_500k_large_segments",
        dim=32,
        n_rows=500_000,
        insert_batch_size=10_000,
        memtable_limit=100_000,
        nlist=128,
        nprobe=32,
        upsert_every=101,
        delete_every=89,
        file_count_limit=16,
        churn_rounds=500,
        churn_upsert_batch=30,
        churn_delete_batch=20,
    )


def test_soak_ivf_sq8_128d_100k_many_segments(tmp_path):
    _run_scale_soak(
        tmp_path=tmp_path,
        name="ivf_sq8_128d_100k_many_segments",
        dim=128,
        n_rows=100_000,
        insert_batch_size=2_000,
        memtable_limit=2_000,
        nlist=64,
        nprobe=24,
        upsert_every=97,
        delete_every=83,
        file_count_limit=100,
        churn_rounds=400,
        churn_upsert_batch=20,
        churn_delete_batch=15,
    )


def test_soak_ivf_sq8_128d_500k_many_segments(tmp_path):
    _run_scale_soak(
        tmp_path=tmp_path,
        name="ivf_sq8_128d_500k_many_segments",
        dim=128,
        n_rows=500_000,
        insert_batch_size=5_000,
        memtable_limit=5_000,
        nlist=128,
        nprobe=32,
        upsert_every=97,
        delete_every=83,
        file_count_limit=180,
        churn_rounds=700,
        churn_upsert_batch=24,
        churn_delete_batch=18,
    )


def test_soak_ivf_sq8_128d_500k_large_segments(tmp_path):
    _run_scale_soak(
        tmp_path=tmp_path,
        name="ivf_sq8_128d_500k_large_segments",
        dim=128,
        n_rows=500_000,
        insert_batch_size=10_000,
        memtable_limit=100_000,
        nlist=128,
        nprobe=32,
        upsert_every=97,
        delete_every=83,
        file_count_limit=16,
        churn_rounds=500,
        churn_upsert_batch=24,
        churn_delete_batch=18,
    )
