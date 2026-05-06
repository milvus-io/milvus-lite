"""Nightly-style indexed long-running stability workloads.

These tests are intentionally marked slow and excluded from the regular CI
gate. They exercise many more state transitions than focused unit tests while
searching through FAISS scalar-quantized indexes instead of the unindexed
NumPy path.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.index.factory import is_faiss_available
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_faiss_available(), reason="faiss-cpu is not installed"),
]


def _schema(dim: int) -> CollectionSchema:
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ])


def _vec(pk: int, version: int, dim: int) -> List[float]:
    # Deterministic high-dimensional vectors with enough variation to make
    # index build/search meaningful without storing a large fixture.
    base = pk * 0.017 + version * 0.003
    return [
        math.sin(base * (i + 1)) + 0.5 * math.cos(base * (i + 3) * 0.37)
        for i in range(dim)
    ]


def _record(pk: int, version: int, dim: int) -> dict:
    return {
        "id": pk,
        "vec": _vec(pk, version, dim),
        "tag": f"tag_{pk % 11}",
        "score": float((pk * 11 + version * 5) % 100),
    }


def _data_file_count(data_dir: str) -> int:
    data_path = os.path.join(data_dir, "partitions", "_default", "data")
    if not os.path.exists(data_path):
        return 0
    return len([name for name in os.listdir(data_path) if name.endswith(".parquet")])


def _open_loaded(name: str, data_dir: str, dim: int) -> Collection:
    col = Collection(name, data_dir, _schema(dim))
    col.load()
    return col


def _run_indexed_churn(
    *,
    tmp_path,
    monkeypatch,
    name: str,
    dim: int,
    index_params: dict,
    pk_space: int,
    steps: int,
    min_batch: int,
    max_batch: int,
    memtable_limit: int,
    seed: int,
    file_count_limit: int,
) -> None:
    monkeypatch.setattr(
        "milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", memtable_limit,
    )
    monkeypatch.setattr(
        "milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 3,
    )

    rng = random.Random(seed)
    data_dir = str(tmp_path / name)
    model: Dict[int, dict] = {}
    version = 0

    col = Collection(name, data_dir, _schema(dim))
    col.create_index("vec", index_params)
    col.load()

    try:
        for step in range(steps):
            action = rng.choices(
                ["insert", "delete", "search", "flush", "reopen"],
                weights=[66, 18, 8, 5, 3],
            )[0]

            if action == "insert":
                batch = []
                for pk in rng.sample(range(pk_space), rng.randint(min_batch, max_batch)):
                    version += 1
                    rec = _record(pk, version, dim)
                    batch.append(rec)
                    model[pk] = rec
                col.insert(batch)
            elif action == "delete":
                pks = rng.sample(range(pk_space), rng.randint(min_batch, max_batch))
                col.delete(pks)
                for pk in pks:
                    model.pop(pk, None)
            elif action == "search":
                if model:
                    pk = rng.choice(list(model))
                    hits = col.search(
                        [model[pk]["vec"]],
                        top_k=16,
                        metric_type=index_params["metric_type"],
                    )[0]
                    # Approximate quantized indexes need not return the
                    # self-query at rank 1 under churn, but they must never
                    # return deleted or unknown pks.
                    assert {hit["id"] for hit in hits}.issubset(set(model))
                else:
                    hits = col.search(
                        [_vec(0, 1, dim)],
                        top_k=16,
                        metric_type=index_params["metric_type"],
                    )[0]
                    assert hits == []
            elif action == "flush":
                col.flush()
            else:
                col.close()
                col = _open_loaded(name, data_dir, dim)

            if step % 1_000 == 0:
                col._wait_for_bg(timeout=60)
                rows = col.query("id >= 0", output_fields=["score"], limit=pk_space + 100)
                assert {row["id"] for row in rows} == set(model)
                assert _data_file_count(data_dir) <= file_count_limit

        col.flush()
        col._wait_for_bg(timeout=60)
        col.close()
        col = _open_loaded(name, data_dir, dim)

        rows = col.query(
            "id >= 0",
            output_fields=["tag", "score", "vec"],
            limit=pk_space + 100,
        )
        by_id = {row["id"]: row for row in rows}
        assert set(by_id) == set(model)
        for pk, expected in model.items():
            assert by_id[pk]["tag"] == expected["tag"]
            assert by_id[pk]["score"] == expected["score"]
            assert by_id[pk]["vec"] == pytest.approx(expected["vec"])
        assert _data_file_count(data_dir) <= file_count_limit
    finally:
        col.close()


def test_longrun_ivf_sq8_32d_large_churn(tmp_path, monkeypatch):
    """More rows at 32 dimensions using IVF_SQ8 scalar quantization."""
    _run_indexed_churn(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name="ivf_sq8_32d",
        dim=32,
        index_params={
            "index_type": "IVF_SQ8",
            "metric_type": "L2",
            "params": {"nlist": 96},
            "search_params": {"nprobe": 24},
        },
        pk_space=20_000,
        steps=15_000,
        min_batch=40,
        max_batch=120,
        memtable_limit=500,
        seed=2026042701,
        file_count_limit=160,
    )


def test_longrun_ivf_sq8_128d_medium_churn(tmp_path, monkeypatch):
    """Fewer rows at 128 dimensions using IVF_SQ8 scalar quantization."""
    _run_indexed_churn(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name="ivf_sq8_128d",
        dim=128,
        index_params={
            "index_type": "IVF_SQ8",
            "metric_type": "L2",
            "params": {"nlist": 64},
            "search_params": {"nprobe": 16},
        },
        pk_space=8_000,
        steps=4_000,
        min_batch=20,
        max_batch=60,
        memtable_limit=300,
        seed=2026042702,
        file_count_limit=120,
    )
