"""Randomized Collection state-machine tests.

This is a lightweight property-style test without an extra dependency.
It drives the Collection through mixed inserts, upserts, deletes, flushes,
reopens, and release/load cycles, then compares read APIs against a simple
in-memory reference model.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def _schema() -> CollectionSchema:
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ])


def _record(pk: str, version: int) -> dict:
    pk_num = int(pk.split("_")[1])
    return {
        "id": pk,
        "vec": [float(pk_num), float(version) / 10.0],
        "title": f"{pk}:v{version}",
        "score": float((pk_num * 17 + version * 7) % 100),
    }


def _ids(rows: Iterable[dict]) -> List[str]:
    return sorted(row["id"] for row in rows)


def _expected_l2_top_ids(model: Dict[str, dict], query: list, k: int) -> List[str]:
    ranked = []
    for pk, rec in model.items():
        dist = math.sqrt(
            (rec["vec"][0] - query[0]) ** 2
            + (rec["vec"][1] - query[1]) ** 2
        )
        ranked.append((dist, pk))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [pk for _dist, pk in ranked[:k]]


def _assert_collection_matches_model(col: Collection, model: Dict[str, dict]) -> None:
    all_pks = [f"pk_{i}" for i in range(8)]

    got = col.get(all_pks, output_fields=["title", "score", "vec"])
    assert _ids(got) == sorted(model)
    by_id = {row["id"]: row for row in got}
    for pk, rec in model.items():
        assert by_id[pk]["title"] == rec["title"]
        assert by_id[pk]["score"] == rec["score"]
        assert by_id[pk]["vec"] == pytest.approx(rec["vec"])

    threshold = 50.0
    query_rows = col.query(
        f"score >= {threshold}",
        output_fields=["score"],
        limit=100,
    )
    expected_query_ids = sorted(
        pk for pk, rec in model.items()
        if rec["score"] >= threshold
    )
    assert _ids(query_rows) == expected_query_ids

    if model:
        query = [3.0, 0.0]
        top_k = min(4, len(model))
        hits = col.search(
            [query],
            top_k=top_k,
            metric_type="L2",
            output_fields=["score"],
        )[0]
        assert [hit["id"] for hit in hits] == _expected_l2_top_ids(model, query, top_k)
        for hit in hits:
            assert hit["entity"]["score"] == model[hit["id"]]["score"]
    else:
        assert col.search([[3.0, 0.0]], top_k=4, metric_type="L2") == [[]]


@pytest.mark.parametrize("seed", range(8))
def test_collection_randomized_state_machine(tmp_path, monkeypatch, seed):
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    rng = random.Random(seed)
    data_dir = str(tmp_path / f"db_{seed}")
    model: Dict[str, dict] = {}

    col = Collection("c", data_dir, _schema())
    version = 0
    try:
        for step in range(60):
            action = rng.choices(
                ["insert", "delete", "flush", "reopen", "release_load"],
                weights=[55, 25, 8, 7, 5],
            )[0]

            if action == "insert":
                batch = []
                for _ in range(rng.randint(1, 3)):
                    pk = f"pk_{rng.randrange(8)}"
                    version += 1
                    rec = _record(pk, version)
                    batch.append(rec)
                    model[pk] = rec
                col.insert(batch)
            elif action == "delete":
                pks = [f"pk_{rng.randrange(8)}" for _ in range(rng.randint(1, 3))]
                col.delete(pks)
                for pk in pks:
                    model.pop(pk, None)
            elif action == "flush":
                col.flush()
            elif action == "reopen":
                col.close()
                col = Collection("c", data_dir, _schema())
            else:
                col.release()
                col.load()
                assert col.load_state == "loaded"

            if step % 5 == 0:
                _assert_collection_matches_model(col, model)

        col.flush()
        col.close()
        col = Collection("c", data_dir, _schema())
        _assert_collection_matches_model(col, model)
    finally:
        col.close()
