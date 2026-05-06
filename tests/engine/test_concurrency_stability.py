"""Concurrency stability tests for sustained mixed workloads.

The engine contract is single-writer per Collection, but readers should
remain stable while that writer inserts, upserts, deletes, flushes, and
background compaction/index work runs.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Dict

import numpy as np
import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ])


def _vec(pk: int, version: int) -> list[float]:
    rng = np.random.default_rng(pk * 1009 + version)
    return rng.random(4).astype(np.float32).tolist()


def _record(pk: int, version: int) -> dict:
    return {
        "id": pk,
        "vec": _vec(pk, version),
        "tag": f"tag_{pk % 5}",
        "score": float((pk * 13 + version * 7) % 100),
    }


def test_sustained_readers_with_single_writer_flush_and_compaction(
    tmp_path, schema, monkeypatch
):
    """Readers continuously call get/query/search while one writer mutates.

    This intentionally mixes:
    - upserts of existing pks
    - deletes of existing and missing pks
    - frequent memtable flushes
    - background compaction/index maintenance

    During the race readers assert local API invariants. After all
    background work drains, the final Collection state is compared exactly
    against a reference dict updated after each successful write.
    """
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 9)
    monkeypatch.setattr(
        "milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 2,
    )

    col = Collection("c", str(tmp_path / "d"), schema)
    col.create_index("vec", {
        "index_type": "BRUTE_FORCE",
        "metric_type": "L2",
        "params": {},
    })
    col.load()

    expected: Dict[int, dict] = {}
    expected_lock = threading.Lock()
    errors: list[str] = []
    stop = threading.Event()

    def writer():
        rng = random.Random(20260427)
        version = 0
        try:
            for step in range(80):
                action = rng.choices(
                    ["insert", "delete", "flush"],
                    weights=[70, 25, 5],
                )[0]

                if action == "insert":
                    batch = []
                    changed = {}
                    for _ in range(rng.randint(1, 4)):
                        pk = rng.randrange(40)
                        version += 1
                        rec = _record(pk, version)
                        batch.append(rec)
                        changed[pk] = rec
                    col.insert(batch)
                    with expected_lock:
                        expected.update(changed)
                elif action == "delete":
                    pks = [rng.randrange(40) for _ in range(rng.randint(1, 4))]
                    col.delete(pks)
                    with expected_lock:
                        for pk in pks:
                            expected.pop(pk, None)
                else:
                    col.flush()

                if step % 16 == 0:
                    col._wait_for_bg(timeout=10)
                time.sleep(0.0005)
        except Exception as exc:  # pragma: no cover - surfaced by assertion
            errors.append(f"writer: {exc!r}")
        finally:
            stop.set()

    def reader(reader_id: int):
        rng = random.Random(9000 + reader_id)
        try:
            while not stop.is_set():
                query_rows = col.query("id >= 0", output_fields=["tag"], limit=200)
                query_ids = [row["id"] for row in query_rows]
                assert len(query_ids) == len(set(query_ids))
                assert all("tag" in row for row in query_rows)

                pks = rng.sample(range(40), 8)
                got_rows = col.get(pks, output_fields=["score"])
                got_ids = [row["id"] for row in got_rows]
                assert set(got_ids).issubset(set(pks))
                assert len(got_ids) == len(set(got_ids))
                assert all("score" in row for row in got_rows)

                hits = col.search(
                    [_vec(rng.randrange(40), rng.randrange(200))],
                    top_k=6,
                    metric_type="L2",
                    output_fields=["tag"],
                )[0]
                hit_ids = [hit["id"] for hit in hits]
                assert len(hit_ids) == len(set(hit_ids))
                assert all("distance" in hit and "entity" in hit for hit in hits)
                assert all("tag" in hit["entity"] for hit in hits)
                time.sleep(0.001)
        except Exception as exc:
            errors.append(f"reader-{reader_id}: {exc!r}")
            stop.set()

    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader, args=(i,)) for i in range(2)]
    threads = [writer_thread, *reader_threads]

    for t in threads:
        t.start()
    writer_thread.join(timeout=20)
    stop.set()
    for t in reader_threads:
        t.join(timeout=5)

    assert not any(t.is_alive() for t in threads), "concurrency test thread hung"
    assert not errors, "\n".join(errors)

    col.flush()
    col._wait_for_bg(timeout=30)

    with expected_lock:
        expected_snapshot = dict(expected)

    rows = col.query("id >= 0", output_fields=["tag", "score", "vec"], limit=1000)
    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == set(expected_snapshot)
    for pk, rec in expected_snapshot.items():
        assert by_id[pk]["tag"] == rec["tag"]
        assert by_id[pk]["score"] == rec["score"]
        assert by_id[pk]["vec"] == pytest.approx(rec["vec"])

    col.close()

    reopened = Collection("c", str(tmp_path / "d"), schema)
    try:
        reopened.load()
        rows_after_reopen = reopened.query("id >= 0", output_fields=["tag"], limit=1000)
        assert {row["id"] for row in rows_after_reopen} == set(expected_snapshot)
    finally:
        reopened.close()
