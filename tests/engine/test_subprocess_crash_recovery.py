"""Process-level crash recovery tests.

The regular crash tests monkeypatch internal functions and raise
SystemExit. These tests kill the writer process with os._exit(0), so
file descriptors, WAL writers, and background executors are not closed
through normal Python cleanup paths.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def _schema() -> CollectionSchema:
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


def _run_crashing_writer(script: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr


def test_unflushed_wal_survives_real_process_exit(tmp_path):
    data_dir = str(tmp_path / "db")
    script = textwrap.dedent(f"""
        import os
        from milvus_lite.engine.collection import Collection
        from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
            FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        ])
        col = Collection("c", {data_dir!r}, schema)
        col.insert([
            {{"id": "a", "vec": [0.1, 0.2], "title": "alpha"}},
            {{"id": "b", "vec": [0.3, 0.4], "title": "beta"}},
        ])
        os._exit(0)
    """)

    _run_crashing_writer(script)

    col = Collection("c", data_dir, _schema())
    try:
        assert col.get(["a"])[0]["title"] == "alpha"
        assert col.get(["b"])[0]["title"] == "beta"
    finally:
        col.close()


def test_delete_wal_hides_flushed_segment_after_real_process_exit(tmp_path):
    data_dir = str(tmp_path / "db")
    script = textwrap.dedent(f"""
        import os
        from milvus_lite.engine.collection import Collection
        from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
            FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
        ])
        col = Collection("c", {data_dir!r}, schema)
        col.insert([
            {{"id": "a", "vec": [0.1, 0.2], "title": "alpha"}},
            {{"id": "b", "vec": [0.3, 0.4], "title": "beta"}},
        ])
        col.flush()
        col.delete(["a"])
        os._exit(0)
    """)

    _run_crashing_writer(script)

    col = Collection("c", data_dir, _schema())
    try:
        assert col.get(["a"]) == []
        assert col.get(["b"])[0]["title"] == "beta"
    finally:
        col.close()
