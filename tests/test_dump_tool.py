"""Tests for the milvus-lite dump command."""

from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace

import pytest

from milvus_lite import cmdline


class _FakeIterator:
    def __init__(self, batches):
        self._batches = list(batches)
        self.closed = False

    def next(self):
        if self._batches:
            return self._batches.pop(0)
        return []

    def close(self):
        self.closed = True


class _FakeCollection:
    last = None

    def __init__(self, name, using=None):
        self.name = name
        self.using = using
        self.primary_field = SimpleNamespace(name="id", auto_id=True)
        self.schema = SimpleNamespace(fields=[
            SimpleNamespace(name="id", dtype="INT64"),
            SimpleNamespace(name="vec", dtype="FLOAT_VECTOR"),
            SimpleNamespace(name="text", dtype="VARCHAR"),
        ])
        self.loaded = False
        self.iterator = _FakeIterator([
            [
                {"id": 10, "vec": [1.0, 0.0], "text": "a"},
                {"id": 11, "vec": [0.0, 1.0], "text": "b"},
            ],
            [
                {"id": 12, "vec": [0.5, 0.5], "text": "c"},
            ],
        ])
        _FakeCollection.last = self

    def load(self):
        self.loaded = True

    def query(self, expr, output_fields=None):
        assert expr == ""
        assert output_fields == ["count(*)"]
        return [{"count(*)": 3}]

    def query_iterator(self, batch_size=None, output_fields=None):
        assert batch_size == 2
        assert output_fields == ["*"]
        return self.iterator


class _FakeConnections:
    def __init__(self):
        self.connected = []
        self.disconnected = []

    def connect(self, alias, uri=None):
        self.connected.append((alias, uri))

    def disconnect(self, alias):
        self.disconnected.append(alias)


class _FakeUtility:
    def has_collection(self, name, using=None):
        return name == "docs" and using == cmdline.DUMP_ALIAS


class _FakeDataType:
    BFLOAT16_VECTOR = "BFLOAT16_VECTOR"
    BINARY_VECTOR = "BINARY_VECTOR"


class _FakeBulkFileType:
    JSON = "json"


class _FakeWriter:
    instances = []

    def __init__(self, schema, local_path, chunk_size, file_type):
        self.schema = schema
        self.local_path = local_path
        self.chunk_size = chunk_size
        self.file_type = file_type
        self.rows = []
        self.committed = False
        _FakeWriter.instances.append(self)

    def append_row(self, row):
        self.rows.append(row)

    def commit(self):
        self.committed = True


@pytest.fixture
def fake_dump_deps(monkeypatch):
    connections = _FakeConnections()
    api = SimpleNamespace(
        Collection=_FakeCollection,
        DataType=_FakeDataType,
        connections=connections,
        utility=_FakeUtility(),
    )
    _FakeWriter.instances = []
    monkeypatch.setattr(cmdline, "_load_pymilvus_api", lambda: api)
    monkeypatch.setattr(
        cmdline,
        "_load_bulk_writer",
        lambda: (_FakeWriter, _FakeBulkFileType),
    )
    return connections


def test_dump_collection_writes_rows_and_drops_auto_id(tmp_path, fake_dump_deps):
    db_path = tmp_path / "demo.db"
    db_path.mkdir()
    out_path = tmp_path / "dump"

    summary = cmdline.dump_collection(
        str(db_path),
        "docs",
        str(out_path),
        batch_size=2,
        chunk_size=1024,
    )

    assert summary == {
        "collection": "docs",
        "path": str(out_path),
        "row_count": 3,
    }
    assert fake_dump_deps.connected == [
        (cmdline.DUMP_ALIAS, str(db_path)),
    ]
    assert fake_dump_deps.disconnected == [cmdline.DUMP_ALIAS]
    assert _FakeCollection.last.loaded is True
    assert _FakeCollection.last.iterator.closed is True

    writer = _FakeWriter.instances[0]
    assert writer.local_path == str(out_path)
    assert writer.chunk_size == 1024
    assert writer.file_type == _FakeBulkFileType.JSON
    assert writer.committed is True
    assert writer.rows == [
        {"vec": [1.0, 0.0], "text": "a"},
        {"vec": [0.0, 1.0], "text": "b"},
        {"vec": [0.5, 0.5], "text": "c"},
    ]


def test_dump_collection_accepts_uri_without_local_path(tmp_path, fake_dump_deps):
    out_path = tmp_path / "dump"

    summary = cmdline.dump_collection(
        None,
        "docs",
        str(out_path),
        uri="http://127.0.0.1:19530",
        batch_size=2,
    )

    assert summary["row_count"] == 3
    assert fake_dump_deps.connected == [
        (cmdline.DUMP_ALIAS, "http://127.0.0.1:19530"),
    ]


def test_dump_collection_rejects_missing_db_file(tmp_path):
    with pytest.raises(RuntimeError, match="not exists"):
        cmdline.dump_collection(
            str(tmp_path / "missing.db"),
            "docs",
            str(tmp_path / "dump"),
        )


def test_dump_collection_rejects_unknown_collection(tmp_path, monkeypatch):
    db_path = tmp_path / "demo.db"
    db_path.mkdir()
    connections = _FakeConnections()
    api = SimpleNamespace(
        Collection=_FakeCollection,
        DataType=_FakeDataType,
        connections=connections,
        utility=SimpleNamespace(has_collection=lambda name, using=None: False),
    )
    monkeypatch.setattr(cmdline, "_load_pymilvus_api", lambda: api)
    monkeypatch.setattr(
        cmdline,
        "_load_bulk_writer",
        lambda: (_FakeWriter, _FakeBulkFileType),
    )

    with pytest.raises(RuntimeError, match="Collection: docs not exists"):
        cmdline.dump_collection(str(db_path), "docs", str(tmp_path / "dump"))

    assert connections.disconnected == [cmdline.DUMP_ALIAS]


def test_main_dispatches_dump(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_dump_collection(db_file, collection_name, path, **kwargs):
        calls.append((db_file, collection_name, path, kwargs))
        return {"collection": collection_name, "path": path, "row_count": 7}

    monkeypatch.setattr(cmdline, "dump_collection", fake_dump_collection)

    rc = cmdline.main([
        "dump",
        "-d", str(tmp_path / "demo.db"),
        "-c", "docs",
        "-p", str(tmp_path / "dump"),
        "--batch-size", "9",
        "--chunk-size", "123",
    ])

    assert rc == 0
    assert calls == [(
        str(tmp_path / "demo.db"),
        "docs",
        str(tmp_path / "dump"),
        {"uri": None, "batch_size": 9, "chunk_size": 123},
    )]
    assert "Dump collection docs success: 7 rows" in capsys.readouterr().out


def test_main_dispatches_server(monkeypatch, tmp_path):
    calls = []

    def fake_run_server(data_dir, host, port, max_workers):
        calls.append({
            "data_dir": data_dir,
            "host": host,
            "port": port,
            "max_workers": max_workers,
        })

    monkeypatch.setattr(cmdline, "run_server", fake_run_server)

    rc = cmdline.main([
        "server",
        "--data-dir", str(tmp_path / "data"),
        "--host", "127.0.0.1",
        "--port", "19531",
        "--max-workers", "3",
    ])

    assert rc == 0
    assert calls == [{
        "data_dir": str(tmp_path / "data"),
        "host": "127.0.0.1",
        "port": 19531,
        "max_workers": 3,
    }]


def test_package_module_main_delegates_to_cmdline(monkeypatch):
    calls = []

    def fake_main():
        calls.append(True)
        return 23

    monkeypatch.setattr(cmdline, "main", fake_main)
    monkeypatch.setitem(sys.modules, "milvus_lite.cmdline", cmdline)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("milvus_lite.__main__", run_name="__main__")

    assert exc_info.value.code == 23
    assert calls == [True]
