"""Persistence barriers must precede publication and obsolete-file cleanup."""

import errno
import os
import stat

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from milvus_lite.storage import durability
from milvus_lite.storage.data_file import write_data_file
from milvus_lite.storage.delta_file import write_delta_file
from milvus_lite.storage.manifest import Manifest


@pytest.mark.parametrize("writer,kind", [(write_data_file, "data"), (write_delta_file, "delta")])
def test_parquet_syncs_contents_and_directory_chain(tmp_path, monkeypatch, writer, kind):
    partition = tmp_path / "partitions" / "p"
    events = []
    real_fsync = os.fsync

    def sync_file(fd):
        # The complete footer must be visible before the durability barrier.
        assert stat.S_ISREG(os.fstat(fd).st_mode)
        assert pq.read_table(partition / kind / f"{kind}_000001_000001.parquet").num_rows == 1
        real_fsync(fd)
        events.append("file")

    monkeypatch.setattr(durability.os, "fsync", sync_file)
    monkeypatch.setattr(durability, "fsync_directory", lambda path: events.append(path))
    writer(pa.table({"id": [1], "_seq": [1]}), str(partition), 1, 1)
    assert events == ["file", str(partition / kind), str(partition), str(partition.parent)]


@pytest.mark.parametrize("error", [errno.EIO, errno.ENOSPC])
def test_directory_sync_propagates_io_errors(tmp_path, monkeypatch, error):
    if os.name == "nt":
        pytest.skip("directory fsync unavailable")

    def fail(fd):
        raise OSError(error, "injected sync failure")

    monkeypatch.setattr(durability.os, "fsync", fail)
    with pytest.raises(OSError) as exc:
        durability.fsync_directory(str(tmp_path))
    assert exc.value.errno == error


def test_manifest_directory_sync_failure_is_not_success(tmp_path, monkeypatch):
    from milvus_lite.storage import manifest

    def fail(path):
        raise OSError(errno.EIO, "injected directory sync failure")

    monkeypatch.setattr(manifest, "fsync_directory", fail)
    with pytest.raises(OSError):
        Manifest(str(tmp_path)).save()


def test_directory_sync_unsupported_is_best_effort(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("directory fsync unavailable")

    def unsupported(fd):
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(durability.os, "fsync", unsupported)
    durability.fsync_directory(str(tmp_path))
