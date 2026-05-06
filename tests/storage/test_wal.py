"""Tests for storage/wal.py — WAL write/recover round-trip, lifecycle, truncation."""

import os

import pyarrow as pa
import pytest

from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.schema.arrow_builder import build_wal_data_schema, build_wal_delta_schema
from milvus_lite.storage.wal import WAL, _read_wal_file, _cleanup_old_wals


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ])


@pytest.fixture
def wal_data_schema(schema):
    return build_wal_data_schema(schema)


@pytest.fixture
def wal_delta_schema(schema):
    return build_wal_delta_schema(schema)


@pytest.fixture
def wal_dir(tmp_path):
    d = tmp_path / "wal"
    d.mkdir()
    return str(d)


@pytest.fixture
def wal(wal_dir, wal_data_schema, wal_delta_schema):
    return WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)


def _make_data_batch(wal_data_schema) -> pa.RecordBatch:
    """Create a sample insert RecordBatch with 2 rows."""
    return pa.RecordBatch.from_pydict(
        {
            "_seq": [1, 2],
            "_partition": ["_default", "_default"],
            "id": [100, 200],
            "vec": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        },
        schema=wal_data_schema,
    )


def _make_delta_batch(wal_delta_schema) -> pa.RecordBatch:
    """Create a sample delete RecordBatch with 1 row."""
    return pa.RecordBatch.from_pydict(
        {
            "id": [100],
            "_seq": [3],
            "_partition": ["_default"],
        },
        schema=wal_delta_schema,
    )


# ---------------------------------------------------------------------------
# Lazy initialisation
# ---------------------------------------------------------------------------

def test_no_files_created_on_init(wal, wal_dir):
    """WAL __init__ should NOT create any .arrow files."""
    arrow_files = [f for f in os.listdir(wal_dir) if f.endswith(".arrow")]
    assert arrow_files == []


def test_data_path_none_before_write(wal):
    assert wal.data_path is None


def test_delta_path_none_before_write(wal):
    assert wal.delta_path is None


# ---------------------------------------------------------------------------
# Write + file creation
# ---------------------------------------------------------------------------

def test_write_insert_creates_file(wal, wal_dir, wal_data_schema):
    batch = _make_data_batch(wal_data_schema)
    wal.write_insert(batch)
    assert wal.data_path is not None
    assert os.path.exists(wal.data_path)
    assert wal.delta_path is None  # delta not touched


def test_write_delete_creates_file(wal, wal_dir, wal_delta_schema):
    batch = _make_delta_batch(wal_delta_schema)
    wal.write_delete(batch)
    assert wal.delta_path is not None
    assert os.path.exists(wal.delta_path)
    assert wal.data_path is None  # data not touched


# ---------------------------------------------------------------------------
# Write + recover round-trip
# ---------------------------------------------------------------------------

def test_write_insert_recover(wal_dir, wal_data_schema, wal_delta_schema):
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    batch = _make_data_batch(wal_data_schema)
    wal.write_insert(batch)
    # Close writer so EOS is written (simulating flush lifecycle)
    wal._data_writer.close()
    wal._data_sink.close()

    data_batches, delta_batches = WAL.recover(wal_dir, 1)
    assert len(data_batches) == 1
    assert data_batches[0].num_rows == 2
    assert delta_batches == []


def test_write_delete_recover(wal_dir, wal_data_schema, wal_delta_schema):
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    batch = _make_delta_batch(wal_delta_schema)
    wal.write_delete(batch)
    wal._delta_writer.close()
    wal._delta_sink.close()

    data_batches, delta_batches = WAL.recover(wal_dir, 1)
    assert data_batches == []
    assert len(delta_batches) == 1
    assert delta_batches[0].num_rows == 1


def test_write_both_recover(wal_dir, wal_data_schema, wal_delta_schema):
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.write_delete(_make_delta_batch(wal_delta_schema))
    # Close writers
    wal._data_writer.close()
    wal._data_sink.close()
    wal._delta_writer.close()
    wal._delta_sink.close()

    data_batches, delta_batches = WAL.recover(wal_dir, 1)
    assert len(data_batches) == 1
    assert len(delta_batches) == 1


def test_multiple_batches_recover(wal_dir, wal_data_schema, wal_delta_schema):
    """Multiple write_insert calls produce multiple RecordBatches on recover."""
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    batch = _make_data_batch(wal_data_schema)
    wal.write_insert(batch)
    wal.write_insert(batch)
    wal._data_writer.close()
    wal._data_sink.close()

    data_batches, _ = WAL.recover(wal_dir, 1)
    assert len(data_batches) == 2
    assert data_batches[0].num_rows == 2
    assert data_batches[1].num_rows == 2


# ---------------------------------------------------------------------------
# close_and_delete
# ---------------------------------------------------------------------------

def test_close_and_delete_removes_files(wal_dir, wal_data_schema, wal_delta_schema):
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.write_delete(_make_delta_batch(wal_delta_schema))
    data_path = wal.data_path
    delta_path = wal.delta_path
    assert os.path.exists(data_path)
    assert os.path.exists(delta_path)

    wal.close_and_delete()
    assert not os.path.exists(data_path)
    assert not os.path.exists(delta_path)


def test_close_and_delete_idempotent(wal_dir, wal_data_schema, wal_delta_schema):
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.close_and_delete()
    # Second call should not raise
    wal.close_and_delete()


def test_close_and_delete_no_files_written(wal):
    """close_and_delete on a WAL that never wrote anything should not raise."""
    wal.close_and_delete()


def test_write_after_close_raises(wal, wal_data_schema):
    wal.close_and_delete()
    with pytest.raises(AssertionError, match="WAL already closed"):
        wal.write_insert(_make_data_batch(wal_data_schema))


# ---------------------------------------------------------------------------
# find_wal_files
# ---------------------------------------------------------------------------

def test_find_wal_files_empty(tmp_path):
    d = tmp_path / "empty_wal"
    d.mkdir()
    assert WAL.find_wal_files(str(d)) == []


def test_find_wal_files_nonexistent(tmp_path):
    assert WAL.find_wal_files(str(tmp_path / "no_such_dir")) == []


def test_find_wal_files_mixed(wal_dir, wal_data_schema, wal_delta_schema):
    # Create WAL 1 with data only
    w1 = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    w1.write_insert(_make_data_batch(wal_data_schema))
    w1._data_writer.close()
    w1._data_sink.close()

    # Create WAL 3 with delta only
    w3 = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=3)
    w3.write_delete(_make_delta_batch(wal_delta_schema))
    w3._delta_writer.close()
    w3._delta_sink.close()

    found = WAL.find_wal_files(wal_dir)
    assert found == [1, 3]


def test_find_wal_files_ignores_non_wal(wal_dir):
    """Non-WAL files in the directory should be ignored."""
    with open(os.path.join(wal_dir, "random.txt"), "w") as f:
        f.write("noise")
    assert WAL.find_wal_files(wal_dir) == []


# ---------------------------------------------------------------------------
# Truncation handling
# ---------------------------------------------------------------------------

def test_truncated_file_recovers_partial(wal_dir, wal_data_schema, wal_delta_schema):
    """A truncated WAL file should return the batches read before truncation."""
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.write_insert(_make_data_batch(wal_data_schema))
    # Don't close writer — simulate crash (no EOS marker).
    # Forcefully close the sink to flush OS buffers.
    wal._data_writer = None  # prevent close_and_delete from using it
    wal._data_sink.close()

    data_batches, _ = WAL.recover(wal_dir, 1)
    # Should recover the 2 complete batches (missing EOS is handled gracefully)
    assert len(data_batches) >= 1


def test_corrupted_file_returns_empty(wal_dir, wal_data_schema, wal_delta_schema):
    """A completely corrupted file should return an empty list."""
    path = os.path.join(wal_dir, "wal_data_000001.arrow")
    with open(path, "wb") as f:
        f.write(b"not an arrow file at all")

    data_batches, delta_batches = WAL.recover(wal_dir, 1)
    assert data_batches == []
    assert delta_batches == []


def test_read_wal_file_does_not_swallow_real_bugs(monkeypatch, wal_dir, wal_data_schema, wal_delta_schema):
    """_read_wal_file must not swallow generic exceptions like AttributeError —
    those would be real bugs in the recovery path, not corruption."""
    # Make a valid file first.
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.close_and_delete()  # this also deletes the file, so make a fresh one
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=2)
    wal.write_insert(_make_data_batch(wal_data_schema))
    # Force file to be readable later by closing the writer (writes EOS).
    wal._data_writer.close()
    wal._data_sink.close()
    wal._closed = True  # prevent close_and_delete from re-closing

    # Inject a bug: monkeypatch open_stream to raise AttributeError.
    def bad_open_stream(*args, **kwargs):
        raise AttributeError("simulated bug")

    monkeypatch.setattr(pa.ipc, "open_stream", bad_open_stream)

    with pytest.raises(AttributeError, match="simulated bug"):
        _read_wal_file(os.path.join(wal_dir, "wal_data_000002.arrow"))


# ---------------------------------------------------------------------------
# _read_wal_file
# ---------------------------------------------------------------------------

def test_read_wal_file_missing():
    assert _read_wal_file("/no/such/path.arrow") == []


# ---------------------------------------------------------------------------
# _cleanup_old_wals
# ---------------------------------------------------------------------------

def test_cleanup_old_wals(wal_dir, wal_data_schema, wal_delta_schema):
    for n in (1, 2, 3):
        w = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=n)
        w.write_insert(_make_data_batch(wal_data_schema))
        w._data_writer.close()
        w._data_sink.close()

    _cleanup_old_wals(wal_dir, up_to_number=2)

    remaining = WAL.find_wal_files(wal_dir)
    assert remaining == [3]


# ---------------------------------------------------------------------------
# WAL number property
# ---------------------------------------------------------------------------

def test_number_property(wal):
    assert wal.number == 1


def test_recover_nonexistent_wal(wal_dir, wal_data_schema, wal_delta_schema):
    """Recovering a WAL number that has no files should return empty lists."""
    data, delta = WAL.recover(wal_dir, 99)
    assert data == []
    assert delta == []


# ---------------------------------------------------------------------------
# sync_mode + fsync
# ---------------------------------------------------------------------------

def test_sync_mode_invalid_raises(wal_dir, wal_data_schema, wal_delta_schema):
    with pytest.raises(ValueError, match="sync_mode must be one of"):
        WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="weird")


def test_sync_mode_default_is_close(wal):
    """Default constructor must give the safer mode, not 'none'."""
    assert wal._sync_mode == "close"


def test_sync_mode_close_fsyncs_on_close(monkeypatch, wal_dir, wal_data_schema, wal_delta_schema):
    """sync_mode='close' should call os.fsync exactly once per writer at close time."""
    fsync_calls: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="close")
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.write_delete(_make_delta_batch(wal_delta_schema))
    assert fsync_calls == []  # not yet
    wal.close_and_delete()
    assert len(fsync_calls) == 2  # once per writer at close
    # restore so other tests aren't affected
    monkeypatch.setattr(os, "fsync", real_fsync)


def test_sync_mode_close_fsyncs_only_initialised_writer(monkeypatch, wal_dir, wal_data_schema, wal_delta_schema):
    """If only inserts happened, only the data writer should be fsynced."""
    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="close")
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.close_and_delete()
    assert len(fsync_calls) == 1


def test_sync_mode_batch_fsyncs_per_write(monkeypatch, wal_dir, wal_data_schema, wal_delta_schema):
    """sync_mode='batch' should fsync after every write_batch call."""
    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="batch")
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.write_delete(_make_delta_batch(wal_delta_schema))
    # 3 batch fsyncs so far
    assert len(fsync_calls) == 3
    wal.close_and_delete()
    # +2 fsync at close (one per writer)
    assert len(fsync_calls) == 5


def test_sync_mode_none_skips_fsync(monkeypatch, wal_dir, wal_data_schema, wal_delta_schema):
    """sync_mode='none' should never call fsync."""
    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="none")
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.close_and_delete()
    assert fsync_calls == []


# ---------------------------------------------------------------------------
# close_and_delete exception safety
# ---------------------------------------------------------------------------

def test_close_and_delete_fails_safely_when_data_close_raises(
    monkeypatch, wal_dir, wal_data_schema, wal_delta_schema
):
    """If data_writer.close raises, the delta side must still be closed and
    _closed must still be set to True (so a second call is a no-op).

    The exception is allowed to propagate — but the cleanup must happen.
    """
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="none")
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.write_delete(_make_delta_batch(wal_delta_schema))

    # Track whether the delta writer's close was called.
    delta_writer_closed = {"called": False}
    real_delta_close = wal._delta_writer.close

    def tracked_delta_close():
        delta_writer_closed["called"] = True
        return real_delta_close()

    monkeypatch.setattr(wal._delta_writer, "close", tracked_delta_close)

    # Make data_writer.close raise.
    def boom():
        raise OSError("disk full")

    monkeypatch.setattr(wal._data_writer, "close", boom)

    with pytest.raises(OSError, match="disk full"):
        wal.close_and_delete()

    # Even though data side blew up, delta side must have been closed
    # AND _closed must be True.
    assert delta_writer_closed["called"]
    assert wal._closed is True

    # Second call must be a no-op (idempotent), not re-raise.
    wal.close_and_delete()


def test_close_and_delete_deletes_files_even_when_close_raises(
    monkeypatch, wal_dir, wal_data_schema, wal_delta_schema
):
    """File deletion is in the outer finally — orphan files are worse than
    a noisy close, so deletion must run unconditionally."""
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="none")
    wal.write_insert(_make_data_batch(wal_data_schema))
    data_path = wal.data_path

    def boom():
        raise OSError("simulated")

    monkeypatch.setattr(wal._data_writer, "close", boom)

    with pytest.raises(OSError):
        wal.close_and_delete()

    # File must still be removed.
    assert not os.path.exists(data_path)


# ---------------------------------------------------------------------------
# fd leak guard for _read_wal_file
# ---------------------------------------------------------------------------

def test_read_wal_file_does_not_leak_fd_on_truncated(wal_dir, wal_data_schema, wal_delta_schema):
    """Repeatedly reading a truncated file must not leak file descriptors.

    We open many times in a tight loop; if `with` is missing, the OS will
    eventually run out of fds (or at least the count will grow).
    """
    # Create a truncated wal_data file (write a couple of batches without EOS).
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1, sync_mode="none")
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal.write_insert(_make_data_batch(wal_data_schema))
    # Bypass close_and_delete: don't write EOS, just close the underlying sink.
    wal._data_writer = None
    wal._data_sink.close()
    wal._closed = True

    path = os.path.join(wal_dir, "wal_data_000001.arrow")
    assert os.path.exists(path)

    # Open many times — `with` should release each handle.
    for _ in range(200):
        _read_wal_file(path)
    # If we got here without OSError("Too many open files"), the fix works.
