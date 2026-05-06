"""Write-Ahead Log — Arrow IPC Streaming, dual-file (data + delta).

Each WAL round corresponds to a pair of files (wal_data_{N}.arrow + wal_delta_{N}.arrow).
After a successful flush the pair is deleted.  Writers are lazily initialised on
first write so that unused files are never created.
"""

from __future__ import annotations

import io
import os
import re
from typing import BinaryIO, List, Optional, Tuple

import pyarrow as pa

from milvus_lite.constants import SEQ_FORMAT_WIDTH, WAL_DATA_TEMPLATE, WAL_DELTA_TEMPLATE


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _read_wal_file(path: str) -> List[pa.RecordBatch]:
    """Read a single WAL file and return its RecordBatch list.

    * File does not exist → []
    * File is complete   → all batches
    * File is truncated  → batches read before the truncation point
    * File severely corrupted (schema unreadable) → []

    The file handle is always released, even on the truncation path —
    this matters because crash-recovery hits the truncation path on
    every dirty shutdown.
    """
    if not os.path.exists(path):
        return []

    batches: list[pa.RecordBatch] = []
    try:
        with pa.OSFile(path, "rb") as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                batches.append(batch)
    except pa.ArrowInvalid:
        # Truncated RecordBatch — keep whatever was successfully read.
        # Open-stream succeeded (schema is fine), some later batch was cut.
        pass
    except (OSError, IOError):
        # File-level IO error (permissions, broken pipe, etc.) — give up,
        # return whatever was read before the failure.
        pass
    # NOTE: deliberately do NOT catch generic Exception — that would hide
    # real bugs (AttributeError / TypeError / KeyError) in the recovery path.

    return batches


def _cleanup_old_wals(wal_dir: str, up_to_number: int) -> None:
    """Delete all WAL files whose number is <= *up_to_number*."""
    for n in WAL.find_wal_files(wal_dir):
        if n <= up_to_number:
            data_path = os.path.join(
                wal_dir,
                WAL_DATA_TEMPLATE.format(n=n, w=SEQ_FORMAT_WIDTH),
            )
            delta_path = os.path.join(
                wal_dir,
                WAL_DELTA_TEMPLATE.format(n=n, w=SEQ_FORMAT_WIDTH),
            )
            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(delta_path):
                os.remove(delta_path)


# ---------------------------------------------------------------------------
# WAL class
# ---------------------------------------------------------------------------

SYNC_MODES = ("none", "close", "batch")


class WAL:
    """Write-Ahead Log with lazy dual-file initialisation.

    sync_mode controls fsync behaviour (see wal-design.md §8):
      - "none"  : never fsync (testing / benchmarks only)
      - "close" : fsync once before close_and_delete (default)
      - "batch" : fsync after every write_batch (strongest, slowest)

    The default "close" mode covers the container-OOM-restart case where
    a new process picks up the same volume before the OS has flushed the
    old process's page cache.
    """

    def __init__(
        self,
        wal_dir: str,
        wal_data_schema: pa.Schema,
        wal_delta_schema: pa.Schema,
        wal_number: int,
        sync_mode: str = "close",
    ) -> None:
        if sync_mode not in SYNC_MODES:
            raise ValueError(
                f"sync_mode must be one of {SYNC_MODES}, got {sync_mode!r}"
            )

        self.wal_dir = wal_dir
        self._wal_data_schema = wal_data_schema
        self._wal_delta_schema = wal_delta_schema
        self._number = wal_number
        self._sync_mode = sync_mode

        self._data_writer: Optional[pa.ipc.RecordBatchStreamWriter] = None
        self._delta_writer: Optional[pa.ipc.RecordBatchStreamWriter] = None
        # Use Python's built-in open() rather than pa.OSFile so that fileno()
        # is available for fsync. The buffering cost vs OSFile is negligible
        # for sequential WAL writes (vector batches dominate).
        self._data_sink: Optional[BinaryIO] = None
        self._delta_sink: Optional[BinaryIO] = None
        self._closed = False

        os.makedirs(wal_dir, exist_ok=True)

    # ── properties ──────────────────────────────────────────────

    @property
    def number(self) -> int:
        return self._number

    @property
    def data_path(self) -> Optional[str]:
        if self._data_writer is None:
            return None
        return os.path.join(
            self.wal_dir,
            WAL_DATA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
        )

    @property
    def delta_path(self) -> Optional[str]:
        if self._delta_writer is None:
            return None
        return os.path.join(
            self.wal_dir,
            WAL_DELTA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
        )

    # ── write ───────────────────────────────────────────────────

    def write_insert(self, record_batch: pa.RecordBatch) -> None:
        """Append *record_batch* to the wal_data file (lazy init).

        After every write we flush Python's userspace buffer so that the
        bytes reach the OS page cache. This is the minimum guarantee
        needed for "any process death (not just OS crash) preserves the
        write" — without it, a SIGKILL or finalizer-skipping path could
        lose data still buffered in the Python file object.
        ``sync_mode="batch"`` additionally fsyncs to disk for OS-crash
        durability.
        """
        assert not self._closed, "WAL already closed"

        if self._data_writer is None:
            path = os.path.join(
                self.wal_dir,
                WAL_DATA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
            )
            self._data_sink = open(path, "wb")
            self._data_writer = pa.ipc.new_stream(self._data_sink, self._wal_data_schema)

        self._data_writer.write_batch(record_batch)
        # Always flush Python buffer → OS. Cheap (one syscall) and
        # required for the recovery contract under any crash semantics.
        self._data_sink.flush()
        if self._sync_mode == "batch":
            os.fsync(self._data_sink.fileno())

    def write_delete(self, record_batch: pa.RecordBatch) -> None:
        """Append *record_batch* to the wal_delta file (lazy init)."""
        assert not self._closed, "WAL already closed"

        if self._delta_writer is None:
            path = os.path.join(
                self.wal_dir,
                WAL_DELTA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
            )
            self._delta_sink = open(path, "wb")
            self._delta_writer = pa.ipc.new_stream(self._delta_sink, self._wal_delta_schema)

        self._delta_writer.write_batch(record_batch)
        self._delta_sink.flush()
        if self._sync_mode == "batch":
            os.fsync(self._delta_sink.fileno())

    # ── lifecycle ───────────────────────────────────────────────

    def close_and_delete(self) -> None:
        """Close writers, fsync (if sync_mode requires), and delete both WAL files.

        Idempotent and exception-safe: every cleanup step is attempted
        regardless of earlier failures, exceptions are collected, and the
        first one is re-raised at the end. ``_closed`` is set unconditionally
        so a second call short-circuits.
        """
        if self._closed:
            return
        self._closed = True

        errors: list[BaseException] = []

        def _safe(action):
            try:
                action()
            except BaseException as e:  # noqa: BLE001 - we re-raise below
                errors.append(e)

        # Each (writer.close, sink.close) pair runs independently. We always
        # try to close the sink even if the writer's close blew up — leaking
        # an fd is worse than a duplicated exception.
        if self._data_writer is not None:
            _safe(self._data_writer.close)
            if self._sync_mode in ("close", "batch"):
                _safe(self._data_sink.flush)
                _safe(lambda: os.fsync(self._data_sink.fileno()))
        if self._data_sink is not None:
            _safe(self._data_sink.close)

        if self._delta_writer is not None:
            _safe(self._delta_writer.close)
            if self._sync_mode in ("close", "batch"):
                _safe(self._delta_sink.flush)
                _safe(lambda: os.fsync(self._delta_sink.fileno()))
        if self._delta_sink is not None:
            _safe(self._delta_sink.close)

        # File deletion happens unconditionally — orphan WAL files are worse
        # than a noisy close. The manifest already says these are gone (we
        # only get here from flush Step 6, after manifest commit).
        data_path = os.path.join(
            self.wal_dir,
            WAL_DATA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
        )
        delta_path = os.path.join(
            self.wal_dir,
            WAL_DELTA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
        )
        for path in (data_path, delta_path):
            if os.path.exists(path):
                _safe(lambda p=path: os.remove(p))

        if errors:
            raise errors[0]

    # ── static helpers ──────────────────────────────────────────

    @staticmethod
    def find_wal_files(wal_dir: str) -> List[int]:
        """Scan *wal_dir* and return a sorted list of WAL numbers found."""
        if not os.path.exists(wal_dir):
            return []

        numbers: set[int] = set()
        pattern = re.compile(r"^wal_(data|delta)_(\d+)\.arrow$")
        for filename in os.listdir(wal_dir):
            m = pattern.match(filename)
            if m:
                numbers.add(int(m.group(2)))

        return sorted(numbers)

    @staticmethod
    def recover(
        wal_dir: str,
        wal_number: int,
    ) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        """Read WAL files for *wal_number* and return (data_batches, delta_batches).

        Arrow IPC streams self-describe their schema, so no schema arguments
        are needed. The Phase-2 ``read_operations`` interface will replace
        this method once the Operation abstraction lands.
        """
        data_path = os.path.join(
            wal_dir,
            WAL_DATA_TEMPLATE.format(n=wal_number, w=SEQ_FORMAT_WIDTH),
        )
        delta_path = os.path.join(
            wal_dir,
            WAL_DELTA_TEMPLATE.format(n=wal_number, w=SEQ_FORMAT_WIDTH),
        )

        data_batches = _read_wal_file(data_path)
        delta_batches = _read_wal_file(delta_path)

        return data_batches, delta_batches
