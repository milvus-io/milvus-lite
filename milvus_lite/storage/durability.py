"""Durability barriers shared by immutable Parquet writers and the manifest."""

from __future__ import annotations

import errno
import os

import pyarrow as pa
import pyarrow.parquet as pq


def fsync_directory(path: str) -> None:
    """Persist directory entries; propagate real I/O failures.

    Windows cannot open directories this way. Some filesystems explicitly
    reject directory fsync; those platforms retain best-effort durability.
    """
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
                raise
    finally:
        os.close(fd)


def write_parquet_durable(table: pa.Table, path: str) -> None:
    """Persist a partition's Parquet file before it can enter the manifest.

    Layout: collection/partitions/partition/{data,delta}/file.parquet.
    Sync entries bottom-up through partitions, including directories that
    the caller may have just created. The manifest commit syncs collection.
    """
    with open(path, "wb") as sink:
        pq.write_table(table, sink)
        sink.flush()
        os.fsync(sink.fileno())

    directory = os.path.dirname(os.path.abspath(path))
    for _ in range(3):
        fsync_directory(directory)
        directory = os.path.dirname(directory)
