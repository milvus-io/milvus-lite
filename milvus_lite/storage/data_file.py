"""Data Parquet file IO — stateless helpers.

Path layout:
    {data_dir}/partitions/{partition}/data/data_{seq_min:06d}_{seq_max:06d}.parquet

Files are immutable: written once, never modified, eventually deleted by
compaction. The on-disk schema is build_data_schema(schema): _seq +
user fields + [$meta]. No _partition column (partition is encoded in
the path).
"""

from __future__ import annotations

import os
import re
from typing import Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from milvus_lite.constants import DATA_FILE_TEMPLATE, SEQ_FORMAT_WIDTH


_DATA_NAME_RE = re.compile(r"^data_(\d+)_(\d+)\.parquet$")
_DELTA_NAME_RE = re.compile(r"^delta_(\d+)_(\d+)\.parquet$")


def write_data_file(
    table: pa.Table,
    partition_dir: str,
    seq_min: int,
    seq_max: int,
) -> str:
    """Write *table* to a data Parquet file.

    Args:
        table: Arrow Table matching ``build_data_schema(schema)``.
        partition_dir: absolute path of the partition directory
            (typically ``{data_dir}/partitions/{partition}``).
        seq_min: smallest _seq in the table (used in filename).
        seq_max: largest _seq in the table.

    Returns:
        Relative path of the written file inside ``partition_dir``,
        i.e. ``"data/data_000001_000010.parquet"``. The Manifest stores
        these relative paths so the data_dir is relocatable.
    """
    rel_dir = "data"
    abs_dir = os.path.join(partition_dir, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)

    filename = DATA_FILE_TEMPLATE.format(
        min=seq_min, max=seq_max, w=SEQ_FORMAT_WIDTH
    )
    rel_path = os.path.join(rel_dir, filename)
    abs_path = os.path.join(partition_dir, rel_path)

    pq.write_table(table, abs_path)
    return rel_path


def read_data_file(path: str) -> pa.Table:
    """Read a data Parquet file and return its Arrow Table."""
    return pq.read_table(path)


def parse_seq_range(filename: str) -> Tuple[int, int]:
    """Extract (seq_min, seq_max) from a data or delta filename.

    Accepts both ``data_NNN_MMM.parquet`` and ``delta_NNN_MMM.parquet``.
    """
    base = os.path.basename(filename)
    m = _DATA_NAME_RE.match(base) or _DELTA_NAME_RE.match(base)
    if m is None:
        raise ValueError(f"not a data/delta filename: {filename!r}")
    return int(m.group(1)), int(m.group(2))


def get_file_size(path: str) -> int:
    """Return file size in bytes — used by compaction bucketing."""
    return os.path.getsize(path)
