"""Delta Parquet file IO — stateless helpers, mirror of data_file.py.

Path layout:
    {data_dir}/partitions/{partition}/delta/delta_{seq_min:06d}_{seq_max:06d}.parquet

The on-disk schema is build_delta_schema(schema): {pk} + _seq.
No _partition column (partition is encoded in the path).
"""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

from milvus_lite.constants import DELTA_FILE_TEMPLATE, SEQ_FORMAT_WIDTH


def write_delta_file(
    table: pa.Table,
    partition_dir: str,
    seq_min: int,
    seq_max: int,
) -> str:
    """Write *table* (delta_schema) to a delta Parquet file.

    Returns relative path inside ``partition_dir`` —
    e.g. ``"delta/delta_000005_000005.parquet"``.
    """
    rel_dir = "delta"
    abs_dir = os.path.join(partition_dir, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)

    filename = DELTA_FILE_TEMPLATE.format(
        min=seq_min, max=seq_max, w=SEQ_FORMAT_WIDTH
    )
    rel_path = os.path.join(rel_dir, filename)
    abs_path = os.path.join(partition_dir, rel_path)

    pq.write_table(table, abs_path)
    return rel_path


def read_delta_file(path: str) -> pa.Table:
    """Read a delta Parquet file and return its Arrow Table."""
    return pq.read_table(path)
