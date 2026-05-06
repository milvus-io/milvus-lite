"""Tests for storage/segment.py"""

import os

import numpy as np
import pyarrow as pa
import pytest

from milvus_lite.schema.arrow_builder import build_data_schema
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.storage.data_file import write_data_file
from milvus_lite.storage.segment import Segment


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=3),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


@pytest.fixture
def parquet_file(tmp_path, schema):
    table = pa.Table.from_pydict(
        {
            "_seq": [1, 2, 3, 4],
            "id": ["a", "b", "c", "d"],
            "vec": [
                [0.5, 0.25, 0.125],
                [0.75, 0.0, 0.5],
                [1.0, 1.0, 0.0],
                [0.0625, 1.5, 0.5],
            ],
            "title": ["t1", "t2", None, "t4"],
        },
        schema=build_data_schema(schema),
    )
    partition_dir = tmp_path / "partitions" / "_default"
    partition_dir.mkdir(parents=True)
    rel = write_data_file(table, str(partition_dir), seq_min=1, seq_max=4)
    return str(partition_dir / rel)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def test_load_basic(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    assert seg.num_rows == 4
    assert seg.partition == "_default"
    assert seg.file_path == parquet_file
    assert seg.vector_dim == 3


def test_load_pks_preserved(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    assert seg.pks == ["a", "b", "c", "d"]


def test_load_seqs_numpy_uint64(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    assert isinstance(seg.seqs, np.ndarray)
    assert seg.seqs.dtype == np.uint64
    assert list(seg.seqs) == [1, 2, 3, 4]


def test_load_vectors_shape_and_dtype(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    assert seg.vectors.shape == (4, 3)
    assert seg.vectors.dtype == np.float32


def test_load_vectors_values(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    np.testing.assert_array_equal(
        seg.vectors,
        np.array([
            [0.5, 0.25, 0.125],
            [0.75, 0.0, 0.5],
            [1.0, 1.0, 0.0],
            [0.0625, 1.5, 0.5],
        ], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# pk_to_row + find_row
# ---------------------------------------------------------------------------

def test_pk_to_row_index(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    assert seg.pk_to_row == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_find_row_hit(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    assert seg.find_row("c") == 2


def test_find_row_miss(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    assert seg.find_row("ghost") is None


# ---------------------------------------------------------------------------
# row_to_dict
# ---------------------------------------------------------------------------

def test_row_to_dict_strips_seq(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    rec = seg.row_to_dict(0)
    assert "_seq" not in rec
    assert rec["id"] == "a"
    assert rec["title"] == "t1"
    assert rec["vec"] == [0.5, 0.25, 0.125]


def test_row_to_dict_handles_null_field(parquet_file):
    seg = Segment.load(parquet_file, "_default", "id", "vec")
    rec = seg.row_to_dict(2)
    assert rec["id"] == "c"
    assert rec["title"] is None


# ---------------------------------------------------------------------------
# Empty file
# ---------------------------------------------------------------------------

def test_load_empty_file(tmp_path, schema):
    table = pa.Table.from_pydict(
        {"_seq": [], "id": [], "vec": [], "title": []},
        schema=build_data_schema(schema),
    )
    partition_dir = tmp_path / "partitions" / "_default"
    partition_dir.mkdir(parents=True)
    rel = write_data_file(table, str(partition_dir), seq_min=0, seq_max=0)
    seg = Segment.load(str(partition_dir / rel), "_default", "id", "vec")
    assert seg.num_rows == 0
    assert seg.pks == []
    assert seg.seqs.shape == (0,)
