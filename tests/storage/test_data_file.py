"""Tests for storage/data_file.py and storage/delta_file.py"""

import os

import pyarrow as pa
import pytest

from milvus_lite.schema.arrow_builder import build_data_schema, build_delta_schema
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.storage.data_file import (
    get_file_size,
    parse_seq_range,
    read_data_file,
    write_data_file,
)
from milvus_lite.storage.delta_file import read_delta_file, write_delta_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


@pytest.fixture
def data_table(schema):
    return pa.Table.from_pydict(
        {
            "_seq": [1, 2, 3],
            "id": ["a", "b", "c"],
            "vec": [[0.5, 0.25], [0.75, 0.125], [0.0625, 1.5]],
            "title": ["x", "y", None],
        },
        schema=build_data_schema(schema),
    )


@pytest.fixture
def delta_table(schema):
    return pa.Table.from_pydict(
        {"id": ["a", "b"], "_seq": [10, 10]},
        schema=build_delta_schema(schema),
    )


@pytest.fixture
def partition_dir(tmp_path):
    d = tmp_path / "partitions" / "_default"
    d.mkdir(parents=True)
    return str(d)


# ---------------------------------------------------------------------------
# data_file: write + read round-trip
# ---------------------------------------------------------------------------

def test_write_data_file_creates_subdir(partition_dir, data_table):
    rel = write_data_file(data_table, partition_dir, seq_min=1, seq_max=3)
    assert rel == "data/data_000001_000003.parquet"
    abs_path = os.path.join(partition_dir, rel)
    assert os.path.exists(abs_path)


def test_data_file_roundtrip(partition_dir, data_table):
    rel = write_data_file(data_table, partition_dir, seq_min=1, seq_max=3)
    abs_path = os.path.join(partition_dir, rel)

    loaded = read_data_file(abs_path)
    assert loaded.num_rows == 3
    assert loaded.column("id").to_pylist() == ["a", "b", "c"]
    assert loaded.column("_seq").to_pylist() == [1, 2, 3]


def test_data_file_schema_preserved(partition_dir, data_table, schema):
    rel = write_data_file(data_table, partition_dir, seq_min=1, seq_max=3)
    loaded = read_data_file(os.path.join(partition_dir, rel))
    assert loaded.schema == build_data_schema(schema)


def test_data_file_naming_zero_padded(partition_dir, data_table):
    rel = write_data_file(data_table, partition_dir, seq_min=1, seq_max=10)
    assert "data_000001_000010.parquet" in rel


def test_data_file_seq_range_large_numbers(partition_dir, data_table):
    rel = write_data_file(data_table, partition_dir, seq_min=999999, seq_max=1234567)
    assert "data_999999_1234567.parquet" in rel


# ---------------------------------------------------------------------------
# delta_file: write + read round-trip
# ---------------------------------------------------------------------------

def test_delta_file_roundtrip(partition_dir, delta_table):
    rel = write_delta_file(delta_table, partition_dir, seq_min=10, seq_max=10)
    assert rel == "delta/delta_000010_000010.parquet"
    abs_path = os.path.join(partition_dir, rel)
    assert os.path.exists(abs_path)

    loaded = read_delta_file(abs_path)
    assert loaded.num_rows == 2
    assert loaded.column("id").to_pylist() == ["a", "b"]
    assert loaded.column("_seq").to_pylist() == [10, 10]


def test_delta_and_data_in_separate_subdirs(partition_dir, data_table, delta_table):
    write_data_file(data_table, partition_dir, 1, 3)
    write_delta_file(delta_table, partition_dir, 10, 10)
    assert os.path.exists(os.path.join(partition_dir, "data"))
    assert os.path.exists(os.path.join(partition_dir, "delta"))


# ---------------------------------------------------------------------------
# parse_seq_range
# ---------------------------------------------------------------------------

def test_parse_seq_range_data():
    assert parse_seq_range("data_000001_000500.parquet") == (1, 500)


def test_parse_seq_range_delta():
    assert parse_seq_range("delta_000501_000503.parquet") == (501, 503)


def test_parse_seq_range_full_path():
    assert parse_seq_range("/some/abs/path/data/data_000001_000500.parquet") == (1, 500)


def test_parse_seq_range_invalid():
    with pytest.raises(ValueError, match="not a data/delta filename"):
        parse_seq_range("random.txt")
    with pytest.raises(ValueError):
        parse_seq_range("data_abc_def.parquet")


# ---------------------------------------------------------------------------
# get_file_size
# ---------------------------------------------------------------------------

def test_get_file_size(partition_dir, data_table):
    rel = write_data_file(data_table, partition_dir, 1, 3)
    abs_path = os.path.join(partition_dir, rel)
    size = get_file_size(abs_path)
    assert size > 0
