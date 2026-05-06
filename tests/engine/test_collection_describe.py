"""Phase 9.1.2 — Collection.num_entities + Collection.describe tests.

num_entities is the live row count after dedup + tombstone filter.
describe() is the dict shape pymilvus's describe_collection consumes.
"""

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=128, nullable=True),
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ])


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    yield c
    c.close()


def _vec(i: float):
    return [i, i + 1, i + 2, i + 3]


# ---------------------------------------------------------------------------
# num_entities
# ---------------------------------------------------------------------------

def test_num_entities_empty(col):
    assert col.num_entities == 0


def test_num_entities_memtable_only(col):
    col.insert([{"id": i, "vec": _vec(i), "score": 0.1} for i in range(5)])
    assert col.num_entities == 5


def test_num_entities_after_flush(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    c.insert([{"id": i, "vec": _vec(i), "score": 0.1} for i in range(10)])
    c.flush()
    assert c.num_entities == 10
    c.close()


def test_num_entities_mixed_memtable_and_segment(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    c.insert([{"id": i, "vec": _vec(i), "score": 0.1} for i in range(5)])
    c.flush()
    c.insert([{"id": i, "vec": _vec(i), "score": 0.1} for i in range(5, 8)])
    assert c.num_entities == 8
    c.close()


def test_num_entities_dedup_upsert(tmp_path, schema):
    """Re-inserting the same pk should not double-count."""
    c = Collection("test", str(tmp_path / "data"), schema)
    c.insert([{"id": 1, "vec": _vec(1), "score": 0.1}])
    c.flush()
    # Upsert id=1 with a higher seq
    c.insert([{"id": 1, "vec": _vec(2), "score": 0.5}])
    assert c.num_entities == 1
    c.close()


def test_num_entities_excludes_deleted(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    c.insert([{"id": i, "vec": _vec(i), "score": 0.1} for i in range(5)])
    c.flush()
    c.delete(pks=[1, 3])
    c.flush()
    assert c.num_entities == 3
    c.close()


def test_num_entities_after_delete_in_memtable(col):
    col.insert([{"id": i, "vec": _vec(i), "score": 0.1} for i in range(5)])
    col.delete(pks=[2])
    assert col.num_entities == 4


def test_num_entities_persists_across_restart(tmp_path, schema):
    c = Collection("test", str(tmp_path / "data"), schema)
    c.insert([{"id": i, "vec": _vec(i), "score": 0.1} for i in range(7)])
    c.flush()
    c.close()

    c2 = Collection("test", str(tmp_path / "data"), schema)
    try:
        assert c2.num_entities == 7
    finally:
        c2.close()


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_basic(col):
    d = col.describe()
    assert d["name"] == "test"
    assert d["partitions"] == ["_default"]
    assert d["num_entities"] == 0
    assert d["schema"]["enable_dynamic_field"] is False
    field_names = [f["name"] for f in d["schema"]["fields"]]
    assert field_names == ["id", "vec", "title", "score"]


def test_describe_field_metadata(col):
    d = col.describe()
    by_name = {f["name"]: f for f in d["schema"]["fields"]}
    assert by_name["id"]["dtype"] == "INT64"
    assert by_name["id"]["is_primary"] is True
    assert by_name["vec"]["dtype"] == "FLOAT_VECTOR"
    assert by_name["vec"]["dim"] == 4
    assert by_name["title"]["nullable"] is True
    assert by_name["title"]["max_length"] == 128


def test_describe_with_dynamic_field(tmp_path):
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ],
        enable_dynamic_field=True,
    )
    c = Collection("dyn", str(tmp_path / "data"), schema)
    try:
        d = c.describe()
        assert d["schema"]["enable_dynamic_field"] is True
    finally:
        c.close()


def test_describe_after_data_and_partitions(col):
    col.create_partition("p1")
    col.insert([{"id": i, "vec": _vec(i), "score": 0.1} for i in range(3)])
    d = col.describe()
    assert sorted(d["partitions"]) == ["_default", "p1"]
    assert d["num_entities"] == 3


def test_name_and_schema_properties(col):
    assert col.name == "test"
    assert col.schema is col._schema  # noqa: SLF001 — testing the public alias
