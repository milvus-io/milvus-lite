import numpy as np
import pyarrow as pa
import pytest

from milvus_lite.index.scalar import ScalarInvertedIndex, ScalarPredicate, plan_indexed_filter
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.search.filter import compile_filter


def _table():
    return pa.Table.from_pydict({
        "age": [18, 25, 30, 25, None],
        "category": ["tech", "news", "tech", None, "blog"],
        "active": [True, False, True, None, False],
    })


def test_int_equality_and_range():
    idx = ScalarInvertedIndex.build(_table(), "age", DataType.INT64)

    assert np.flatnonzero(idx.match(ScalarPredicate("==", "age", value=25))).tolist() == [1, 3]
    assert np.flatnonzero(idx.match(ScalarPredicate(">", "age", value=24))).tolist() == [1, 2, 3]
    assert np.flatnonzero(idx.match(ScalarPredicate("<=", "age", value=25))).tolist() == [0, 1, 3]


def test_in_and_null_semantics():
    idx = ScalarInvertedIndex.build(_table(), "category", DataType.VARCHAR)

    mask = idx.match(ScalarPredicate("in", "category", values=("tech", "blog")))
    assert np.flatnonzero(mask).tolist() == [0, 2, 4]

    mask = idx.match(ScalarPredicate("!=", "category", value="tech"))
    assert np.flatnonzero(mask).tolist() == [1, 4]

    mask = idx.match(ScalarPredicate("not in", "category", values=("tech",)))
    assert np.flatnonzero(mask).tolist() == [1, 3, 4]

    mask = idx.match(ScalarPredicate("is null", "category"))
    assert np.flatnonzero(mask).tolist() == [3]


def test_save_load_round_trip(tmp_path):
    idx = ScalarInvertedIndex.build(_table(), "age", DataType.INT64)
    path = tmp_path / "age.sidx"
    idx.save(str(path))

    loaded = ScalarInvertedIndex.load(str(path))
    mask = loaded.match(ScalarPredicate("in", "age", values=(18, 30)))
    assert np.flatnonzero(mask).tolist() == [0, 2]


def test_load_rejects_corrupt_file(tmp_path):
    path = tmp_path / "age.sidx"
    path.write_bytes(b"not an arrow ipc file")

    with pytest.raises(Exception):
        ScalarInvertedIndex.load(str(path))


def test_load_rejects_unsupported_version(tmp_path):
    idx = ScalarInvertedIndex.build(_table(), "age", DataType.INT64)
    path = tmp_path / "age.sidx"
    idx.save(str(path))

    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.RecordBatchFileReader(source).read_all()
    metadata = dict(table.schema.metadata or {})
    metadata[b"milvus_lite.scalar_index.version"] = b"999"
    table = table.replace_schema_metadata(metadata)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    with pytest.raises(ValueError, match="unsupported scalar index file version"):
        ScalarInvertedIndex.load(str(path))


def test_planner_does_not_plan_generic_not():
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="age", dtype=DataType.INT64, nullable=True),
    ])
    compiled = compile_filter("not (age == 25)", schema)

    assert plan_indexed_filter(compiled, {"age"}) is None


def test_planner_falls_back_for_dynamic_field_in_and_null():
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="age", dtype=DataType.INT64, nullable=True),
        ],
        enable_dynamic_field=True,
    )

    assert plan_indexed_filter(compile_filter("color in ['red']", schema), {"age"}) is None
    assert plan_indexed_filter(compile_filter("color is null", schema), {"age"}) is None
