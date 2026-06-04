import os

import pyarrow as pa
import pytest

from milvus_lite import Collection, CollectionSchema, DataType, FieldSchema
from milvus_lite.exceptions import SchemaValidationError


def _schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="age", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="category", dtype=DataType.VARCHAR, nullable=True, max_length=64),
        FieldSchema(name="active", dtype=DataType.BOOL, nullable=True),
    ])


def _rows():
    return [
        {"id": 1, "vec": [1, 0, 0, 0], "age": 18, "category": "tech", "active": True},
        {"id": 2, "vec": [0, 1, 0, 0], "age": 25, "category": "news", "active": False},
        {"id": 3, "vec": [0, 0, 1, 0], "age": 30, "category": "tech", "active": True},
        {"id": 4, "vec": [0, 0, 0, 1], "age": 50, "category": None, "active": None},
        {"id": 5, "vec": [1, 1, 0, 0], "age": None, "category": "blog", "active": False},
    ]


def _ids(rows):
    return [r["id"] for r in rows]


def _index_files(data_dir, partition="_default"):
    index_dir = os.path.join(data_dir, "partitions", partition, "indexes")
    if not os.path.isdir(index_dir):
        return []
    return sorted(os.listdir(index_dir))


def test_scalar_index_query_matches_scan(tmp_path):
    scan = Collection("scan", str(tmp_path / "scan"), _schema())
    indexed = Collection("indexed", str(tmp_path / "indexed"), _schema())
    try:
        scan.insert(_rows())
        scan.flush()
        indexed.insert(_rows())
        indexed.flush()
        indexed.create_index("age", {"index_type": "INVERTED"})
        indexed.load()

        exprs = [
            "age == 25",
            "age != 25",
            "age in [18, 30]",
            "age not in [18, 30]",
            "age >= 25 and age < 50",
            "age is null",
            "age is not null",
            "not (age == 25)",
        ]
        for expr in exprs:
            assert _ids(indexed.query(expr, output_fields=["id"])) == _ids(
                scan.query(expr, output_fields=["id"])
            )
    finally:
        scan.close()
        indexed.close()


def test_scalar_index_search_matches_scan(tmp_path):
    scan = Collection("scan", str(tmp_path / "scan"), _schema())
    indexed = Collection("indexed", str(tmp_path / "indexed"), _schema())
    try:
        scan.insert(_rows())
        scan.flush()
        indexed.insert(_rows())
        indexed.flush()
        indexed.create_index("category", {"index_type": "INVERTED"})
        indexed.load()

        query = [[1, 0, 0, 0]]
        exprs = [
            "category == 'tech'",
            "category in ['tech', 'blog']",
            "category != 'tech'",
            "category is null",
        ]
        for expr in exprs:
            actual = indexed.search(query, top_k=10, metric_type="L2", expr=expr)
            expected = scan.search(query, top_k=10, metric_type="L2", expr=expr)
            assert [h["id"] for h in actual[0]] == [h["id"] for h in expected[0]]
    finally:
        scan.close()
        indexed.close()


def test_scalar_index_persistence_and_sidecar(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    try:
        col.insert(_rows())
        col.flush()
        col.create_index("age", {"index_type": "INVERTED"})
        col.load()
        files = os.listdir(data_dir / "partitions" / "_default" / "indexes")
        assert any(f.endswith(".age.inverted.sidx") for f in files)
    finally:
        col.close()

    reopened = Collection("c", str(data_dir), _schema())
    try:
        reopened.load()
        assert _ids(reopened.query("age >= 25", output_fields=["id"])) == [2, 3, 4]
    finally:
        reopened.close()


def test_scalar_index_compaction_replaces_sidx_files(tmp_path, monkeypatch):
    monkeypatch.setattr("milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 2)

    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    try:
        col.create_index("age", {"index_type": "INVERTED"})
        col.load()

        for batch in range(3):
            rows = []
            for row in _rows():
                item = dict(row)
                item["id"] = batch * 100 + item["id"]
                rows.append(item)
            col.insert(rows)
            col.flush()
        col._wait_for_bg()

        data_files = sorted(os.listdir(data_dir / "partitions" / "_default" / "data"))
        sidx_files = [f for f in _index_files(str(data_dir)) if f.endswith(".age.inverted.sidx")]
        data_stems = {os.path.splitext(f)[0] for f in data_files}
        sidx_stems = {f[: -len(".age.inverted.sidx")] for f in sidx_files}
        assert data_stems == sidx_stems
        assert len(sidx_files) < 3
    finally:
        col.close()


def test_scalar_index_query_and_search_match_scan_after_compaction(tmp_path, monkeypatch):
    monkeypatch.setattr("milvus_lite.engine.compaction.COMPACTION_MIN_FILES_PER_BUCKET", 2)

    scan = Collection("scan", str(tmp_path / "scan"), _schema())
    indexed = Collection("indexed", str(tmp_path / "indexed"), _schema())
    try:
        for batch in range(3):
            rows = []
            for row in _rows():
                item = dict(row)
                item["id"] = batch * 100 + item["id"]
                rows.append(item)
            scan.insert(rows)
            scan.flush()
            indexed.insert(rows)
            indexed.flush()

        indexed.create_index("age", {"index_type": "INVERTED"})
        indexed.load()
        scan._wait_for_bg()
        indexed._wait_for_bg()

        exprs = [
            "age == 25",
            "age != 25",
            "age in [18, 30]",
            "age not in [18, 30]",
            "age >= 25 and age < 50",
            "age is null",
            "age is not null",
        ]
        for expr in exprs:
            assert _ids(indexed.query(expr, output_fields=["id"])) == _ids(
                scan.query(expr, output_fields=["id"])
            )

            actual = indexed.search([[1, 0, 0, 0]], top_k=20, metric_type="L2", expr=expr)
            expected = scan.search([[1, 0, 0, 0]], top_k=20, metric_type="L2", expr=expr)
            assert [h["id"] for h in actual[0]] == [h["id"] for h in expected[0]]
    finally:
        scan.close()
        indexed.close()


def test_unimplemented_milvus_scalar_index_type_is_rejected(tmp_path):
    col = Collection("c", str(tmp_path / "data"), _schema())
    try:
        with pytest.raises(SchemaValidationError, match="not implemented"):
            col.create_index("age", {"index_type": "BITMAP"})
    finally:
        col.close()


def test_scalar_index_params_do_not_expose_internal_dtype(tmp_path):
    col = Collection("c", str(tmp_path / "data"), _schema())
    try:
        col.create_index("age", {"index_type": "INVERTED"})
        info = col.get_index_info("age")
        assert info is not None
        assert info["build_params"] == {}
    finally:
        col.close()


_ALL_SCALAR_FIELDS = [
    "int8_value",
    "int16_value",
    "int32_value",
    "int64_value",
    "float_value",
    "double_value",
    "string_value",
    "bool_value",
]


def _all_scalar_schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="int8_value", dtype=DataType.INT8, nullable=True),
        FieldSchema(name="int16_value", dtype=DataType.INT16, nullable=True),
        FieldSchema(name="int32_value", dtype=DataType.INT32, nullable=True),
        FieldSchema(name="int64_value", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="float_value", dtype=DataType.FLOAT, nullable=True),
        FieldSchema(name="double_value", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="string_value", dtype=DataType.VARCHAR, nullable=True, max_length=64),
        FieldSchema(name="bool_value", dtype=DataType.BOOL, nullable=True),
    ])


def _all_scalar_rows(count=12):
    rows = []
    for i in range(count):
        is_null = i % 5 == 0
        rows.append({
            "id": i,
            "vec": [float(i == 0), float(i == 1), float(i == 2), float(i == 3)],
            "int8_value": None if is_null else i % 7,
            "int16_value": None if is_null else i * 2,
            "int32_value": None if is_null else i * 3,
            "int64_value": None if is_null else i * 10,
            "float_value": None if is_null else float(i % 4) + 0.5,
            "double_value": None if is_null else float(i % 6) + 0.25,
            "string_value": None if is_null else f"str_{i % 4}",
            "bool_value": None if is_null else i % 2 == 0,
        })
    return rows


def test_create_inverted_index_on_all_supported_scalar_fields(tmp_path):
    col = Collection("c", str(tmp_path / "data"), _all_scalar_schema())
    try:
        col.insert(_all_scalar_rows())
        col.flush()
        for field_name in _ALL_SCALAR_FIELDS:
            col.create_index(field_name, {"index_type": "INVERTED"})
            assert col.has_index(field_name)
            info = col.get_index_info(field_name)
            assert info is not None
            assert info["index_type"] == "INVERTED"
            assert info["metric_type"] == "NONE"
        assert col.list_indexes() == sorted(_ALL_SCALAR_FIELDS)
    finally:
        col.close()


def test_create_multiple_inverted_indexes_builds_sidecars(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _all_scalar_schema())
    indexed_fields = ["int8_value", "int32_value", "string_value"]
    try:
        col.insert(_all_scalar_rows())
        col.flush()
        for field_name in indexed_fields:
            col.create_index(field_name, {"index_type": "INVERTED"})
        col.load()

        files = _index_files(str(data_dir))
        for field_name in indexed_fields:
            assert any(f.endswith(f".{field_name}.inverted.sidx") for f in files)
    finally:
        col.close()


def test_create_string_index_before_load(tmp_path):
    col = Collection("c", str(tmp_path / "data"), _all_scalar_schema())
    try:
        col.insert(_all_scalar_rows())
        col.flush()
        col.release()
        col.create_index("string_value", {"index_type": "INVERTED"})
        col.create_index("vec", {"index_type": "FLAT", "metric_type": "L2"})
        col.load()

        assert _ids(col.query("string_value == 'str_1'", output_fields=["id"])) == [1, 9]
    finally:
        col.close()


def test_load_after_create_string_index(tmp_path):
    col = Collection("c", str(tmp_path / "data"), _all_scalar_schema())
    try:
        col.insert(_all_scalar_rows())
        col.flush()
        col.release()
        col.create_index("vec", {"index_type": "FLAT", "metric_type": "L2"})
        col.create_index("string_value", {"index_type": "INVERTED"})
        col.load()

        actual = col.search(
            [[1, 0, 0, 0]],
            top_k=10,
            metric_type="L2",
            expr="string_value in ['str_0', 'str_2']",
            output_fields=["id", "string_value"],
        )
        assert {hit["id"] for hit in actual[0]} == {2, 4, 6, 8}
    finally:
        col.close()


def test_scalar_index_consistency_l1_core_types(tmp_path):
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="int64_no_index", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="int64_inverted", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="varchar_no_index", dtype=DataType.VARCHAR, nullable=True, max_length=64),
        FieldSchema(name="varchar_inverted", dtype=DataType.VARCHAR, nullable=True, max_length=64),
        FieldSchema(name="float_no_index", dtype=DataType.FLOAT, nullable=True),
        FieldSchema(name="float_inverted", dtype=DataType.FLOAT, nullable=True),
        FieldSchema(name="bool_no_index", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="bool_inverted", dtype=DataType.BOOL, nullable=True),
    ])
    rows = []
    for i in range(30):
        is_null = i % 7 == 0
        int_value = None if is_null else i % 9
        varchar_value = None if is_null else f"str_{i % 5}"
        float_value = None if is_null else float(i % 6) + 0.5
        bool_value = None if is_null else i % 2 == 0
        rows.append({
            "id": i,
            "vec": [float(i == 0), float(i == 1), float(i == 2), float(i == 3)],
            "int64_no_index": int_value,
            "int64_inverted": int_value,
            "varchar_no_index": varchar_value,
            "varchar_inverted": varchar_value,
            "float_no_index": float_value,
            "float_inverted": float_value,
            "bool_no_index": bool_value,
            "bool_inverted": bool_value,
        })

    col = Collection("c", str(tmp_path / "data"), schema)
    try:
        col.insert(rows)
        col.flush()
        for field_name in ["int64_inverted", "varchar_inverted", "float_inverted", "bool_inverted"]:
            col.create_index(field_name, {"index_type": "INVERTED"})
        col.load()

        cases = [
            ("int64_no_index", "int64_inverted", ["{f} > 3", "{f} <= 4", "{f} == 5", "{f} in [1, 3, 5]", "{f} is null", "{f} is not null"]),
            ("varchar_no_index", "varchar_inverted", ["{f} == 'str_2'", "{f} != 'str_2'", "{f} in ['str_1', 'str_3']", "{f} not in ['str_4']", "{f} is null", "{f} is not null"]),
            ("float_no_index", "float_inverted", ["{f} > 2.5", "{f} <= 3.5", "{f} == 4.5", "{f} in [1.5, 3.5]", "{f} is null", "{f} is not null"]),
            ("bool_no_index", "bool_inverted", ["{f} == true", "{f} == false", "{f} is null", "{f} is not null"]),
        ]
        for scan_field, indexed_field, templates in cases:
            for template in templates:
                expected = sorted(_ids(col.query(template.format(f=scan_field), output_fields=["id"])))
                actual = sorted(_ids(col.query(template.format(f=indexed_field), output_fields=["id"])))
                assert actual == expected
    finally:
        col.close()


def test_drop_index_removes_scalar_sidx_files(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    try:
        col.insert(_rows())
        col.flush()
        col.create_index("age", {"index_type": "INVERTED"})
        col.load()
        assert any(f.endswith(".age.inverted.sidx") for f in _index_files(str(data_dir)))

        col.release()
        col.drop_index("age")

        assert not any(f.endswith(".age.inverted.sidx") for f in _index_files(str(data_dir)))
    finally:
        col.close()


def test_recovery_cleans_orphan_scalar_sidx(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    col.insert(_rows())
    col.flush()
    col.create_index("age", {"index_type": "INVERTED"})
    col.load()
    col.close()

    index_dir = data_dir / "partitions" / "_default" / "indexes"
    orphan = index_dir / "data_999999_999999.age.inverted.sidx"
    orphan.write_bytes(b"orphan")
    assert orphan.exists()

    reopened = Collection("c", str(data_dir), _schema())
    try:
        assert not orphan.exists()
        assert any(f.endswith(".age.inverted.sidx") for f in _index_files(str(data_dir)))
    finally:
        reopened.close()


def test_recovery_keeps_valid_scalar_sidx(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    col.insert(_rows())
    col.flush()
    col.create_index("age", {"index_type": "INVERTED"})
    col.load()
    files_before = set(_index_files(str(data_dir)))
    col.close()

    reopened = Collection("c", str(data_dir), _schema())
    try:
        assert set(_index_files(str(data_dir))) == files_before
        reopened.load()
        assert _ids(reopened.query("age >= 25", output_fields=["id"])) == [2, 3, 4]
    finally:
        reopened.close()


def test_flush_after_load_indexes_new_scalar_segments(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    try:
        col.create_index("age", {"index_type": "INVERTED"})
        col.load()

        col.insert(_rows())
        col.flush()
        col._wait_for_bg()
        files_after_first_flush = set(_index_files(str(data_dir)))
        assert any(f.endswith(".age.inverted.sidx") for f in files_after_first_flush)

        newer = []
        for row in _rows():
            item = dict(row)
            item["id"] += 100
            item["age"] = 60
            newer.append(item)
        col.insert(newer)
        col.flush()
        col._wait_for_bg()

        files_after_second_flush = set(_index_files(str(data_dir)))
        assert files_after_second_flush - files_after_first_flush
        assert _ids(col.query("age == 60", output_fields=["id"])) == [101, 102, 103, 104, 105]
    finally:
        col.close()


def test_load_rejects_scalar_sidx_field_metadata_mismatch(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    try:
        col.insert(_rows())
        col.flush()
        col.create_index("age", {"index_type": "INVERTED"})
        col.load()
        col.release()

        index_dir = data_dir / "partitions" / "_default" / "indexes"
        sidx_path = next(index_dir.glob("*.age.inverted.sidx"))
        with pa.memory_map(str(sidx_path), "r") as source:
            table = pa.ipc.RecordBatchFileReader(source).read_all()
        metadata = dict(table.schema.metadata or {})
        metadata[b"field_name"] = b"category"
        table = table.replace_schema_metadata(metadata)
        with pa.OSFile(str(sidx_path), "wb") as sink:
            with pa.ipc.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        with pytest.raises(ValueError, match="metadata does not match"):
            col.load()
    finally:
        col.close()


def test_load_rejects_scalar_sidx_dtype_metadata_mismatch(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    try:
        col.insert(_rows())
        col.flush()
        col.create_index("age", {"index_type": "INVERTED"})
        col.load()
        col.release()

        index_dir = data_dir / "partitions" / "_default" / "indexes"
        sidx_path = next(index_dir.glob("*.age.inverted.sidx"))
        with pa.memory_map(str(sidx_path), "r") as source:
            table = pa.ipc.RecordBatchFileReader(source).read_all()
        metadata = dict(table.schema.metadata or {})
        metadata[b"dtype"] = b"VARCHAR"
        table = table.replace_schema_metadata(metadata)
        with pa.OSFile(str(sidx_path), "wb") as sink:
            with pa.ipc.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        with pytest.raises(ValueError, match="metadata does not match"):
            col.load()
    finally:
        col.close()


def test_drop_scalar_index_keeps_vector_idx_files(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    try:
        col.insert(_rows())
        col.flush()
        col.create_index("vec", {"index_type": "FLAT", "metric_type": "L2"})
        col.create_index("age", {"index_type": "INVERTED"})
        col.load()
        index_files_before = set(_index_files(str(data_dir)))
        assert any(f.endswith(".vec.flat.idx") for f in index_files_before)
        assert any(f.endswith(".age.inverted.sidx") for f in index_files_before)

        col.release()
        col.drop_index("age")

        index_files_after = set(_index_files(str(data_dir)))
        assert any(f.endswith(".vec.flat.idx") for f in index_files_after)
        assert not any(f.endswith(".age.inverted.sidx") for f in index_files_after)
        col.load()
        hits = col.search([[1, 0, 0, 0]], top_k=1, metric_type="L2")
        assert hits[0][0]["id"] == 1
    finally:
        col.close()


def test_drop_vector_index_keeps_scalar_sidx_files(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    try:
        col.insert(_rows())
        col.flush()
        col.create_index("vec", {"index_type": "FLAT", "metric_type": "L2"})
        col.create_index("age", {"index_type": "INVERTED"})
        col.load()
        index_files_before = set(_index_files(str(data_dir)))
        assert any(f.endswith(".vec.flat.idx") for f in index_files_before)
        assert any(f.endswith(".age.inverted.sidx") for f in index_files_before)

        col.release()
        col.drop_index("vec")

        index_files_after = set(_index_files(str(data_dir)))
        assert not any(f.endswith(".vec.flat.idx") for f in index_files_after)
        assert any(f.endswith(".age.inverted.sidx") for f in index_files_after)
        col.load()
        assert _ids(col.query("age >= 25", output_fields=["id"])) == [2, 3, 4]
    finally:
        col.close()


def test_recovery_cleans_orphan_idx_and_sidx_independently(tmp_path):
    data_dir = tmp_path / "data"
    col = Collection("c", str(data_dir), _schema())
    col.insert(_rows())
    col.flush()
    col.create_index("vec", {"index_type": "FLAT", "metric_type": "L2"})
    col.create_index("age", {"index_type": "INVERTED"})
    col.load()
    valid_files = set(_index_files(str(data_dir)))
    col.close()

    index_dir = data_dir / "partitions" / "_default" / "indexes"
    orphan_idx = index_dir / "data_999999_999999.vec.flat.idx"
    orphan_sidx = index_dir / "data_999999_999999.age.inverted.sidx"
    orphan_idx.write_bytes(b"orphan")
    orphan_sidx.write_bytes(b"orphan")

    reopened = Collection("c", str(data_dir), _schema())
    try:
        assert not orphan_idx.exists()
        assert not orphan_sidx.exists()
        assert set(_index_files(str(data_dir))) == valid_files
        reopened.load()
        assert _ids(reopened.query("age >= 25", output_fields=["id"])) == [2, 3, 4]
        hits = reopened.search([[1, 0, 0, 0]], top_k=1, metric_type="L2")
        assert hits[0][0]["id"] == 1
    finally:
        reopened.close()
