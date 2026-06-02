import json
import os

import pytest

from milvus_lite.exceptions import CollectionAlreadyExistsError, CollectionNotFoundError
from milvus_lite.storage.manifest import Manifest

from milvus_lite.db import MilvusLite
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


def _record(i: int) -> dict:
    return {
        "id": f"doc_{i:04d}",
        "vec": [float(i), 0.0, 0.0, 0.0],
        "title": f"title-{i}",
    }


def test_create_list_drop_snapshot(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1), _record(2)])

        snap = db.create_snapshot("docs", "snap_a", description="before mutation")

        assert snap["name"] == "snap_a"
        assert snap["description"] == "before mutation"
        assert snap["collection_name"] == "docs"
        assert snap["created_seq"] >= 2
        assert snap["data_files"]
        assert db.list_snapshots("docs")[0]["name"] == "snap_a"

        db.drop_snapshot("docs", "snap_a")
        assert db.list_snapshots("docs") == []
    finally:
        db.close()


def test_create_duplicate_snapshot_name_fails(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        db.create_snapshot("docs", "snap_a")

        with pytest.raises(FileExistsError):
            db.create_snapshot("docs", "snap_a")
    finally:
        db.close()


def test_snapshot_empty_collection_can_restore(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        db.create_collection("docs", schema)
        snap = db.create_snapshot("docs", "snap_empty")

        assert snap["data_files"] == {}
        restored = db.restore_snapshot("docs", "snap_empty", "docs_restored")
        assert restored.query() == []
    finally:
        db.close()


def test_same_snapshot_name_allowed_across_collections(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col_a = db.create_collection("docs_a", schema)
        col_b = db.create_collection("docs_b", schema)
        col_a.insert([_record(1)])
        col_b.insert([_record(2)])

        snap_a = db.create_snapshot("docs_a", "snap_same")
        snap_b = db.create_snapshot("docs_b", "snap_same")

        assert snap_a["collection_name"] == "docs_a"
        assert snap_b["collection_name"] == "docs_b"
        assert db.restore_snapshot("docs_a", "snap_same", "docs_a_restored").query()[0]["id"] == "doc_0001"
        assert db.restore_snapshot("docs_b", "snap_same", "docs_b_restored").query()[0]["id"] == "doc_0002"
    finally:
        db.close()


def test_restore_snapshot_to_new_collection(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1), _record(2)])
        db.create_snapshot("docs", "snap_a")

        col.insert([_record(3)])
        restored = db.restore_snapshot("docs", "snap_a", "docs_restored")

        assert db.has_collection("docs_restored")
        assert sorted(r["id"] for r in restored.query()) == ["doc_0001", "doc_0002"]
        assert sorted(r["id"] for r in col.query()) == [
            "doc_0001", "doc_0002", "doc_0003",
        ]
    finally:
        db.close()


def test_restore_snapshot_rejects_missing_snapshot(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        db.create_collection("docs", schema)

        with pytest.raises(FileNotFoundError):
            db.restore_snapshot("docs", "missing", "docs_restored")
    finally:
        db.close()


def test_restore_snapshot_rejects_existing_collection(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        db.create_snapshot("docs", "snap_a")
        db.create_collection("docs_restored", schema)

        with pytest.raises(CollectionAlreadyExistsError):
            db.restore_snapshot("docs", "snap_a", "docs_restored")
    finally:
        db.close()


def test_restore_deleted_snapshot_fails(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        db.create_snapshot("docs", "snap_a")
        db.drop_snapshot("docs", "snap_a")

        with pytest.raises(FileNotFoundError):
            db.restore_snapshot("docs", "snap_a", "docs_restored")
    finally:
        db.close()


def test_create_snapshot_rejects_missing_collection(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        with pytest.raises(CollectionNotFoundError):
            db.create_snapshot("missing", "snap_a")
    finally:
        db.close()


def test_multiple_snapshots_restore_different_time_points(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        db.create_snapshot("docs", "snap_one")
        col.insert([_record(2)])
        db.create_snapshot("docs", "snap_two")
        col.insert([_record(3)])

        restored_one = db.restore_snapshot("docs", "snap_one", "docs_one")
        restored_two = db.restore_snapshot("docs", "snap_two", "docs_two")

        assert sorted(r["id"] for r in restored_one.query()) == ["doc_0001"]
        assert sorted(r["id"] for r in restored_two.query()) == ["doc_0001", "doc_0002"]
        assert sorted(r["id"] for r in col.query()) == ["doc_0001", "doc_0002", "doc_0003"]
    finally:
        db.close()


def test_snapshot_preserves_delete_state(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1), _record(2), _record(3)])
        col.delete(["doc_0002"])
        db.create_snapshot("docs", "snap_after_delete")

        restored = db.restore_snapshot("docs", "snap_after_delete", "docs_restored")

        assert sorted(r["id"] for r in restored.query()) == ["doc_0001", "doc_0003"]
    finally:
        db.close()


def test_restored_snapshot_preserves_scalar_and_vector_data(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1), _record(2), _record(3)])
        db.create_snapshot("docs", "snap_a")

        restored = db.restore_snapshot("docs", "snap_a", "docs_restored")
        rows = sorted(restored.query(), key=lambda r: r["id"])
        hits = restored.search([[2.0, 0.0, 0.0, 0.0]], top_k=1, metric_type="L2")

        assert [r["title"] for r in rows] == ["title-1", "title-2", "title-3"]
        assert rows[1]["vec"] == [2.0, 0.0, 0.0, 0.0]
        assert hits[0][0]["id"] == "doc_0002"
    finally:
        db.close()


def test_snapshot_restore_preserves_dropped_partition_data(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.create_partition("archive")
        col.insert([_record(1)])
        col.insert([_record(2)], partition_name="archive")
        snap = db.create_snapshot("docs", "snap_with_partition")
        pinned = snap["data_files"]["archive"][0]
        pinned_path = tmp_path / "db" / "collections" / "docs" / "partitions" / "archive" / pinned

        col.drop_partition("archive")

        assert os.path.exists(pinned_path)
        assert "archive" not in col.list_partitions()
        restored = db.restore_snapshot("docs", "snap_with_partition", "docs_restored")
        assert "archive" in restored.list_partitions()
        assert sorted(r["id"] for r in restored.query()) == ["doc_0001", "doc_0002"]
    finally:
        db.close()


def test_snapshot_pins_files_after_compaction(tmp_path, monkeypatch, schema):
    import milvus_lite.engine.compaction as compaction

    monkeypatch.setattr(compaction, "COMPACTION_MIN_FILES_PER_BUCKET", 2)
    monkeypatch.setattr(compaction, "MAX_DATA_FILES", 2)

    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        col.flush()
        snap = db.create_snapshot("docs", "snap_a")
        pinned = snap["data_files"]["_default"][0]
        pinned_path = tmp_path / "db" / "collections" / "docs" / "partitions" / "_default" / pinned

        col.insert([_record(2)])
        col.flush()
        col._wait_for_bg()

        assert os.path.exists(pinned_path)
        restored = db.restore_snapshot("docs", "snap_a", "docs_restored")
        assert [r["id"] for r in restored.query()] == ["doc_0001"]
    finally:
        db.close()


def test_startup_maintenance_compacts_existing_files(tmp_path, monkeypatch, schema):
    import milvus_lite.engine.compaction as compaction

    monkeypatch.setattr(compaction, "COMPACTION_MIN_FILES_PER_BUCKET", 10)
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        col.flush()
        col.insert([_record(2)])
        col.flush()
        col._wait_for_bg()
        assert len(col._manifest.get_data_files("_default")) == 2
    finally:
        db.close()

    monkeypatch.setattr(compaction, "COMPACTION_MIN_FILES_PER_BUCKET", 2)
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.get_collection("docs")
        col._wait_for_bg()
        assert len(col._manifest.get_data_files("_default")) == 1
    finally:
        db.close()


def test_snapshot_pins_index_files_after_drop_index(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1), _record(2)])
        col.flush()
        col.create_index("vec", {
            "index_type": "BRUTE_FORCE",
            "metric_type": "COSINE",
            "params": {},
        })
        snap = db.create_snapshot("docs", "snap_a")
        pinned = snap["index_files"]["_default"][0]
        pinned_path = tmp_path / "db" / "collections" / "docs" / "partitions" / "_default" / pinned

        col.release()
        col.drop_index("vec")

        assert os.path.exists(pinned_path)
        restored = db.restore_snapshot("docs", "snap_a", "docs_restored")
        restored.load()
        hits = restored.search([[1.0, 0.0, 0.0, 0.0]], top_k=1)
        assert hits[0][0]["id"] == "doc_0001"
    finally:
        db.close()


def test_restored_snapshot_starts_with_clean_wal_state(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        db.create_snapshot("docs", "snap_a")

        restored = db.restore_snapshot("docs", "snap_a", "docs_restored")
        restored.insert([_record(2)])
        restored.flush()

        restored_dir = tmp_path / "db" / "collections" / "docs_restored"
        manifest = Manifest.load(str(restored_dir))
        assert manifest.active_wal_number == 2
        assert sorted(r["id"] for r in restored.query()) == ["doc_0001", "doc_0002"]
    finally:
        db.close()


def test_create_snapshot_drains_background_before_collecting_indexes(
    tmp_path, monkeypatch, schema
):
    import milvus_lite.db as db_module

    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        drained = {"value": False}
        original_collect = db_module.collect_index_files

        def fake_wait_for_bg(*_args, **_kwargs):
            drained["value"] = True

        def fake_collect_index_files(collection_dir, data_files):
            assert drained["value"]
            return original_collect(collection_dir, data_files)

        monkeypatch.setattr(col, "_wait_for_bg", fake_wait_for_bg)
        monkeypatch.setattr(db_module, "collect_index_files", fake_collect_index_files)

        db.create_snapshot("docs", "snap_a")
    finally:
        db.close()


def test_failed_snapshot_creation_cleans_partial_manifest_dir(tmp_path, monkeypatch, schema):
    import milvus_lite.storage.snapshots as snapshots

    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        original_copy2 = snapshots.shutil.copy2

        def fail_on_manifest(src, dst):
            if str(src).endswith("manifest.json"):
                raise OSError("copy failed")
            return original_copy2(src, dst)

        monkeypatch.setattr(snapshots.shutil, "copy2", fail_on_manifest)
        with pytest.raises(OSError):
            db.create_snapshot("docs", "snap_a")

        snap_dir = tmp_path / "db" / "collections" / "docs" / "snapshots" / "manifests" / "snap_a"
        meta_path = tmp_path / "db" / "collections" / "docs" / "snapshots" / "metadata" / "snap_a.json"
        assert not snap_dir.exists()
        assert not meta_path.exists()
    finally:
        db.close()


def test_create_snapshot_removes_stale_manifest_dir_without_metadata(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        snap_dir = tmp_path / "db" / "collections" / "docs" / "snapshots" / "manifests" / "snap_a"
        snap_dir.mkdir(parents=True)
        (snap_dir / "stale.txt").write_text("stale", encoding="utf-8")

        snap = db.create_snapshot("docs", "snap_a")

        assert snap["name"] == "snap_a"
        assert not (snap_dir / "stale.txt").exists()
        assert (snap_dir / "schema.json").exists()
        assert (snap_dir / "manifest.json").exists()
    finally:
        db.close()


def test_restore_snapshot_rejects_unsafe_metadata_paths(tmp_path, schema):
    db = MilvusLite(str(tmp_path / "db"))
    try:
        col = db.create_collection("docs", schema)
        col.insert([_record(1)])
        snap = db.create_snapshot("docs", "snap_a")
        meta_path = tmp_path / "db" / "collections" / "docs" / "snapshots" / "metadata" / "snap_a.json"
        snap["data_files"] = {"_default": ["../outside.parquet"]}
        meta_path.write_text(json.dumps(snap), encoding="utf-8")

        with pytest.raises(ValueError, match="snapshot path"):
            db.restore_snapshot("docs", "snap_a", "docs_restored")
        assert not db.has_collection("docs_restored")
    finally:
        db.close()
