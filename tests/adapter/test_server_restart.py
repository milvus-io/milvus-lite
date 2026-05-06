"""Server restart persistence tests.

Verifies that data survives a full gRPC server stop → restart cycle,
as a user would experience when restarting the process.

Covers:
1. Insert → stop → restart → query returns same data
2. Insert + index + search → stop → restart → load → search returns same results
3. Multiple collections survive restart
4. Delete persists across restart
5. Dynamic fields survive restart
6. MilvusClient .db mode restart
"""

import tempfile
import os

import pytest

pymilvus = pytest.importorskip("pymilvus")
pytest.importorskip("grpc")

from pymilvus import MilvusClient, DataType
from milvus_lite.adapter.grpc.server import start_server_in_thread


def _start(data_dir, **kwargs):
    """Start a gRPC server, return (server, db, uri)."""
    server, db, port = start_server_in_thread(data_dir, **kwargs)
    return server, db, f"http://127.0.0.1:{port}"


def _stop(server, db):
    server.stop(grace=2)
    db.close()


# ---------------------------------------------------------------------------
# 1. Basic data persistence across restart
# ---------------------------------------------------------------------------

def test_insert_survives_restart():
    """Insert data, stop server, restart, query returns same data."""
    with tempfile.TemporaryDirectory() as d:
        # Phase 1: insert data
        server, db, uri = _start(d)
        client = MilvusClient(uri=uri)

        schema = MilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        schema.add_field("text", DataType.VARCHAR, max_length=128)
        client.create_collection("persist_test", schema=schema)
        client.insert("persist_test", [
            {"id": 1, "vec": [1, 0, 0, 0], "text": "hello"},
            {"id": 2, "vec": [0, 1, 0, 0], "text": "world"},
            {"id": 3, "vec": [0, 0, 1, 0], "text": "foo"},
        ])

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT",
                      metric_type="COSINE", params={})
        client.create_index("persist_test", idx)
        client.load_collection("persist_test")

        # Verify before stop
        rows = client.query("persist_test", filter="id >= 1",
                            output_fields=["text"], limit=10)
        assert len(rows) == 3

        _stop(server, db)

        # Phase 2: restart and verify
        server2, db2, uri2 = _start(d)
        client2 = MilvusClient(uri=uri2)

        assert client2.has_collection("persist_test")

        # Index spec persists in manifest — just load
        client2.load_collection("persist_test")

        rows = client2.query("persist_test", filter="id >= 1",
                             output_fields=["text"], limit=10)
        assert len(rows) == 3
        texts = {r["text"] for r in rows}
        assert texts == {"hello", "world", "foo"}

        _stop(server2, db2)


# ---------------------------------------------------------------------------
# 2. Search results consistent across restart
# ---------------------------------------------------------------------------

def test_search_consistent_after_restart():
    """Search results should be identical before and after restart."""
    with tempfile.TemporaryDirectory() as d:
        # Phase 1: build index and search
        server, db, uri = _start(d)
        client = MilvusClient(uri=uri)

        schema = MilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        client.create_collection("search_persist", schema=schema)
        client.insert("search_persist", [
            {"id": i, "vec": [float(i == j) for j in range(4)]}
            for i in range(4)
        ])

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT",
                      metric_type="COSINE", params={})
        client.create_index("search_persist", idx)
        client.load_collection("search_persist")

        results_before = client.search("search_persist",
                                        data=[[1, 0, 0, 0]], limit=4)
        ids_before = [h["id"] for h in results_before[0]]

        _stop(server, db)

        # Phase 2: restart and search again
        server2, db2, uri2 = _start(d)
        client2 = MilvusClient(uri=uri2)

        client2.load_collection("search_persist")

        results_after = client2.search("search_persist",
                                        data=[[1, 0, 0, 0]], limit=4)
        ids_after = [h["id"] for h in results_after[0]]

        assert ids_before == ids_after

        _stop(server2, db2)


# ---------------------------------------------------------------------------
# 3. Multiple collections survive restart
# ---------------------------------------------------------------------------

def test_multiple_collections_persist():
    """All collections should survive restart."""
    with tempfile.TemporaryDirectory() as d:
        server, db, uri = _start(d)
        client = MilvusClient(uri=uri)

        for name in ["col_a", "col_b", "col_c"]:
            schema = MilvusClient.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
            client.create_collection(name, schema=schema)
            client.insert(name, [{"id": 1, "vec": [1, 0, 0, 0]}])

        _stop(server, db)

        server2, db2, uri2 = _start(d)
        client2 = MilvusClient(uri=uri2)

        collections = client2.list_collections()
        assert set(collections) == {"col_a", "col_b", "col_c"}

        _stop(server2, db2)


# ---------------------------------------------------------------------------
# 4. Deletes persist across restart
# ---------------------------------------------------------------------------

def test_delete_persists():
    """Deleted records should stay deleted after restart."""
    with tempfile.TemporaryDirectory() as d:
        server, db, uri = _start(d)
        client = MilvusClient(uri=uri)

        schema = MilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        client.create_collection("del_test", schema=schema)
        client.insert("del_test", [
            {"id": 1, "vec": [1, 0, 0, 0]},
            {"id": 2, "vec": [0, 1, 0, 0]},
            {"id": 3, "vec": [0, 0, 1, 0]},
        ])
        client.delete("del_test", ids=[2])

        _stop(server, db)

        server2, db2, uri2 = _start(d)
        client2 = MilvusClient(uri=uri2)

        client2.load_collection("del_test")

        rows = client2.query("del_test", filter="id >= 1",
                             output_fields=["id"], limit=10)
        ids = {r["id"] for r in rows}
        assert ids == {1, 3}  # id=2 was deleted

        _stop(server2, db2)


# ---------------------------------------------------------------------------
# 5. Dynamic fields survive restart
# ---------------------------------------------------------------------------

def test_dynamic_fields_persist():
    """Dynamic field values should survive restart."""
    with tempfile.TemporaryDirectory() as d:
        server, db, uri = _start(d)
        client = MilvusClient(uri=uri)

        schema = MilvusClient.create_schema(enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        client.create_collection("dyn_persist", schema=schema)
        client.insert("dyn_persist", [
            {"id": 1, "vec": [1, 0, 0, 0], "color": "red", "score": 95},
            {"id": 2, "vec": [0, 1, 0, 0], "color": "blue", "score": 80},
        ])

        _stop(server, db)

        server2, db2, uri2 = _start(d)
        client2 = MilvusClient(uri=uri2)

        client2.load_collection("dyn_persist")

        rows = client2.query("dyn_persist", filter="id >= 1",
                             output_fields=["color", "score"], limit=10)
        assert len(rows) == 2
        colors = {r["color"] for r in rows}
        assert colors == {"red", "blue"}

        _stop(server2, db2)


# ---------------------------------------------------------------------------
# 6. MilvusClient .db mode restart
# ---------------------------------------------------------------------------

def test_db_mode_restart():
    """Data persists across .db mode server restarts."""
    from milvus_lite.server_manager import ServerManager

    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "restart.db")

        # Phase 1: start via ServerManager, insert data
        mgr = ServerManager()
        uri = mgr.start_and_get_uri(db_path)
        client = MilvusClient(uri=uri)

        client.create_collection("db_persist", dimension=4)
        client.insert("db_persist", [
            {"id": 1, "vector": [1, 0, 0, 0]},
            {"id": 2, "vector": [0, 1, 0, 0]},
        ])
        mgr.release_all()

        # Phase 2: restart
        mgr2 = ServerManager()
        uri2 = mgr2.start_and_get_uri(db_path)
        client2 = MilvusClient(uri=uri2)

        assert client2.has_collection("db_persist")

        client2.load_collection("db_persist")
        results = client2.search("db_persist", data=[[1, 0, 0, 0]], limit=2)
        assert len(results[0]) == 2
        assert results[0][0]["id"] == 1

        mgr2.release_all()
