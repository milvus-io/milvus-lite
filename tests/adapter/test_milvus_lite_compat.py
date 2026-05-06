"""Test the milvus_lite compatibility layer.

Verifies that pymilvus can connect via MilvusClient("./demo.db") using
our ServerManager, and perform basic CRUD operations.
"""

import tempfile
import os

import pytest

# Skip entire module if grpc deps are not installed
pymilvus = pytest.importorskip("pymilvus")
pytest.importorskip("grpc")

from pymilvus import MilvusClient, DataType
from milvus_lite.server_manager import ServerManager


class TestServerManager:
    """Unit tests for ServerManager lifecycle."""

    def test_start_and_get_uri(self):
        mgr = ServerManager()
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            uri = mgr.start_and_get_uri(db_path)
            try:
                assert uri is not None
                assert uri.startswith("http://127.0.0.1:")
                port = int(uri.split(":")[-1])
                assert port > 0
            finally:
                mgr.release_all()

    def test_reuse_same_path(self):
        mgr = ServerManager()
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            uri1 = mgr.start_and_get_uri(db_path)
            uri2 = mgr.start_and_get_uri(db_path)
            try:
                assert uri1 == uri2
            finally:
                mgr.release_all()

    def test_different_paths(self):
        mgr = ServerManager()
        with tempfile.TemporaryDirectory() as d:
            uri1 = mgr.start_and_get_uri(os.path.join(d, "a.db"))
            uri2 = mgr.start_and_get_uri(os.path.join(d, "b.db"))
            try:
                assert uri1 != uri2
            finally:
                mgr.release_all()

    def test_release_server(self):
        mgr = ServerManager()
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            mgr.start_and_get_uri(db_path)
            mgr.release_server(db_path)
            assert os.path.abspath(db_path) not in mgr._servers


class TestPymilvusIntegration:
    """End-to-end: pymilvus MilvusClient → .db path → our engine."""

    def test_full_lifecycle_via_server_manager(self):
        """Manually start server and connect pymilvus to it."""
        mgr = ServerManager()
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "demo.db")
            uri = mgr.start_and_get_uri(db_path)
            assert uri is not None

            try:
                client = MilvusClient(uri=uri)

                # Create collection
                schema = MilvusClient.create_schema()
                schema.add_field("id", DataType.INT64, is_primary=True)
                schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
                schema.add_field("text", DataType.VARCHAR, max_length=128)
                client.create_collection("test_col", schema=schema)

                # Insert
                client.insert("test_col", [
                    {"id": 1, "vec": [1, 0, 0, 0], "text": "hello"},
                    {"id": 2, "vec": [0, 1, 0, 0], "text": "world"},
                    {"id": 3, "vec": [0, 0, 1, 0], "text": "foo"},
                ])

                # Create index + load
                idx = client.prepare_index_params()
                idx.add_index(
                    field_name="vec", index_type="FLAT",
                    metric_type="COSINE", params={},
                )
                client.create_index("test_col", idx)
                client.load_collection("test_col")

                # Search
                results = client.search(
                    "test_col", data=[[1, 0, 0, 0]], limit=3,
                    output_fields=["text"],
                )
                assert len(results) == 1
                assert len(results[0]) == 3
                assert results[0][0]["id"] == 1

                # Query
                rows = client.query(
                    "test_col", filter="id < 3",
                    output_fields=["text"], limit=10,
                )
                assert len(rows) == 2

                # Delete
                client.delete("test_col", ids=[1])
                rows = client.query(
                    "test_col", filter="id >= 1",
                    output_fields=["id"], limit=10,
                )
                assert {r["id"] for r in rows} == {2, 3}

                # Drop
                client.drop_collection("test_col")
                assert not client.has_collection("test_col")

            finally:
                mgr.release_all()

    def test_shorthand_create_collection(self):
        """Test pymilvus shorthand: create_collection("name", dimension=N)."""
        mgr = ServerManager()
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "short.db")
            uri = mgr.start_and_get_uri(db_path)
            assert uri is not None

            try:
                client = MilvusClient(uri=uri)
                client.create_collection("quick", dimension=8)
                client.insert("quick", [
                    {"id": i, "vector": [float(i)] * 8}
                    for i in range(10)
                ])
                results = client.search("quick", data=[[1.0] * 8], limit=3)
                assert len(results[0]) == 3
                client.drop_collection("quick")
            finally:
                mgr.release_all()

    def test_dot_db_uri_via_pymilvus(self):
        """Test that MilvusClient("./xxx.db") works end-to-end.

        pymilvus detects .db suffix → imports milvus_lite.server_manager
        → starts our gRPC server → connects. Full round-trip.
        """
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "pymilvus_test.db")
            client = MilvusClient(uri=db_path)

            try:
                # Create + insert + index + load + search
                client.create_collection("e2e", dimension=4)
                client.insert("e2e", [
                    {"id": 1, "vector": [1, 0, 0, 0]},
                    {"id": 2, "vector": [0, 1, 0, 0]},
                    {"id": 3, "vector": [0, 0, 1, 0]},
                ])
                results = client.search("e2e", data=[[1, 0, 0, 0]], limit=3)
                assert len(results[0]) == 3
                assert results[0][0]["id"] == 1

                # Query
                rows = client.query("e2e", filter="id >= 2",
                                    output_fields=["id"], limit=10)
                assert {r["id"] for r in rows} == {2, 3}

                # Delete + verify
                client.delete("e2e", ids=[2])
                rows = client.query("e2e", filter="id >= 1",
                                    output_fields=["id"], limit=10)
                assert {r["id"] for r in rows} == {1, 3}

                client.drop_collection("e2e")
            finally:
                client.close()
                # Clean up server started by pymilvus
                from milvus_lite.server_manager import server_manager_instance
                server_manager_instance.release_server(db_path)
