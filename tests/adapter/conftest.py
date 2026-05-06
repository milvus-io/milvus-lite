"""Shared fixtures for the gRPC adapter test suite.

Uses a session-scoped gRPC server to avoid file descriptor exhaustion
(each server opens sockets + LOCK files; 100+ function-scoped servers
hit the OS fd limit). Collections are cleaned up after each test to
maintain isolation.
"""

import tempfile

import pytest

# Probe pymilvus + grpcio at import time. If either is missing, skip
# the entire adapter test directory.
pymilvus = pytest.importorskip("pymilvus")
grpc = pytest.importorskip("grpc")


@pytest.fixture(scope="session")
def grpc_server():
    """Start a single MilvusLite gRPC server for the entire adapter
    test session. Avoids fd exhaustion from per-test server creation."""
    from milvus_lite.adapter.grpc.server import start_server_in_thread

    tmpdir = tempfile.mkdtemp(prefix="milvus_lite_test_")
    server, db, port = start_server_in_thread(tmpdir)
    yield port, db
    server.stop(grace=2)
    db.close()
    # tmpdir cleanup is best-effort; OS cleans /tmp on reboot
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def milvus_client(grpc_server):
    """A pymilvus MilvusClient connected to the session server.

    Cleans up all collections after each test to maintain isolation.
    """
    from pymilvus import MilvusClient
    port, _db = grpc_server
    client = MilvusClient(uri=f"http://127.0.0.1:{port}")
    try:
        yield client
    finally:
        # Clean up all collections created during this test
        for name in client.list_collections():
            try:
                client.drop_collection(name)
            except Exception:
                pass
        client.close()
