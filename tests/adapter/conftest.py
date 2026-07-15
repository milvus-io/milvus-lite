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


def _cleanup_all_databases(client):
    try:
        database_names = client.list_databases()
    except Exception:
        database_names = ["default"]

    for database_name in database_names:
        try:
            client.using_database(database_name)
        except Exception:
            continue
        try:
            collection_names = client.list_collections()
        except Exception:
            collection_names = []
        for collection_name in collection_names:
            try:
                client.drop_collection(collection_name)
            except Exception:
                pass

    try:
        client.using_database("default")
    except Exception:
        pass

    for database_name in database_names:
        if database_name != "default":
            try:
                client.drop_database(database_name)
            except Exception:
                pass


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

    Teardown best-effort cleans all collections in every database, then drops
    all non-default databases to maintain isolation.
    """
    from pymilvus import MilvusClient
    port, _db = grpc_server
    client = MilvusClient(uri=f"http://127.0.0.1:{port}")
    try:
        yield client
    finally:
        _cleanup_all_databases(client)
        client.close()
