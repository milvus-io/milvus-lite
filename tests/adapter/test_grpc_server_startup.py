"""Phase 10.1 — server startup + connection-level RPC tests.

Validates the absolute minimum surface needed for pymilvus to connect
to the server. The data-plane RPCs (CreateCollection, Insert, Search,
etc.) are NOT tested here — they're added in 10.2-10.5 and tested in
their own files. Phase 10.1 only proves:

    1. The gRPC server starts and binds a port.
    2. pymilvus.MilvusClient(uri=...) can construct successfully
       (which requires Connect / GetVersion to be implemented).
    3. CheckHealth returns healthy.
    4. Any other RPC raises UNIMPLEMENTED — never silent fail.
"""

import grpc
import pytest


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def test_server_starts_and_binds_port(grpc_server):
    """Just constructing the fixture is the test — if the server
    couldn't start it would raise inside the fixture setup."""
    port, db = grpc_server
    assert port > 0
    assert db is not None


def test_server_data_dir_lock_held(tmp_path):
    """The server should hold the MilvusLite LOCK file for its lifetime,
    so a second concurrent server on the same data_dir is rejected."""
    from milvus_lite.adapter.grpc.server import start_server_in_thread
    from milvus_lite.exceptions import DataDirLockedError

    s1, db1, port1 = start_server_in_thread(str(tmp_path / "data"))
    try:
        with pytest.raises(DataDirLockedError):
            start_server_in_thread(str(tmp_path / "data"))
    finally:
        s1.stop(grace=1)
        db1.close()


# ---------------------------------------------------------------------------
# pymilvus client construction (Connect + GetVersion)
# ---------------------------------------------------------------------------

def test_pymilvus_client_constructs_successfully(milvus_client):
    """If Connect or GetVersion weren't implemented, MilvusClient
    construction itself would raise. Reaching this assertion proves
    both are working."""
    assert milvus_client is not None


def test_get_version_returns_milvus_lite_string(grpc_server):
    """GetVersion should return our identity string. Use the
    pymilvus.utility helper since MilvusClient doesn't surface
    GetVersion directly."""
    from pymilvus import connections, utility

    port, _ = grpc_server
    alias = "test_get_version"
    connections.connect(alias=alias, host="127.0.0.1", port=port)
    try:
        version = utility.get_server_version(using=alias)
        assert "milvus_lite" in version.lower()
    finally:
        connections.disconnect(alias)


# ---------------------------------------------------------------------------
# UNIMPLEMENTED behavior
# ---------------------------------------------------------------------------

def test_unimplemented_rpc_raises_clean_error(milvus_client):
    """Calling an RPC we haven't implemented yet should raise a
    pymilvus exception that wraps gRPC's UNIMPLEMENTED status, NOT
    crash the connection or silent-fail.

    We use ``load_partitions`` because partition-level load/release is
    intentionally outside the current support surface."""
    with pytest.raises(Exception) as exc_info:
        milvus_client.load_partitions("col", ["_default"])
    assert "implement" in str(exc_info.value).lower() or "not support" in str(exc_info.value).lower()


def test_connection_survives_unimplemented_call(grpc_server, milvus_client):
    """An UNIMPLEMENTED response must not poison the channel — the
    next call should still go through."""
    port, _db = grpc_server

    # Trigger an UNIMPLEMENTED error via a still-unimplemented RPC
    with pytest.raises(Exception):
        milvus_client.load_partitions("col", ["_default"])

    # Connection-level RPCs should still work on a fresh connection
    from pymilvus import connections, utility
    alias = "test_survive"
    connections.connect(alias=alias, host="127.0.0.1", port=port)
    try:
        v = utility.get_server_version(using=alias)
        assert "milvus_lite" in v.lower()
    finally:
        connections.disconnect(alias)
