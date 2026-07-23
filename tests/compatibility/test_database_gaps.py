"""Executable gaps in database-name compatibility with upstream Milvus.

The naming cases execute only the upstream
``test_milvus_client_create_database_invalid_db_name`` create scenario.
"""

import shutil
import sys
import tempfile

import pytest
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from milvus_lite.adapter.grpc.server import start_server_in_thread


@pytest.fixture(scope="session")
def grpc_server():
    data_dir = tempfile.mkdtemp(prefix="milvus_lite_database_gaps_")
    server = None
    db = None
    try:
        server, db, port = start_server_in_thread(data_dir)
        yield port
    finally:
        if server is not None:
            server.stop(grace=2)
        if db is not None:
            db.close()
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def database_client(grpc_server):
    client = MilvusClient(uri=f"http://127.0.0.1:{grpc_server}")
    try:
        yield client
    finally:
        try:
            database_names = client.list_databases()
        except Exception:
            database_names = ["default"]

        for database_name in database_names:
            if database_name == "default":
                continue
            try:
                client.using_database(database_name)
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
            if database_name == "default":
                continue
            try:
                client.drop_database(database_name)
            except Exception:
                pass

        client.close()


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason=(
        "Upstream Milvus rejects names outside its identifier grammar, while "
        "Milvus Lite currently accepts filesystem-safe names"
    ),
)
@pytest.mark.parametrize(
    "database_name",
    [
        "12-s",
        "12 s",
        "(mn)",
        "中文",
        "%$#",
        pytest.param(
            "  ",
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="Windows cannot represent a directory name of only spaces",
            ),
        ),
    ],
)
def test_create_database_rejects_upstream_invalid_names(
    database_client,
    database_name,
):
    databases_before = database_client.list_databases()
    try:
        database_client.create_database(database_name)
    except MilvusException:
        assert database_client.list_databases() == databases_before
        return

    pytest.fail("Milvus Lite accepted upstream-invalid database name")


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason=(
        "Upstream test_connect_not_existed_db rejects client construction for "
        "a missing database, while Milvus Lite currently constructs the client"
    ),
)
def test_client_constructor_rejects_missing_database(database_client, grpc_server):
    missing_database = f"missing_{id(database_client)}"
    databases_before = database_client.list_databases()
    new_client = None
    try:
        new_client = MilvusClient(
            uri=f"http://127.0.0.1:{grpc_server}",
            db_name=missing_database,
        )
    except MilvusException:
        assert database_client.list_databases() == databases_before
        return

    new_client.close()
    pytest.fail("Milvus Lite constructed a client for a missing database")
