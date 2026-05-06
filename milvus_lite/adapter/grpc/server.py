"""gRPC server lifecycle for the Milvus-protocol adapter.

Two entry points:
    run_server(data_dir, host, port) — blocking; for CLI use
    start_server_in_thread(data_dir, port) — non-blocking; for tests

Both share the same construction path: open a MilvusLite on
``data_dir``, instantiate ``MilvusServicer``, register it on a
ThreadPoolExecutor-backed grpc.server, bind to host:port, and start.

Concurrency model: gRPC's threadpool dispatches requests across
worker threads. The engine layer is single-writer per Collection,
so concurrent reads are safe but concurrent writes against the
same Collection are not. For Phase 10 MVP we accept that — the
target use case is "one local Python process drives the server",
and concurrency hardening is a separate phase.

Lifetime: ``run_server`` blocks until KeyboardInterrupt, then closes
the MilvusLite (which releases the data_dir LOCK). The thread variant
returns a (server, db) handle so tests can call ``server.stop()`` and
``db.close()`` themselves.
"""

from __future__ import annotations

import logging
from concurrent import futures
from typing import Optional, Tuple

import grpc
from pymilvus.grpc_gen import milvus_pb2_grpc

from milvus_lite.adapter.grpc.servicer import MilvusServicer
from milvus_lite.db import MilvusLite

logger = logging.getLogger(__name__)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 19530
DEFAULT_MAX_WORKERS = 10


def run_server(
    data_dir: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Start a blocking gRPC server bound to *host:port*.

    Holds the data_dir LOCK for the lifetime of the server. On
    KeyboardInterrupt, gracefully shuts down (drains in-flight
    requests for up to 5 seconds, then closes the DB).
    """
    db = MilvusLite(data_dir)
    server = _build_server(db, max_workers)

    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    print(f"MilvusLite gRPC server listening on {addr} (data_dir={data_dir!r})")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop(grace=5)
        db.close()


def start_server_in_thread(
    data_dir: str,
    host: str = "127.0.0.1",
    port: int = 0,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Tuple[grpc.Server, MilvusLite, int]:
    """Start a non-blocking gRPC server. Used by integration tests.

    Args:
        data_dir: backing data directory
        host:     bind host (default 127.0.0.1 for tests)
        port:     0 = pick a free port automatically; the actual
                  port is returned in the third tuple slot
        max_workers: thread pool size

    Returns:
        (server, db, port). Caller is responsible for shutdown:
            server.stop(grace=2)
            db.close()
    """
    db = MilvusLite(data_dir)
    server = _build_server(db, max_workers)

    addr = f"{host}:{port}"
    bound_port = server.add_insecure_port(addr)
    server.start()
    return server, db, bound_port


_MAX_MESSAGE_SIZE = 256 * 1024 * 1024  # 256 MB, matching Milvus Standalone


def _build_server(db: MilvusLite, max_workers: int) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_receive_message_length", _MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", _MAX_MESSAGE_SIZE),
        ],
    )
    servicer = MilvusServicer(db)
    milvus_pb2_grpc.add_MilvusServiceServicer_to_server(servicer, server)
    return server
