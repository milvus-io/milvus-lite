"""Milvus-protocol gRPC adapter (Phase 10).

Provides a gRPC server that pymilvus clients can connect to without
modification. Built on top of ``pymilvus.grpc_gen`` so the wire format
is identical to a real Milvus deployment by construction.

Public API:
    run_server(data_dir, host, port) — start the server in the
        current thread; blocks until KeyboardInterrupt.
    MilvusServicer(db) — the gRPC servicer class. Internal — most
        callers should go through run_server.

Phase 10.1 ships an empty skeleton: the server starts, pymilvus
clients can connect, but every RPC returns UNIMPLEMENTED. Phase
10.2-10.6 fills in the quickstart subset (collection lifecycle,
CRUD, search, index, partitions, error mapping).
"""

from milvus_lite.adapter.grpc.server import run_server
from milvus_lite.adapter.grpc.servicer import MilvusServicer

__all__ = ["run_server", "MilvusServicer"]
