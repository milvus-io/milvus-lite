"""External protocol adapters (Phase 10).

Adapters translate external protocols (gRPC today; HTTP / OpenAPI in
the future) into engine API calls. Adapters NEVER add capability —
they're pure protocol shims. Unsupported RPCs return UNIMPLEMENTED
with a friendly message rather than silent-failing.

Subpackages:
    grpc/  — Milvus-protocol-compatible gRPC server (Phase 10).
             Implemented on top of pymilvus.grpc_gen so the wire
             protocol stays in sync with pymilvus by construction.
"""
