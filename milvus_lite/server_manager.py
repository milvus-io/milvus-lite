"""ServerManager — pymilvus integration point for .db URIs.

pymilvus calls ``server_manager_instance.start_and_get_uri("./demo.db")``
when it detects a ``.db`` URI. This module:

1. Uses the ``.db`` path as the MilvusLite data directory
2. Starts a gRPC server in a background thread on a free port
3. Returns ``http://127.0.0.1:{port}`` so pymilvus can connect

Multiple calls with the same path reuse the same server. Servers are
cleaned up on process exit via atexit.

Unlike milvus-lite v1 (which spawns a ~200MB C++ subprocess), this
runs entirely in-process as pure Python threads.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ServerManager:
    """Manages per-path gRPC server instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # path → (server, db, port)
        self._servers: Dict[str, Tuple] = {}
        atexit.register(self.release_all)

    def start_and_get_uri(
        self,
        path: str,
        args: Optional[dict] = None,
    ) -> Optional[str]:
        """Start a gRPC server for the given .db path (or reuse existing).

        Args:
            path: the ``.db`` URI from pymilvus (used as data_dir)
            args: optional extra arguments (unused, kept for API compat)

        Returns:
            ``http://127.0.0.1:{port}`` on success, None on failure
        """
        abs_path = os.path.abspath(path)

        with self._lock:
            if abs_path in self._servers:
                _, _, port = self._servers[abs_path]
                return f"http://127.0.0.1:{port}"

        try:
            from milvus_lite.adapter.grpc.server import start_server_in_thread

            server, db, port = start_server_in_thread(
                data_dir=abs_path,
                host="127.0.0.1",
                port=0,  # auto-select free port
            )

            with self._lock:
                # Double-check: another thread may have raced us
                if abs_path in self._servers:
                    server.stop(grace=0)
                    db.close()
                    _, _, port = self._servers[abs_path]
                    return f"http://127.0.0.1:{port}"

                self._servers[abs_path] = (server, db, port)

            logger.info(
                "MilvusLite server started for %s on port %d", abs_path, port
            )
            return f"http://127.0.0.1:{port}"

        except Exception:
            logger.exception("Failed to start MilvusLite server for %s", abs_path)
            return None

    def release_server(self, path: str) -> None:
        """Stop the server for a specific .db path."""
        abs_path = os.path.abspath(path)
        with self._lock:
            entry = self._servers.pop(abs_path, None)
        if entry:
            server, db, port = entry
            try:
                server.stop(grace=2)
            except Exception:
                pass
            try:
                db.close()
            except Exception:
                pass
            logger.info("MilvusLite server stopped for %s (port %d)", abs_path, port)

    def release_all(self) -> None:
        """Stop all running servers. Called automatically at process exit."""
        with self._lock:
            paths = list(self._servers.keys())
        for path in paths:
            self.release_server(path)


server_manager_instance = ServerManager()
