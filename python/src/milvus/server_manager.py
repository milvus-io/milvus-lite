from typing import Optional
import threading
from milvus.server import Server
import logging
import pathlib


logger = logging.getLogger()


class ServerManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._servers = {}

    def start_and_get_uri(self, path: str, args=None) -> Optional[str]:
        path = pathlib.Path(path).absolute().resolve()
        with self._lock:
            if str(path) not in self._servers:
                s = Server(str(path), args)
                if not s.init():
                    return None
                self._servers[str(path)] = s
                if not self._servers[str(path)].start():
                    logger.error("Start local milvus failed")
                    return None
            return self._servers[str(path)].uds_path

    def release_server(self, path: str):
        path = pathlib.Path(path).absolute().resolve()
        with self._lock:
            if str(path) not in self._servers:
                logger.warning("No local milvus in path %s", str(path))
                return
            self._servers[str(path)].stop()
            del self._servers[str(path)]

    def release_all(self):
        for s in self._servers.values():
            s.stop()

    def __del__(self):
        with self._lock:
            self.release_all()


server_manager_instance = ServerManager()
