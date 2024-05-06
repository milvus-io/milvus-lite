import os
import subprocess
import pathlib
import logging
import fcntl
import re


BIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')


logger = logging.getLogger()


class Server:
    """
    """
    MILVUS_BIN = 'milvus'
    

    def __init__(self, db_file: str, args=None):
        if os.environ.get('BIN_PATH') is not None:
            self._bin_path = pathlib.Path(os.environ['BIN_PATH']).absolute()
        else:
            self._bin_path = pathlib.Path(BIN_PATH).absolute()
        self._db_file = pathlib.Path(db_file).absolute()
        if not re.match(r'^[a-zA-Z0-9.\-_]+$', self._db_file.name):
            raise RuntimeError(f"Unsupport db name {self._db_file.name}, the name must match ^[a-zA-Z0-9.\-_]+$")
        self._work_dir = self._db_file.parent
        self._args = args
        self._p = None
        self._uds_path = str(self._db_file.parent / f'.{self._db_file.name}.sock')
        self._lock_path = str(self._db_file.parent / f'.{self._db_file.name}.lock')
        self._lock_fd = None

    def init(self) -> bool:
        if not self._bin_path.exists():
            logger.error("Bin path not exists")
            return False
        if not self._work_dir.exists():
            logger.error("Dir %s not exist", self._work_dir)
        return True

    @property
    def milvus_bin(self):
        return str(self._bin_path / 'milvus')

    @property
    def log_level(self):
        return os.environ.get("LOG_LEVEL", "ERROR")

    @property
    def uds_path(self):
        return f'unix:{self._uds_path}'

    @property
    def args(self):
        if self._args is not None:
            return self._args
        return [self.milvus_bin, self._db_file, self.uds_path, self.log_level, self._lock_path]

    def start(self) -> bool:
        assert self._p is None, "Server already started"
        self._lock_fd = open(self._lock_path, 'a')
        try:
            fcntl.lockf(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._p = subprocess.Popen(
                args=self.args,
                env={"LD_LIBRARY_PATH": str(self._bin_path)},
                cwd=str(self._work_dir),
            )
            return True
        except BlockingIOError:
            logger.error("Open %s failed, the file has been opened by another program", self._db_file)
            return False

    def stop(self):
        if self._p is not None:
            logger.info("Stop milvus...")
            try:
                self._p.terminate()
                self._p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._p.kill()
                self._p.wait(timeout=3)
            self._p = None
        if self._lock_fd:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None
        pathlib.Path(self._uds_path).unlink(missing_ok=True)
        pathlib.Path(self._lock_path).unlink(missing_ok=True)
