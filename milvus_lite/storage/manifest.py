"""Single-source-of-truth manifest, atomically updated.

Phase-2 minimal subset:
- load / save (with .prev backup + fallback)
- partition CRUD: add_partition / has_partition / list_partitions
- counters: version, current_seq, active_wal_number
- file lists: stored as {partition: {"data_files": [], "delta_files": []}}
  but Phase 2 adds no files — only Phase 3 flush will populate them.

Disk layout:
    data_dir/
      ├── manifest.json          # current
      └── manifest.json.prev     # previous version (single-step backup)

Atomic update protocol (save):
    1. dump payload to manifest.json.tmp
    2. fsync the tmp file
    3. if manifest.json exists, copy it → manifest.json.prev (overwriting)
    4. os.rename(manifest.json.tmp, manifest.json)
    5. fsync the parent directory

Load fallback:
    1. try manifest.json
    2. on JSONDecodeError or missing keys → try manifest.json.prev (with warning)
    3. if both fail → ManifestCorruptedError
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from milvus_lite.constants import DEFAULT_PARTITION
from milvus_lite.exceptions import (
    DefaultPartitionError,
    ManifestCorruptedError,
    PartitionAlreadyExistsError,
    PartitionNotFoundError,
)

if TYPE_CHECKING:
    from milvus_lite.index.spec import IndexSpec

logger = logging.getLogger(__name__)


MANIFEST_FILENAME = "manifest.json"
MANIFEST_PREV_FILENAME = "manifest.json.prev"
MANIFEST_TMP_FILENAME = "manifest.json.tmp"

MANIFEST_FORMAT_VERSION = 2


class Manifest:
    """Single-source-of-truth state snapshot for a Collection.

    Phase-2 holds only minimal state. Phase 3 will add data/delta file
    tracking when flush lands.
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        # Persistent state.
        self._version: int = 0
        self._current_seq: int = 0
        self._schema_version: int = 1
        self._active_wal_number: Optional[int] = None
        self._partitions: Dict[str, Dict[str, List[str]]] = {
            DEFAULT_PARTITION: {"data_files": [], "delta_files": []},
        }
        # Phase 9.3 / Phase 18: per-field IndexSpec dict.
        # Keys are field_name strings, values are IndexSpec instances.
        # Empty dict = no indexes created yet.
        self._index_specs: Dict[str, "IndexSpec"] = {}

    # ── persistence ─────────────────────────────────────────────

    def save(self) -> None:
        """Atomically write manifest.json. Bumps version by 1.

        Steps:
            1. write manifest.json.tmp + fsync
            2. cp manifest.json → manifest.json.prev (if it exists)
            3. rename manifest.json.tmp → manifest.json  ← commit point
            4. bump in-memory version (only after rename succeeds)
            5. fsync parent dir so the rename is durable

        Crash safety:
            - If any step before rename fails, .tmp is cleaned up and
              in-memory version is unchanged.
            - The .prev copy failing is non-fatal: we still proceed with
              the rename because the .tmp is already durable. Losing the
              backup is better than failing the entire save.
        """
        os.makedirs(self._data_dir, exist_ok=True)

        new_version = self._version + 1
        # Build payload with new_version without mutating self._version yet.
        payload = self._to_payload()
        payload["version"] = new_version

        target_path = os.path.join(self._data_dir, MANIFEST_FILENAME)
        prev_path = os.path.join(self._data_dir, MANIFEST_PREV_FILENAME)
        tmp_path = os.path.join(self._data_dir, MANIFEST_TMP_FILENAME)

        try:
            # 1. dump tmp + fsync
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            # 2. copy current → prev (best-effort; failure here is non-fatal
            #    because the .tmp is already durable and the rename can proceed)
            if os.path.exists(target_path):
                try:
                    shutil.copy2(target_path, prev_path)
                except OSError as e:
                    logger.warning("manifest: failed to create .prev backup: %s", e)

            # 3. atomic rename — this is the commit point
            os.rename(tmp_path, target_path)
        except BaseException:
            # Clean up orphaned .tmp on any failure before commit point.
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

        # 4. Version bump only after the commit point succeeds.
        self._version = new_version

        # 5. fsync the directory so the rename survives a crash
        try:
            dir_fd = os.open(self._data_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is not supported on every platform; best effort.
            pass

    @classmethod
    def load(cls, data_dir: str) -> "Manifest":
        """Load manifest from data_dir. If no manifest exists, return a fresh one.

        Fallback chain:
            1. manifest.json
            2. manifest.json.prev (with warning)
            3. raise ManifestCorruptedError
        """
        target_path = os.path.join(data_dir, MANIFEST_FILENAME)
        prev_path = os.path.join(data_dir, MANIFEST_PREV_FILENAME)

        if not os.path.exists(target_path) and not os.path.exists(prev_path):
            return cls(data_dir)

        # Try current.
        if os.path.exists(target_path):
            try:
                return cls._load_path(data_dir, target_path)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "manifest.json failed to load (%s), falling back to manifest.json.prev",
                    e,
                )

        # Fallback to prev.
        if os.path.exists(prev_path):
            try:
                m = cls._load_path(data_dir, prev_path)
                logger.warning(
                    "loaded manifest from manifest.json.prev — last save likely corrupted"
                )
                return m
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                raise ManifestCorruptedError(
                    f"both manifest.json and manifest.json.prev failed to load in {data_dir!r}: {e}"
                ) from e

        raise ManifestCorruptedError(
            f"manifest.json failed to load and no manifest.json.prev exists in {data_dir!r}"
        )

    @classmethod
    def _load_path(cls, data_dir: str, path: str) -> "Manifest":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"manifest at {path!r} is not an object")

        m = cls(data_dir)
        m._version = int(payload.get("version", 0))
        m._current_seq = int(payload.get("current_seq", 0))
        m._schema_version = int(payload.get("schema_version", 1))
        wal_num = payload.get("active_wal_number")
        m._active_wal_number = int(wal_num) if wal_num is not None else None

        partitions = payload.get("partitions", {})
        if not isinstance(partitions, dict):
            raise ValueError(f"manifest at {path!r} has non-dict 'partitions'")
        m._partitions = {}
        for name, contents in partitions.items():
            if not isinstance(contents, dict):
                raise ValueError(
                    f"manifest at {path!r} partition {name!r} contents not a dict"
                )
            m._partitions[name] = {
                "data_files": list(contents.get("data_files", [])),
                "delta_files": list(contents.get("delta_files", [])),
            }
        if DEFAULT_PARTITION not in m._partitions:
            m._partitions[DEFAULT_PARTITION] = {"data_files": [], "delta_files": []}

        from milvus_lite.index.spec import IndexSpec
        specs_dict = payload.get("index_specs")
        if specs_dict and isinstance(specs_dict, dict):
            m._index_specs = {
                k: IndexSpec.from_dict(v) for k, v in specs_dict.items()
            }
        else:
            m._index_specs = {}

        return m

    def _to_payload(self) -> Dict[str, Any]:
        import copy
        return {
            "manifest_format_version": MANIFEST_FORMAT_VERSION,
            "version": self._version,
            "current_seq": self._current_seq,
            "schema_version": self._schema_version,
            "active_wal_number": self._active_wal_number,
            "partitions": copy.deepcopy(self._partitions),
            "index_specs": {
                k: v.to_dict() for k, v in self._index_specs.items()
            },
        }

    # ── partition CRUD ──────────────────────────────────────────

    def add_partition(self, name: str) -> None:
        if name in self._partitions:
            raise PartitionAlreadyExistsError(name)
        self._partitions[name] = {"data_files": [], "delta_files": []}

    def remove_partition(self, name: str) -> None:
        if name == DEFAULT_PARTITION:
            raise DefaultPartitionError(
                f"cannot remove default partition {DEFAULT_PARTITION!r}"
            )
        if name not in self._partitions:
            raise PartitionNotFoundError(name)
        del self._partitions[name]

    def has_partition(self, name: str) -> bool:
        return name in self._partitions

    def list_partitions(self) -> List[str]:
        return sorted(self._partitions.keys())

    # ── data file CRUD ──────────────────────────────────────────

    def add_data_file(self, partition: str, filename: str) -> None:
        """Register a new data Parquet file for *partition*."""
        if partition not in self._partitions:
            raise PartitionNotFoundError(partition)
        self._partitions[partition]["data_files"].append(filename)

    def remove_data_files(self, partition: str, filenames: List[str]) -> None:
        """Unregister data files (used by compaction in Phase 6)."""
        if partition not in self._partitions:
            raise PartitionNotFoundError(partition)
        bucket = self._partitions[partition]["data_files"]
        for fn in filenames:
            try:
                bucket.remove(fn)
            except ValueError:
                pass  # already absent — idempotent

    def get_data_files(self, partition: str) -> List[str]:
        """Relative paths of data files in *partition*."""
        if partition not in self._partitions:
            raise PartitionNotFoundError(partition)
        return list(self._partitions[partition]["data_files"])

    def get_all_data_files(self) -> Dict[str, List[str]]:
        """{partition: [data files]} for all partitions."""
        return {p: list(c["data_files"]) for p, c in self._partitions.items()}

    # ── delta file CRUD ─────────────────────────────────────────

    def add_delta_file(self, partition: str, filename: str) -> None:
        if partition not in self._partitions:
            raise PartitionNotFoundError(partition)
        self._partitions[partition]["delta_files"].append(filename)

    def remove_delta_files(self, partition: str, filenames: List[str]) -> None:
        if partition not in self._partitions:
            raise PartitionNotFoundError(partition)
        bucket = self._partitions[partition]["delta_files"]
        for fn in filenames:
            try:
                bucket.remove(fn)
            except ValueError:
                pass

    def get_delta_files(self, partition: str) -> List[str]:
        if partition not in self._partitions:
            raise PartitionNotFoundError(partition)
        return list(self._partitions[partition]["delta_files"])

    def get_all_delta_files(self) -> Dict[str, List[str]]:
        return {p: list(c["delta_files"]) for p, c in self._partitions.items()}

    # ── counters / properties ───────────────────────────────────

    @property
    def version(self) -> int:
        return self._version

    @property
    def current_seq(self) -> int:
        return self._current_seq

    @current_seq.setter
    def current_seq(self, value: int) -> None:
        if value < self._current_seq:
            raise ValueError(
                f"current_seq is monotonic; tried to set {value} < {self._current_seq}"
            )
        self._current_seq = value

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @schema_version.setter
    def schema_version(self, value: int) -> None:
        self._schema_version = value

    @property
    def active_wal_number(self) -> Optional[int]:
        return self._active_wal_number

    @active_wal_number.setter
    def active_wal_number(self, value: Optional[int]) -> None:
        self._active_wal_number = value

    @property
    def data_dir(self) -> str:
        return self._data_dir

    # ── index specs ─────────────────────────────────────────────

    @property
    def index_specs(self) -> Dict[str, "IndexSpec"]:
        """All persisted IndexSpecs, keyed by field_name (defensive copy)."""
        return dict(self._index_specs)

    def set_index_spec(self, spec: Optional["IndexSpec"]) -> None:
        """Set or clear an IndexSpec by field_name.
        Pass None to clear ALL specs. Caller must call save()."""
        if spec is None:
            self._index_specs = {}
        else:
            self._index_specs[spec.field_name] = spec

    def remove_index_spec(self, field_name: str) -> None:
        """Remove a single IndexSpec by field_name."""
        self._index_specs.pop(field_name, None)

    @property
    def format_version(self) -> int:
        """The on-disk manifest format version."""
        return MANIFEST_FORMAT_VERSION
