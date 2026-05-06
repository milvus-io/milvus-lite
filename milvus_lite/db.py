"""DB layer — multi-Collection lifecycle management.

MilvusLite is the top-level entry point. It manages multiple Collections
under one ``data_dir`` and ensures only one process can hold the dir at
a time via an advisory file lock.

Layout:

    data_dir/
    ├── LOCK                      # advisory flock — held while DB is open
    └── collections/
        ├── col_a/
        │   ├── schema.json       # Collection schema
        │   ├── manifest.json     # Manifest (single source of truth)
        │   ├── manifest.json.prev
        │   ├── wal/              # WAL files
        │   └── partitions/       # data + delta Parquet by partition
        └── col_b/
            └── ...

Collection instances are cached so that calling ``get_collection`` twice
with the same name returns the same object (and the same in-memory
MemTable / WAL state).
"""

from __future__ import annotations

import os
import sys
import shutil
import json
from typing import Any, Dict, List, Optional

from milvus_lite.engine.collection import Collection
from milvus_lite.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DataDirLockedError,
)
from milvus_lite.schema.persistence import load_schema, save_schema
from milvus_lite.schema.types import CollectionSchema
from milvus_lite.schema.validation import validate_schema


COLLECTIONS_DIRNAME = "collections"
LOCK_FILENAME = "LOCK"
SCHEMA_FILENAME = "schema.json"
ALIASES_FILENAME = "aliases.json"


class MilvusLite:
    """Top-level entry point. Open one of these per process per data_dir.

    Usage:

        db = MilvusLite("/path/to/data")
        col = db.create_collection("docs", schema)
        col.insert([...])
        results = col.search([...])
        db.close()

    Or as a context manager:

        with MilvusLite("/path/to/data") as db:
            col = db.get_collection("docs")
            ...

    Multi-process safety: ``__init__`` acquires an advisory ``flock`` on
    ``{data_dir}/LOCK``. If another process already holds the lock,
    construction raises ``DataDirLockedError``. The lock is released by
    ``close()`` (or by process exit, since the OS reclaims it).
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self._collections_root(), exist_ok=True)

        self._lock_path = os.path.join(data_dir, LOCK_FILENAME)
        self._lock_fd: Optional[int] = None
        self._acquire_lock()

        # Cache of opened Collections, keyed by name. Collections are
        # only created when explicitly requested (lazy load on get).
        self._collections: Dict[str, Collection] = {}
        self._aliases: Dict[str, str] = self._load_aliases()
        self._closed = False

    # ── public API ──────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        schema: CollectionSchema,
    ) -> Collection:
        """Create a new Collection. Raises if a Collection with this
        name already exists."""
        self._check_open()
        self._validate_name(name)

        # Validate the schema BEFORE touching disk so we don't leave a
        # half-initialized collection directory if validation fails.
        validate_schema(schema)

        if self.has_collection(name):
            raise CollectionAlreadyExistsError(
                f"collection {name!r} already exists"
            )

        col_dir = self._collection_dir(name)
        os.makedirs(col_dir, exist_ok=False)

        # Persist schema first — this is the marker that the Collection
        # exists. has_collection() looks for it.
        save_schema(schema, name, os.path.join(col_dir, SCHEMA_FILENAME))

        try:
            col = Collection(name, col_dir, schema)
        except Exception:
            # Clean up orphan directory if Collection init fails
            shutil.rmtree(col_dir, ignore_errors=True)
            raise
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        """Open an existing Collection. Subsequent calls return the same
        cached instance."""
        self._check_open()
        name = self.resolve_collection_name(name)
        if name in self._collections:
            return self._collections[name]
        if not self.has_collection(name):
            raise CollectionNotFoundError(
                f"collection {name!r} does not exist"
            )

        col_dir = self._collection_dir(name)
        _name, schema = load_schema(os.path.join(col_dir, SCHEMA_FILENAME))
        col = Collection(name, col_dir, schema)
        self._collections[name] = col
        return col

    def drop_collection(self, name: str) -> None:
        """Close and delete a Collection. No-op if it does not exist."""
        self._check_open()
        name = self.resolve_collection_name(name)
        if not self.has_collection(name):
            return

        # Close the cached instance first so its WAL writers release
        # any open file handles before we rmtree the directory.
        if name in self._collections:
            self._collections[name].close()

        col_dir = self._collection_dir(name)
        shutil.rmtree(col_dir, ignore_errors=False)
        # Only remove from cache after successful rmtree
        self._collections.pop(name, None)
        removed = [alias for alias, target in self._aliases.items() if target == name]
        for alias in removed:
            del self._aliases[alias]
        if removed:
            self._save_aliases()

    def rename_collection(self, old_name: str, new_name: str) -> None:
        """Rename a collection on disk and in the cache."""
        self._check_open()
        if not self.has_collection(old_name):
            raise CollectionNotFoundError(
                f"collection {old_name!r} does not exist"
            )
        self._validate_name(new_name)
        if self.has_collection(new_name):
            raise CollectionAlreadyExistsError(
                f"collection {new_name!r} already exists"
            )

        # Close the cached instance so WAL / file handles are released.
        if old_name in self._collections:
            self._collections[old_name].close()
            del self._collections[old_name]

        old_dir = self._collection_dir(old_name)
        new_dir = self._collection_dir(new_name)
        os.rename(old_dir, new_dir)

        # Update collection_name inside schema.json.
        schema_path = os.path.join(new_dir, SCHEMA_FILENAME)
        _name, schema = load_schema(schema_path)
        save_schema(schema, new_name, schema_path)

        for alias, target in list(self._aliases.items()):
            if target == old_name:
                self._aliases[alias] = new_name
        self._save_aliases()

    def has_collection(self, name: str) -> bool:
        """True iff a Collection with this name exists on disk."""
        name = self.resolve_collection_name(name)
        return os.path.exists(
            os.path.join(self._collection_dir(name), SCHEMA_FILENAME)
        )

    def list_collections(self) -> List[str]:
        """Return all Collection names, sorted."""
        root = self._collections_root()
        if not os.path.exists(root):
            return []
        names: List[str] = []
        for entry in os.listdir(root):
            sub = os.path.join(root, entry)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, SCHEMA_FILENAME)):
                names.append(entry)
        return sorted(names)

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """Phase 9.1: Return basic stats for a collection.

        Currently returns ``{"row_count": int}``. The Phase 10 gRPC
        adapter maps this directly into Milvus's
        ``GetCollectionStatistics`` response (a list of KeyValuePair
        with the single ``row_count`` entry).

        Loads the Collection if not already cached. Raises
        ``CollectionNotFoundError`` if it doesn't exist.
        """
        col = self.get_collection(name)
        return {"row_count": col.num_entities}

    def get_partition_stats(self, collection_name: str, partition_name: str) -> Dict[str, Any]:
        """Return basic stats for one partition."""
        col = self.get_collection(collection_name)
        return {"row_count": col.partition_num_entities(partition_name)}

    def truncate_collection(self, name: str) -> None:
        """Delete all data in a collection while preserving schema and aliases."""
        self._check_open()
        name = self.resolve_collection_name(name)
        if not self.has_collection(name):
            raise CollectionNotFoundError(f"collection {name!r} does not exist")

        col_dir = self._collection_dir(name)
        _schema_name, schema = load_schema(os.path.join(col_dir, SCHEMA_FILENAME))
        if name in self._collections:
            self._collections[name].close()
            del self._collections[name]

        shutil.rmtree(col_dir, ignore_errors=False)
        os.makedirs(col_dir, exist_ok=False)
        save_schema(schema, name, os.path.join(col_dir, SCHEMA_FILENAME))

    def create_alias(self, collection_name: str, alias: str) -> None:
        """Create a collection alias."""
        self._check_open()
        collection_name = self.resolve_collection_name(collection_name)
        self._validate_name(alias)
        if not self.has_collection(collection_name):
            raise CollectionNotFoundError(
                f"collection {collection_name!r} does not exist"
            )
        if os.path.exists(os.path.join(self._collection_dir(alias), SCHEMA_FILENAME)):
            raise CollectionAlreadyExistsError(
                f"alias {alias!r} conflicts with an existing collection"
            )
        if alias in self._aliases:
            raise CollectionAlreadyExistsError(f"alias {alias!r} already exists")
        self._aliases[alias] = collection_name
        self._save_aliases()

    def alter_alias(self, collection_name: str, alias: str) -> None:
        """Point an existing alias at another collection."""
        self._check_open()
        collection_name = self.resolve_collection_name(collection_name)
        if alias not in self._aliases:
            raise CollectionNotFoundError(f"alias {alias!r} does not exist")
        if not self.has_collection(collection_name):
            raise CollectionNotFoundError(
                f"collection {collection_name!r} does not exist"
            )
        self._aliases[alias] = collection_name
        self._save_aliases()

    def drop_alias(self, alias: str) -> None:
        """Drop an alias. No-op if it does not exist, matching drop_collection."""
        self._check_open()
        if alias in self._aliases:
            del self._aliases[alias]
            self._save_aliases()

    def describe_alias(self, alias: str) -> Dict[str, str]:
        """Return alias metadata."""
        self._check_open()
        if alias not in self._aliases:
            raise CollectionNotFoundError(f"alias {alias!r} does not exist")
        return {"alias": alias, "collection": self._aliases[alias]}

    def list_aliases(self, collection_name: Optional[str] = None) -> List[str]:
        """Return aliases, optionally filtered by target collection."""
        self._check_open()
        if collection_name:
            collection_name = self.resolve_collection_name(collection_name)
            return sorted(
                alias for alias, target in self._aliases.items()
                if target == collection_name
            )
        return sorted(self._aliases)

    def resolve_collection_name(self, name: str) -> str:
        """Resolve a collection alias to its target collection name."""
        return self._aliases.get(name, name)

    def close(self) -> None:
        """Close every cached Collection and release the LOCK.

        Idempotent. After close(), this DB instance is unusable; create
        a new one to reopen the data_dir.
        """
        if self._closed:
            return
        for name in list(self._collections.keys()):
            try:
                self._collections[name].close()
            except Exception:
                pass
        self._collections.clear()
        self._release_lock()
        self._closed = True

    # ── context manager ─────────────────────────────────────────

    def __enter__(self) -> "MilvusLite":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── properties ──────────────────────────────────────────────

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def closed(self) -> bool:
        return self._closed

    # ── internals ───────────────────────────────────────────────

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("MilvusLite is closed")

    def _collections_root(self) -> str:
        return os.path.join(self._data_dir, COLLECTIONS_DIRNAME)

    def _collection_dir(self, name: str) -> str:
        return os.path.join(self._collections_root(), name)

    def _aliases_path(self) -> str:
        return os.path.join(self._data_dir, ALIASES_FILENAME)

    def _load_aliases(self) -> Dict[str, str]:
        path = self._aliases_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {
            str(alias): str(target)
            for alias, target in data.items()
            if isinstance(alias, str) and isinstance(target, str)
        }

    def _save_aliases(self) -> None:
        path = self._aliases_path()
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._aliases, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"collection name must be a string, got {type(name).__name__}")
        if not name:
            raise ValueError("collection name must not be empty")
        # Forbid path separators and dot segments — names map directly to
        # filesystem paths, so leakage outside collections/ is unsafe.
        if "/" in name or "\\" in name or name in (".", "..") or "\x00" in name:
            raise ValueError(f"invalid collection name: {name!r}")
        if len(name.encode("utf-8")) > 255:
            raise ValueError(
                f"collection name too long ({len(name.encode('utf-8'))} bytes, max 255)"
            )

    def _acquire_lock(self) -> None:
        """Acquire an exclusive non-blocking lock on LOCK_FILENAME.

        Uses fcntl.flock on Unix and msvcrt.locking on Windows.
        Raises DataDirLockedError if another process holds it.
        """
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as e:
            os.close(fd)
            raise DataDirLockedError(
                f"another process holds the lock on {self._data_dir!r}: {e}"
            ) from e
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._lock_fd)
        except OSError:
            pass
        self._lock_fd = None
