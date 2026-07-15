"""DB layer — multi-Collection lifecycle management.

MilvusLite is the top-level entry point. It manages multiple Collections
under one ``data_dir`` and ensures only one process can hold the dir at
a time via an advisory file lock.

Layout:

    data_dir/
    ├── LOCK                      # advisory flock — held while DB is open
    ├── collections/              # default database collections (legacy layout)
    ├── aliases.json              # default database aliases
    ├── database_properties.json  # default database properties
    ├── .database-staging/        # incomplete DB creations, cleaned on startup
    └── databases/                # non-default databases
        ├── db_a/
        │   ├── collections/
        │   ├── aliases.json
        │   └── database_properties.json
        └── db_b/
            └── ...

Collection instances are cached so that calling ``get_collection`` twice
with the same database/name returns the same object (and the same in-memory
MemTable / WAL state).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional

from milvus_lite.engine.collection import Collection
from milvus_lite.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DatabaseAlreadyExistsError,
    DatabaseNotEmptyError,
    DatabaseNotFoundError,
    DataDirLockedError,
)
from milvus_lite.schema.persistence import load_schema, save_schema
from milvus_lite.schema.types import CollectionSchema
from milvus_lite.schema.validation import validate_schema
from milvus_lite.schema.timestamptz import validate_timezone_name
from milvus_lite.storage.manifest import MANIFEST_FILENAME, Manifest
from milvus_lite.storage.snapshots import (
    collect_index_files,
    create_snapshot as create_snapshot_metadata,
    drop_snapshot as drop_snapshot_metadata,
    list_snapshots as list_snapshot_metadata,
    load_snapshot,
)


COLLECTIONS_DIRNAME = "collections"
DATABASES_DIRNAME = "databases"
DATABASE_STAGING_DIRNAME = ".database-staging"
LOCK_FILENAME = "LOCK"
SCHEMA_FILENAME = "schema.json"
ALIASES_FILENAME = "aliases.json"
DATABASE_PROPERTIES_FILENAME = "database_properties.json"
DEFAULT_DATABASE_NAME = "default"


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
        os.makedirs(self._collections_root(DEFAULT_DATABASE_NAME), exist_ok=True)
        os.makedirs(self._databases_root(), exist_ok=True)

        self._lock_path = os.path.join(data_dir, LOCK_FILENAME)
        self._lock_fd: Optional[int] = None
        self._acquire_lock()
        os.makedirs(self._database_staging_root(), exist_ok=True)
        self._cleanup_database_staging()

        self._collections: Dict[tuple[str, str], Collection] = {}
        self._aliases_by_db: Dict[str, Dict[str, str]] = {}
        self._database_properties_by_db: Dict[str, Dict[str, Any]] = {}
        self._closed = False

    # ── public API ──────────────────────────────────────────────

    def create_database(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._check_open()
        name = self._normalize_database_name(name)
        self._validate_database_name(name)
        if self.has_database(name):
            raise DatabaseAlreadyExistsError(f"database {name!r} already exists")

        props = self._normalize_database_properties(properties or {})
        db_dir = self._database_dir(name)
        staging_dir = tempfile.mkdtemp(
            prefix="database-",
            dir=self._database_staging_root(),
        )
        try:
            os.makedirs(os.path.join(staging_dir, COLLECTIONS_DIRNAME))
            self._write_database_properties_file(
                os.path.join(staging_dir, DATABASE_PROPERTIES_FILENAME),
                props,
            )
            os.rename(staging_dir, db_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        self._database_properties_by_db[name] = props

    def drop_database(self, name: str) -> None:
        self._check_open()
        name = self._normalize_database_name(name)
        self._validate_database_name(name)
        if name == DEFAULT_DATABASE_NAME:
            raise ValueError("cannot drop default database")
        if not self.has_database(name):
            raise DatabaseNotFoundError(f"database {name!r} does not exist")
        if self._database_has_collections(name):
            raise DatabaseNotEmptyError(f"database {name!r} is not empty")

        for key in [key for key in self._collections if key[0] == name]:
            self._collections[key].close()
            del self._collections[key]
        shutil.rmtree(self._database_dir(name), ignore_errors=False)
        self._aliases_by_db.pop(name, None)
        self._database_properties_by_db.pop(name, None)

    def has_database(self, name: str) -> bool:
        name = self._normalize_database_name(name)
        self._validate_database_name(name)
        if name == DEFAULT_DATABASE_NAME:
            return True
        return os.path.isdir(self._database_dir(name))

    def list_databases(self) -> List[str]:
        self._check_open()
        names = [DEFAULT_DATABASE_NAME]
        root = self._databases_root()
        if os.path.exists(root):
            for entry in os.listdir(root):
                sub = os.path.join(root, entry)
                if os.path.isdir(sub):
                    names.append(entry)
        return sorted(names)

    def create_collection(
        self,
        name: str,
        schema: CollectionSchema,
        properties: Optional[Dict[str, Any]] = None,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> Collection:
        """Create a new Collection. Raises if a Collection with this
        name already exists."""
        self._check_open()
        database_name = self._require_database(database_name)
        self._validate_name(name)

        if properties:
            merged = dict(schema.properties)
            merged.update(properties)
            schema.properties = merged

        db_properties = self._database_properties(database_name)
        validate_schema(schema, default_properties=db_properties)

        if self.has_collection(name, database_name=database_name):
            raise CollectionAlreadyExistsError(
                f"collection {name!r} already exists"
            )

        col_dir = self._collection_dir(name, database_name)
        os.makedirs(col_dir, exist_ok=False)

        save_schema(schema, name, os.path.join(col_dir, SCHEMA_FILENAME))

        try:
            col = Collection(
                name,
                col_dir,
                schema,
                database_properties=db_properties,
            )
        except Exception:
            shutil.rmtree(col_dir, ignore_errors=True)
            raise
        self._collections[(database_name, name)] = col
        return col

    def get_collection(
        self,
        name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> Collection:
        """Open an existing Collection. Subsequent calls return the same
        cached instance."""
        self._check_open()
        database_name = self._require_database(database_name)
        name = self.resolve_collection_name(name, database_name=database_name)
        key = (database_name, name)
        if key in self._collections:
            return self._collections[key]
        if not self.has_collection(name, database_name=database_name):
            raise CollectionNotFoundError(
                f"collection {name!r} does not exist"
            )

        col_dir = self._collection_dir(name, database_name)
        _name, schema = load_schema(os.path.join(col_dir, SCHEMA_FILENAME))
        col = Collection(
            name,
            col_dir,
            schema,
            database_properties=self._database_properties(database_name),
        )
        self._collections[key] = col
        return col

    def drop_collection(
        self,
        name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Close and delete a Collection. No-op if it does not exist."""
        self._check_open()
        database_name = self._require_database(database_name)
        name = self.resolve_collection_name(name, database_name=database_name)
        if not self.has_collection(name, database_name=database_name):
            return

        key = (database_name, name)
        if key in self._collections:
            self._collections[key].close()

        col_dir = self._collection_dir(name, database_name)
        shutil.rmtree(col_dir, ignore_errors=False)
        self._collections.pop(key, None)
        aliases = self._aliases(database_name)
        removed = [alias for alias, target in aliases.items() if target == name]
        for alias in removed:
            del aliases[alias]
        if removed:
            self._save_aliases(database_name)

    def rename_collection(
        self,
        old_name: str,
        new_name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Rename a collection on disk and in the cache."""
        self._check_open()
        database_name = self._require_database(database_name)
        old_name = self.resolve_collection_name(old_name, database_name=database_name)
        if not self.has_collection(old_name, database_name=database_name):
            raise CollectionNotFoundError(
                f"collection {old_name!r} does not exist"
            )
        self._validate_name(new_name)
        if self.has_collection(new_name, database_name=database_name):
            raise CollectionAlreadyExistsError(
                f"collection {new_name!r} already exists"
            )

        old_key = (database_name, old_name)
        if old_key in self._collections:
            self._collections[old_key].close()
            del self._collections[old_key]

        old_dir = self._collection_dir(old_name, database_name)
        new_dir = self._collection_dir(new_name, database_name)
        os.rename(old_dir, new_dir)

        schema_path = os.path.join(new_dir, SCHEMA_FILENAME)
        _name, schema = load_schema(schema_path)
        save_schema(schema, new_name, schema_path)

        aliases = self._aliases(database_name)
        for alias, target in list(aliases.items()):
            if target == old_name:
                aliases[alias] = new_name
        self._save_aliases(database_name)

    def has_collection(
        self,
        name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> bool:
        """True iff a Collection with this name exists on disk."""
        database_name = self._require_database(database_name)
        name = self.resolve_collection_name(name, database_name=database_name)
        return os.path.exists(
            os.path.join(self._collection_dir(name, database_name), SCHEMA_FILENAME)
        )

    def list_collections(
        self,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> List[str]:
        """Return all Collection names in one database, sorted."""
        database_name = self._require_database(database_name)
        root = self._collections_root(database_name)
        if not os.path.exists(root):
            return []
        names: List[str] = []
        for entry in os.listdir(root):
            sub = os.path.join(root, entry)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, SCHEMA_FILENAME)):
                names.append(entry)
        return sorted(names)

    def create_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
        description: str = "",
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> Dict[str, Any]:
        """Create an immutable manifest snapshot for a collection."""
        self._check_open()
        database_name = self._require_database(database_name)
        collection_name = self.resolve_collection_name(
            collection_name, database_name=database_name
        )
        col = self.get_collection(collection_name, database_name=database_name)
        col.flush()
        col._wait_for_bg()

        col_dir = self._collection_dir(collection_name, database_name)
        with col._maintenance_lock:
            manifest_path = os.path.join(col_dir, MANIFEST_FILENAME)
            if not os.path.exists(manifest_path):
                col._manifest.save()
            data_files = col._manifest.get_all_data_files()
            delta_files = col._manifest.get_all_delta_files()
            index_files = collect_index_files(col_dir, data_files)
            return create_snapshot_metadata(
                collection_dir=col_dir,
                collection_name=collection_name,
                snapshot_name=snapshot_name,
                description=description,
                schema_filename=SCHEMA_FILENAME,
                manifest_filename=MANIFEST_FILENAME,
                data_files=data_files,
                delta_files=delta_files,
                index_files=index_files,
                current_seq=col._manifest.current_seq,
                manifest_version=col._manifest.version,
            )

    def list_snapshots(
        self,
        collection_name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> List[Dict[str, Any]]:
        """List snapshots for a collection."""
        self._check_open()
        database_name = self._require_database(database_name)
        collection_name = self.resolve_collection_name(
            collection_name, database_name=database_name
        )
        if not self.has_collection(collection_name, database_name=database_name):
            raise CollectionNotFoundError(f"collection {collection_name!r} does not exist")
        return list_snapshot_metadata(self._collection_dir(collection_name, database_name))

    def drop_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Drop snapshot metadata and release its pinned files."""
        self._check_open()
        database_name = self._require_database(database_name)
        collection_name = self.resolve_collection_name(
            collection_name, database_name=database_name
        )
        if not self.has_collection(collection_name, database_name=database_name):
            raise CollectionNotFoundError(f"collection {collection_name!r} does not exist")
        drop_snapshot_metadata(self._collection_dir(collection_name, database_name), snapshot_name)

    def restore_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
        new_collection_name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> Collection:
        """Restore a snapshot into a new collection in the same database."""
        self._check_open()
        database_name = self._require_database(database_name)
        collection_name = self.resolve_collection_name(
            collection_name, database_name=database_name
        )
        self._validate_name(new_collection_name)
        if not self.has_collection(collection_name, database_name=database_name):
            raise CollectionNotFoundError(f"collection {collection_name!r} does not exist")
        if self.has_collection(new_collection_name, database_name=database_name):
            raise CollectionAlreadyExistsError(
                f"collection {new_collection_name!r} already exists"
            )

        src_dir = self._collection_dir(collection_name, database_name)
        dst_dir = self._collection_dir(new_collection_name, database_name)
        snap = load_snapshot(src_dir, snapshot_name)
        os.makedirs(dst_dir, exist_ok=False)
        try:
            schema_src = os.path.join(src_dir, snap["schema_file"])
            manifest_src = os.path.join(src_dir, snap["manifest_file"])
            shutil.copy2(schema_src, os.path.join(dst_dir, SCHEMA_FILENAME))
            shutil.copy2(manifest_src, os.path.join(dst_dir, MANIFEST_FILENAME))
            restored_manifest = Manifest.load(dst_dir)
            restored_manifest.active_wal_number = None
            restored_manifest.save()

            for key in ("data_files", "delta_files", "index_files"):
                file_map = snap.get(key, {})
                if not isinstance(file_map, dict):
                    continue
                for partition, rels in file_map.items():
                    if not isinstance(partition, str) or not isinstance(rels, list):
                        continue
                    for rel in rels:
                        if not isinstance(rel, str):
                            continue
                        src = os.path.join(src_dir, "partitions", partition, rel)
                        dst = os.path.join(dst_dir, "partitions", partition, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)

            _old_name, schema = load_schema(os.path.join(dst_dir, SCHEMA_FILENAME))
            save_schema(schema, new_collection_name, os.path.join(dst_dir, SCHEMA_FILENAME))
            col = Collection(
                new_collection_name,
                dst_dir,
                schema,
                database_properties=self._database_properties(database_name),
            )
        except Exception:
            shutil.rmtree(dst_dir, ignore_errors=True)
            raise
        self._collections[(database_name, new_collection_name)] = col
        return col

    def get_collection_stats(
        self,
        name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> Dict[str, Any]:
        col = self.get_collection(name, database_name=database_name)
        return {"row_count": col.num_entities}

    def get_partition_stats(
        self,
        collection_name: str,
        partition_name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> Dict[str, Any]:
        """Return basic stats for one partition."""
        col = self.get_collection(collection_name, database_name=database_name)
        return {"row_count": col.partition_num_entities(partition_name)}

    def truncate_collection(
        self,
        name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Delete all data in a collection while preserving schema and aliases."""
        self._check_open()
        database_name = self._require_database(database_name)
        name = self.resolve_collection_name(name, database_name=database_name)
        if not self.has_collection(name, database_name=database_name):
            raise CollectionNotFoundError(f"collection {name!r} does not exist")

        col_dir = self._collection_dir(name, database_name)
        _schema_name, schema = load_schema(os.path.join(col_dir, SCHEMA_FILENAME))
        key = (database_name, name)
        if key in self._collections:
            self._collections[key].close()
            del self._collections[key]

        shutil.rmtree(col_dir, ignore_errors=False)
        os.makedirs(col_dir, exist_ok=False)
        save_schema(schema, name, os.path.join(col_dir, SCHEMA_FILENAME))

    def alter_collection_properties(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        delete_keys: Optional[List[str]] = None,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Update mutable collection properties and persist schema.json."""
        self._check_open()
        database_name = self._require_database(database_name)
        name = self.resolve_collection_name(name, database_name=database_name)
        if not self.has_collection(name, database_name=database_name):
            raise CollectionNotFoundError(f"collection {name!r} does not exist")

        col = self.get_collection(name, database_name=database_name)
        col.alter_properties(properties=properties, delete_keys=delete_keys)
        save_schema(
            col.schema,
            name,
            os.path.join(self._collection_dir(name, database_name), SCHEMA_FILENAME),
        )

    def describe_database(self, name: str = DEFAULT_DATABASE_NAME) -> Dict[str, Any]:
        """Return metadata for a database."""
        self._check_open()
        name = self._require_database(name)
        return {
            "name": name,
            "properties": dict(self._database_properties(name)),
        }

    def alter_database_properties(
        self,
        name: str = DEFAULT_DATABASE_NAME,
        properties: Optional[Dict[str, Any]] = None,
        delete_keys: Optional[List[str]] = None,
    ) -> None:
        """Update database-level default properties."""
        self._check_open()
        name = self._require_database(name)

        props = self._database_properties(name)
        next_props = dict(props)
        for key in delete_keys or []:
            next_props.pop(str(key), None)
        for key, value in (properties or {}).items():
            next_props[str(key)] = value
        next_props = self._normalize_database_properties(next_props)
        if next_props == props:
            return

        old_timezone = props.get("timezone")
        self._persist_database_properties(name, next_props)
        props.clear()
        props.update(next_props)
        if old_timezone != props.get("timezone"):
            for (db_name, _), col in self._collections.items():
                if db_name == name:
                    col._filter_cache.clear()  # noqa: SLF001

    def drop_database_properties(
        self,
        name: str = DEFAULT_DATABASE_NAME,
        property_keys: Optional[List[str]] = None,
    ) -> None:
        """Delete database-level properties."""
        self.alter_database_properties(name=name, delete_keys=property_keys or [])

    def create_alias(
        self,
        collection_name: str,
        alias: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Create a collection alias."""
        self._check_open()
        database_name = self._require_database(database_name)
        collection_name = self.resolve_collection_name(
            collection_name, database_name=database_name
        )
        self._validate_name(alias)
        if not self.has_collection(collection_name, database_name=database_name):
            raise CollectionNotFoundError(
                f"collection {collection_name!r} does not exist"
            )
        if os.path.exists(os.path.join(self._collection_dir(alias, database_name), SCHEMA_FILENAME)):
            raise CollectionAlreadyExistsError(
                f"alias {alias!r} conflicts with an existing collection"
            )
        aliases = self._aliases(database_name)
        if alias in aliases:
            raise CollectionAlreadyExistsError(f"alias {alias!r} already exists")
        aliases[alias] = collection_name
        self._save_aliases(database_name)

    def alter_alias(
        self,
        collection_name: str,
        alias: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Point an existing alias at another collection."""
        self._check_open()
        database_name = self._require_database(database_name)
        collection_name = self.resolve_collection_name(
            collection_name, database_name=database_name
        )
        aliases = self._aliases(database_name)
        if alias not in aliases:
            raise CollectionNotFoundError(f"alias {alias!r} does not exist")
        if not self.has_collection(collection_name, database_name=database_name):
            raise CollectionNotFoundError(
                f"collection {collection_name!r} does not exist"
            )
        aliases[alias] = collection_name
        self._save_aliases(database_name)

    def drop_alias(
        self,
        alias: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Drop an alias. No-op if it does not exist, matching drop_collection."""
        self._check_open()
        database_name = self._require_database(database_name)
        aliases = self._aliases(database_name)
        if alias in aliases:
            del aliases[alias]
            self._save_aliases(database_name)

    def describe_alias(
        self,
        alias: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> Dict[str, str]:
        """Return alias metadata."""
        self._check_open()
        database_name = self._require_database(database_name)
        aliases = self._aliases(database_name)
        if alias not in aliases:
            raise CollectionNotFoundError(f"alias {alias!r} does not exist")
        return {"alias": alias, "collection": aliases[alias]}

    def list_aliases(
        self,
        collection_name: Optional[str] = None,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> List[str]:
        """Return aliases, optionally filtered by target collection."""
        self._check_open()
        database_name = self._require_database(database_name)
        aliases = self._aliases(database_name)
        if collection_name:
            collection_name = self.resolve_collection_name(
                collection_name, database_name=database_name
            )
            return sorted(
                alias for alias, target in aliases.items()
                if target == collection_name
            )
        return sorted(aliases)

    def resolve_collection_name(
        self,
        name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> str:
        """Resolve a collection alias to its target collection name."""
        database_name = self._normalize_database_name(database_name)
        return self._aliases(database_name).get(name, name)

    def close(self) -> None:
        """Close every cached Collection and release the LOCK.

        Idempotent. After close(), this DB instance is unusable; create
        a new one to reopen the data_dir.
        """
        if self._closed:
            return
        for key in list(self._collections.keys()):
            try:
                self._collections[key].close()
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

    def _require_database(self, name: Optional[str]) -> str:
        name = self._normalize_database_name(name)
        self._validate_database_name(name)
        if not self.has_database(name):
            raise DatabaseNotFoundError(f"database {name!r} does not exist")
        return name

    @staticmethod
    def _normalize_database_name(name: Optional[str]) -> str:
        return name or DEFAULT_DATABASE_NAME

    def _databases_root(self) -> str:
        return os.path.join(self._data_dir, DATABASES_DIRNAME)

    def _database_staging_root(self) -> str:
        return os.path.join(self._data_dir, DATABASE_STAGING_DIRNAME)

    def _cleanup_database_staging(self) -> None:
        root = self._database_staging_root()
        for entry in os.listdir(root):
            path = os.path.join(root, entry)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError:
                pass

    def _database_dir(self, database_name: str) -> str:
        if database_name == DEFAULT_DATABASE_NAME:
            return self._data_dir
        return os.path.join(self._databases_root(), database_name)

    def _collections_root(self, database_name: str = DEFAULT_DATABASE_NAME) -> str:
        return os.path.join(self._database_dir(database_name), COLLECTIONS_DIRNAME)

    def _database_has_collections(self, database_name: str) -> bool:
        root = self._collections_root(database_name)
        if not os.path.exists(root):
            return False
        for entry in os.listdir(root):
            sub = os.path.join(root, entry)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, SCHEMA_FILENAME)):
                return True
        return False

    def _collection_dir(
        self,
        name: str,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> str:
        return os.path.join(self._collections_root(database_name), name)

    def _aliases_path(self, database_name: str = DEFAULT_DATABASE_NAME) -> str:
        return os.path.join(self._database_dir(database_name), ALIASES_FILENAME)

    def _database_properties_path(
        self,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> str:
        return os.path.join(self._database_dir(database_name), DATABASE_PROPERTIES_FILENAME)

    def _aliases(self, database_name: str) -> Dict[str, str]:
        database_name = self._normalize_database_name(database_name)
        if database_name not in self._aliases_by_db:
            self._aliases_by_db[database_name] = self._load_aliases(database_name)
        return self._aliases_by_db[database_name]

    def _load_aliases(self, database_name: str = DEFAULT_DATABASE_NAME) -> Dict[str, str]:
        path = self._aliases_path(database_name)
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

    def _save_aliases(self, database_name: str = DEFAULT_DATABASE_NAME) -> None:
        path = self._aliases_path(database_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                self._aliases(database_name),
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        os.replace(tmp_path, path)

    def _database_properties(self, database_name: str) -> Dict[str, Any]:
        database_name = self._normalize_database_name(database_name)
        if database_name not in self._database_properties_by_db:
            self._database_properties_by_db[database_name] = (
                self._load_database_properties(database_name)
            )
        return self._database_properties_by_db[database_name]

    def _load_database_properties(
        self,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> Dict[str, Any]:
        path = self._database_properties_path(database_name)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return self._normalize_database_properties(data)

    def _save_database_properties(
        self,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        self._persist_database_properties(
            database_name,
            self._database_properties(database_name),
        )

    def _persist_database_properties(
        self,
        database_name: str,
        properties: Dict[str, Any],
    ) -> None:
        path = self._database_properties_path(database_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._write_database_properties_file(path, properties)

    @staticmethod
    def _write_database_properties_file(
        path: str,
        properties: Dict[str, Any],
    ) -> None:
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    properties,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _normalize_database_properties(values: Dict[str, Any]) -> Dict[str, Any]:
        out = {str(key): value for key, value in values.items()}
        if "timezone" in out:
            out["timezone"] = validate_timezone_name(out["timezone"])
        return out

    @classmethod
    def _validate_database_name(cls, name: str) -> None:
        cls._validate_filesystem_name(name, "database")

    @classmethod
    def _validate_name(cls, name: str) -> None:
        cls._validate_filesystem_name(name, "collection")

    @staticmethod
    def _validate_filesystem_name(name: str, kind: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"{kind} name must be a string, got {type(name).__name__}")
        if not name:
            raise ValueError(f"{kind} name must not be empty")
        if "/" in name or "\\" in name or name in (".", "..") or "\x00" in name:
            raise ValueError(f"invalid {kind} name: {name!r}")
        if len(name.encode("utf-8")) > 255:
            raise ValueError(
                f"{kind} name too long ({len(name.encode('utf-8'))} bytes, max 255)"
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
