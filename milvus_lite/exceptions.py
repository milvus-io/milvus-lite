class MilvusLiteError(Exception):
    """Base exception for MilvusLite."""


class SchemaValidationError(MilvusLiteError):
    """Schema definition or record validation failed."""


class CollectionNotFoundError(MilvusLiteError):
    """Collection does not exist."""


class CollectionAlreadyExistsError(MilvusLiteError):
    """Collection already exists."""


class PartitionNotFoundError(MilvusLiteError):
    """Partition does not exist."""


class PartitionAlreadyExistsError(MilvusLiteError):
    """Partition already exists."""


class DefaultPartitionError(MilvusLiteError):
    """Illegal operation on the default partition."""


class WALCorruptedError(MilvusLiteError):
    """WAL file is corrupted and cannot be fully recovered."""


class ManifestCorruptedError(MilvusLiteError):
    """Both manifest.json and manifest.json.prev failed to load."""


class DataDirLockedError(MilvusLiteError):
    """Another process holds the data_dir LOCK file."""


# ── Phase 9.3 — index lifecycle ─────────────────────────────────────


class CollectionNotLoadedError(MilvusLiteError):
    """search / get / query was called on a Collection that is not in
    the 'loaded' state. Mirrors Milvus behavior — the user must call
    Collection.load() (or pymilvus client.load_collection) first.
    Caused by either having never called load(), or having explicitly
    called release()."""


class IndexAlreadyExistsError(MilvusLiteError):
    """create_index was called on a Collection that already has an
    index. Drop the existing one with drop_index first."""


class IndexNotFoundError(MilvusLiteError):
    """drop_index / describe_index was called for a field that has no
    index attached."""


class IndexBackendUnavailableError(MilvusLiteError):
    """Requested index_type requires an optional dependency that is
    not installed (e.g. faiss-cpu for HNSW)."""
