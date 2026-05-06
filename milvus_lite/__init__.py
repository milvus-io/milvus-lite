"""MilvusLite — local embedded vector database.

Public API:

    from milvus_lite import MilvusLite, CollectionSchema, FieldSchema, DataType

    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128),
    ])

    with MilvusLite("/path/to/data") as db:
        col = db.create_collection("docs", schema)
        col.insert([{"id": "doc1", "vec": [...]}, ...])
        results = col.search([[query_vector]], top_k=10)
"""

from milvus_lite.db import MilvusLite
from milvus_lite.engine.collection import Collection
from milvus_lite.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    CollectionNotLoadedError,
    DataDirLockedError,
    DefaultPartitionError,
    IndexAlreadyExistsError,
    IndexBackendUnavailableError,
    IndexNotFoundError,
    MilvusLiteError,
    ManifestCorruptedError,
    PartitionAlreadyExistsError,
    PartitionNotFoundError,
    SchemaValidationError,
    WALCorruptedError,
)
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.schema.types import Function, FunctionType
from milvus_lite.search.filter import (
    FilterError,
    FilterFieldError,
    FilterParseError,
    FilterTypeError,
)
from milvus_lite.server_manager import server_manager_instance
from milvus_lite._version import get_version

__version__ = get_version()

__all__ = [
    "__version__",
    # Top-level entry point
    "MilvusLite",
    # Collection (returned from db.get_collection / db.create_collection)
    "Collection",
    # Schema types
    "CollectionSchema",
    "FieldSchema",
    "DataType",
    # Exception hierarchy
    "MilvusLiteError",
    "SchemaValidationError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "PartitionNotFoundError",
    "PartitionAlreadyExistsError",
    "DefaultPartitionError",
    "WALCorruptedError",
    "ManifestCorruptedError",
    "DataDirLockedError",
    # Phase 9 index exceptions
    "CollectionNotLoadedError",
    "IndexAlreadyExistsError",
    "IndexNotFoundError",
    "IndexBackendUnavailableError",
    # Filter expression errors (Phase 8)
    "FilterError",
    "FilterParseError",
    "FilterFieldError",
    "FilterTypeError",
    # pymilvus integration
    "server_manager_instance",
]
