"""MilvusLiteError → Milvus ErrorCode translation.

Maps the MilvusLite exception hierarchy onto Milvus's TWO concurrent
error code systems:

    - status.error_code (legacy enum, from common.proto's ErrorCode):
        UnexpectedError=1, CollectionNotExists=4, IllegalArgument=6,
        IndexNotExist=11, IndexBuildFailed=26, IndexAlreadyExists=35,
        CollectionNotLoaded=101, PartitionNotExists=200, ...
        Used by pymilvus 2.3 / 2.4 client code paths.

    - status.code (new int, post 2.3.2):
        SUCCESS=0, UNEXPECTED_ERROR=1,
        COLLECTION_NOT_FOUND=100, INDEX_NOT_FOUND=700, ...
        Used by newer pymilvus paths and the milvus-lite Server.

We set BOTH fields on every Status response so old AND new pymilvus
client code paths recognize the error correctly. This is critical
for ``has_collection`` (which checks legacy error_code first) and
``describe_collection`` (which is used internally by has_collection).

Unknown exception types map to UnexpectedError(1, 1) so the caller
still sees something rather than crashing.
"""

from __future__ import annotations

from typing import Type

from milvus_lite.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    CollectionNotLoadedError,
    DefaultPartitionError,
    IndexAlreadyExistsError,
    IndexBackendUnavailableError,
    IndexNotFoundError,
    MilvusLiteError,
    PartitionAlreadyExistsError,
    PartitionNotFoundError,
    SchemaValidationError,
)
from milvus_lite.search.filter.exceptions import (
    FilterFieldError,
    FilterParseError,
    FilterTypeError,
)


# ── Legacy ErrorCode (status.error_code) ────────────────────────────
# From pymilvus.grpc_gen.common_pb2.ErrorCode enum.

SUCCESS = 0
LEGACY_UNEXPECTED_ERROR = 1
LEGACY_COLLECTION_NOT_EXISTS = 4
LEGACY_ILLEGAL_ARGUMENT = 6
LEGACY_INDEX_NOT_EXIST = 11
LEGACY_INDEX_BUILD_FAILED = 26
LEGACY_INDEX_ALREADY_EXISTS = 35
LEGACY_COLLECTION_NOT_LOADED = 101
LEGACY_PARTITION_NOT_EXISTS = 200


# ── New code (status.code) ──────────────────────────────────────────
# From pymilvus.exceptions.ErrorCode + the milvus-proto 2.x extended
# range. Many overlap with the legacy values; the ones that don't are
# called out below.

CODE_SUCCESS = 0
CODE_UNEXPECTED_ERROR = 1
CODE_COLLECTION_NOT_FOUND = 100   # NOT same as legacy 4
CODE_INDEX_NOT_FOUND = 700        # NOT same as legacy 11


# Map: exception type → (legacy_error_code, new_code, label).
# When a code doesn't have a distinct "new" form yet, fall back to
# the legacy value — most pymilvus paths still inspect error_code.
_EXCEPTION_TO_CODE: dict[Type[Exception], tuple[int, int, str]] = {
    CollectionNotFoundError:        (LEGACY_COLLECTION_NOT_EXISTS,    CODE_COLLECTION_NOT_FOUND, "CollectionNotExists"),
    CollectionAlreadyExistsError:   (LEGACY_UNEXPECTED_ERROR,         CODE_UNEXPECTED_ERROR,     "CollectionAlreadyExists"),
    PartitionNotFoundError:         (LEGACY_PARTITION_NOT_EXISTS,     LEGACY_PARTITION_NOT_EXISTS, "PartitionNotExists"),
    PartitionAlreadyExistsError:    (LEGACY_UNEXPECTED_ERROR,         CODE_UNEXPECTED_ERROR,     "PartitionAlreadyExists"),
    DefaultPartitionError:          (LEGACY_ILLEGAL_ARGUMENT,         LEGACY_ILLEGAL_ARGUMENT,   "DefaultPartitionError"),
    SchemaValidationError:          (LEGACY_ILLEGAL_ARGUMENT,         LEGACY_ILLEGAL_ARGUMENT,   "IllegalArgument"),
    FilterParseError:               (LEGACY_ILLEGAL_ARGUMENT,         LEGACY_ILLEGAL_ARGUMENT,   "IllegalArgument"),
    FilterTypeError:                (LEGACY_ILLEGAL_ARGUMENT,         LEGACY_ILLEGAL_ARGUMENT,   "IllegalArgument"),
    FilterFieldError:               (LEGACY_ILLEGAL_ARGUMENT,         LEGACY_ILLEGAL_ARGUMENT,   "IllegalArgument"),
    CollectionNotLoadedError:       (LEGACY_COLLECTION_NOT_LOADED,    LEGACY_COLLECTION_NOT_LOADED, "CollectionNotLoaded"),
    IndexAlreadyExistsError:        (LEGACY_INDEX_ALREADY_EXISTS,     LEGACY_INDEX_ALREADY_EXISTS,  "IndexAlreadyExists"),
    IndexNotFoundError:             (LEGACY_INDEX_NOT_EXIST,          CODE_INDEX_NOT_FOUND,         "IndexNotExist"),
    IndexBackendUnavailableError:   (LEGACY_INDEX_BUILD_FAILED,       LEGACY_INDEX_BUILD_FAILED,    "IndexBuildFailed"),
}


# ── Public API used by the servicer ─────────────────────────────────
# UNEXPECTED_ERROR is exported under its short name for back-compat
# with the Phase 10.1 servicer skeleton.
UNEXPECTED_ERROR = LEGACY_UNEXPECTED_ERROR


def to_status_kwargs(exc: BaseException) -> dict:
    """Translate an exception into kwargs for ``common_pb2.Status(...)``.

    Returns a dict with all four fields pymilvus may inspect:
        error_code (legacy enum, used by pymilvus 2.4-)
        code       (new int, used by pymilvus 2.5+)
        reason     (human-readable message — exception's str())

    Setting BOTH error_code AND code is required for compatibility.
    pymilvus's has_collection, for instance, only consults the
    legacy error_code field on the response from DescribeCollection.
    """
    legacy, new_code, _label = _EXCEPTION_TO_CODE.get(
        type(exc), (LEGACY_UNEXPECTED_ERROR, CODE_UNEXPECTED_ERROR, "UnexpectedError"),
    )
    return {
        "error_code": legacy,
        "code": new_code,
        "reason": str(exc),
    }


def success_status_kwargs() -> dict:
    """kwargs for a successful Status. error_code AND code are both 0."""
    return {"error_code": SUCCESS, "code": CODE_SUCCESS, "reason": ""}
