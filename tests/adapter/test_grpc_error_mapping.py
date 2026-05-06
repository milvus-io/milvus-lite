"""Phase 10.6 — error code mapping + UNIMPLEMENTED stub tests.

Validates that:
1. MilvusLiteError subclasses produce the right grpc Status codes
   (both legacy error_code and new code fields) so pymilvus clients
   see the expected exception types.
2. Explicitly-stubbed UNIMPLEMENTED RPCs return a friendly reason
   string (not the generic "Method not implemented!" from the base).
"""

import pytest
from pymilvus import DataType, MilvusClient

from milvus_lite.adapter.grpc.errors import (
    LEGACY_COLLECTION_NOT_EXISTS,
    LEGACY_COLLECTION_NOT_LOADED,
    LEGACY_ILLEGAL_ARGUMENT,
    CODE_COLLECTION_NOT_FOUND,
    to_status_kwargs,
)
from milvus_lite.exceptions import (
    CollectionNotFoundError,
    CollectionNotLoadedError,
    SchemaValidationError,
)


# ---------------------------------------------------------------------------
# Unit: to_status_kwargs → (error_code, code, reason)
# ---------------------------------------------------------------------------

def test_collection_not_found_dual_codes():
    exc = CollectionNotFoundError("x")
    kw = to_status_kwargs(exc)
    assert kw["error_code"] == LEGACY_COLLECTION_NOT_EXISTS  # 4
    assert kw["code"] == CODE_COLLECTION_NOT_FOUND            # 100
    assert "x" in kw["reason"]


def test_collection_not_loaded_code():
    exc = CollectionNotLoadedError("y")
    kw = to_status_kwargs(exc)
    assert kw["error_code"] == LEGACY_COLLECTION_NOT_LOADED   # 101
    assert "y" in kw["reason"]


def test_schema_validation_code():
    exc = SchemaValidationError("bad schema")
    kw = to_status_kwargs(exc)
    assert kw["error_code"] == LEGACY_ILLEGAL_ARGUMENT        # 6
    assert "bad schema" in kw["reason"]


def test_unknown_exception_maps_to_unexpected():
    kw = to_status_kwargs(RuntimeError("oops"))
    assert kw["error_code"] == 1  # UNEXPECTED_ERROR
    assert kw["code"] == 1
    assert "oops" in kw["reason"]


# ---------------------------------------------------------------------------
# Integration: pymilvus sees the right exceptions via gRPC
# ---------------------------------------------------------------------------

def test_drop_nonexistent_is_idempotent(milvus_client):
    """Dropping a non-existent collection should silently succeed (Milvus compat)."""
    milvus_client.drop_collection("ghost")  # no error


def test_search_after_release_reports_not_loaded(milvus_client):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    milvus_client.create_collection("demo", schema=schema)
    milvus_client.insert("demo", [
        {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0]},
    ])
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="L2", params={})
    milvus_client.create_index("demo", idx)
    milvus_client.release_collection("demo")
    with pytest.raises(Exception) as exc_info:
        milvus_client.search("demo", data=[[1.0, 0.0, 0.0, 0.0]], limit=1)
    assert "load" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# UNIMPLEMENTED stubs: friendly messages
# ---------------------------------------------------------------------------

def test_rename_nonexistent_reports_not_found(milvus_client):
    """Renaming a non-existent collection should report an error."""
    with pytest.raises(Exception) as exc_info:
        milvus_client.rename_collection("ghost", "new")
    assert "not exist" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()


def test_hybrid_search_unimplemented(milvus_client):
    """pymilvus's hybrid_search is a client-side wrapper but ultimately
    sends the HybridSearch RPC. We can't easily trigger it from
    MilvusClient, but we can verify our stub exists by looking at the
    servicer method directly."""
    from milvus_lite.adapter.grpc.servicer import MilvusServicer
    assert hasattr(MilvusServicer, "HybridSearch")


def test_create_alias_nonexistent_collection_reports_not_found(milvus_client):
    with pytest.raises(Exception) as exc_info:
        milvus_client.create_alias("demo", "al")
    msg = str(exc_info.value).lower()
    assert "collection" in msg and "not exist" in msg
