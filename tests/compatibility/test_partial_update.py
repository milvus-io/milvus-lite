"""
Partial Update (Upsert) compatibility tests — verify MilvusLite's partial
upsert functionality through the pymilvus MilvusClient.

Partial Update semantics:
  - Only provide a subset of fields + the primary key during upsert
  - If pk already exists: merge with old record, new fields overwrite old ones,
    unprovided fields remain unchanged
  - If pk does not exist: treat as a new record insert (all required fields must
    be provided)

Test coverage:
  1. Basic partial update — update only some scalar fields
  2. Partial update preserves vectors unchanged
  3. Partial update updates vector while preserving scalar fields
  4. Mixed batch — some existing + some new records
  5. Multiple consecutive partial updates
  6. Partial update after flush (reading old records goes through segments)
  7. Dynamic field partial update
  8. JSON field partial update
  9. ARRAY field partial update
 10. Nullable field partial update (set to None)
 11. Search results correct after partial update
 12. Filter query correct after partial update
 13. Partial update + delete interaction
 14. VARCHAR primary key partial update
 15. Partial update does not affect other records
 16. Full-field upsert degrades to complete overwrite
 17. Row count unchanged after partial update
 18. Partial update within a partition
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from milvus_lite.adapter.grpc.server import start_server_in_thread

DIM = 8
SEED = 42


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="partial_test_")
    server, db, port = start_server_in_thread(data_dir)
    yield port, data_dir
    server.stop(grace=2)
    db.close()
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def client(server):
    port, _ = server
    c = MilvusClient(uri=f"http://127.0.0.1:{port}")
    yield c
    for name in c.list_collections():
        c.drop_collection(name)


def rvecs(n: int, dim: int = DIM, seed: int = SEED) -> list[list[float]]:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32).tolist()


def make_standard_collection(client, name):
    """Create a standard test collection: pk(INT64) + vec + title(nullable) + score(nullable)"""
    schema = client.create_schema()
    schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
    schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", MilvusDataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field("score", MilvusDataType.FLOAT, nullable=True)
    client.create_collection(name, schema=schema)


# ====================================================================
# 1. Basic partial update — update only some scalar fields
# ====================================================================

class TestBasicPartialUpdate:

    def test_update_single_scalar_field(self, client: MilvusClient):
        """Update only title, keeping vec and score unchanged"""
        make_standard_collection(client, "partial_basic")
        vecs = rvecs(1)
        client.insert("partial_basic", [
            {"pk": 1, "vec": vecs[0], "title": "original", "score": 88.5},
        ])

        client.upsert("partial_basic", [
            {"pk": 1, "title": "updated"},
        ], partial_update=True)

        got = client.get("partial_basic", ids=[1])
        assert len(got) == 1
        assert got[0]["title"] == "updated"
        assert got[0]["score"] == pytest.approx(88.5)
        # Vector should also remain unchanged
        assert got[0]["vec"] == pytest.approx(vecs[0], rel=1e-5)

    def test_update_multiple_scalar_fields(self, client: MilvusClient):
        """Update both title and score simultaneously"""
        make_standard_collection(client, "partial_multi")
        vecs = rvecs(1)
        client.insert("partial_multi", [
            {"pk": 1, "vec": vecs[0], "title": "old", "score": 10.0},
        ])

        client.upsert("partial_multi", [
            {"pk": 1, "title": "new", "score": 99.0},
        ], partial_update=True)

        got = client.get("partial_multi", ids=[1])
        assert got[0]["title"] == "new"
        assert got[0]["score"] == pytest.approx(99.0)
        assert got[0]["vec"] == pytest.approx(vecs[0], rel=1e-5)


# ====================================================================
# 2. Partial update preserves vectors unchanged
# ====================================================================

class TestVectorPreservation:

    def test_vector_unchanged_after_partial_update(self, client: MilvusClient):
        """After partial update of scalar fields, vector search still correctly finds the record"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("label", MilvusDataType.VARCHAR, max_length=64, nullable=True)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("partial_vec_keep", schema=schema, index_params=idx)

        vecs = rvecs(5)
        client.insert("partial_vec_keep", [
            {"pk": i, "vec": vecs[i], "label": f"orig_{i}"} for i in range(5)
        ])

        # Partial update label of pk=2
        client.upsert("partial_vec_keep", [{"pk": 2, "label": "changed"}],
                      partial_update=True)

        client.load_collection("partial_vec_keep")

        # Search using pk=2's original vector, it should still be the nearest neighbor
        results = client.search("partial_vec_keep", data=[vecs[2]], limit=1,
                                output_fields=["pk", "label"])
        assert results[0][0]["entity"]["pk"] == 2
        assert results[0][0]["entity"]["label"] == "changed"
        assert results[0][0]["distance"] < 1e-4  # Distance should be close to 0


# ====================================================================
# 3. Partial update updates vector while preserving scalar fields
# ====================================================================

class TestVectorUpdate:

    def test_update_vector_only(self, client: MilvusClient):
        """Update only the vector, keeping scalar fields unchanged"""
        make_standard_collection(client, "partial_vec_update")
        old_vec = rvecs(1, seed=1)[0]
        client.insert("partial_vec_update", [
            {"pk": 1, "vec": old_vec, "title": "keep_me", "score": 42.0},
        ])

        new_vec = rvecs(1, seed=999)[0]
        client.upsert("partial_vec_update", [
            {"pk": 1, "vec": new_vec},
        ], partial_update=True)

        got = client.get("partial_vec_update", ids=[1])
        assert got[0]["title"] == "keep_me"
        assert got[0]["score"] == pytest.approx(42.0)
        assert got[0]["vec"] == pytest.approx(new_vec, rel=1e-5)


# ====================================================================
# 4. Mixed batch — some existing + some new records
# ====================================================================

class TestMixedBatch:

    def test_mixed_existing_and_new(self, client: MilvusClient):
        """Partial update existing records + full upsert new records (requires two batches)"""
        make_standard_collection(client, "partial_mixed")
        vecs = rvecs(3)
        client.insert("partial_mixed", [
            {"pk": 1, "vec": vecs[0], "title": "one", "score": 1.0},
            {"pk": 2, "vec": vecs[1], "title": "two", "score": 2.0},
        ])

        # pymilvus partial_update requires consistent field count within a batch, need two batches
        # Batch 1: partial update existing records
        client.upsert("partial_mixed", [
            {"pk": 1, "title": "one_updated"},
        ], partial_update=True)

        # Batch 2: full upsert new records
        new_vecs = rvecs(2, seed=88)
        client.upsert("partial_mixed", [
            {"pk": 3, "vec": new_vecs[0], "title": "three", "score": 3.0},
        ])

        r1 = client.get("partial_mixed", ids=[1])[0]
        assert r1["title"] == "one_updated"
        assert r1["score"] == pytest.approx(1.0)  # Unchanged
        assert r1["vec"] == pytest.approx(vecs[0], rel=1e-5)  # Unchanged

        r2 = client.get("partial_mixed", ids=[2])[0]
        assert r2["title"] == "two"  # Not touched

        r3 = client.get("partial_mixed", ids=[3])[0]
        assert r3["title"] == "three"


# ====================================================================
# 5. Multiple consecutive partial updates
# ====================================================================

class TestConsecutiveUpdates:

    def test_multiple_partial_updates(self, client: MilvusClient):
        """Perform multiple partial updates on the same record"""
        make_standard_collection(client, "partial_multi_round")
        vecs = rvecs(1)
        client.insert("partial_multi_round", [
            {"pk": 1, "vec": vecs[0], "title": "v1", "score": 10.0},
        ])

        # Round 1: update only title
        client.upsert("partial_multi_round", [{"pk": 1, "title": "v2"}],
                      partial_update=True)
        got = client.get("partial_multi_round", ids=[1])[0]
        assert got["title"] == "v2"
        assert got["score"] == pytest.approx(10.0)

        # Round 2: update only score
        client.upsert("partial_multi_round", [{"pk": 1, "score": 20.0}],
                      partial_update=True)
        got = client.get("partial_multi_round", ids=[1])[0]
        assert got["title"] == "v2"  # Title from previous round still present
        assert got["score"] == pytest.approx(20.0)

        # Round 3: update title + score
        client.upsert("partial_multi_round", [{"pk": 1, "title": "v3", "score": 30.0}],
                      partial_update=True)
        got = client.get("partial_multi_round", ids=[1])[0]
        assert got["title"] == "v3"
        assert got["score"] == pytest.approx(30.0)
        assert got["vec"] == pytest.approx(vecs[0], rel=1e-5)  # Vector always unchanged


# ====================================================================
# 6. Partial update after flush
# ====================================================================

class TestPartialUpdateAfterFlush:

    def test_partial_update_reads_from_segments(self, client: MilvusClient):
        """After flush, old records reside in segments; partial update still merges correctly"""
        make_standard_collection(client, "partial_flush")
        vecs = rvecs(1)
        client.insert("partial_flush", [
            {"pk": 1, "vec": vecs[0], "title": "persisted", "score": 77.0},
        ])
        client.flush("partial_flush")

        # partial update
        client.upsert("partial_flush", [{"pk": 1, "score": 88.0}],
                      partial_update=True)

        got = client.get("partial_flush", ids=[1])[0]
        assert got["title"] == "persisted"  # Read from segment
        assert got["score"] == pytest.approx(88.0)


# ====================================================================
# 7. Dynamic field partial update
# ====================================================================

class TestDynamicFieldPartialUpdate:

    def test_partial_update_preserves_dynamic_fields(self, client: MilvusClient):
        """Partial update preserves old dynamic fields and updates the specified ones"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.enable_dynamic_field = True

        client.create_collection("partial_dyn", schema=schema)
        vecs = rvecs(1)
        client.insert("partial_dyn", [
            {"pk": 1, "vec": vecs[0], "color": "red", "size": 42, "tag": "important"},
        ])

        # Partial update: only update color
        client.upsert("partial_dyn", [{"pk": 1, "color": "blue"}],
                      partial_update=True)

        # Dynamic fields require explicit output_fields specification
        got = client.get("partial_dyn", ids=[1],
                         output_fields=["pk", "vec", "color", "size", "tag"])[0]
        assert got["color"] == "blue"
        assert int(got["size"]) == 42   # Dynamic field numbers may be serialized as strings by JSON
        assert got["tag"] == "important"  # Unchanged

    def test_partial_update_adds_new_dynamic_field(self, client: MilvusClient):
        """Partial update can add new dynamic fields"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.enable_dynamic_field = True

        client.create_collection("partial_dyn_new", schema=schema)
        vecs = rvecs(1)
        client.insert("partial_dyn_new", [
            {"pk": 1, "vec": vecs[0], "color": "red"},
        ])

        # Partial update: add new field priority
        client.upsert("partial_dyn_new", [{"pk": 1, "priority": "high"}],
                      partial_update=True)

        got = client.get("partial_dyn_new", ids=[1],
                         output_fields=["pk", "vec", "color", "priority"])[0]
        assert got["color"] == "red"       # Preserved
        assert got["priority"] == "high"   # Newly added


# ====================================================================
# 8. JSON field partial update
# ====================================================================

class TestJsonPartialUpdate:

    def test_json_field_replacement(self, client: MilvusClient):
        """Partial update replaces the entire JSON field"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("meta", MilvusDataType.JSON, nullable=True)
        schema.add_field("name", MilvusDataType.VARCHAR, max_length=64, nullable=True)

        client.create_collection("partial_json", schema=schema)
        vecs = rvecs(1)
        client.insert("partial_json", [
            {"pk": 1, "vec": vecs[0], "meta": {"env": "dev", "v": 1}, "name": "test"},
        ])

        # Only update meta
        client.upsert("partial_json", [
            {"pk": 1, "meta": {"env": "prod", "v": 2, "region": "us"}},
        ], partial_update=True)

        got = client.get("partial_json", ids=[1])[0]
        assert got["meta"]["env"] == "prod"
        assert got["meta"]["v"] == 2
        assert got["meta"]["region"] == "us"
        assert got["name"] == "test"  # Unchanged


# ====================================================================
# 9. ARRAY field partial update
# ====================================================================

class TestArrayPartialUpdate:

    def test_array_field_replacement(self, client: MilvusClient):
        """Partial update replaces the entire ARRAY field"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tags", MilvusDataType.ARRAY,
                         element_type=MilvusDataType.VARCHAR,
                         max_capacity=10, max_length=64,
                         nullable=True)
        schema.add_field("score", MilvusDataType.FLOAT, nullable=True)

        client.create_collection("partial_arr", schema=schema)
        vecs = rvecs(1)
        client.insert("partial_arr", [
            {"pk": 1, "vec": vecs[0], "tags": ["a", "b"], "score": 1.0},
        ])

        client.upsert("partial_arr", [
            {"pk": 1, "tags": ["x", "y", "z"]},
        ], partial_update=True)

        got = client.get("partial_arr", ids=[1])[0]
        assert got["tags"] == ["x", "y", "z"]
        assert got["score"] == pytest.approx(1.0)  # Unchanged


# ====================================================================
# 10. Nullable field set to None
# ====================================================================

class TestNullablePartialUpdate:

    def test_set_field_to_none(self, client: MilvusClient):
        """Partial update explicitly sets a nullable field to None"""
        make_standard_collection(client, "partial_null")
        vecs = rvecs(1)
        client.insert("partial_null", [
            {"pk": 1, "vec": vecs[0], "title": "has_value", "score": 50.0},
        ])

        # Set title to None
        client.upsert("partial_null", [
            {"pk": 1, "title": None},
        ], partial_update=True)

        got = client.get("partial_null", ids=[1])[0]
        assert got["title"] is None
        assert got["score"] == pytest.approx(50.0)  # Unchanged

    def test_set_none_to_value(self, client: MilvusClient):
        """Update a field that was previously None to a real value"""
        make_standard_collection(client, "partial_fill")
        vecs = rvecs(1)
        client.insert("partial_fill", [
            {"pk": 1, "vec": vecs[0], "title": None, "score": None},
        ])

        client.upsert("partial_fill", [
            {"pk": 1, "title": "now_has_value"},
        ], partial_update=True)

        got = client.get("partial_fill", ids=[1])[0]
        assert got["title"] == "now_has_value"
        assert got["score"] is None  # Unchanged


# ====================================================================
# 11. Search results correct after partial update
# ====================================================================

class TestSearchAfterPartialUpdate:

    def test_search_returns_updated_fields(self, client: MilvusClient):
        """Search returns updated field values after partial update"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("category", MilvusDataType.VARCHAR, max_length=32, nullable=True)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("partial_search", schema=schema, index_params=idx)

        vecs = rvecs(5)
        client.insert("partial_search", [
            {"pk": i, "vec": vecs[i], "category": "old"} for i in range(5)
        ])

        # Partial update category of pk=0
        client.upsert("partial_search", [{"pk": 0, "category": "new"}],
                      partial_update=True)

        client.load_collection("partial_search")

        results = client.search("partial_search", data=[vecs[0]], limit=1,
                                output_fields=["pk", "category"])
        assert results[0][0]["entity"]["pk"] == 0
        assert results[0][0]["entity"]["category"] == "new"


# ====================================================================
# 12. Filter query correct after partial update
# ====================================================================

class TestFilterAfterPartialUpdate:

    def test_filter_sees_updated_values(self, client: MilvusClient):
        """Filter can find updated values after partial update"""
        make_standard_collection(client, "partial_filter")
        vecs = rvecs(3)
        client.insert("partial_filter", [
            {"pk": 1, "vec": vecs[0], "title": "a", "score": 10.0},
            {"pk": 2, "vec": vecs[1], "title": "b", "score": 20.0},
            {"pk": 3, "vec": vecs[2], "title": "c", "score": 30.0},
        ])

        # Change pk=2's score from 20 to 99
        client.upsert("partial_filter", [{"pk": 2, "score": 99.0}],
                      partial_update=True)

        r = client.query("partial_filter", filter="score > 50",
                         output_fields=["pk", "score"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [2]  # Only pk=2 has score=99 > 50


# ====================================================================
# 13. Partial update + delete interaction
# ====================================================================

class TestPartialUpdateAndDelete:

    def test_partial_update_deleted_record(self, client: MilvusClient):
        """Upsert the same pk after delete: should be treated as a fresh insert"""
        make_standard_collection(client, "partial_del")
        vecs = rvecs(2)
        client.insert("partial_del", [
            {"pk": 1, "vec": vecs[0], "title": "alive", "score": 1.0},
        ])

        client.delete("partial_del", ids=[1])

        # Upsert a deleted pk -- since no old record is found, treat as a new record
        new_vec = rvecs(1, seed=99)[0]
        client.upsert("partial_del", [
            {"pk": 1, "vec": new_vec, "title": "resurrected", "score": 2.0},
        ])

        got = client.get("partial_del", ids=[1])
        assert len(got) == 1
        assert got[0]["title"] == "resurrected"

    def test_delete_after_partial_update(self, client: MilvusClient):
        """Delete after partial update, record should disappear"""
        make_standard_collection(client, "partial_then_del")
        vecs = rvecs(1)
        client.insert("partial_then_del", [
            {"pk": 1, "vec": vecs[0], "title": "orig", "score": 1.0},
        ])

        client.upsert("partial_then_del", [{"pk": 1, "title": "updated"}],
                      partial_update=True)
        client.delete("partial_then_del", ids=[1])

        got = client.get("partial_then_del", ids=[1])
        assert len(got) == 0


# ====================================================================
# 14. VARCHAR primary key partial update
# ====================================================================

class TestVarcharPKPartialUpdate:

    def test_varchar_pk_partial_update(self, client: MilvusClient):
        """Partial update with VARCHAR primary key"""
        schema = client.create_schema()
        schema.add_field("id", MilvusDataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", MilvusDataType.VARCHAR, max_length=32, nullable=True)
        schema.add_field("count", MilvusDataType.INT64, nullable=True)

        client.create_collection("partial_vpk", schema=schema)
        vecs = rvecs(2)
        client.insert("partial_vpk", [
            {"id": "doc_a", "vec": vecs[0], "status": "draft", "count": 0},
            {"id": "doc_b", "vec": vecs[1], "status": "published", "count": 100},
        ])

        # partial update doc_a
        client.upsert("partial_vpk", [{"id": "doc_a", "status": "published"}],
                      partial_update=True)

        got = client.get("partial_vpk", ids=["doc_a"])[0]
        assert got["status"] == "published"
        assert got["count"] == 0  # Unchanged
        assert got["vec"] == pytest.approx(vecs[0], rel=1e-5)

        # doc_b is not affected
        got_b = client.get("partial_vpk", ids=["doc_b"])[0]
        assert got_b["status"] == "published"
        assert got_b["count"] == 100


# ====================================================================
# 15. Partial update does not affect other records
# ====================================================================

class TestNoSideEffects:

    def test_other_records_unchanged(self, client: MilvusClient):
        """Partial update of one record does not affect other records"""
        make_standard_collection(client, "partial_no_side")
        vecs = rvecs(5)
        original = [
            {"pk": i, "vec": vecs[i], "title": f"t{i}", "score": float(i * 10)}
            for i in range(5)
        ]
        client.insert("partial_no_side", original)

        # Only update pk=2
        client.upsert("partial_no_side", [{"pk": 2, "title": "changed"}],
                      partial_update=True)

        # Verify other records are unchanged
        for i in [0, 1, 3, 4]:
            got = client.get("partial_no_side", ids=[i])[0]
            assert got["title"] == f"t{i}"
            assert got["score"] == pytest.approx(float(i * 10))


# ====================================================================
# 16. Full-field upsert degrades to complete overwrite
# ====================================================================

class TestFullUpsertFallback:

    def test_full_upsert_replaces_all(self, client: MilvusClient):
        """Upsert with all fields provided is equivalent to a complete overwrite"""
        make_standard_collection(client, "partial_full")
        vecs = rvecs(2)
        client.insert("partial_full", [
            {"pk": 1, "vec": vecs[0], "title": "old", "score": 1.0},
        ])

        new_vec = rvecs(1, seed=88)[0]
        client.upsert("partial_full", [
            {"pk": 1, "vec": new_vec, "title": "new", "score": 99.0},
        ])

        got = client.get("partial_full", ids=[1])[0]
        assert got["title"] == "new"
        assert got["score"] == pytest.approx(99.0)
        assert got["vec"] == pytest.approx(new_vec, rel=1e-5)


# ====================================================================
# 17. Row count unchanged after partial update
# ====================================================================

class TestCountAfterPartialUpdate:

    def test_row_count_stable(self, client: MilvusClient):
        """Partial update of existing records does not increase row_count"""
        make_standard_collection(client, "partial_count")
        vecs = rvecs(5)
        client.insert("partial_count", [
            {"pk": i, "vec": vecs[i], "title": f"t{i}", "score": 0.0}
            for i in range(5)
        ])

        stats_before = client.get_collection_stats("partial_count")
        assert int(stats_before["row_count"]) == 5

        # Partial update 3 records
        client.upsert("partial_count", [
            {"pk": 0, "title": "u0"},
            {"pk": 2, "title": "u2"},
            {"pk": 4, "title": "u4"},
        ], partial_update=True)

        stats_after = client.get_collection_stats("partial_count")
        assert int(stats_after["row_count"]) == 5  # Still 5


# ====================================================================
# 18. Partial update within a partition
# ====================================================================

class TestPartitionPartialUpdate:

    def test_partial_update_in_partition(self, client: MilvusClient):
        """Perform partial update within a specified partition"""
        make_standard_collection(client, "partial_part")
        client.create_partition("partial_part", "region_a")

        vecs = rvecs(2)
        client.insert("partial_part", [
            {"pk": 1, "vec": vecs[0], "title": "orig", "score": 5.0},
        ], partition_name="region_a")

        client.upsert("partial_part", [
            {"pk": 1, "title": "updated_in_partition"},
        ], partition_name="region_a", partial_update=True)

        got = client.get("partial_part", ids=[1])
        assert got[0]["title"] == "updated_in_partition"
        assert got[0]["score"] == pytest.approx(5.0)


# ====================================================================
# 19. Bulk partial update
# ====================================================================

class TestBulkPartialUpdate:

    def test_bulk_partial_update(self, client: MilvusClient):
        """Bulk partial update of 100 records"""
        make_standard_collection(client, "partial_bulk")
        vecs = rvecs(100, seed=1)
        client.insert("partial_bulk", [
            {"pk": i, "vec": vecs[i], "title": f"orig_{i}", "score": float(i)}
            for i in range(100)
        ])

        # Partial update title of all even-numbered pks
        updates = [{"pk": i, "title": f"updated_{i}"} for i in range(0, 100, 2)]
        client.upsert("partial_bulk", updates, partial_update=True)

        # Verify even pks are updated, odd ones unchanged
        for i in [0, 10, 50, 98]:
            got = client.get("partial_bulk", ids=[i])[0]
            assert got["title"] == f"updated_{i}"
            assert got["score"] == pytest.approx(float(i))

        for i in [1, 11, 51, 99]:
            got = client.get("partial_bulk", ids=[i])[0]
            assert got["title"] == f"orig_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
