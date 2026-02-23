import unittest
import os
import tempfile
from pymilvus import MilvusClient, MilvusException, DataType


class TestNullableInsertAndQuery(unittest.TestCase):
    """End-to-end tests for nullable field support in Milvus Lite.

    Each test gets its own unique DB file to avoid milvus-lite server
    lifecycle issues between test methods.
    """

    def setUp(self):
        self._db_fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(self._db_fd)
        os.remove(self._db_path)
        self.client = MilvusClient(self._db_path)

    def tearDown(self):
        self.client.close()
        for suffix in ("", "-wal", "-shm"):
            path = self._db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    def _create_nullable_collection(self, name="test_nullable"):
        if self.client.has_collection(name):
            self.client.drop_collection(name)
        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        schema.add_field("nullable_int", DataType.INT64, nullable=True)
        schema.add_field("nullable_str", DataType.VARCHAR, nullable=True, max_length=256)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vec", metric_type="L2")
        self.client.create_collection(name, schema=schema, index_params=index_params)
        return name

    def test_create_collection_with_nullable_field(self):
        """Creating a collection with nullable=True should succeed."""
        name = self._create_nullable_collection()
        self.assertTrue(self.client.has_collection(name))

    def test_insert_explicit_none(self):
        """Insert rows where nullable fields are explicitly set to None."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": 10, "nullable_str": "hello"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "nullable_int": None, "nullable_str": None},
            {"id": 3, "vec": [0.0, 0.0, 1.0, 0.0], "nullable_int": 30, "nullable_str": "world"},
        ]
        result = self.client.insert(name, rows)
        self.assertEqual(result["insert_count"], 3)

    def test_insert_omitted_nullable_key(self):
        """Insert rows where nullable fields are omitted entirely from the dict."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": 10, "nullable_str": "hello"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0]},
            {"id": 3, "vec": [0.0, 0.0, 1.0, 0.0], "nullable_int": 30, "nullable_str": "world"},
        ]
        result = self.client.insert(name, rows)
        self.assertEqual(result["insert_count"], 3)

    def test_query_returns_none_at_correct_positions(self):
        """Query should return None for null fields, real values for non-null."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": 10, "nullable_str": "hello"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "nullable_int": None, "nullable_str": None},
            {"id": 3, "vec": [0.0, 0.0, 1.0, 0.0], "nullable_int": 30, "nullable_str": "world"},
        ]
        self.client.insert(name, rows)

        results = self.client.query(name, filter="id >= 1", output_fields=["id", "nullable_int", "nullable_str"])
        results_by_id = {r["id"]: r for r in results}

        self.assertEqual(results_by_id[1]["nullable_int"], 10)
        self.assertEqual(results_by_id[1]["nullable_str"], "hello")
        self.assertIsNone(results_by_id[2]["nullable_int"])
        self.assertIsNone(results_by_id[2]["nullable_str"])
        self.assertEqual(results_by_id[3]["nullable_int"], 30)
        self.assertEqual(results_by_id[3]["nullable_str"], "world")

    def test_default_value_materialized_on_insert(self):
        """Fields with default_value should get the default when omitted."""
        name = "test_default_val"
        if self.client.has_collection(name):
            self.client.drop_collection(name)
        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        schema.add_field("with_default", DataType.INT64, default_value=42)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vec", metric_type="L2")
        self.client.create_collection(name, schema=schema, index_params=index_params)

        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0]},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "with_default": 99},
        ]
        self.client.insert(name, rows)

        results = self.client.query(name, filter="id >= 1", output_fields=["id", "with_default"])
        results_by_id = {r["id"]: r for r in results}

        self.assertEqual(results_by_id[1]["with_default"], 42)
        self.assertEqual(results_by_id[2]["with_default"], 99)

    def test_restart_persistence(self):
        """Null values should survive a close/reopen cycle."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": None, "nullable_str": None},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "nullable_int": 20, "nullable_str": "ok"},
        ]
        self.client.insert(name, rows)
        self.client.close()

        self.client = MilvusClient(self._db_path)
        results = self.client.query(name, filter="id >= 1", output_fields=["id", "nullable_int", "nullable_str"])
        results_by_id = {r["id"]: r for r in results}

        self.assertIsNone(results_by_id[1]["nullable_int"])
        self.assertIsNone(results_by_id[1]["nullable_str"])
        self.assertEqual(results_by_id[2]["nullable_int"], 20)
        self.assertEqual(results_by_id[2]["nullable_str"], "ok")


    def test_filter_is_null(self):
        """Filtering with 'nullable_int is null' should return only null rows."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": 10, "nullable_str": "hello"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "nullable_int": None, "nullable_str": None},
            {"id": 3, "vec": [0.0, 0.0, 1.0, 0.0], "nullable_int": 30, "nullable_str": "world"},
            {"id": 4, "vec": [0.0, 0.0, 0.0, 1.0], "nullable_int": None, "nullable_str": "present"},
        ]
        self.client.insert(name, rows)

        results = self.client.query(name, filter="nullable_int is null", output_fields=["id", "nullable_int"])
        ids = sorted(r["id"] for r in results)
        self.assertEqual(ids, [2, 4])
        for r in results:
            self.assertIsNone(r["nullable_int"])

    def test_filter_is_not_null(self):
        """Filtering with 'nullable_int is not null' should return only non-null rows."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": 10, "nullable_str": "hello"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "nullable_int": None, "nullable_str": None},
            {"id": 3, "vec": [0.0, 0.0, 1.0, 0.0], "nullable_int": 30, "nullable_str": "world"},
        ]
        self.client.insert(name, rows)

        results = self.client.query(name, filter="nullable_int is not null", output_fields=["id", "nullable_int"])
        ids = sorted(r["id"] for r in results)
        self.assertEqual(ids, [1, 3])
        for r in results:
            self.assertIsNotNone(r["nullable_int"])

    def test_filter_is_null_varchar(self):
        """Filtering with 'nullable_str is null' works for VARCHAR fields."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": None, "nullable_str": "hello"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "nullable_int": 10, "nullable_str": None},
        ]
        self.client.insert(name, rows)

        results = self.client.query(name, filter="nullable_str is null", output_fields=["id", "nullable_str"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 2)
        self.assertIsNone(results[0]["nullable_str"])

    def test_search_with_null_filter(self):
        """Search with a null-aware filter expression should work correctly."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": 10, "nullable_str": "a"},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "nullable_int": None, "nullable_str": None},
            {"id": 3, "vec": [0.0, 0.0, 1.0, 0.0], "nullable_int": 30, "nullable_str": "b"},
        ]
        self.client.insert(name, rows)

        results = self.client.search(
            name,
            data=[[1.0, 0.0, 0.0, 0.0]],
            filter="nullable_int is not null",
            output_fields=["id", "nullable_int"],
            limit=10,
        )
        result_ids = [hit["id"] for hit in results[0]]
        self.assertNotIn(2, result_ids)
        for hit in results[0]:
            self.assertIsNotNone(hit["entity"]["nullable_int"])

    def test_combined_null_and_value_filter(self):
        """Combining 'is not null' with a value comparison in the same filter."""
        name = self._create_nullable_collection()
        rows = [
            {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0], "nullable_int": 10},
            {"id": 2, "vec": [0.0, 1.0, 0.0, 0.0], "nullable_int": None},
            {"id": 3, "vec": [0.0, 0.0, 1.0, 0.0], "nullable_int": 30},
            {"id": 4, "vec": [0.0, 0.0, 0.0, 1.0], "nullable_int": 5},
        ]
        self.client.insert(name, rows)

        results = self.client.query(
            name,
            filter="nullable_int is not null and nullable_int > 8",
            output_fields=["id", "nullable_int"],
        )
        ids = sorted(r["id"] for r in results)
        self.assertEqual(ids, [1, 3])


if __name__ == "__main__":
    unittest.main()
