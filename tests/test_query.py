# Copyright (C) 2019-2024 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import unittest
from pymilvus import MilvusClient, DataType


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.collection_name = 'default'
        self.milvus_client = MilvusClient('./local_test.db')
        has_collection = self.milvus_client.has_collection(self.collection_name, timeout=5)
        if has_collection:
            self.milvus_client.drop_collection(self.collection_name)
        self.milvus_client.create_collection(self.collection_name, self.dim)

    def tearDown(self):
        self.milvus_client.drop_collection(self.collection_name)

    def test_query_empty(self):
        self.milvus_client.query(self.collection_name, limit=3)
        rows = [
            {'id': 1, 'vector': [0.0, 1], 'a': 100},
        ]
        insert_result = self.milvus_client.insert(self.collection_name, rows)
        self.assertEqual(insert_result['insert_count'], 1)
        self.milvus_client.delete(self.collection_name, ids=[1])
        self.milvus_client.query(self.collection_name, limit=3)

    def test_query(self):
        rows = [
            {'id': 1, 'vector': [0.0, 1], 'a': 100},
            {'id': 2, 'vector': [0.1, 0.9], 'b': 200},
            {'id': 3, 'vector': [0.2, 0.8], 'c': 300},
            {'id': 4, 'vector': [0.3, 0.7], 'd': 400},
            {'id': 5, 'vector': [0.8, 0.2], 'e': 500},
            {'id': 6, 'vector': [1.0, 0.0], 'f': 600},
        ]

        insert_result = self.milvus_client.insert(self.collection_name, rows)
        self.assertEqual(insert_result['insert_count'], 6)
        result = self.milvus_client.query(self.collection_name, filter="", limit=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[1]['id'], 2)
        self.assertEqual(result[2]['id'], 3)

        result = self.milvus_client.query(self.collection_name, filter="", limit=3, offset=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 6)

        result = self.milvus_client.query(self.collection_name, filter="a == 100", limit=3, offset=5)
        self.assertEqual(len(result), 0)

        result = self.milvus_client.query(self.collection_name, filter="a == 100", limit=3)
        self.assertEqual(len(result), 1)


class TestArray(unittest.TestCase):
    def test_query(self):
        self.dim = 2
        self.collection_name = 'default'
        self.milvus_client = MilvusClient('./local_test.db')
        has_collection = self.milvus_client.has_collection(self.collection_name, timeout=5)
        if has_collection:
            self.milvus_client.drop_collection(self.collection_name)
        schema = self.milvus_client.create_schema(enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=2)
        schema.add_field("array", DataType.ARRAY, element_type=DataType.INT64, max_capacity=5)
        self.milvus_client.create_collection(self.collection_name, self.dim)
        rows = [
            {'id': 1, 'vector': [0.0, 1], 'array': [100, 1]},
            {'id': 2, 'vector': [0.1, 0.9], 'array': [200, 2]},
            {'id': 3, 'vector': [0.2, 0.8], 'array': [300, 1]},
            {'id': 4, 'vector': [0.3, 0.7], 'array': [400, 3]},
            {'id': 5, 'vector': [0.8, 0.2], 'array': [500]},
            {'id': 6, 'vector': [1.0, 0.0], 'array': [600]},
        ]
        insert_result = self.milvus_client.insert(self.collection_name, rows)
        self.assertEqual(insert_result['insert_count'], 6)
        result = self.milvus_client.query(
                collection_name="default",
                filter="ARRAY_CONTAINS_ANY(array, [1, 2])",
                output_fields=["id"],
                limit=3
            )
        self.assertEqual(len(result), 3)

        result = self.milvus_client.query(
                collection_name="default",
                filter="ARRAY_CONTAINS_ALL(array, [100, 1])",
                output_fields=["id"],
                limit=3
            )
        self.assertEqual(len(result), 1)

        result = self.milvus_client.query(
                collection_name="default",
                filter="ARRAY_LENGTH(array) == 2",
                output_fields=["id"],
                limit=4
            )
        self.assertEqual(len(result), 4)
        self.milvus_client.drop_collection(self.collection_name)


if __name__ == '__main__':
    unittest.main()        
