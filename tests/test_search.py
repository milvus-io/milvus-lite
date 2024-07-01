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
from pymilvus import MilvusClient, MilvusException


class TestDefaultSearch(unittest.TestCase):
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

    def test_index_info(self):
        index_info = self.milvus_client.describe_index(self.collection_name, 'vector')
        self.assertEqual(index_info['metric_type'], 'COSINE')
        self.assertEqual(int(index_info['dim']), self.dim)

    def test_error_metrics(self):
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
        with self.assertRaises(MilvusException):
            self.milvus_client.search(self.collection_name, [[0.0, 1.0]], limit=3,
                                      search_params={'metric_type': 'IP'},
                                      consistency_level='Strong')

    def test_search(self):
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
        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0]], limit=3,
                                      search_params={'metric_type': 'COSINE'},
                                      consistency_level='Strong', output_fields=['vector'])
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(result[0][0]['distance'], 1.0)
        self.assertEqual([item['id']for item in result[0]], [1, 2, 3])
        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0]], limit=2, offset=10)
        self.assertEqual(len(result[0]), 0)
        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0]], limit=2, offset=5)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0]['id'], 6)

        # range search
        result = self.milvus_client.search(
            self.collection_name,
            [[0.0, 1.0]],
            limit=3,
            search_params={
                'metric_type': 'COSINE',
                "params": {
                    "radius": 0.99,
                    "range_filter": 1.0
                }
            }
        )
        self.assertEqual(len(result[0]), 2)
        self.assertTrue(0.99 <= result[0][0]['distance'] <= 1.0)
        self.assertTrue(0.99 <= result[0][1]['distance'] <= 1.0)

        # range search
        result = self.milvus_client.search(
            self.collection_name,
            [[0.0, 1.0]],
            limit=3,
            search_params={
                'metric_type': 'COSINE',
                "params": {
                    "radius": 0.9,
                    "range_filter": 1.0
                }
            }
        )
        self.assertEqual(len(result[0]), 3)
        self.assertTrue(0.9 <= result[0][0]['distance'] <= 1.0)
        self.assertTrue(0.9 <= result[0][1]['distance'] <= 1.0)
        self.assertTrue(0.9 <= result[0][2]['distance'] <= 1.0)


    def test_multi_search(self):
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
        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0], [1.0, 0.0]], limit=3,
                                      search_params={'metric_type': 'COSINE'},
                                      consistency_level='Strong')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]['distance'], 1.0)
        self.assertEqual([item['id']for item in result[0]], [1, 2, 3])
        self.assertEqual(result[1][0]['distance'], 1.0)
        self.assertEqual([item['id']for item in result[1]], [6, 5, 4])

        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0], [1.0, 0.0]],
                                      consistency_level='Strong')
        self.assertEqual(len(result), 2)
        self.assertEqual([item['id']for item in result[0]], [1, 2, 3, 4, 5, 6])
        self.assertEqual([item['id']for item in result[1]], [6, 5, 4, 3, 2, 1])


class TestIndexMetric(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.collection_name = 'hello_milvus'
        self.milvus_client = MilvusClient('./local_test.db')
        has_collection = self.milvus_client.has_collection(self.collection_name, timeout=5)
        if has_collection:
            self.milvus_client.drop_collection(self.collection_name)

    def tearDown(self):
        self.milvus_client.drop_collection(self.collection_name)

    def test_L2(self):
        self.milvus_client.create_collection(self.collection_name, self.dim, metric_type="L2")
        index_info = self.milvus_client.describe_index(self.collection_name, 'vector')
        self.assertEqual(index_info['metric_type'], 'L2')
        self.assertEqual(int(index_info['dim']), self.dim)
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
        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0], [1.0, 0.0]], limit=3,
                                      search_params={'metric_type': 'L2'},
                                      consistency_level='Strong')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]['distance'], 0.0)
        self.assertEqual([item['id']for item in result[0]], [1, 2, 3])
        self.assertEqual(result[1][0]['distance'], 0.0)
        self.assertEqual([item['id']for item in result[1]], [6, 5, 4])

        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0], [1.0, 0.0]],
                                      consistency_level='Strong')
        self.assertEqual(len(result), 2)
        self.assertEqual([item['id']for item in result[0]], [1, 2, 3, 4, 5, 6])
        self.assertEqual([item['id']for item in result[1]], [6, 5, 4, 3, 2, 1])

    def test_IP(self):
        self.milvus_client.create_collection(self.collection_name, self.dim, metric_type="IP")
        index_info = self.milvus_client.describe_index(self.collection_name, 'vector')
        self.assertEqual(index_info['metric_type'], 'IP')
        self.assertEqual(int(index_info['dim']), self.dim)
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
        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0], [1.0, 0.0]], limit=3,
                                      search_params={'metric_type': 'IP'},
                                      consistency_level='Strong')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]['distance'], 1.0)
        self.assertEqual([item['id']for item in result[0]], [1, 2, 3])
        self.assertEqual(result[1][0]['distance'], 1.0)
        self.assertEqual([item['id']for item in result[1]], [6, 5, 4])

        result = self.milvus_client.search(self.collection_name, [[0.0, 1.0], [1.0, 0.0]],
                                      consistency_level='Strong')
        self.assertEqual(len(result), 2)
        self.assertEqual([item['id']for item in result[0]], [1, 2, 3, 4, 5, 6])
        self.assertEqual([item['id']for item in result[1]], [6, 5, 4, 3, 2, 1])


if __name__ == '__main__':
    unittest.main()
