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
from pymilvus import MilvusClient


class TestDefaultSearch(unittest.TestCase):
    def test_delete_by_ids(self):
        dim = 2
        collection_name = 'default'
        milvus_client = MilvusClient('./local_test.db')
        has_collection = milvus_client.has_collection(collection_name, timeout=5)
        if has_collection:
            milvus_client.drop_collection(collection_name)
        milvus_client.create_collection(collection_name, dim, id_type='string', max_length=1000)
        
        rows = [
            {'id': '-xf%^@#$%^&***)(*/.', 'vector': [0.0, 1.0], 'a': 100},
            {'id': '中文id', 'vector': [0.1, 0.9], 'a': 100},
            {'id': 'Título', 'vector': [0.2, 0.8], 'a': 200},
            {'id': 'Cien años de soledad', 'vector': [0.3, 0.7], 'a': 200},
        ]
        insert_result = milvus_client.insert(collection_name, rows)
        self.assertEqual(insert_result['insert_count'], 4)

        result = milvus_client.search(collection_name, [[0.0, 1.0]], limit=3)
        self.assertEqual(result[0][0]['distance'], 1.0)
        self.assertEqual([item['id']for item in result[0]], ['-xf%^@#$%^&***)(*/.', '中文id', 'Título'])

        result = milvus_client.delete(collection_name, ids=['-xf%^@#$%^&***)(*/.', '中文id'])
        result = milvus_client.search(collection_name, [[0.0, 1.0]], limit=3)
        self.assertEqual([item['id']for item in result[0]], ['Título', 'Cien años de soledad'])
        milvus_client.release_collection(collection_name)        
        del milvus_client

        local_client = MilvusClient('./local_test.db')
        result = local_client.search(collection_name, [[0.0, 1.0]], limit=3)
        self.assertEqual([item['id']for item in result[0]], ['Título', 'Cien años de soledad'])

        result = local_client.delete(collection_name, ids=['Cien años de soledad'])
        result = local_client.search(collection_name, [[0.0, 1.0]], limit=3)
        self.assertEqual([item['id']for item in result[0]], ['Título'])

        local_client.drop_collection(collection_name)

    def test_delete_by_filter(self):
        dim = 2
        collection_name = 'hello_milvus'
        milvus_client = MilvusClient('./local_test.db')
        has_collection = milvus_client.has_collection(collection_name, timeout=5)
        if has_collection:
            milvus_client.drop_collection(collection_name)
        milvus_client.create_collection(collection_name, dim, id_type='string', max_length=1000)
        
        rows = [
            {'id': '-xf%^@#$%^&***)(*/.', 'vector': [0.0, 1.0], 'a': 100, 'b': 300},
            {'id': '中文id', 'vector': [0.1, 0.9], 'a': 100},
            {'id': 'Título', 'vector': [0.2, 0.8], 'b': 200},
            {'id': 'Cien años de soledad', 'vector': [0.3, 0.7], 'b': 200},
        ]
        insert_result = milvus_client.insert(collection_name, rows)
        self.assertEqual(insert_result['insert_count'], 4)

        result = milvus_client.search(collection_name, [[0.0, 1.0]], limit=3)
        self.assertEqual(result[0][0]['distance'], 1.0)
        self.assertEqual([item['id']for item in result[0]], ['-xf%^@#$%^&***)(*/.', '中文id', 'Título'])

        result = milvus_client.delete(collection_name, filter='(a==100) && (b==300)')
        result = milvus_client.search(collection_name, [[0.0, 1.0]], limit=3)
        self.assertEqual([item['id']for item in result[0]], ['中文id', 'Título', 'Cien años de soledad'])
        milvus_client.release_collection(collection_name)
        del milvus_client

        local_client = MilvusClient('./local_test.db')
        result = local_client.search(collection_name, [[0.0, 1.0]], limit=3)
        self.assertEqual([item['id']for item in result[0]], ['中文id', 'Título', 'Cien años de soledad'])        

        local_client.delete(collection_name, filter='(a==100) || (b==200)')
        result = local_client.search(collection_name, [[0.0, 1.0]], limit=3)
        self.assertEqual(result, [[]])

        local_client.drop_collection(collection_name)        


if __name__ == '__main__':
    unittest.main()        
