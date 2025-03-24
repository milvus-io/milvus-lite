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
from pymilvus import MilvusClient, MilvusException, DataType


class TestDefaultSearch(unittest.TestCase):
    def test_schema_field_limits(self):
        collection_name = "hello_milvus"
        milvus_client = MilvusClient("./local_test.db")
        has_collection = milvus_client.has_collection(collection_name)        
        if has_collection:
            milvus_client.drop_collection(collection_name)        
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=2)
        for i in range(62):
            schema.add_field('a' + str(i), DataType.INT64)
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name = "embeddings", metric_type="L2")
        milvus_client.create_collection(collection_name, schema=schema, index_params=index_params)

    def test_schema_field_out_limits(self):
        collection_name = "hello_milvus"
        milvus_client = MilvusClient("./local_test.db")
        has_collection = milvus_client.has_collection(collection_name)        
        if has_collection:
            milvus_client.drop_collection(collection_name)        
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=2)
        for i in range(63):
            schema.add_field('a' + str(i), DataType.INT64)
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name = "embeddings", metric_type="L2")
        with self.assertRaises(MilvusException):
            milvus_client.create_collection(collection_name, schema=schema, index_params=index_params)

    def test_varchar_field_maxlen(self):
        collection_name = "hello_milvus"
        milvus_client = MilvusClient("./local_test.db")
        has_collection = milvus_client.has_collection(collection_name)        
        if has_collection:
            milvus_client.drop_collection(collection_name)        
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=2)
        schema.add_field("string", DataType.VARCHAR, max_length=65535)
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name = "embeddings", metric_type="L2")
        milvus_client.create_collection(collection_name, schema=schema, index_params=index_params)

    def test_varchar_field_out_maxlen(self):
        collection_name = "hello_milvus"
        milvus_client = MilvusClient("./local_test.db")
        has_collection = milvus_client.has_collection(collection_name)        
        if has_collection:
            milvus_client.drop_collection(collection_name)        
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=2)
        schema.add_field("string", DataType.VARCHAR, max_length=65536)
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name = "embeddings", metric_type="L2")
        with self.assertRaises(MilvusException):
            milvus_client.create_collection(collection_name, schema=schema, index_params=index_params)                    

    def test_collection_name(self):
        collection_name = "test_collection_escck"
        milvus_client = MilvusClient("./local_test.db")
        has_collection = milvus_client.has_collection(collection_name)
        if has_collection:
            milvus_client.drop_collection(collection_name)
        milvus_client.create_collection(collection_name, dimension=384)


if __name__ == '__main__':
    unittest.main()
