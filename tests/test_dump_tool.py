# Copyright (C) 2019-2023 Zilliz. All rights reserved.
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
import json
import os
import random
import pandas as pd
import numpy as np
import tempfile


from pymilvus import (
    MilvusClient,
    FieldSchema, CollectionSchema, DataType
)


from milvus_lite.cmdline import dump_collection

DIM = 512
DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'train_embeddings.csv')

def gen_binary_vector():
    raw_vector = [random.randint(0, 1) for i in range(DIM)]
    binary_vectors = np.packbits(raw_vector, axis=-1).tolist()
    return binary_vectors

def gen_float_vector():
    return [random.random() for _ in range(DIM)]

def all_types_data(bin_vec: bool)->list:
    rows = []
    count = 100
    for i in range(count):
        row = {
            "id": i,
            "bool": True if i%5 == 0 else False,
            "int8": i%128,
            "int16": i%1000,
            "int32": i%100000,
            "int64": i,
            "float": i/3,
            "double": i/7,
            "varchar": f"varchar_{i}",
            "json": {"dummy": i, "ok": f"name_{i}"},
            "vector": gen_binary_vector() if bin_vec else gen_float_vector(),
            f"dynamic_{i}": i,
            # bulkinsert doesn't support import npy with array field, the below values will be stored into dynamic field
            "array_str": [f"str_{k}" for k in range(5)],
            "array_int": [k for k in range(10)],
        }
        rows.append(row)
    return rows


class TestDumpTool(unittest.TestCase):
    def setUp(self):
        self.collection_name = 'hello_milvus'
        self.milvus_client = MilvusClient('./local_test.db')
        has_collection = self.milvus_client.has_collection(self.collection_name, timeout=5)
        if has_collection:
            self.milvus_client.drop_collection(self.collection_name)

    def tearDown(self):
        # self.milvus_client.drop_collection(self.collection_name)
        pass

    @unittest.skip("")
    def test_simple_collection(self):
        has_collection = self.milvus_client.has_collection(self.collection_name)
        if has_collection:
            self.milvus_client.drop_collection(self.collection_name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields=fields)
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(field_name = "vector", metric_type="L2")
        self.milvus_client.create_collection(self.collection_name, schema=schema, index_params=index_params, consistency_level="Strong")

        csv_data = pd.read_csv(DATA_FILE)
        for i in range(csv_data.shape[0]):
            row = {}
            for col in csv_data.columns.values:
                if col == "vector":
                    vec = json.loads(csv_data[col][i]) # convert the string format vector to List[float]
                    row[col] = vec
                else:
                    row[col] = csv_data[col][i]
            self.milvus_client.insert(self.collection_name, row)

        # dump
        with tempfile.TemporaryDirectory() as temp_dir:
            dump_collection('./local_test.db', self.collection_name, temp_dir)

    @unittest.skip("")            
    def test_all_type_collection(self):
        has_collection = self.milvus_client.has_collection(self.collection_name)
        if has_collection:
            self.milvus_client.drop_collection(self.collection_name)        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="bool", dtype=DataType.BOOL),
            FieldSchema(name="int8", dtype=DataType.INT8),
            FieldSchema(name="int16", dtype=DataType.INT16),
            FieldSchema(name="int32", dtype=DataType.INT32),
            FieldSchema(name="int64", dtype=DataType.INT64),
            FieldSchema(name="float", dtype=DataType.FLOAT),
            FieldSchema(name="double", dtype=DataType.DOUBLE),
            FieldSchema(name="varchar", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="json", dtype=DataType.JSON),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM)
        ]

        fields.append(FieldSchema(name="array_str", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.VARCHAR, max_length=128))
        fields.append(FieldSchema(name="array_int", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64))
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(field_name = "vector", metric_type="L2")
        self.milvus_client.create_collection(self.collection_name, schema=schema, index_params=index_params, consistency_level="Strong")
        rows = all_types_data(False)
        self.milvus_client.insert(self.collection_name, rows)
        # dump
        with tempfile.TemporaryDirectory() as temp_dir:
            dump_collection('./local_test.db', self.collection_name, temp_dir)

    def test_all_type_bin_vec_collection(self):
        has_collection = self.milvus_client.has_collection(self.collection_name)
        if has_collection:
            self.milvus_client.drop_collection(self.collection_name)                
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="bool", dtype=DataType.BOOL),
            FieldSchema(name="int8", dtype=DataType.INT8),
            FieldSchema(name="int16", dtype=DataType.INT16),
            FieldSchema(name="int32", dtype=DataType.INT32),
            FieldSchema(name="int64", dtype=DataType.INT64),
            FieldSchema(name="float", dtype=DataType.FLOAT),
            FieldSchema(name="double", dtype=DataType.DOUBLE),
            FieldSchema(name="varchar", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="json", dtype=DataType.JSON),
            FieldSchema(name="vector", dtype=DataType.BINARY_VECTOR, dim=DIM)
        ]

        fields.append(FieldSchema(name="array_str", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.VARCHAR, max_length=128))
        fields.append(FieldSchema(name="array_int", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64))
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        self.milvus_client.create_collection(self.collection_name, schema=schema, consistency_level="Strong")
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index("vector", "BIN_FLAT", metric_type="HAMMING")
        # index_params = [{"index_type": "BIN_FLAT", "metric_type": "HAMMING"}]
        self.milvus_client.create_index(self.collection_name, index_params=index_params)
        rows = all_types_data(True)
        self.milvus_client.insert(self.collection_name, rows)
        # dump
        with tempfile.TemporaryDirectory() as temp_dir:            
            dump_collection('./local_test.db', self.collection_name, temp_dir)


if __name__ == '__main__':
    unittest.main()
