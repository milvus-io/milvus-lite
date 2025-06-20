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
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker


class TestBM25(unittest.TestCase):
    def test_single_function(self):
        collection_name = "hello_milvus"
        milvus_client = MilvusClient("./local_test.db")
        has_collection = milvus_client.has_collection(collection_name)        
        if has_collection:
            milvus_client.drop_collection(collection_name)
        schema = milvus_client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field(
            "document_content", DataType.VARCHAR, max_length=9000, enable_analyzer=True
        )
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        
        bm25_function = Function(
            name="bm25_fn",
            input_field_names=["document_content"],
            output_field_names="sparse_vector",
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"bm25_k1": 1.2, "bm25_b": 0.75},
        )

        milvus_client.create_collection(
            collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

        rows = [
            {"id": 1, "document_content": "hello world"},
            {"id": 2, "document_content": "hello milvus"},
            {"id": 3, "document_content": "hello zilliz"},
        ]

        insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
        texts_to_search = ["hello"]
        search_params = {"metric_type": "BM25"}

        result = milvus_client.search(
            collection_name,
            texts_to_search,
            limit=3,
            output_fields=["document_content"],
            search_params=search_params,
        )
        scores = [hit['distance'] for hit in result[0]]
        self.assertAlmostEqual(scores, [0.133531391620636, 0.133531391620636, 0.133531391620636])

        texts_to_search = ["hello", "hello, zilliz"]
        search_params = {"metric_type": "BM25"}

        result = milvus_client.search(
            collection_name,
            texts_to_search,
            limit=3,
            output_fields=["document_content"],
            search_params=search_params,
        )
        scores1 = [hit['distance'] for hit in result[0]]
        self.assertAlmostEqual(scores1, [0.133531391620636, 0.133531391620636, 0.133531391620636])

        scores2 = [hit['distance'] for hit in result[1]]
        self.assertAlmostEqual(scores2, [1.1143605709075928, 0.133531391620636, 0.133531391620636])

    def test_multi_functions(self):
        collection_name = "hello_milvus"
        milvus_client = MilvusClient("./local_test.db")
        has_collection = milvus_client.has_collection(collection_name)        
        if has_collection:
            milvus_client.drop_collection(collection_name)
        schema = milvus_client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field(
            "document_content", DataType.VARCHAR, max_length=9000, enable_analyzer=True
        )
        schema.add_field("sparse_vector_1", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("sparse_vector_2", DataType.SPARSE_FLOAT_VECTOR)        
        
        f1 = Function(
            name="bm25_fn",
            input_field_names=["document_content"],
            output_field_names="sparse_vector_1",
            function_type=FunctionType.BM25,
        )
        schema.add_function(f1)

        f2 = Function(
            name="bm25_fn",
            input_field_names=["document_content"],
            output_field_names="sparse_vector_2",
            function_type=FunctionType.BM25,
        )
        schema.add_function(f2)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector_1",
            index_name="sparse_inverted_index_1",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"bm25_k1": 1.2, "bm25_b": 0.75},
        )
        index_params.add_index(
            field_name="sparse_vector_2",
            index_name="sparse_inverted_index_2",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"bm25_k1": 1.2, "bm25_b": 0.75},
        )

        milvus_client.create_collection(
            collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

        rows = [
            {"id": 1, "document_content": "hello world"},
            {"id": 2, "document_content": "hello milvus"},
            {"id": 3, "document_content": "hello zilliz"},
        ]

        milvus_client.insert(collection_name, rows, progress_bar=True)
        texts_to_search = ["hello"]
        search_params = {"metric_type": "BM25"}

        result = milvus_client.search(
            collection_name,
            texts_to_search,
            anns_field="sparse_vector_1",
            limit=3,
            output_fields=["document_content"],
            search_params=search_params,
        )
        scores = [hit['distance'] for hit in result[0]]
        self.assertAlmostEqual(scores, [0.133531391620636, 0.133531391620636, 0.133531391620636])

        texts_to_search = ["hello", "hello, zilliz"]
        search_params = {"metric_type": "BM25"}

        result = milvus_client.search(
            collection_name,
            texts_to_search,
            anns_field="sparse_vector_2",
            limit=3,
            output_fields=["document_content"],
            search_params=search_params,
        )
        scores1 = [hit['distance'] for hit in result[0]]
        self.assertAlmostEqual(scores1, [0.133531391620636, 0.133531391620636, 0.133531391620636])

        scores2 = [hit['distance'] for hit in result[1]]
        self.assertAlmostEqual(scores2, [1.1143605709075928, 0.133531391620636, 0.133531391620636])

    def test_bm25_hybrid_search(self):
        collection_name = "hello_milvus"
        milvus_client = MilvusClient("./local_test.db")
        has_collection = milvus_client.has_collection(collection_name)        
        if has_collection:
            milvus_client.drop_collection(collection_name)
        schema = milvus_client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field(
            "document_content", DataType.VARCHAR, max_length=9000, enable_analyzer=True
        )
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("sparse_vector_2", DataType.SPARSE_FLOAT_VECTOR)
        
        bm25_function = Function(
            name="bm25_fn",
            input_field_names=["document_content"],
            output_field_names="sparse_vector",
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        f2 = Function(
            name="bm25_fn",
            input_field_names=["document_content"],
            output_field_names="sparse_vector_2",
            function_type=FunctionType.BM25,
        )
        schema.add_function(f2)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"bm25_k1": 1.2, "bm25_b": 0.75},
        )
        index_params.add_index(
            field_name="sparse_vector_2",
            index_name="sparse_inverted_index_2",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"bm25_k1": 1.2, "bm25_b": 0.75},
        )

        milvus_client.create_collection(
            collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

        rows = [
            {"id": 1, "document_content": "hello world"},
            {"id": 2, "document_content": "hello milvus"},
            {"id": 3, "document_content": "hello zilliz"},
        ]

        milvus_client.insert(collection_name, rows, progress_bar=True)

        texts_to_search = ["hello"]
        search_params = {"metric_type": "BM25"}

        req1 = AnnSearchRequest(
            texts_to_search,
            "sparse_vector",
            search_params,
            limit=3,
        )
        req2 = AnnSearchRequest(
            texts_to_search,
            "sparse_vector_2",
            search_params,
            limit=3,
        )
        reqs = [req1, req2]
        result = milvus_client.hybrid_search(collection_name, reqs, ranker=RRFRanker(), limit=3, output_fields=["document_content"])
        print(result)



if __name__ == '__main__':
    unittest.main()
