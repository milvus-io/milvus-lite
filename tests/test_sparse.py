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
from pymilvus import MilvusClient, MilvusException, FieldSchema, CollectionSchema, DataType
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction


def csr_to_dict(csr):
    coo = csr.tocoo()
    ret = {}
    for i, j, v in zip(coo.row, coo.col, coo.data):
        ret[j] = v
    return ret


class TestSparse(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.collection_name = 'hello_milvus'
        self.milvus_client = MilvusClient('./local_test.db')
        has_collection = self.milvus_client.has_collection(self.collection_name, timeout=5)
        if has_collection:
            self.milvus_client.drop_collection(self.collection_name)
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR,
                        is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="num", dtype=DataType.INT64),
            FieldSchema(name="embeddings", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(
            fields, "demo for using sparse float vector with milvus client")
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(field_name="embeddings", index_name="sparse_inverted_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})
        self.milvus_client.create_collection(self.collection_name, schema=schema,
                                             index_params=index_params, timeout=5, consistency_level="Strong")

    def compare(self, crs, sparse_return):
        src = csr_to_dict(crs)
        for index, score in sparse_return.items():
            self.assertAlmostEqual(score, src[index])    

    def tearDown(self):
        self.milvus_client.drop_collection(self.collection_name)

    def test_query(self):
        analyzer = build_default_analyzer(language="en")
        corpus = [
            "Artificial intelligence was founded as an academic discipline in 1956.",
            "Alan Turing was the first person to conduct substantial research in AI.",
            "Born in Maida Vale, London, Turing was raised in southern England. Artificial intelligence",
        ]

        bm25_ef = BM25EmbeddingFunction(analyzer)
        bm25_ef.fit(corpus)
        rows = [{"num": i, "embeddings": bm25_ef.encode_documents([corpus[i]])} for i in range(len(corpus))]
        insert_result = self.milvus_client.insert(self.collection_name, rows, progress_bar=True)
        self.assertEqual(insert_result['insert_count'], 3)

        query_results = self.milvus_client.query(self.collection_name, filter="num <= 1")
        self.assertEqual([item['num'] for item in query_results], [0, 1])
        for i in range(len(query_results)):
            self.compare(rows[i]['embeddings'], query_results[i]['embeddings'])

        query_results = self.milvus_client.query(self.collection_name, filter="num <= 2", offset=1)
        self.assertEqual([item['num'] for item in query_results], [1, 2])
        for i in range(len(query_results)):
            self.compare(rows[i+1]['embeddings'], query_results[i]['embeddings'])        

        query_results = self.milvus_client.query(self.collection_name, filter="num <= 100")
        self.assertEqual([item['num'] for item in query_results], [0, 1, 2])
        for i in range(len(query_results)):
            self.compare(rows[i]['embeddings'], query_results[i]['embeddings'])        

        query_results = self.milvus_client.query(self.collection_name, filter="num <= 100", offset=10)
        self.assertEqual([item['num'] for item in query_results], [])

    def test_search(self):
        analyzer = build_default_analyzer(language="en")
        corpus = [
            "Artificial intelligence was founded as an academic discipline in 1956.",
            "Alan Turing was the first person to conduct substantial research in AI.",
            "Born in Maida Vale, London, Turing was raised in southern England. Artificial intelligence",
        ]

        bm25_ef = BM25EmbeddingFunction(analyzer)
        bm25_ef.fit(corpus)
        rows = [{"num": i, "embeddings": bm25_ef.encode_documents([corpus[i]])} for i in range(len(corpus))]
        insert_result = self.milvus_client.insert(self.collection_name, rows, progress_bar=True)
        self.assertEqual(insert_result['insert_count'], 3)

        # single search
        vectors_to_search = bm25_ef.encode_queries([corpus[0]])
        search_params = {
            "metric_type": "IP",
            "params": {
                "drop_ratio_search": 0.0,
            }
        }
        result = self.milvus_client.search(self.collection_name, vectors_to_search, limit=3, output_fields=[
            "pk", "num", "embeddings"], search_params=search_params)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual([item['entity']['num'] for item in result[0]], [0, 2])
        self.assertEqual([item['distance'] for item in result[0]], [2.507876396179199, 0.16398896276950836])
        result_rows = [rows[0], rows[2]]
        for i in range(len(result[0])):
            self.compare(result_rows[i]['embeddings'], result[0][i]['entity']['embeddings'])

        # multi search
        vectors_to_search = bm25_ef.encode_queries(corpus)
        search_params = {
            "metric_type": "IP",
            "params": {
                "drop_ratio_search": 0.0,
            }
        }
        result = self.milvus_client.search(self.collection_name, vectors_to_search, limit=3, output_fields=[
            "pk", "num", "embeddings"], search_params=search_params)

        index = 0
        self.assertEqual(len(result[index]), 2)
        self.assertEqual([item['entity']['num'] for item in result[index]], [0, 2])
        self.assertEqual([item['distance'] for item in result[index]], [2.507876396179199, 0.16398896276950836])
        result_rows = [rows[0], rows[2]]
        for i in range(len(result[index])):
            self.compare(result_rows[i]['embeddings'], result[index][i]['entity']['embeddings'])

        index = 1            
        self.assertEqual(len(result[index]), 2)
        self.assertEqual([item['entity']['num'] for item in result[index]], [1, 2])
        self.assertEqual([item['distance'] for item in result[index]], [3.6669983863830566, 0.08199448138475418])
        result_rows = [rows[1], rows[2]]
        for i in range(len(result[index])):
            self.compare(result_rows[i]['embeddings'], result[index][i]['entity']['embeddings'])

        index = 2
        self.assertEqual(len(result[index]), 3)
        self.assertEqual([item['entity']['num'] for item in result[index]], [2, 0, 1])
        self.assertEqual([item['distance'] for item in result[index]], [3.460167407989502, 0.20556363463401794, 0.09121885895729065])
        result_rows = [rows[2], rows[0], rows[1]]
        for i in range(len(result[index])):
            self.compare(result_rows[i]['embeddings'], result[index][i]['entity']['embeddings'])

        result = self.milvus_client.search(self.collection_name, vectors_to_search, limit=3, output_fields=[
            "pk", "num", "embeddings"], search_params=search_params, offset=2)
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(len(result[1]), 0)
        index = 2
        self.assertEqual(len(result[index]), 1)
        self.assertEqual([item['entity']['num'] for item in result[index]], [1])
        self.assertEqual([item['distance'] for item in result[index]], [0.09121885895729065])
        result_rows = [rows[1]]
        for i in range(len(result[index])):
            self.compare(result_rows[i]['embeddings'], result[index][i]['entity']['embeddings'])


if __name__ == '__main__':
    unittest.main()
