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
from pymilvus import (
    FieldSchema, CollectionSchema, DataType, utility,
    Collection, AnnSearchRequest, RRFRanker, connections, WeightedRanker
)

from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

connections.connect("default", uri="milvus.db")
analyzer = build_default_analyzer(language="en")
bm25_ef = BM25EmbeddingFunction(analyzer)


class TestHybridSearch(unittest.TestCase):
    def test_hybrid_search(self):
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR,
                        is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                        dim=2),
        ]
        schema = CollectionSchema(fields, "")
        col_name = 'hybrid_test'
        utility.drop_collection(col_name)
        col = Collection(col_name, schema, consistency_level="Strong")
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "FLAT", "metric_type": "L2"}
        col.create_index("dense_vector", dense_index)
        col.load()
        docs = [
            "Artificial intelligence was founded as an academic discipline in 1956.",
            "Alan Turing was the first person to conduct substantial research in AI.",
            "Born in Maida Vale, London, Turing was raised in southern England.",
        ]
        query = "Who started AI research?"
        bm25_ef.fit(docs)
        docs_embeddings = bm25_ef.encode_documents(docs)
        entities = [docs, docs_embeddings, [[0.1, 0.9], [0.0, 1.0], [1.0, 0.0]]]

        k = 2
        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(bm25_ef.encode_queries([query]),
                                      "sparse_vector", sparse_search_params, limit=k)
        dense_search_params = {"metric_type": "L2"}
        dense_req = AnnSearchRequest([[0.0, 0.1]],
                                     "dense_vector", dense_search_params, limit=k)

        # empty table
        res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=k, output_fields=['text'])
        self.assertEqual(res[0], [])

        ################################# nq = 1 #####################
        col.insert(entities)

        k = 2
        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(bm25_ef.encode_queries([query]),
                                      "sparse_vector", sparse_search_params, limit=k)
        dense_search_params = {"metric_type": "L2"}
        dense_req = AnnSearchRequest([[0.0, 0.1]],
                                     "dense_vector", dense_search_params, limit=k)

        res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=k, output_fields=['text'])
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(res[0][0].fields['text'], docs[1])
        self.assertEqual(res[0][1].fields['text'], docs[0])

        ##################### offset #########################
        res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=k, offset=10, output_fields=['text'])
        self.assertEqual(res[0], [])

        res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=k, offset=1, output_fields=['text'])
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertEqual(res[0][0].fields['text'], docs[0])

        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(bm25_ef.encode_queries([query]),
                                      "sparse_vector", sparse_search_params, limit=10)
        dense_search_params = {"metric_type": "L2"}
        dense_req = AnnSearchRequest([[0.0, 0.1]],
                                     "dense_vector", dense_search_params, limit=10)

        res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=k, offset=1, output_fields=['text'])
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(res[0][0].fields['text'], docs[0])
        self.assertEqual(res[0][1].fields['text'], docs[2])

        ############################### nq = 2 ###################
        k = 3
        dense_search_params = {"metric_type": "L2"}
        dense_req1 = AnnSearchRequest([[0.9, 0.1], [0.1, 0.9]],
                                     "dense_vector", dense_search_params, limit=k)

        dense_req2 = AnnSearchRequest([[0.9, 0.1], [0.1, 0.9]],
                                      "dense_vector", dense_search_params, limit=k)

        res = col.hybrid_search([dense_req1, dense_req2], rerank=WeightedRanker(0.9, 0.1),
                                limit=k, output_fields=['text'])
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 3)
        self.assertEqual(len(res[1]), 3)
        self.assertEqual(res[0][0].fields['text'], docs[2])
        self.assertEqual(res[0][1].fields['text'], docs[0])
        self.assertEqual(res[0][2].fields['text'], docs[1])

        self.assertEqual(res[1][0].fields['text'], docs[0])
        self.assertEqual(res[1][1].fields['text'], docs[1])
        self.assertEqual(res[1][2].fields['text'], docs[2])

        ########################## offset ####################

        res = col.hybrid_search([dense_req1, dense_req2], rerank=RRFRanker(),
                                limit=k, offset=10, output_fields=['text'])
        self.assertEqual(res[0], [])

        res = col.hybrid_search([dense_req1, dense_req2], rerank=RRFRanker(),
                                limit=k, offset=1, output_fields=['text'])
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(len(res[1]), 2)        
        self.assertEqual(res[0][0].fields['text'], docs[0])
        self.assertEqual(res[1][0].fields['text'], docs[1])        
        col.drop()

    def test_dynamic(self):
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64,
                        is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                        dim=2)
        ]
        schema = CollectionSchema(fields, "", enable_dynamic_field=True)
        col_name = 'hybrid_test_dynamic'
        utility.drop_collection(col_name)
        col = Collection(col_name, schema, consistency_level="Strong")
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "FLAT", "metric_type": "L2"}
        col.create_index("dense_vector", dense_index)
        col.load()
        docs = [
            "Artificial intelligence was founded as an academic discipline in 1956.",
            "Alan Turing was the first person to conduct substantial research in AI.",
            "Born in Maida Vale, London, Turing was raised in southern England.",
        ]
        query = "Who started AI research?"
        bm25_ef.fit(docs)
        docs_embeddings = bm25_ef.encode_documents(docs)
        data_rows = [
            {"pk": 1, "text": docs[0], "sparse_vector": docs_embeddings[0], "dense_vector": [0.1, 0.9], "a": 1, "c": 2},
            {"pk": 2, "text": docs[1], "sparse_vector": docs_embeddings[1], "dense_vector": [0.0, 1.0], "a": 4, "d": 2},
            {"pk": 3, "text": docs[2], "sparse_vector": docs_embeddings[2], "dense_vector": [1.0, 0.0], "d": 1, "c": 4},
        ]

        k = 2
        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(bm25_ef.encode_queries([query]),
                                      "sparse_vector", sparse_search_params, limit=k)
        dense_search_params = {"metric_type": "L2"}
        dense_req = AnnSearchRequest([[0.0, 0.1]],
                                     "dense_vector", dense_search_params, limit=k)

        # empty table
        res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=k, output_fields=['text'])
        self.assertEqual(res[0], [])

        # nq = 1
        col.insert(data_rows)
        
        k = 2
        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(bm25_ef.encode_queries([query]),
                                      "sparse_vector", sparse_search_params, limit=k)
        dense_search_params = {"metric_type": "L2"}
        dense_req = AnnSearchRequest([[0.0, 0.1]],
                                     "dense_vector", dense_search_params, limit=k)

        res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=k, output_fields=['text', 'a'])

        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(res[0][0].fields['text'], docs[1])
        self.assertEqual(res[0][0].fields['a'], 4)
        self.assertEqual(res[0][1].fields['text'], docs[0])
        self.assertEqual(res[0][1].fields['a'], 1)

        # nq = 2
        k = 10
        dense_search_params = {"metric_type": "L2"}
        dense_req1 = AnnSearchRequest([[0.9, 0.1], [0.1, 0.9]],
                                     "dense_vector", dense_search_params, limit=k)

        dense_req2 = AnnSearchRequest([[0.9, 0.1], [0.1, 0.9]],
                                      "dense_vector", dense_search_params, limit=k)        

        res = col.hybrid_search([dense_req1, dense_req2], rerank=WeightedRanker(0.9, 0.1),
                                limit=k, output_fields=['text', 'c'])

        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 3)
        self.assertEqual(len(res[1]), 3)
        self.assertEqual(res[0][0].fields['text'], docs[2])
        self.assertEqual(res[0][0].fields['c'], 4)
        self.assertEqual(res[0][1].fields['text'], docs[0])
        self.assertEqual(res[0][1].fields['c'], 2)
        self.assertEqual(res[0][2].fields['text'], docs[1])
        self.assertEqual(res[0][2].fields.get('c'), None)

        self.assertEqual(res[1][0].fields['text'], docs[0])
        self.assertEqual(res[1][0].fields['c'], 2)
        self.assertEqual(res[1][1].fields['text'], docs[1])
        self.assertEqual(res[1][1].get('c'), None)
        self.assertEqual(res[1][2].fields['text'], docs[2])
        self.assertEqual(res[1][2].fields.get('c'), 4)
        col.drop()        


if __name__ == '__main__':
    unittest.main()        
