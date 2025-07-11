# A demo showing hybrid semantic search with dense and full text search with BM25
# using Milvus.
#
# You can optionally choose to use the BGE-M3 model to embed the text as dense
# vectors, or simply use random generated vectors as an example.
#
# You can also use the BGE CrossEncoder model to rerank the search results.
#
# Note that the full text search feature is only available in Milvus 2.4.0 or
# higher version. Make sure you follow https://milvus.io/docs/install_standalone-docker.md
# to set up the latest version of Milvus in your local environment.

# To connect to Milvus server, you need the python client library called pymilvus.
# To use BGE-M3 model, you need to install the optional `model` module in pymilvus.
# You can get them by simply running the following commands:
#
# pip install pymilvus
# pip install pymilvus[model]

# If true, use BGE-M3 model to generate dense vectors.
# If false, use random numbers to compose dense vectors.
use_bge_m3 = False
# If true, the search result will be reranked using BGE CrossEncoder model.
use_reranker = False

# The overall steps are as follows:
# 1. embed the text as dense and sparse vectors
# 2. setup a Milvus collection to store the dense and sparse vectors
# 3. insert the data to Milvus
# 4. search and inspect the result!
import random
import string
import numpy as np

from pymilvus import (
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    RRFRanker,
    connections,
    Function,
    FunctionType,
)

# 1. prepare a small corpus to search
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
# add some randomly generated texts
docs.extend(
    [
        " ".join(
            "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(1, 8)))
            for _ in range(10)
        )
        for _ in range(1000)
    ]
)
query = "Who started AI research?"


def random_embedding(texts):
    rng = np.random.default_rng()
    return {
        "dense": np.random.rand(len(texts), 768),
    }


dense_dim = 768
ef = random_embedding

if use_bge_m3:
    # BGE-M3 model is included in the optional `model` module in pymilvus, to
    # install it, simply run "pip install pymilvus[model]".
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim["dense"]

docs_embeddings = ef(docs)
query_embeddings = ef([query])

# 2. setup Milvus collection and index
connections.connect("default", uri="milvus.db")

# Specify the data schema for the new Collection.
fields = [
    # Use auto generated id as primary key
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    # Store the original text to retrieve based on semantically distance
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512, enable_analyzer=True),
    # We need a sparse vector field to perform full text search with BM25,
    # but you don't need to provide data for it when inserting data.
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
functions = [
    Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names="sparse_vector",
    )
]
schema = CollectionSchema(fields, "", functions=functions)
col_name = "hybrid_bm25_demo"
# Now we can create the new collection with above name and schema.
col = Collection(col_name, schema, consistency_level="Strong")

# We need to create indices for the vector fields. The indices will be loaded
# into memory for efficient search.
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "FLAT", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)
col.load()

# 3. insert text and sparse/dense vector representations into the collection
entities = [docs, docs_embeddings["dense"]]
col.insert(entities)
col.flush()

# 4. search and inspect the result!
k = 2  # we want to get the top 2 docs closest to the query

# Prepare the search requests for both full text search and dense vector search
full_text_search_params = {"metric_type": "BM25"}
# provide raw text query for full text search, while use the sparse vector as
# ANNS field
full_text_search_req = AnnSearchRequest([query], "sparse_vector", full_text_search_params, limit=k)
dense_search_params = {"metric_type": "IP"}
dense_req = AnnSearchRequest(
    query_embeddings["dense"], "dense_vector", dense_search_params, limit=k
)

# Search topK docs based on dense and sparse vectors and rerank with RRF.
res = col.hybrid_search(
    [full_text_search_req, dense_req], rerank=RRFRanker(), limit=k, output_fields=["text"]
)

# Currently Milvus only support 1 query in the same hybrid search request, so
# we inspect res[0] directly. In future release Milvus will accept batch
# hybrid search queries in the same call.
res = res[0]

if use_reranker:
    result_texts = [hit.fields["text"] for hit in res]
    from pymilvus.model.reranker import BGERerankFunction

    bge_rf = BGERerankFunction(device="cpu")
    # rerank the results using BGE CrossEncoder model
    results = bge_rf(query, result_texts, top_k=2)
    for hit in results:
        print(f"text: {hit.text} distance {hit.score}")
else:
    for hit in res:
        print(f'text: {hit.fields["text"]} distance {hit.distance}')

# If you used both BGE-M3 and the reranker, you should see the following:
# text: Alan Turing was the first person to conduct substantial research in AI. distance 0.9306981017573297
# text: Artificial intelligence was founded as an academic discipline in 1956. distance 0.03217001154515051
#
# If you used only BGE-M3, you should see the following:
# text: Alan Turing was the first person to conduct substantial research in AI. distance 0.032786883413791656
# text: Artificial intelligence was founded as an academic discipline in 1956. distance 0.016129031777381897

# In this simple example the reranker yields the same result as the embedding based hybrid search, but in more complex
# scenarios the reranker can provide more accurate results.

# If you used random vectors, the result will be different each time you run the script.

# Drop the collection to clean up the data.
utility.drop_collection(col_name)
