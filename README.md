# Introduction
Milvus Lite is the lightweight version of [Milvus](https://github.com/milvus-io/milvus), an open-source vector database that powers AI applications with vector embeddings and similarity search.

Milvus Lite can be imported into your Python application, providing the core vector search functionality of Milvus. Milvus Lite is included in the [Python SDK of Milvus](https://github.com/milvus-io/pymilvus), thus it can be simply deployed with `pip install pymilvus`. This repo contains the core components of Milvus Lite.

Milvus Lite shares the same API and covers most of the features of Milvus. Together, they provide a consistent user experience across different types of environments, fitting use cases of different size. With the same client-side code, you can run a quick demo of less than a million vectors with Milvus Lite, or a small scale app with Milvus Docker container hosted on a single machine, and eventually to a large scale production deployment on Kubenetes serving billions of vectors at thousands of QPS. 

With Milvus Lite, you can start building an AI application with vector similarity search within minutes! Milvus Lite is good for running in the following environment:
- Jupyter Notebook / Google Colab
- Laptops
- Edge Devices that runs Linux

# Requirements
Milvus Lite supports the following OS distributions and sillicon types:
- Ubuntu >= 20.04 (x86_64)
- MacOS >= 11.0 (Apple Silicon and x86_64)

Please note that Milvus Lite is good for getting started with vector search or building demos and prototypes. For a production use case, we recommend using Milvus on [Docker](https://milvus.io/docs/install_standalone-docker.md) or [Kubenetes](https://milvus.io/docs/install_cluster-milvusoperator.md), or considering the fully-managed Milvus on [Zilliz Cloud](https://zilliz.com/cloud).

# Installation
Note that milvus-lite is included in `pymilvus` since version 2.4.2, so you can install with `pymilvus`
```shell
pip install "pymilvus>=2.4.2"
```

# Usage
In `pymilvus`, specify a local file name as uri parameter of MilvusClient to use Milvus Lite.
```python
from pymilvus import MilvusClient
client = MilvusClient("./milvus_demo.db")
```
Or with old `connections.connect` API (not recommended):
```python
from pymilvus import connections
connections.connect(uri="./milvus_demo.db")
```

# Examples
Following is a simple demo showing the idea of using Milvus Lite for text search. There are more comprehensive [examples](https://github.com/milvus-io/bootcamp/tree/master/bootcamp/tutorials) for using Milvus Lite to build applications
such as [RAG](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/build_RAG_with_milvus.ipynb), [image search](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/image_search_with_milvus.ipynb), and used with popular framework such as [LangChain](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/integration/rag_with_milvus_and_langchain.ipynb), [LlamaIndex](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/integration/rag_with_milvus_and_llamaindex.ipynb) and more!

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("./milvus_demo.db")
client.create_collection(
    collection_name="demo_collection",
    dimension=384  # The vectors we will use in this demo has 384 dimensions
)

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
# For illustration, here we use fake vectors with random numbers (384 dimension).

vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]
data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]
res = client.insert(
    collection_name="demo_collection",
    data=data
)

# This will exclude any text in "history" subject despite close to the query vector.
res = client.search(
    collection_name="demo_collection",
    data=[vectors[0]],
    filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],
)
print(res)

# a query that retrieves all entities matching filter expressions.
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
print(res)

# delete
res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'history'",
)
print(res)
```

# Contributing
If you want to contribute to Milvus Lite, please read the [Contributing Guide](https://github.com/milvus-io/milvus-lite/blob/main/CONTRIBUTING.md) first.

# Report a bug
When you use or develop milvus-lite, if you find any bug, please report it to us. You could submit an issue in [milvus-lite](https://github.com/milvus-io/milvus-lite/issues/new/choose) or report you [milvus](https://github.com/milvus-io/milvus/issues/new/choose) repo if you think it is a Milvus issue.

# License
Milvus Lite is under the Apache 2.0 license. See the [LICENSE](https://github.com/milvus-io/milvus-lite/blob/main/LICENSE) file for details.
