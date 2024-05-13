# Introduction
Milvus is an open-source vector database built to power embedding similarity search and AI applications. Milvus makes unstructured data search more accessible, and provides a consistent user experience regardless of the deployment environment.
Milvus Lite is a lightweight version of Milvus that can be embedded into your Python application, integrating the core vector search functionality of Milvus.
Please note that it is not suggested to use Milvus Lite in a production environment. Consider using Milvus clustered or the fully managed Milvus on Cloud.

# Requirements
Milvus Lite is available in:
- Google Colab
- Jupyter Notebook

Here's also a list of verified OS types where Milvus Lite can successfully build and run:
- Ubuntu >= 20.04 (x86_64)
- MacOS >= 11.0 (Apple Silicon)

# Installation
Note that milvus-lite is included in pymilvus since version 2.4.2.
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
