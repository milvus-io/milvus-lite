<div align="center">
    <img src="milvus_lite_log.png#gh-light-mode-only" width="60%"/>
</div>

<h3 align="center">
    <p style="text-align: center;"> <span style="font-weight: bold; font: Arial, sans-serif;"></span>A lightweight version of Milvus</p>
</h3>

<div class="column" align="middle">
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/license-apache2.0-green?style=flat" alt="license"/>
  </a>
  <a href="https://pypi.org/project/milvus-lite/">
    <img src="https://img.shields.io/pypi/v/milvus-lite?label=Release&color&logo=Python" alt="github actions">
  </a>
    <a href="https://pypi.org/project/milvus-lite/">
    <img src="https://img.shields.io/pypi/dm/milvus-lite.svg?color=bright-green&logo=Pypi" alt="github actions">
  </a>
</div>


# Introduction
Milvus Lite is the lightweight version of [Milvus](https://github.com/milvus-io/milvus), a high-performance vector database that powers AI applications with vector similarity search. This repo contains the core components of Milvus Lite. 

With Milvus Lite, you can start building an AI application with vector similarity search within minutes! Milvus Lite is good for running in the following environment:
- Jupyter Notebook / Google Colab
- Laptops
- Edge Devices

Milvus Lite can be imported into your Python application, providing the core vector search functionality of Milvus. Milvus Lite is already included in the [Python SDK of Milvus](https://github.com/milvus-io/pymilvus). It can be simply deployed with `pip install pymilvus`. 

Milvus Lite shares the same API with Milvus Standalone and Distributed, and covers most of the features such as vector data persistence and management, vector CRUD operations, sparse and dense vector search, metadata filtering, multi-vector and hybrid_search. Together, they provide a consistent experience across different types of environments, from edge devices to clusters in cloud, fitting use cases of different size. With the same client-side code, you can run GenAI apps with Milvus Lite on a laptop or Jupyter Notebook, or Milvus Standalone on Docker container, or Milvus Distributed on massive scale Kubernetes cluster serving billions of vectors in production.

# Requirements
Milvus Lite currently supports the following environmnets:
- Ubuntu >= 20.04 (x86_64 and arm64)
- MacOS >= 11.0 (Apple Silicon M1/M2 and x86_64)

***Note:*** Windows is not yet supported.

Please note that Milvus Lite is only suitable for small scale vector search use cases. For a large scale use case, we recommend using [Milvus Standalone](https://milvus.io/docs/install-overview.md#Milvus-Standalone) or [Milvus Distributed](https://milvus.io/docs/install-overview.md#Milvus-Distributed). You can also consider the fully-managed Milvus on [Zilliz Cloud](https://zilliz.com/cloud).

# Installation
```shell
pip install -U pymilvus
```
We recommend using `pymilvus`. Since `milvus-lite` is included in `pymilvus` version 2.4.2 or above, you can `pip install` with `-U` to force update to the latest version and `milvus-lite` is automatically installed.


If you want to explicitly install `milvus-lite` package, or you have installed an older version of `milvus-lite` and would like to update it, you can do `pip install -U milvus-lite`.

# Usage
In `pymilvus`, specify a local file name as uri parameter of MilvusClient will use Milvus Lite.
```python
from pymilvus import MilvusClient
client = MilvusClient("./milvus_demo.db")
```

> **_NOTE:_**  Note that the same API also applies to Milvus Standalone, Milvus Distributed and Zilliz Cloud, the only difference is to replace local file name to remote server endpoint and credentials, e.g. 
`client = MilvusClient(uri="http://localhost:19530", token="username:password")`.

# Examples
Following is a simple demo showing how to use Milvus Lite for text search. There are more comprehensive [examples](https://github.com/milvus-io/bootcamp/tree/master/bootcamp/tutorials) for using Milvus Lite to build applications
such as [RAG](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/build_RAG_with_milvus.ipynb), [image search](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/image_search_with_milvus.ipynb), and using Milvus Lite in popular RAG framework such as [LangChain](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/integration/rag_with_milvus_and_langchain.ipynb) and [LlamaIndex](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/integration/rag_with_milvus_and_llamaindex.ipynb)!

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

# Supported Features

# Known Limitations
Milvus Lite only supports [FLAT](https://milvus.io/docs/index.md?tab=floating#FLAT) index type. It uses FLAT type regardless of the specified index type in collection.

Milvus Lite does not support partitions, users/roles/RBAC, alias. To use those features, please choose other Milvus deployment types such as [Standalone](https://milvus.io/docs/install-overview.md#Milvus-Standalone), [Distributed](https://milvus.io/docs/install-overview.md#Milvus-Distributed) or [Zilliz Cloud](https://zilliz.com/cloud) (fully-managed Milvus).

# Migrating data from Milvus Lite

All data stored in Milvus Lite can be easily exported and loaded into other types of Milvus deployment, such as Milvus Standalone on Docker, Milvus Distributed on K8s, or fully-managed Milvus on [Zilliz Cloud](https://zilliz.com/cloud).

Milvus Lite provides a command line tool that can dump data into a json file, which can be imported into [milvus](https://github.com/milvus-io/milvus) and [Zilliz Cloud](https://zilliz.com/cloud)(the fully managed cloud service for Milvus). The milvus-lite command will be installed together with milvus-lite python package 

```shell
# Install
pip install -U "pymilvus[bulk_writer]"

milvus-lite dump -h

usage: milvus-lite dump [-h] [-d DB_FILE] [-c COLLECTION] [-p PATH]

optional arguments:
  -h, --help            show this help message and exit
  -d DB_FILE, --db-file DB_FILE
                        milvus lite db file
  -c COLLECTION, --collection COLLECTION
                        collection that need to be dumped
  -p PATH, --path PATH  dump file storage dir
```
The following example dumps all data from `demo_collection` collection that's stored in `./milvus_demo.db` (Milvus Lite database file)

To export data:

```shell
milvus-lite dump -d ./milvus_demo.db -c demo_collection -p ./data_dir
# ./milvus_demo.db: milvus lite db file
# demo_collection: collection that need to be dumped
#./data_dir : dump file storage dir
```

With the dump file, you can upload data to Zilliz Cloud via [Data Import](https://docs.zilliz.com/docs/data-import), or upload data to Milvus servers via [Bulk Insert](https://milvus.io/docs/import-data.md).
# Contributing
If you want to contribute to Milvus Lite, please read the [Contributing Guide](https://github.com/milvus-io/milvus-lite/blob/main/CONTRIBUTING.md) first.

# Report a bug
For any bug or feature request, please report it by submitting an issue in [milvus-lite](https://github.com/milvus-io/milvus-lite/issues/new/choose) repo.

# License
Milvus Lite is under the Apache 2.0 license. See the [LICENSE](https://github.com/milvus-io/milvus-lite/blob/main/LICENSE) file for details.
p
