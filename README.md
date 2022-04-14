# Introduction

The embedded Milvus broughts up a Milvus instance on that starts and exits whenever you wish it to, while keeping all data and logs persistent. You can have it work within two simple steps:

```python
$ pip install milvus
$ (in python) >>> import milvus
```

Embedded milvus does not have any other dependencies and do not require anything pre-installed, including Etcd, MinIO, etc.

Everything you do with embedded Milvus, every piece of code you write for embedded Milvus can be safely migrated to other forms of Milvus (standalone version, cluster version, cloud version, etc.) or simply "Write once, run anywhere".

Please note that it is not suggested to use embedded Milvus in a production environment. Consider using Milvus clustered or the fully managed Milvus on Cloud. 

## Configuration

A configurable file will be created on initial start located at `/tmp/milvus/configs/embedded-milvus.yaml`

## Data and Log Persistence

All data and logs are persistent and will be stored under `/tmp/milvus/` by default. If you want them somewhere else, you can update the embedded Milvus configuration file.

## Working with PyMilvus

Embedded Milvus depends on PyMilvus. We are keeping our PyMilvus version consistent with the embedded Milvus version.

## Release Plan

Embedded Milvus is released together with the main Milvus project and will adopt Milvus's version as its own version.

Embedded Milvus always depends on the most suitable PyMilvus version when released.

# Requirements

```shell
python >= 3.9
```

# Installation

You can install embedded Milvus via `pip` or `pip3` for Python 3.6+:

```shell
$ pip3 install milvus
```

You can install a specific version of embedded Milvus by:

```shell
$ pip3 install milvus
```

You can upgrade embedded Milvus by:

```shell
$ pip3 install --upgrade milvus
```

# Running Embedded Milvus

1. Before you run, install dependencies for Milvus:
```shell
wget -O - https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/install_deps.sh | bash
```

2. Preload and set environment variables.
```shell
$ python3
Python 3.9.10 (main, Jan 15 2022, 11:40:53)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import milvus
>>> milvus.preload()
please run:
export LD_PRELOAD=${YOUR_LD_PRELOAD_VALUE}
export LD_LIBRARY_PATH=${YOUR_LD_LIBRARY_PATH_VALUE}
>>> exit()

$ export LD_PRELOAD=${${YOUR_LD_PRELOAD_VALUE}}
$ export LD_LIBRARY_PATH=${YOUR_LD_LIBRARY_PATH_VALUE}
```

3. Start milvus:

```python
$ python3
Python 3.9.10 (main, Jan 15 2022, 11:40:53)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import milvus
>>> milvus.start()
>>>
```

Milvus is now ready and you can start interacting with it. For a full example, you can look at [Hello Milvus](https://milvus.io/docs/v2.0.0/example_code.md).

```python
$ python3
Python 3.9.10 (main, Jan 15 2022, 11:40:53)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import milvus
>>> milvus.start()
>>>
>>> import random
>>> from pymilvus import (
...     connections,
...     utility,
...     FieldSchema, CollectionSchema, DataType,
...     Collection,
... )
>>> connections.connect("default", host="localhost", port="19530")
>>> has = utility.has_collection("hello_milvus")
>>> fields = [
...     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
...     FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
... ]
>>> schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
>>> hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")
>>> num_entities = 3000
>>> entities = [
...     [i for i in range(num_entities)], # provide the pk field because `auto_id` is set to False
...     [[random.random() for _ in range(8)] for _ in range(num_entities)],  # field embeddings
... ]
>>> insert_result = hello_milvus.insert(entities)
>>> index = {
...     "index_type": "IVF_FLAT",
...     "metric_type": "L2",
...     "params": {"nlist": 128},
... }
>>> hello_milvus.create_index("embeddings", index)
>>> hello_milvus.load()
>>> vectors_to_search = entities[-1][-2:]
>>> search_params = {
...     "metric_type": "l2",
...     "params": {"nprobe": 10},
... }
>>> result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3)
>>> for hits in result:
...     for hit in hits:
...         print(f"hit: {hit}")
...
hit: (distance: 0.0, id: 2998)
hit: (distance: 0.1088758111000061, id: 2345)
hit: (distance: 0.12012234330177307, id: 1172)
hit: (distance: 0.0, id: 2999)
hit: (distance: 0.0297045037150383, id: 2000)
hit: (distance: 0.16927233338356018, id: 560)
>>> utility.drop_collection("hello_milvus")
```

You can also start another python script or SDK to work with embedded Milvus while it is not closed. For example, in a new shell window, you could:

```shell
$ python3 ./examples/hello_milvus
```

Finally, when you are done, simply do exit().

```python
>>> exit()
$ 
```

# Building the Package

1. In Milvus repository, build it with:
```shell
$ make embd-milvus
```

2. Upon successful make, a dynamic library (embd-milvus.so and embd-milvus.h) will be created, create a new folder `bin` and put these two files in (See below for a complete directory structure). 

3. Upon successful make, Milvus related shared libraries will be created under `internal/core/output/lib`, create a new folder `lib` and put these files in (See below for a complete directory structure).

4. After the steps above, your repository should have the structure like below:
```shell
embd-milvus/
├── LICENSE
├── README.md
├── examples
│   └── hello_milvus.py
├── milvus
│   ├── __init__.py
│   ├── bin
│   │   ├── embd-milvus.h
│   │   └── embd-milvus.so
│   ├── configs
│   │   └── embedded-milvus.yaml
│   ├── lib
│   │   ├── libfaiss.a
│   │   ├── libknowhere.dylib               # (or .so)
│   │   ├── libmilvus_common.dylib          # (or .so)
│   │   ├── libmilvus_index.dylib           # (or .so)
│   │   ├── libmilvus_indexbuilder.dylib    # (or .so)
│   │   └── libmilvus_segcore.dylib         # (or .so)
│   └── milvus.py
└── setup.py
```

5. Run the following command to update environment variables:
```shell
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/tmp/milvus/lib"
$ export LD_PRELOAD="/tmp/milvus/lib/embd-milvus.so"
```

6. Build the wheel:
```shell
$ python3 setup.py bdist_wheel
```

7. Double check that the wheel has the right files included:
$ unzip -l dist/milvus-{version}-{python}-{abi}-{platform}.whl


8. Test it locally

Under the embd-milvus directory, start a virtual environment:

```shell
$ python3 -m pip install virtualenv
$ virtualenv venv
$ source venv/bin/activate
# Force install the wheel you just built in the last step.
(venv) $ pip install --upgrade --force-reinstall --no-deps ./dist/milvus-{version}-{python}-{abi}-{platform}.whl
(venv) $ python3
Python 3.9.10 (main, Jan 15 2022, 11:40:53)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import milvus
...
```

9. If everything's good. Upload it to TestPyPI and PyPI.
```shell
python3 -m twine upload --repository testpypi dist/*
```

10. Your package will be downloadable and installable now.
```shell
$ python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps milvus
```