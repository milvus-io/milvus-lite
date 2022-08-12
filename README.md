# Introduction

The embedded Milvus brings up a Milvus instance on that starts and exits whenever you wish it to, while keeping all data and logs persistent. You can have it work within two simple steps:

```python
$ python3 -m pip install milvus
$ (in python) >>> import milvus
```

Embedded milvus does not have any other dependencies and do not require anything pre-installed, including Etcd, MinIO, etc.

Everything you do with embedded Milvus, every piece of code you write for embedded Milvus can be safely migrated to other forms of Milvus (standalone version, cluster version, cloud version, etc.) or simply "Write once, run anywhere".

Please note that it is not suggested to use embedded Milvus in a production environment. Consider using Milvus clustered or the fully managed Milvus on Cloud. 

## Configuration

A configurable file will be created on initial start located at `/var/bin/e-milvus/configs/embedded-milvus.yaml`

## Data and Log Persistence

All data and logs are persistent and will be stored under `/var/bin/e-milvus` by default. If you want them somewhere else, you can update the embedded Milvus configuration file.

## Working with PyMilvus

Embedded Milvus depends on PyMilvus. We are keeping our PyMilvus version consistent with the embedded Milvus version.

## Release Plan

Embedded Milvus is released together with the main Milvus project and will adopt Milvus's version as its own version.

Embedded Milvus always depends on the most suitable PyMilvus version when released.

# Requirements

```shell
supported OS: Ubuntu 18.04, Mac x86_64, Mac M1
python >= 3.6
```

# Installation

* You can install embedded Milvus via `pip` for Python 3.6+:

    ```shell
    $ python3 -m pip install milvus
    ```
    
    Or if you already have required version of PyMilvus installed:
    
    ```shell
    $ python3 -m pip install --no-deps milvus
    ```
    
    You can also install a specific version of embedded Milvus by:
    
    ```shell
    $ python3 -m pip install milvus==2.0.2rc4
    ```
    
    You can upgrade embedded Milvus by:
    
    ```shell
    $ python3 -m pip install --upgrade milvus
    ```
* After installation, you need to create the data folder for Milvus under /var/bin/e-mllvus:
    ```shell
    $ sudo mkdir -p /var/bin/e-milvus
    $ sudo chmod -R 777 /var/bin/e-milvus
    ```

# Running Embedded Milvus

1. If you are running for the first time. Import and then run `milvus.before()` for setup instructions.
```shell
$ python3
Python 3.9.10 (main, Jan 15 2022, 11:40:53)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import milvus
--- if you are running Milvus for the first time, type milvus.before() for pre-run instructions ---
--- otherwise, type milvus.start() ---
>>>
>>> milvus.before()
please do the following if you haven not already done so:
1. install required dependencies: bash /var/bin/e-milvus/lib/install_deps.sh
2. (Linux system only) export LD_PRELOAD=/Users/yuchengao/Documents/GitHub/soothing-rain/embd-milvus/milvus/bin/embd-milvus.so
3. (on Linux systems) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:/var/bin/e-milvus/lib/
   (on MacOS systems) export DYLD_FALLBACK_LIBRARY_PATH=DYLD_FALLBACK_LIBRARY_PATH:/usr/lib:/usr/local/lib:/var/bin/e-milvus/lib/
>>>
```

2. If you have not yet installed the required dependency, do so as instructed in 1.
```bash
# exit() python interactive mode first
# Note that this must be done AFTER `import milvus`
$ bash /var/bin/e-milvus/lib/install_deps.sh
```

3. If you have not yet set the environment variable, do so as instructed in 1.
```bash
# exit() python interactive mode first
# Note that this must be done AFTER `import milvus`
$ (Linux system only) export LD_PRELOAD=/Users/yuchengao/Documents/GitHub/soothing-rain/embd-milvus/milvus/bin/embd-milvus.so
(on Linux systems) $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:/var/bin/e-milvus/lib/
(on MacOS systems) $ export DYLD_FALLBACK_LIBRARY_PATH=DYLD_FALLBACK_LIBRARY_PATH:/usr/lib:/usr/local/lib:/var/bin/e-milvus/lib/
```

4. Start Milvus:

```python
$ python3
Python 3.9.10 (main, Jan 15 2022, 11:40:53)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import milvus
--- if you are running Milvus for the first time, type milvus.before() for pre-run instructions ---
--- otherwise, type milvus.start() ---
>>>
>>> milvus.start()
---Milvus Proxy successfully initialized and ready to serve!---
>>>
```

Milvus is now ready. There are two ways to interact with embedded Milvus, we have included a Hello Milvus test script that you can try out.

(1) You can also connect to embedded Milvus from Milvus SDK. Take PyMilvus SDK for example:

```shell
$ python3 /var/bin/e-milvus/examples/hello_milvus.py
```
    
For a full example, look at [Hello Milvus](https://milvus.io/docs/v2.0.x/example_code.md).

(2) Within in the same python interactive mode terminal, type and run your command directly:
```python
$ python3
Python 3.9.10 (main, Jan 15 2022, 11:40:53)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import milvus
--- if you are running Milvus for the first time, type milvus.before() for pre-run instructions ---
--- otherwise, type milvus.start() ---
>>>
>>> milvus.start()
---Milvus Proxy successfully initialized and ready to serve!---
>>>
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
>>>
```



4. Finally, when you are done, it is highly recommended that you stop Milvus gracefully and use exit() or Ctrl-D (i.e. EOF) to exit.

```python
>>> milvus.stop()
to clean up, run:
(Linux system only) export LD_PRELOAD=
(on Linux) export LD_LIBRARY_PATH=
(on MacOS) export DYLD_FALLBACK_LIBRARY_PATH=
>>>
>>> exit()
```

# FAQ
Q: Embedded-Milvus quited with segmentation error on Linux systems.

A: Start another terminal window to run your Milvus client. This is because setting env variable `LD_PRELOAD` in Linux might cause some conflicts.

---

Q: I have other issues.

A: Please file an issue here: https://github.com/milvus-io/embd-milvus/issues/new

---


# Building the Package

1. In Milvus repository, build it with:
```shell
$ make embd-milvus
```
2. Copy `embedded-milvus.yaml` file and `easylogging.yaml` to `config` folder. (See below for a sample directory structure)

3. Upon successful make, a dynamic library (embd-milvus.so and embd-milvus.h) will be created, create a new folder `bin` and put these two files in (See below for a sample directory structure). 

4. Upon successful make, Milvus related shared libraries will be created under `internal/core/output/lib`, create a new folder `lib` and put *all* these files in. (See below for a sample directory structure)

5. After the steps above, your repository should have the structure like below:
```shell
.
├── LICENSE
├── README.md
├── milvus
│   ├── __init__.py
│   ├── bin
│   │   ├── embd-milvus.h
│   │   └── embd-milvus.so
│   ├── configs
│   │   ├── easylogging.yaml
│   │   └── embedded-milvus.yaml
│   ├── examples
│   │   └── hello_milvus.py
│   └── lib
│       ├── cmake
│       │   ├── ...
│       ├── libarrow.a
│       ├── libarrow_bundled_dependencies.a
│       ├── libfaiss.a
│       ├── libknowhere.dylib
│       ├── libmarisa.0.dylib
│       ├── libmarisa.a
│       ├── libmarisa.dylib -> libmarisa.0.dylib
│       ├── libmarisa.la
│       ├── libmilvus_common.dylib
│       ├── libmilvus_index.dylib
│       ├── libmilvus_indexbuilder.dylib
│       ├── libmilvus_segcore.dylib
│       ├── libmilvus_storage.dylib
│       ├── libparquet.a
│       ├── librocksdb.a
│       └── pkgconfig
│           ├── arrow-compute.pc
│           ├── arrow.pc
│           ├── marisa.pc
│           ├── milvus_common.pc
│           ├── milvus_indexbuilder.pc
│           ├── milvus_segcore.pc
│           ├── milvus_storage.pc
│           ├── parquet.pc
│           └── rocksdb.pc
├── ...
├── myeasylog.log
└── setup.py
```

5. Build the wheel:
```shell
$ python3 setup.py bdist_wheel
```

6. Double check that the wheel has the right files included:
$ unzip -l dist/milvus-{version}-{python}-{abi}-{platform}.whl


7. Test it locally

Under the embd-milvus directory, start a virtual environment:

```shell
$ python3 -m pip install virtualenv
$ virtualenv venv
$ source venv/bin/activate
# Force install the wheel you just built in the last step.
(venv) $ python3 -m pip install --upgrade --force-reinstall --no-deps ./dist/milvus-{version}-{python}-{abi}-{platform}.whl
(venv) $ python3
Python 3.9.10 (main, Jan 15 2022, 11:40:53)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import milvus
...
```

8. If everything's good. Upload it to TestPyPI and PyPI.
```shell
$ python3 -m twine upload --repository testpypi dist/*
$ python3 -m twine upload dist/*
```

9. Your package will be downloadable and installable now.
```shell
$ python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps milvus
$ python3 -m pip install --no-deps milvus
```
