# Embeded Milvus

[![PyPI Version](https://img.shields.io/pypi/v/milvus.svg)](https://pypi.python.org/pypi/milvus)

## Introduction

The embedded Milvus is a lightweight version of Milvus that can be embedded into your Python application. It is a single binary that can be easily installed and run on your machine.

It could be imported as a Python library, as well as use it as a command standalone server.

Thanks to Milvus standalone version could be run with embeded etcd and local storage, embedded milvus does not have any other dependencies.

Everything you do with embedded Milvus, every piece of code you write for embedded Milvus can be safely migrated to other forms of Milvus (standalone version, cluster version, cloud version, etc.).

Please note that it is not suggested to use embedded Milvus in a production environment. Consider using Milvus clustered or the fully managed Milvus on Cloud. 



## Requirements

Here's a list of verified OS types where Embedded Milvus can successfully build and run:
- Ubuntu >= 18.04 (x86_64)
- CentOS >= 7.0 (x86_64)
- MacOS >= 11.0 (Apple Silicon)

For linux we use manylinux2014 as the base image, so it should be able to run on most linux distributions.

## Installation

Embedded Milvus is available on PyPI. You can install it via `pip` for Python 3.6+:

```bash
$ python3 -m pip install milvus
```

Or, install with client(pymilvus):
```bash
$ python3 -m pip install "milvus[client]"
```

## Usage

### Import as Python library
You could load the `default_server` in Python and start it.

```python
from milvus import default_server
from pymilvus import connections

# Optional, if you want store all related data to specific location
# default it wil using:
#   %APPDATA%/milvus-io/milvus-server on windows
#   ~/.milvus-io/milvus-server on linux
default_server.set_base_dir('milvus_data')

# Optional, if you want cleanup previous data
default_server.cleanup()

# star you milvus server
default_server.start()

# Now you could connect with localhost and the port
# The port is in default_server.listen_port
connections.connect(host='127.0.0.1', port=default_server.listen_port)
```

### CLI milvus-server

You could also use the `milvus-server` command to start the server.

```bash
$ milvus-serevr
```

The full options cloud be found by `milvus-server --help`.


## Advanced usage

### Debug startup

You could use `debug_server` instead of `default_server` for checking startup failures.

```python
from milvus import debug_server
```

and you could also try create server instance by your self

```python
from milvus import MilvusServer

server = MilvusServer(debug=True)
```

If you're using CLI `milvus-server`, you could use `--debug` to enable debug mode.

```bash
$ milvus-server --debug
```

### Context

You could close server while you not need it anymore.
Or, you're able to using `with` context to start/stop it.

```python
from milvus import default_server

with default_server:
    # milvus started, using default server here
    ...
```

### Data and Log Persistence

By default all data and logs are stored in the following locations: `~/.milvus.io/milvus-server/VERSION` (VERSION is the versiong string of embedded Milvus).

You could also set it at runtime(before the server started), by Python code:

```python
from milvus import default_server
default_server.set_base_dir('milvus_data')
```

Or with CLI:

```bash
$ milvus-server --data milvus_data
```

### Working with PyMilvus

Embedded Milvus could be run without pymilvus if you just want run as a server.
You could also install with extras `client` to get pymilvus.

```bash
$ python3 -m pip install "milvus[client]"
```

## Examples

Embedded Milvus is friendly with jupyter notebook, you could find more examples under [examples](https://github.com/milvus-io/embd-milvus/blob/main/examples) folder.

## Contributing
If you want to contribute to Embedded Milvus, please read the [Contributing Guide](https://github.com/milvus-io/embd-milvus/blob/main/CONTRIBUTING.md) first.

## License
Embedded Milvus is under the Apache 2.0 license. See the [LICENSE](https://github.com/milvus-io/embd-milvus/blob/main/LICENSE) file for details.
