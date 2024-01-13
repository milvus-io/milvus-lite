# Milvus Lite

[![PyPI Version](https://img.shields.io/pypi/v/milvus.svg)](https://pypi.python.org/pypi/milvus)

## Introduction

Milvus Lite is a lightweight version of Milvus that can be embedded into your Python application. It is a single binary that can be easily installed and run on your machine.

It could be imported as a Python library, as well as use it as a command standalone server.

Thanks to Milvus standalone version could be run with embedded etcd and local storage, Milvus Lite does not have any other dependencies.

Everything you do with Milvus Lite, every piece of code you write for Milvus Lite can be safely migrated to other forms of Milvus (standalone version, cluster version, cloud version, etc.).

Please note that it is not suggested to use Milvus Lite in a production environment. Consider using Milvus clustered or the fully managed Milvus on Cloud. 


## Requirements

Milvus Lite is available in:
- Google Colab [example](https://github.com/milvus-io/milvus-lite/blob/main/examples/example.ipynb)
- Jupyter Notebook

Here's also a list of verified OS types where Milvus Lite can successfully build and run:
- Ubuntu >= 18.04 (x86_64)
- CentOS >= 7.0 (x86_64)
- MacOS >= 11.0 (Apple Silicon)

*NOTE*
* For linux we use manylinux2014 as the base image, so it should be able to run on most linux distributions.
* Milvus Lite can also run on **Windows**. However, this is not strictly verified.

## Installation

Milvus Lite is available on PyPI. You can install it via `pip` for Python 3.6+:

```bash
$ python3 -m pip install milvus
```

Or, install with client(pymilvus):
```bash
$ python3 -m pip install "milvus[client]"
```
Note: pymilvus now requires Python 3.7+

## Usage

### Import as Python library
Simply import `milvus.default_server`.

```python
from milvus import default_server
from pymilvus import connections, utility

# (OPTIONAL) Set if you want store all related data to specific location
# Default location:
#   %APPDATA%/milvus-io/milvus-server on windows
#   ~/.milvus-io/milvus-server on linux
# default_server.set_base_dir('milvus_data')

# (OPTIONAL) if you want cleanup previous data
# default_server.cleanup()

# Start you milvus server
default_server.start()

# Now you could connect with localhost and the given port
# Port is defined by default_server.listen_port
connections.connect(host='127.0.0.1', port=default_server.listen_port)

# Check if the server is ready.
print(utility.get_server_version())
```

### CLI milvus-server

You could also use the `milvus-server` command to start the server.

```bash
$ milvus-server
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

### Configurations for Milvus
Milvus Lite could set configure by API as well as by CLI. We seperate the configurations into two parts: `basic` and `extra`.

#### The basic configurations
You could find available configurations by `milvus-server --help` for got the list of `basic` configurations.

These basic configurations including:
- Some listen ports for service, e.g. `--proxy-port` for specifying the port of proxy service.
- Some storage configurations, e.g. `--data` for specifying the data directory.
- Some log configurations. e.g. `--system-log-level` for specifying the log level.

If you using Python API, you could set these configurations by `MilvusServer.config.set` method.

```python
# this have the same effect as `milvus-server --system-log-level info`
default_server.config.set('system_log_level', 'info')
```

All configuable basic configurations could be found in config yaml template, which is installed with milvus package.

#### The extra configurations
Other configurations are `extra` configurations, which could also be set by `MilvusServer.config.set` method.

for example, if we want to set `dataCoord.segment.maxSize` to 1024, we could do:

```python
default_server.config.set('dataCoord.segment.maxSize', 1024)
```

or by CLI:

``` bash
milvus-server --extra-config dataCoord.segment.maxSize=1024
```

Both of them will update the content of Milvus config yaml with:
``` yaml
dataCoord:
  segment:
    maxSize: 1024
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

By default all data and logs are stored in the following locations: `~/.milvus.io/milvus-server/VERSION` (VERSION is the versiong string of Milvus Lite).

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

Milvus Lite could be run without pymilvus if you just want run as a server.
You could also install with extras `client` to get pymilvus.

```bash
$ python3 -m pip install "milvus[client]"
```

## Examples

Milvus Lite is friendly with jupyter notebook, you could find more examples under [examples](https://github.com/milvus-io/milvus-lite/blob/main/examples) folder.

## Contributing
If you want to contribute to Milvus Lite, please read the [Contributing Guide](https://github.com/milvus-io/milvus-lite/blob/main/CONTRIBUTING.md) first.

## Report a bug
When you use or develop milvus-lite, if you find any bug, please report it to us. You could submit an issue in [milvus-lite](
https://github.com/milvus-io/milvus-lite/issues/new/choose) or report you [milvus](https://github.com/milvus-io/milvus/issues/new/choose) repo if you think is a Milvus issue.

## License
Milvus Lite is under the Apache 2.0 license. See the [LICENSE](https://github.com/milvus-io/milvus-lite/blob/main/LICENSE) file for details.
