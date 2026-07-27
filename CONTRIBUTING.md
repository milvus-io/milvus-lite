# Contributing Guide

Welcome contributors! This guide will help you get started with contributing to Milvus-lite.

Please always find the latest version of this guide at [CONTRIBUTING.md:main](https://github.com/milvus-io/milvus-lite/blob/main/CONTRIBUTING.md)

## How to set up the development environment

Milvus Lite is written in Python and requires Python 3.10 or later. We recommend using a virtual environment for development.

From the repository root, create the environment and install the package with its development dependencies:

```bash
make dev
source .venv/bin/activate
```

Alternatively, install the project in editable mode in an existing virtual environment:

```bash
python -m pip install -e ".[dev]"
```

The core dependencies include PyArrow, NumPy, FAISS, and gRPC. They are installed through `pyproject.toml`; building Milvus Lite does not require a Milvus source checkout or the Milvus C++ build dependencies.

For Python package builds, we use the `build` module with the Hatchling backend configured in `pyproject.toml`.

## Build Milvus-lite
```bash
python3 -m build
```

After build, you should have a wheel package and source distribution under the `dist/` folder. You can also run `make build` from the repository root.
