# Contributing Guide

Welcome contributors! This guide will help you get started with contributing to Milvus Lite.

Please always find the latest version of this guide at [CONTRIBUTING.md:main](https://github.com/milvus-io/milvus-lite/blob/main/CONTRIBUTING.md).

## Development environment

The easiest way to develop Milvus Lite is using the included
[devcontainer](https://containers.dev/), which provides the full C++ and
Python toolchain inside a Docker container. It works with VS Code, Cursor,
GitHub Codespaces, the devcontainer CLI, or any other tool that supports 
the devcontainer spec.

The main build dependency is the Milvus core (included as a git submodule
under `thirdparty/milvus`). The devcontainer handles all native toolchain
setup — Conan, CMake, Rust, OpenBLAS, etc. — so you don't need to install
them on your host machine.

### Prerequisites

- Docker Desktop (macOS / Linux / WSL)
- An editor with devcontainer support (VS Code, Cursor, etc.)

### Getting started

1. Clone the repo with submodules:

```bash
git clone --recurse-submodules https://github.com/milvus-io/milvus-lite.git
```

2. Open the repo in the devcontainer. In VS Code / Cursor, reopen when
   prompted.

3. Build the wheel (first build takes ~13 minutes for Conan dependencies,
   subsequent builds ~30 seconds):

```bash
cd python && python3 -m build --wheel
```

4. Run Python integration tests:

```bash
python3 -m pytest tests/test_nullable.py tests/test_query.py tests/test_search.py -v \
    --noconftest -o "addopts=" -c /dev/null
```

The extra flags bypass `tests/conftest.py` (which imports heavy upstream
dependencies like `jax` that aren't installed) and `tests/pytest.ini`
(which requires `pytest-html`).

5. Run C++ unit tests (after building the wheel):

```bash
cd python/build/bdist.linux-aarch64/build_milvus   # or bdist.linux-x86_64
cmake /workspaces/milvus-lite -DENABLE_UNIT_TESTS=ON -DUSE_SYSTEM_DEPS=OFF
cmake --build . --target milvus_proxy_test storage_test server_test bm25_function_test nullable_test
ctest --output-on-failure
```

The wheel build uses `-DENABLE_UNIT_TESTS=OFF`, so you need to
reconfigure cmake once before building test targets.

### Conan cache

The devcontainer persists Conan packages in a Docker named volume
(`milvus-lite-conan-cache`). If the cache gets corrupted:

```bash
docker volume rm milvus-lite-conan-cache
```

## Building release wheels

To build release wheels for distribution (from a git tag, in a clean
environment), use the release build script:

```bash
# x86_64 wheel from the main branch
./scripts/build.sh scripts/Dockerfile.manylinux.x86_64 main

# aarch64 wheel from a release tag
./scripts/build.sh scripts/Dockerfile.manylinux.aarch64 v2.5.2

# With a Conan cache directory for faster rebuilds
./scripts/build.sh scripts/Dockerfile.manylinux.x86_64 main /tmp/conan-cache
```

## Code style

- **C++**: Google style via [`.clang-format`](.clang-format) (80-col, 4-space indent). Format with `clang-format -i src/*.cpp`.
- **Python**: Google style via [`.pylintrc`](.pylintrc) (120-char lines, 4-space indent). Lint with `pylint`.
