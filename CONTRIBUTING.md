# Contributing Guide

Welcome contributors! This guide will help you get started with contributing to Milvus-lite.

Please always find the latest version of this guide at [CONTRIBUTING.md:main](https://github.com/milvus-io/milvus-lite/blob/main/CONTRIBUTING.md)

## How to set up the development environment
The Milvus-lite project is written in Python. To set up the development environment, you need to install Python 3.8 or later(Our release is supported Python3.6+). We recommend that you use a virtual environment to install the dependencies, although the Milvus-lite project requires a very small number of external packages.

The main dependencies for build Milvus-lite is to install dependencies of milvis, so generally you could refer to Milvus's [install_deps.sh](https://github.com/milvus-io/milvus/blob/master/scripts/install_deps.sh) as a reference. Please note, you should follow the related branch of Milvus. For example, if you want to build Milvus-lite with Milvus 2.2.0, you should checkout the branch of Milvus 2.2.0.

For python3 build wheel, we use the build module and requires newer version of setuptools. So you should install the latest version of setuptools and build.

### Setup development environment under linux
We release the Milvus-lite with CentOS image, which reuses from milvusdb/milvus-env, so the binary distribution is compatiable with manylinux2014.

If you open the project with VSCode, you could alse use devcontainer to setup the development environment(recommanded). That will help you install all dependencies automatically.

### Setup development environment under macOS
As we build Milvus-lite with macos 11 and 12, and all dependencies are resloved during build.
Generally, you need first install [brew](https://brew.sh/), then install the following packages:

```bash
brew install boost libomp ninja tbb openblas ccache pkg-config md5sha1sum llvm@15
```

You could also find more details in [build.sh](milvus_binary/build.sh)


### Setup development environment under Windows/msys2
We only support specific version of msys/mingw, currently we **msys2-base-x86_64-20220603**, which could be found at [MSYS2 Install Release](https://github.com/msys2/msys2-installer/releases/tag/2022-06-03)

After install mingw/msys2, you need install the following packages by pacman:
```bash
pacman -S git patch
pacman -S mingw-w64-x86_64-python mingw-w64-x86_64-python-wheel mingw-w64-x86_64-python-pip
```

## Build Milvus-lite
```bash
python3 -m build --wheel
```

After build, you shoud have wheel package under dist folder.
