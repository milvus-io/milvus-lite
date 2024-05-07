# Contributing Guide

Welcome contributors! This guide will help you get started with contributing to Milvus-lite.

Please always find the latest version of this guide at [CONTRIBUTING.md:main](https://github.com/milvus-io/milvus-lite/blob/main/CONTRIBUTING.md)

## How to set up the development environment
The Milvus-lite project is written in Python. To set up the development environment, you need to install Python 3.8 or later. We recommend that you use a virtual environment to install the dependencies, although the Milvus-lite project requires a very small number of external packages.

The main dependencies for build Milvus-lite is to install dependencies of milvis, so generally you could refer to Milvus's [install_deps.sh](https://github.com/milvus-io/milvus/blob/master/scripts/install_deps.sh) as a reference. Please note, you should follow the related branch of Milvus. For example, if you want to build Milvus-lite with Milvus 2.4.0, you should checkout the branch of Milvus 2.4.0.

For python3 build wheel, we use the build module and requires newer version of setuptools. So you should install the latest version of setuptools and build.

## Build Milvus-lite
```bash
python3 setup.py bdist_wheel
```

After build, you shoud have wheel package under dist folder.