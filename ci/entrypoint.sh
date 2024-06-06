#!/bin/bash
git config --global --add safe.directory /root/milvus-lite/thirdparty/milvus;
cd python
python3 setup.py bdist_wheel

