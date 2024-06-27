#!/bin/bash
git config --global --add safe.directory /root/milvus-lite/thirdparty/milvus;
cd python
python3 -m build --wheel

