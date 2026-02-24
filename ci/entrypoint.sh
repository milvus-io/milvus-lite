#!/bin/bash
set -e
git config --global --add safe.directory /root/milvus-lite/thirdparty/milvus

# Deps missing from the current milvusdb/milvus-env:lite-main image:
#  - openblas-devel: cblas.h header needed by knowhere
#  - libatomic-static: static libatomic.a needed at link time
#  - perl-IPC-Cmd/perl-Digest-SHA: needed when openssl is built from source via Conan
# These can be removed once the base image is rebuilt from scripts/Dockerfile.manylinux.*
yum install -y openblas-devel libatomic-static perl-IPC-Cmd perl-Digest-SHA 2>/dev/null || true
cp /usr/lib/gcc/x86_64-redhat-linux/8/libatomic.a /usr/lib64/ 2>/dev/null || true
# RHEL 8 openblas-devel puts headers in /usr/include/openblas/ not /usr/include/
ln -sf /usr/include/openblas/*.h /usr/include/ 2>/dev/null || true

cd python
python3 -m build --wheel

