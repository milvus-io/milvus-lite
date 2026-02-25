#!/bin/bash
set -e
git config --global --add safe.directory /root/milvus-lite/thirdparty/milvus

# ---- Begin workarounds for stale CI image ----
# The CI images (milvusdb/milvus-env:lite-main, lite-manylinux2014) are missing
# these packages. To fix permanently, rebuild the images from
# scripts/Dockerfile.manylinux.{x86_64,aarch64} and push to DockerHub,
# then remove this block.
#  - openblas-devel: cblas.h header needed by knowhere
#  - libatomic-static: static libatomic.a needed at link time
#  - perl-IPC-Cmd/perl-Digest-SHA: needed when openssl is built from source via Conan
yum install -y openblas-devel libatomic-static perl-IPC-Cmd perl-Digest-SHA perl-Thread-Queue 2>/dev/null || true
cp /usr/lib/gcc/x86_64-redhat-linux/8/libatomic.a /usr/lib64/ 2>/dev/null || true
ln -sf /usr/include/openblas/*.h /usr/include/ 2>/dev/null || true
# ---- End workarounds ----

cd python
python3 -m build --wheel

