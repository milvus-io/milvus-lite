#!/bin/bash

set -e

BLAS_VERSION=0.3.21

wget https://github.com/xianyi/OpenBLAS/archive/v${BLAS_VERSION}.tar.gz
tar zxvf v${BLAS_VERSION}.tar.gz && cd OpenBLAS-${BLAS_VERSION}
make NO_STATIC=1 NO_LAPACK=1 NO_LAPACKE=1 NO_AFFINITY=1 USE_OPENMP=1 \
    TARGET=HASWELL DYNAMIC_ARCH=1 \
    NUM_THREADS=64 MAJOR_VERSION=3 libs shared
make PREFIX=/usr/local NUM_THREADS=64 MAJOR_VERSION=3 install
cd ..
rm -rf OpenBLAS-${BLAS_VERSION} && rm v${BLAS_VERSION}.tar.gz
