#!/bin/bash

set -e

yum -y update && yum -y install wget

if ! /usr/local/go/bin/go version 2>/dev/null | grep -wq go1.18.10 ; then
    rm -fr /usr/local/go && \
    cd /usr/local && \
    (wget https://studygolang.com/dl/golang/go1.18.10.linux-amd64.tar.gz || \
     wget https://go.dev/dl/go1.18.10.linux-amd64.tar.gz) && \
    tar xvf go1.18.10.linux-amd64.tar.gz && \
    rm -fr go1.18.10.linux-amd64.tar.gz
fi

yum -y install make lcov libtool m4 autoconf automake ccache \
    openssl-devel zlib-devel libzstd-devel libcurl-devel \
    libuuid-devel pulseaudio-libs-devel libatomic \
    devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-gcc-gfortran \
    boost169-devel lapack-devel

export PATH=${PATH}:/usr/local/go/bin
export BOOST_INCLUDEDIR=/usr/include/boost169
export BOOST_LIBRARYDIR=/usr/lib64/boost169

# patch for cmake find boost
if ! test -L /usr/include/boost ; then
    ln -s /usr/include/boost169/boost /usr/include/boost
fi

# cleanup cache
yum clean all
rm -fr ~/.cache
