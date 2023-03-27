#!/bin/bash

set -e

# TBB
if ! test -f /usr/local/lib/libtbb.so ; then
    git clone https://github.com/wjakob/tbb.git && \
        cd tbb/build && \
        cmake .. && make -j && \
        make install && \
        cd ../../ && rm -rf tbb/
fi

