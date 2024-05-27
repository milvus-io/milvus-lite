#!/bin/bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TAG="main"

if [ "$#" -eq 0 ]; then
    docker build -t build_milvus_lite:$TAG . \
        && docker run --rm -v $PWD:/workspace/dist build_milvus_lite:$TAG $TAG
elif [ "$#" -eq 1 ]; then
    TAG=$1
    docker build -t build_milvus_lite:$TAG . \
        && docker run --rm -v $PWD:/workspace/dist build_milvus_lite:$TAG $TAG
elif [ "$#" -eq 2 ]; then
    TAG=$1
    CACAN_CACHE=$2
    docker build -t build_milvus_lite:$TAG . \
        && docker run --rm -e CONAN_USER_HOME=/workspace/conan -v $CACAN_CACHE:/workspace/conan -v $PWD:/workspace/dist build_milvus_lite:$TAG $TAG
fi
