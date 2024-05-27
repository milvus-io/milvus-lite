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

tag="main"

if [ "$#" -eq 1 ]; then
    tag=$1
fi

echo "Build milvus-lite:$tag"
echo $CONAN_USER_HOME

git clone --recurse-submodules https://github.com/milvus-io/milvus-lite.git \
    && cd milvus-lite \
    && git checkout $tag \
    && cd python \
    && python3 setup.py bdist_wheel \
    && cp -r dist /workspace/ \
    && cd /workspace && rm -rf milvus-lite    
