# Copyright (C) 2019-2024 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License

file(GLOB_RECURSE SRC_FILES thirdparty/milvus-storage/cpp/src/*.cpp thirdparty/milvus-storage/cpp/src/*.cc)
add_library(milvus-storage ${SRC_FILES})
target_include_directories(milvus-storage PUBLIC BEFORE thirdparty/milvus-storage/cpp/include/milvus-storage thirdparty/milvus-storage/cpp/src)
target_link_libraries(milvus-storage PUBLIC
    arrow::arrow
    Boost::boost
    protobuf::protobuf
    glog::glog
)
