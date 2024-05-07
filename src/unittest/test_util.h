// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "type.h"
#include "pb/milvus.pb.h"

namespace milvus::local {
namespace test {

#define PK_NAME "id"
#define PK_ID 200

#define VEC_NAME "vec"
#define VEC_ID 201
#define VEC_DIM_NAME "dim"
#define VEC_DIM 3

#define SCALAR_NAME "sc"
#define SCALAR_ID 202

std::string
CreateCollection(const std::string& collection_name = "test_schema");

std::string
CreateVectorIndex();

milvus::local::Rows
CreateData(int32_t count);

::milvus::proto::milvus::CreateCollectionRequest
GetCreateCollectionRequestProto(const std::string& collection_name);

::milvus::proto::milvus::LoadCollectionRequest
GetLoadCollectionRequestProto(const std::string& collection_name);

::milvus::proto::milvus::CreateIndexRequest
GetCreateIndexRequestProto(const std::string& collection_name,
                           const std::string& index_name,
                           const std::string& field_name);

::milvus::proto::milvus::InsertRequest
GetInsertRequestProto(const std::string& collection_name, int64_t row_num);

::milvus::proto::milvus::SearchRequest
GetSearchRequestProto(const std::string& collection_name,
                      const std::string& expr,
                      const std::vector<std::vector<float>>& vecs,
                      const std::string& topk,
                      const std::string& metric_type,
                      const std::string& offset);

::milvus::proto::milvus::QueryRequest
GetQueryRequestProto(const std::string& collection_name,
                     const std::string& expr,
                     const std::string& limit,
                     const std::string& offset,
                     const std::vector<std::string>& output_fields);

::milvus::proto::milvus::DeleteRequest
GetDeleteRequestProto(const std::string& collection_name,
                      const std::string& expr);

::milvus::proto::milvus::DescribeCollectionRequest
GetDescribeCollectionRequest(const std::string& collection_name);

inline ::milvus::proto::milvus::DropCollectionRequest
GetDropCollectionRequest(const std::string& collection_name) {
    ::milvus::proto::milvus::DropCollectionRequest request;
    request.set_collection_name(collection_name);
    return request;
}

}  // namespace test
}  // namespace milvus::local
