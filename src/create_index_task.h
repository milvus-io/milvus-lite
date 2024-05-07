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
#include <map>
#include <string>
#include <vector>
#include "common.h"
#include "pb/common.pb.h"
#include "pb/milvus.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "status.h"

namespace milvus::local {

class CreateIndexTask final : NonCopyableNonMovable {
 public:
    CreateIndexTask(
        const ::milvus::proto::milvus::CreateIndexRequest* create_index_request,
        const ::milvus::proto::schema::CollectionSchema* schema)
        : create_index_request_(create_index_request), schema_(schema) {
        field_id_ = 0;
        collectionid_ = 0;
        is_auto_index_ = false;
    }
    virtual ~CreateIndexTask() = default;

 public:
    Status
    Process(milvus::proto::segcore::FieldIndexMeta* field_meta);

 private:
    Status
    ParseIndexParams();

    void
    WrapUserIndexParams(const std::string& metrics_type);

    bool
    AddAutoIndexParams(size_t numberParams, KVMap* index_params);

    Status
    CheckTrain(const ::milvus::proto::schema::FieldSchema& field,
               KVMap& index_params);

    bool
    ValidateStringIndexType(const std::string& index_type) {
        return index_type == kDefaultStringIndexType ||
               index_type == "marisa-trie" || index_type == kInvertedIndexType;
    }

    bool
    ValidateArithmeticIndexType(const std::string& index_type) {
        return index_type == kDefaultStringIndexType ||
               index_type == "Asceneding" || index_type == kInvertedIndexType;
    }

    bool
    FillDimension(const ::milvus::proto::schema::FieldSchema& field,
                  KVMap* index_params);

    bool
    IsVectorIndex(::milvus::proto::schema::DataType dtype);

 private:
    // string of ::milvus::proto::milvus::CreateIndexRequest;
    const ::milvus::proto::milvus::CreateIndexRequest* create_index_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;

    std::vector<::milvus::proto::common::KeyValuePair> new_index_params_;
    std::vector<::milvus::proto::common::KeyValuePair> new_type_params_;
    std::vector<::milvus::proto::common::KeyValuePair> new_extra_params_;

    std::string index_name_;
    int64_t field_id_;
    int64_t collectionid_;
    bool is_auto_index_;
};

}  // namespace milvus::local
