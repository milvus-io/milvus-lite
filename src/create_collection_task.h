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
#include "pb/milvus.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "common.h"
#include "status.h"

namespace milvus::local {

class CreateCollectionTask final : NonCopyableNonMovable {
 public:
    explicit CreateCollectionTask(
        const ::milvus::proto::milvus::CreateCollectionRequest*
            create_collection_request)
        : create_collection_request_(create_collection_request) {
    }
    virtual ~CreateCollectionTask() = default;

 public:
    Status
    Process(::milvus::proto::schema::CollectionSchema* schema);

 private:
    bool
    CheckDefaultValue(const ::milvus::proto::schema::CollectionSchema& schema);

    bool
    HasSystemFields(const ::milvus::proto::schema::CollectionSchema& schema);

    void
    AssignFieldId(::milvus::proto::schema::CollectionSchema* schema);

    void
    AppendDynamicField(::milvus::proto::schema::CollectionSchema* schema);

    void
    AppendSysFields(::milvus::proto::schema::CollectionSchema* schema);

    Status
    GetVarcharFieldMaxLength(const ::milvus::proto::schema::FieldSchema& field,
                             uint64_t* max_len);

    Status
    ValidateSchema(const ::milvus::proto::schema::CollectionSchema& schema);

    Status
    CheckFieldName(const std::string& field_name);

 private:
    const ::milvus::proto::milvus::CreateCollectionRequest*
        create_collection_request_;
};

}  // namespace milvus::local
