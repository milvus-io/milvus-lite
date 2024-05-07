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
#include <memory>
#include <string>
#include "segcore_wrapper.h"
#include "common.h"
#include "status.h"

namespace milvus::local {

class Index final : NonCopyableNonMovable {
 public:
    Index() = default;
    virtual ~Index() = default;

 public:
    // meta interface
    Status
    CreateCollection(const std::string& collection_name,
                     const std::string& schema_proto);
    bool
    DropCollection(const std::string& collection_name);

    bool
    HasLoaded(const std::string& collection_name) {
        return collections_.find(collection_name) != collections_.end();
    }

    Status
    CreateIndex(const std::string& collection_name,
                const std::string& index_proto);

    Status
    Insert(const std::string& collection_name,
           int64_t size,
           const std::string& insert_record_proto);

    Status
    Retrieve(const std::string& collection_name,
             const std::string& expr,
             RetrieveResult* result);

    Status
    Search(const std::string& collection_name,
           const std::string& plan,
           const std::string& placeholder_group,
           SearchResult* result);

    Status
    DeleteByIds(const std::string& collection_name,
                const std::string& ids,
                int64_t size);

 private:
    std::map<std::string, std::unique_ptr<SegcoreWrapper>> collections_;
};

}  // namespace milvus::local
