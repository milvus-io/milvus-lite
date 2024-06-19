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
#include <vector>
#include <optional>
#include "pb/milvus.pb.h"
#include "pb/plan.pb.h"
#include "pb/segcore.pb.h"
#include "schema.pb.h"
#include "search_result.h"
#include "status.h"

namespace milvus::local {

class SearchTask final : NonCopyableNonMovable {
 public:
    SearchTask(::milvus::proto::milvus::SearchRequest* search_reques,
               const ::milvus::proto::schema::CollectionSchema* schema,
               const std::vector<std::string>*);
    virtual ~SearchTask();

 public:
    Status
    Process(::milvus::proto::plan::PlanNode* plan,
            std::string* placeholder_group,
            std::vector<int64_t>* nqs,
            std::vector<int64_t>* topks);

    bool
    PostProcess(const SearchResult& segcore_reaul,
                ::milvus::proto::milvus::SearchResults* search_results);

 private:
    bool
    ParseSearchInfo(::milvus::proto::plan::QueryInfo* info);

    std::optional<std::tuple<std::string, int64_t>>
    GetVectorField();

    void
    FillInFieldInfo(::milvus::proto::schema::SearchResultData* result_data);

 private:
    ::milvus::proto::milvus::SearchRequest* search_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;
    const std::vector<std::string>* all_index_;

    std::vector<std::string> output_fields_;
    std::vector<std::string> user_output_fields_;
    int64_t topk_, offset_;
    std::string ann_field_;
    std::string metric_;
    std::string groupby_field_name_;
};

}  // namespace milvus::local
