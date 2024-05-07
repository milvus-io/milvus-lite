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

#include "pb/milvus.pb.h"
#include "pb/plan.pb.h"
#include <cstdint>
#include <vector>

#include "retrieve_result.h"
#include "status.h"

namespace milvus::local {

class QueryTask : NonCopyableNonMovable {
 public:
    QueryTask(const ::milvus::proto::milvus::QueryRequest* query_request,
              const ::milvus::proto::schema::CollectionSchema*);
    virtual ~QueryTask();

    Status
    Process(::milvus::proto::plan::PlanNode* plan);

    bool
    PostProcess(const RetrieveResult& rt,
                ::milvus::proto::milvus::QueryResults* ret);

 private:
    bool
    GetOutputFieldIds(std::vector<int64_t>* ids);

    void
    FilterSystemField();

    void
    FillInFieldInfo(::milvus::proto::milvus::QueryResults* result_data);

    Status
    ParseQueryParams(::milvus::proto::plan::PlanNode* plan, bool expr_empty);

 private:
    const ::milvus::proto::milvus::QueryRequest* query_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;

    std::vector<std::string> output_fields_;
    std::vector<std::string> user_output_fields_;

    int64_t limit_;
    int64_t offset_;
    bool is_count_;
};

}  // namespace milvus::local
