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
#include <memory>
#include <vector>
#include "pb/milvus.pb.h"
#include "pb/schema.pb.h"
#include "pb/plan.pb.h"
#include "pb/segcore.pb.h"
#include "retrieve_result.h"
#include "common.h"
#include "status.h"
#include "re_scorer.h"

namespace milvus::local {

class HyBridSearchTask : NonCopyableNonMovable {
 public:
    HyBridSearchTask(const ::milvus::proto::schema::CollectionSchema* schema,
                     const std::vector<std::string>* all_index)
        : schema_(schema),
          all_index_(all_index),
          offset_(0),
          limit_(-1),
          round_decimal_(-1),
          nq_(-1) {
    }
    virtual ~HyBridSearchTask() = default;

    Status
    ProcessSearch(
        const ::milvus::proto::milvus::HybridSearchRequest* request,
        std::vector<::milvus::proto::milvus::SearchRequest>* search_requests);

    Status
    PostProcessSearch(
        std::vector<::milvus::proto::milvus::SearchResults>& search_results,
        ::milvus::proto::milvus::SearchResults* search_result);

    void
    ProcessQuery(const ::milvus::proto::milvus::HybridSearchRequest* r,
                 const ::milvus::proto::milvus::SearchResults* search_result,
                 ::milvus::proto::milvus::QueryRequest* query_req);

    Status
    PostProcess(const RetrieveResult& query_result,
                ::milvus::proto::milvus::SearchResults* search_result);

    const ::milvus::proto::schema::CollectionSchema*
    GetSchema() {
        return schema_;
    }

    void
    FillInFieldInfo(::milvus::proto::schema::SearchResultData* result_data);

 private:
    Status
    ParseRankParams(const ::google::protobuf::RepeatedPtrField<
                    ::milvus::proto::common::KeyValuePair>& params);

    Status
    GetMetricType(const ::milvus::proto::milvus::SearchRequest& req,
                  std::string* metric);

 private:
    const ::milvus::proto::schema::CollectionSchema* schema_;
    const std::vector<std::string>* all_index_;
    std::vector<std::string> output_fields_;
    std::vector<std::string> user_output_fields_;

    uint64_t offset_;
    int64_t limit_;
    int64_t round_decimal_;
    int64_t nq_;
    std::vector<std::unique_ptr<ReScorer>> re_scorers_;
};

}  // namespace milvus::local
