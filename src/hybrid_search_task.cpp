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

#include "hybrid_search_task.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
#include "common.h"
#include "common.pb.h"
#include "log/Log.h"
#include "parser/utils.h"
#include "re_scorer.h"
#include "schema_util.h"
#include "status.h"
#include "string_util.hpp"
#include "milvus_id.hpp"

namespace milvus::local {

Status
HyBridSearchTask::ParseRankParams(
    const ::google::protobuf::RepeatedPtrField<
        ::milvus::proto::common::KeyValuePair>& params) {
    for (const auto& param : params) {
        if (param.key() == kLimitKey) {
            try {
                limit_ = std::stoll(param.value());
            } catch (std::exception& e) {
                LOG_ERROR("Parse limit failed, limit: {}, err: {}",
                          param.value(),
                          e.what());
                return Status::ParameterInvalid();
            }
        } else if (param.key() == kOffsetKey) {
            try {
                offset_ = std::stoll(param.value());
            } catch (std::exception& e) {
                LOG_ERROR("Parse offset failed, offset: {}, err: {}",
                          param.value(),
                          e.what());
                return Status::ParameterInvalid();
            }

        } else if (param.key() == kRoundDecimalKey) {
            try {
                round_decimal_ = std::stoll(param.value());
            } catch (std::exception& e) {
                LOG_ERROR(
                    "Parse round_decimal failed, round_decimal: {}, err: {}",
                    param.value(),
                    e.what());
                return Status::ParameterInvalid();
            }
        }
    }

    if (offset_ < 0) {
        return Status::ParameterInvalid(
            "{} {} is invalid, should be gte than 0", kOffsetKey, offset_);
    }

    if (limit_ <= 0) {
        return Status::ParameterInvalid(
            "{} {} is invalid, should be greater than 0", kLimitKey, limit_);
    }

    int64_t depth = offset_ + limit_;
    if (depth <= 0 || depth > kTopkLimit) {
        return Status::ParameterInvalid(
            "(offset+limit) should be in range [1, {}], but got {}",
            kTopkLimit,
            depth);
    }
    return Status::Ok();
}

Status
HyBridSearchTask::GetMetricType(
    const ::milvus::proto::milvus::SearchRequest& req, std::string* metric) {
    std::string ann_field;
    for (const auto& param : req.search_params()) {
        if (param.key() == kMetricTypeKey) {
            *metric = param.value();
        } else if (param.key() == kAnnFieldKey) {
            ann_field = param.value();
        }
    }

    if (ann_field.empty()) {
        return Status::ParameterInvalid("Ann field not found");
    }

    const ::milvus::proto::schema::FieldSchema* field;
    auto s = schema_util::FindVectorField(*schema_, ann_field, &field);
    CHECK_STATUS(s, "");

    std::string index_metric;
    for (const auto& index_str : *all_index_) {
        milvus::proto::segcore::FieldIndexMeta field_index;
        if (!field_index.ParseFromString(index_str)) {
            return Status::ServiceInternal("Error index info in db");
        }
        if (field_index.fieldid() == field->fieldid()) {
            index_metric = schema_util::GetIndexMetricType(field_index);
        }
    }

    if (metric->empty())
        *metric = index_metric;

    if (metric->empty() || *metric != index_metric) {
        return Status::ParameterInvalid(
            "fail to search: metric type not match: invalid "
            "[expected={}][actual={}]",
            index_metric,
            *metric);
    }
    return Status::Ok();
}

Status
HyBridSearchTask::ProcessSearch(
    const ::milvus::proto::milvus::HybridSearchRequest* request,
    std::vector<::milvus::proto::milvus::SearchRequest>* search_requests) {
    if (request->requests_size() <= 0) {
        return Status::ParameterInvalid("No subrequest found");
    }

    if (!schema_util::TranslateOutputFields(request->output_fields(),
                                            *schema_,
                                            false,
                                            &output_fields_,
                                            &user_output_fields_)) {
        return Status::ParameterInvalid();
    }

    CHECK_STATUS(ParseRankParams(request->rank_params()), "");
    CHECK_STATUS(
        NewReScorers(
            request->requests_size(), request->rank_params(), &re_scorers_),
        "");
    nq_ = request->requests(0).nq();

    for (int64_t i = 0; i < request->requests_size(); i++) {
        auto req = request->requests(i);
        if (nq_ != req.nq()) {
            return Status::ParameterInvalid(
                "sub search request nq should be the same: invalid "
                "parameter[expected={}][actual={}]",
                nq_,
                req.nq());
        }
        // search without output_fields
        ::milvus::proto::milvus::SearchRequest search_request;
        search_request.set_collection_name(request->collection_name());
        search_request.mutable_partition_names()->CopyFrom(
            request->partition_names());

        search_request.set_not_return_all_meta(request->not_return_all_meta());
        search_request.set_consistency_level(request->consistency_level());
        search_request.set_use_default_consistency(
            request->use_default_consistency());
        search_request.set_search_by_primary_keys(false);
        search_request.set_dsl(req.dsl());
        search_request.set_placeholder_group(req.placeholder_group());
        search_request.set_dsl_type(req.dsl_type());
        // need remove offset in subrequest
        for (const auto& pair : req.search_params()) {
            if (pair.key() != kOffsetKey) {
                auto new_pair = search_request.add_search_params();
                new_pair->set_key(pair.key());
                new_pair->set_value(pair.value());
            }
        }
        search_request.set_nq(req.nq());
        std::string metric;
        CHECK_STATUS(GetMetricType(search_request, &metric), "");
        re_scorers_[i]->SetMetricType(metric);
        search_requests->push_back(search_request);
    }
    return Status::Ok();
}

Status
HyBridSearchTask::PostProcessSearch(
    std::vector<::milvus::proto::milvus::SearchResults>& search_results,
    ::milvus::proto::milvus::SearchResults* search_result) {
    search_result->mutable_results()->set_num_queries(nq_);
    search_result->mutable_results()->set_top_k(limit_);

    for (size_t i = 0; i < search_results.size(); i++) {
        re_scorers_[i]->ReScore(&search_results[i]);
    }
    std::vector<std::map<MilvusID, float>> accumulated_scores(nq_);
    for (const auto& req : search_results) {
        int64_t start = 0;
        for (int64_t i = 0; i < nq_; i++) {
            int64_t topk = req.results().topks(i);
            for (int64_t j = start; j < start + topk; j++) {
                MilvusID id(req.results().ids(), j);
                if (accumulated_scores[i].find(id) ==
                    accumulated_scores[i].end()) {
                    accumulated_scores[i][id] = req.results().scores(j);
                } else {
                    accumulated_scores[i][id] += req.results().scores(j);
                }
            }
            start += topk;
        }
    }
    for (int64_t i = 0; i < nq_; i++) {
        auto id_scores = accumulated_scores[i];
        if (id_scores.size() < offset_) {
            search_result->mutable_results()->add_topks(0);
            continue;
        }

        std::vector<std::pair<MilvusID, float>> sorted_id_scores(
            id_scores.begin(), id_scores.end());
        std::sort(sorted_id_scores.begin(),
                  sorted_id_scores.end(),
                  [](const std::pair<MilvusID, float>& a,
                     const std::pair<MilvusID, float>& b) {
                      return a.second > b.second;
                  });
        auto begin = sorted_id_scores.begin() + offset_;
        auto end = offset_ + limit_ < sorted_id_scores.size()
                       ? sorted_id_scores.begin() + offset_ + limit_
                       : sorted_id_scores.end();

        auto topk_id_scores =
            std::vector<std::pair<MilvusID, float>>(begin, end);
        search_result->mutable_results()->add_topks(topk_id_scores.size());
        for (const auto& id_score_pair : topk_id_scores) {
            search_result->mutable_results()->add_scores(id_score_pair.second);
            id_score_pair.first.AppendToIDs(
                search_result->mutable_results()->mutable_ids());
        }
    }
    return Status::Ok();
}

void
HyBridSearchTask::ProcessQuery(
    const ::milvus::proto::milvus::HybridSearchRequest* r,
    const ::milvus::proto::milvus::SearchResults* search_result,
    ::milvus::proto::milvus::QueryRequest* query_req) {
    query_req->set_collection_name(r->collection_name());
    query_req->mutable_output_fields()->CopyFrom(r->output_fields());
    query_req->mutable_partition_names()->CopyFrom(r->partition_names());
    query_req->set_not_return_all_meta(true);
    query_req->set_consistency_level(
        ::milvus::proto::common::ConsistencyLevel::Strong);
    query_req->set_use_default_consistency(false);

    std::string pk_name = schema_util::GetPkName(*schema_).value();
    if (search_result->results().ids().has_int_id()) {
        auto ids_str = string_util::Join(
            ",", search_result->results().ids().int_id().data());
        auto expr = string_util::SFormat("{} in [{}]", pk_name, ids_str);
        query_req->set_expr(expr);

    } else {
        std::vector<std::string> ids;
        for (int i = 0; i < search_result->results().ids().str_id().data_size();
             i++) {
            ids.emplace_back(
                "'" + search_result->results().ids().str_id().data(i) + "'");
        }
        auto ids_str = string_util::Join(",", ids);
        auto expr = string_util::SFormat("{} in [{}]", pk_name, ids_str);
        query_req->set_expr(expr);
    }
}

Status
HyBridSearchTask::PostProcess(
    const RetrieveResult& query_result,
    ::milvus::proto::milvus::SearchResults* search_result) {
    milvus::proto::segcore::RetrieveResults seg_ret;
    seg_ret.ParseFromArray(query_result.retrieve_result_.proto_blob,
                           query_result.retrieve_result_.proto_size);

    // Copy field data from query result
    // Reorganize Results. The order of query result ids will be altered and differ from queried ids.
    // We should reorganize query results to keep the order of original queried ids. For example:
    // query expr pk in [3, 5, 1, 3, 5, 2]
    // query result: [3, 5, 1, 2]
    // need reorganize

    std::map<MilvusID, int> offsets;
    int query_size = schema_util::IDsSize(seg_ret.ids());
    for (int i = 0; i < query_size; i++) {
        offsets[MilvusID(seg_ret.ids(), i)] = i;
    }

    int search_size = schema_util::IDsSize(search_result->results().ids());
    std::vector<int64_t> indexes;
    for (int i = 0; i < search_size; i++) {
        indexes.push_back(offsets[MilvusID(search_result->results().ids(), i)]);
    }

    std::vector<int64_t> field_ids;
    schema_util::GetOutputFieldsIds(output_fields_, *schema_, &field_ids);
    for (const auto& field_data : seg_ret.fields_data()) {
        if (std::find(field_ids.begin(),
                      field_ids.end(),
                      field_data.field_id()) == field_ids.end()) {
            continue;
        }
        auto new_data = search_result->mutable_results()->add_fields_data();
        new_data->set_type(field_data.type());
        new_data->set_field_name(field_data.field_name());
        new_data->set_field_id(field_data.field_id());
        new_data->set_is_dynamic(field_data.is_dynamic());
        schema_util::PickFieldDataByIndex(field_data, indexes, new_data);
    }
    for (const auto& name : user_output_fields_) {
        search_result->mutable_results()
            ->mutable_output_fields()
            ->Add()
            ->append(name);
    }
    FillInFieldInfo(search_result->mutable_results());
    return Status::Ok();
}

void
HyBridSearchTask::FillInFieldInfo(
    ::milvus::proto::schema::SearchResultData* result_data) {
    std::variant<::milvus::proto::schema::SearchResultData*,
                 ::milvus::proto::milvus::QueryResults*>
        result_var;
    result_var = result_data;
    schema_util::FillInFieldInfo(output_fields_, *schema_, result_var);
}

}  // namespace milvus::local
