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

#include "search_task.h"
#include <cstdint>
#include <exception>
#include <optional>
#include <tuple>
#include <vector>
#include "common.h"
#include "pb/plan.pb.h"
#include "antlr4-runtime.h"
#include "parser/parser.h"
#include "parser/utils.h"
#include "log/Log.h"
#include "schema.pb.h"
#include "schema_util.h"
#include "status.h"

namespace milvus::local {

SearchTask::SearchTask(::milvus::proto::milvus::SearchRequest* search_reques,
                       const ::milvus::proto::schema::CollectionSchema* schema,
                       const std::vector<std::string>* all_index)
    : search_request_(search_reques),
      schema_(schema),
      all_index_(all_index),
      topk_(-1),
      offset_(0),
      ann_field_(""),
      metric_(""),
      groupby_field_name_("") {
}
SearchTask::~SearchTask() {
}

bool
SearchTask::ParseSearchInfo(::milvus::proto::plan::QueryInfo* info) {
    int64_t round_decimal = -1;
    std::string search_param_str("");
    for (const auto& param : search_request_->search_params()) {
        if (param.key() == kTopkKey) {
            try {
                topk_ = std::stoll(param.value());
            } catch (std::exception& e) {
                LOG_ERROR("Parse topk failed, topk: {}, err: {}",
                          param.value(),
                          e.what());
                return false;
            }

        } else if (param.key() == kOffsetKey) {
            try {
                offset_ = std::stoll(param.value());
            } catch (std::exception& e) {
                LOG_ERROR("Parse offset failed, offset: {}, err: {}",
                          param.value(),
                          e.what());
                return false;
            }
        } else if (param.key() == kMetricTypeKey) {
            metric_ = param.value();
        } else if (param.key() == kRoundDecimalKey) {
            try {
                round_decimal = std::stoll(param.value());
            } catch (std::exception& e) {
                LOG_ERROR(
                    "Parse round_decimal failed, round_decimal: {}, err: {}",
                    param.value(),
                    e.what());
                return false;
            }
        } else if (param.key() == kSearchParamKey) {
            search_param_str = param.value();
        } else if (param.key() == kGroupByFieldKey) {
            groupby_field_name_ = param.value();
        } else if (param.key() == kAnnFieldKey) {
            ann_field_ = param.value();
        }
    }

    // get and validate topk
    if (topk_ <= 0 || topk_ > kTopkLimit) {
        LOG_ERROR(
            "Topk should be in range [1, {}], but got {}", kTopkLimit, topk_);
        return false;
    }

    if (offset_ > kTopkLimit) {
        LOG_ERROR("Offset should be in range [0, {}], but got {}",
                  kTopkLimit,
                  offset_);
        return false;
    }

    if ((topk_ + offset_) > kTopkLimit) {
        LOG_ERROR("topk + offset should be in range [1, {}], but got {}",
                  kTopkLimit,
                  topk_ + offset_);
        return false;
    }

    if (round_decimal != -1 && (round_decimal > 6 || round_decimal < 0)) {
        LOG_ERROR(
            "round_decimal {} is invalid, should be -1 or an integer in "
            "range [0, 6]",
            round_decimal);
        return false;
    }

    int64_t groupby_field_id = kRowIdField;
    if (groupby_field_name_ != "") {
        groupby_field_id = -1;
        for (const auto& field : schema_->fields()) {
            if (groupby_field_name_ == field.name()) {
                groupby_field_id = field.fieldid();
                break;
            }
        }
        if (groupby_field_id == -1) {
            LOG_ERROR("groupBy field {} not found in schema",
                      groupby_field_name_);
            return false;
        }
    }
    info->set_topk(topk_ + offset_);
    info->set_search_params(search_param_str);
    info->set_round_decimal(round_decimal);
    info->set_group_by_field_id(groupby_field_id);
    return true;
}

std::optional<std::tuple<std::string, int64_t>>
SearchTask::GetVectorField() {
    for (const auto& field : schema_->fields()) {
        if (schema_util::IsVectorField(field.data_type())) {
            return std::make_tuple(field.name(), field.data_type());
        }
    }
    LOG_ERROR("Can not found vector field");
    return std::nullopt;
}

Status
SearchTask::Process(::milvus::proto::plan::PlanNode* plan,
                    std::string* placeholder_group,
                    std::vector<int64_t>* nqs,
                    std::vector<int64_t>* topks) {
    if (!schema_util::TranslateOutputFields(search_request_->output_fields(),
                                            *schema_,
                                            false,
                                            &output_fields_,
                                            &user_output_fields_)) {
        return Status::ParameterInvalid();
    }

    std::vector<int64_t> ids;
    if (!schema_util::GetOutputFieldsIds(output_fields_, *schema_, &ids)) {
        return Status::ParameterInvalid();
    }
    for (int64_t id : ids) {
        plan->add_output_field_ids(id);
    }
    auto vector_anns = plan->mutable_vector_anns();
    vector_anns->set_placeholder_tag(kPlaceholderTag);
    if (!ParseSearchInfo(vector_anns->mutable_query_info())) {
        return Status::ParameterInvalid();
    }

    placeholder_group->assign(search_request_->placeholder_group());
    nqs->push_back(search_request_->nq());
    topks->push_back(vector_anns->query_info().topk());

    const ::milvus::proto::schema::FieldSchema* field;
    auto s = schema_util::FindVectorField(*schema_, ann_field_, &field);
    CHECK_STATUS(s, "");
    vector_anns->set_field_id(field->fieldid());
    auto vtype = schema_util::DataTypeToVectorType(field->data_type());
    vector_anns->set_vector_type(*vtype);

    // check metrics
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
    if (index_metric.empty()) {
        return Status::IndexNotFound("No index found in field [{}]",
                                     field->name());
    }

    if (metric_.empty())
        metric_ = index_metric;

    if (metric_.empty() || metric_ != index_metric) {
        return Status::ParameterInvalid(
            "fail to search: metric type not match: invalid "
            "[expected={}][actual={}]",
            index_metric,
            metric_);
    }

    plan->mutable_vector_anns()->mutable_query_info()->set_metric_type(metric_);

    if (!search_request_->dsl().empty()) {
        CHECK_STATUS(schema_util::ParseExpr(
                         search_request_->dsl(),
                         *schema_,
                         plan->mutable_vector_anns()->mutable_predicates()),
                     "");
    }
    return Status::Ok();
}

bool
SearchTask::PostProcess(
    const SearchResult& segcore_reault,
    ::milvus::proto::milvus::SearchResults* search_results) {
    ::milvus::proto::schema::SearchResultData tmp_ret;
    tmp_ret.ParseFromArray(segcore_reault.result_[0].proto_blob,
                           segcore_reault.result_[0].proto_size);
    search_results->mutable_results()->set_num_queries(tmp_ret.num_queries());
    auto ret_size = tmp_ret.scores_size();
    auto nq = tmp_ret.num_queries();
    int score_coefficient = schema_util::PositivelyRelated(metric_) ? 1 : -1;
    search_results->mutable_results()->set_top_k(0);

    int64_t cur_index = 0;
    if (nq * offset_ < ret_size) {
        for (const auto& name : user_output_fields_) {
            search_results->mutable_results()->add_output_fields(name);
        }

        std::vector<std::tuple<int64_t, int64_t>> ranges;
        for (int i = 0; i < nq; i++) {
            int64_t limit = 0;
            if (tmp_ret.topks(i) - offset_ > 0)
                limit = tmp_ret.topks(i) - offset_;
            if (search_results->results().top_k() < limit)
                search_results->mutable_results()->set_top_k(limit);

            search_results->mutable_results()->mutable_topks()->Add(limit);
            ranges.push_back(std::make_tuple(cur_index + offset_, limit));
            // copy topks and scores
            for (int j = cur_index + offset_; j < cur_index + tmp_ret.topks(i);
                 j++) {
                search_results->mutable_results()->mutable_scores()->Add(
                    tmp_ret.scores(j) * score_coefficient);

                // copy ids
                if (tmp_ret.ids().has_int_id()) {
                    search_results->mutable_results()
                        ->mutable_ids()
                        ->mutable_int_id()
                        ->add_data(tmp_ret.ids().int_id().data(j));
                } else {
                    search_results->mutable_results()
                        ->mutable_ids()
                        ->mutable_str_id()
                        ->add_data(tmp_ret.ids().str_id().data(j));
                }
            }
            cur_index += tmp_ret.topks(i);
        }
        // copy fields_data
        for (const auto& field_data : tmp_ret.fields_data()) {
            auto new_data =
                search_results->mutable_results()->add_fields_data();
            new_data->set_field_id(field_data.field_id());
            new_data->set_type(field_data.type());
            new_data->set_field_name(field_data.field_name());
            new_data->set_is_dynamic(field_data.is_dynamic());
            schema_util::SliceFieldData(field_data, ranges, new_data);
        }
        FillInFieldInfo(search_results->mutable_results());
    } else {
        for (int i = 0; i < nq; i++) {
            search_results->mutable_results()->mutable_topks()->Add(0);
        }
    }
    return true;
}

void
SearchTask::FillInFieldInfo(
    ::milvus::proto::schema::SearchResultData* result_data) {
    std::variant<::milvus::proto::schema::SearchResultData*,
                 ::milvus::proto::milvus::QueryResults*>
        result_var;
    result_var = result_data;
    schema_util::FillInFieldInfo(output_fields_, *schema_, result_var);
}

}  // namespace milvus::local
