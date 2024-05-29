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

#include "query_task.h"
#include <cstdint>
#include <tuple>
#include <vector>
#include "antlr4-runtime.h"
#include "parser/parser.h"
#include "parser/utils.h"
#include "common.h"
#include "pb/plan.pb.h"
#include "schema.pb.h"
#include "schema_util.h"
#include "status.h"
#include "string_util.hpp"
#include "pb/segcore.pb.h"
#include "log/Log.h"

namespace milvus::local {

QueryTask::QueryTask(const ::milvus::proto::milvus::QueryRequest* query_request,
                     const ::milvus::proto::schema::CollectionSchema* schema)
    : query_request_(query_request),
      schema_(schema),
      limit_(-1),
      offset_(0),
      is_count_(false) {
}
QueryTask::~QueryTask() {
}

bool
QueryTask::GetOutputFieldIds(std::vector<int64_t>* ids) {
    if (output_fields_.size() == 0) {
        for (const auto& field : schema_->fields()) {
            if (field.fieldid() >= kStartOfUserFieldId &&
                !schema_util::IsVectorField(field.data_type())) {
                ids->push_back(field.fieldid());
            }
        }
    } else {
        std::string pk;
        std::map<std::string, int64_t> name_ids;
        for (const auto& field : schema_->fields()) {
            name_ids[field.name()] = field.fieldid();
            if (field.is_primary_key()) {
                pk = field.name();
                ids->push_back(field.fieldid());
            }
        }
        for (const auto& output_field : output_fields_) {
            if (output_field == pk)
                continue;
            auto it = name_ids.find(output_field);
            if (it == name_ids.end()) {
                LOG_ERROR("Can not find output field {} in schema",
                          output_field);
                return false;
            }
            if (it->second >= kStartOfUserFieldId) {
                ids->push_back(it->second);
            }
        }
    }
    return true;
}

Status
QueryTask::ParseQueryParams(::milvus::proto::plan::PlanNode* plan,
                            bool expr_empty) {
    for (const auto& param : query_request_->query_params()) {
        if (param.key() == kLimitKey) {
            try {
                limit_ = std::stoll(param.value());
            } catch (std::exception& e) {
                auto err = string_util::SFormat("Parse limit failed, limit: {}",
                                                param.value());
                return Status::ParameterInvalid(err);
            }
        } else if (param.key() == kOffsetKey) {
            try {
                offset_ = std::stoll(param.value());
            } catch (std::exception& e) {
                auto err = string_util::SFormat(
                    "Parse offset failed, offset: {}", param.value());
                return Status::ParameterInvalid(err);
            }
        } else if (param.key() == kReduceStopForBestKey) {
            // not used in local
        }
    }
    if (offset_ < 0 || offset_ >= kTopkLimit) {
        return Status::ParameterInvalid(
            "Offset should be in range [0, {}], but got {}",
            kTopkLimit,
            offset_);
    }

    if (limit_ <= 0) {
        if (is_count_) {
            limit_ = -1;
        } else if (expr_empty) {
            return Status::ParameterInvalid(
                "empty expression should be used with limit");
        } else {
            limit_ = kTopkLimit - offset_ - 1;
        }
    }

    // get and validate topk
    if (limit_ >= kTopkLimit) {
        return Status::ParameterInvalid(
            "limit should be in range [1, {}], but got {}", kTopkLimit, limit_);
    }

    if ((limit_ + offset_) >= kTopkLimit) {
        return Status::ParameterInvalid(
            "topk + offset should be in range [1, {}], but got {}",
            kTopkLimit,
            limit_ + offset_);
    }
    plan->mutable_query()->set_limit(limit_ + offset_);
    return Status::Ok();
}

Status
QueryTask::Process(::milvus::proto::plan::PlanNode* plan) {
    if (query_request_->output_fields_size() == 1 &&
        string_util::Trim(string_util::ToLower(
            query_request_->output_fields().Get(0))) == kCountStr) {
        plan->mutable_query()->set_is_count(true);
        is_count_ = true;
    }

    CHECK_STATUS(
        ParseQueryParams(plan, string_util::Trim(query_request_->expr()) == ""),
        "");

    CHECK_STATUS(
        schema_util::ParseExpr(string_util::Trim(query_request_->expr()),
                               *schema_,
                               plan->mutable_query()->mutable_predicates()),
        "");

    if (is_count_) {
        user_output_fields_.push_back(kCountStr);
    } else {
        if (!schema_util::TranslateOutputFields(query_request_->output_fields(),
                                                *schema_,
                                                true,
                                                &output_fields_,
                                                &user_output_fields_)) {
            return Status::ParameterInvalid("Error output fields");
        }

        std::vector<int64_t> output_ids;
        if (!GetOutputFieldIds(&output_ids)) {
            return Status::ParameterInvalid("Error output fields");
        }
        for (auto id : output_ids) {
            plan->add_output_field_ids(id);
        }
    }

    return Status::Ok();
}

bool
QueryTask::PostProcess(const RetrieveResult& rt,
                       ::milvus::proto::milvus::QueryResults* ret) {
    milvus::proto::segcore::RetrieveResults seg_ret;
    seg_ret.ParseFromArray(rt.retrieve_result_.proto_blob,
                           rt.retrieve_result_.proto_size);

    if (is_count_) {
        auto count_data = ret->add_fields_data();
        count_data->CopyFrom(seg_ret.fields_data(0));
        count_data->set_field_name(kCountStr);
        ret->add_output_fields(kCountStr);
        return true;
    }

    // reduce data by id
    std::vector<::milvus::proto::schema::FieldData> reduced_fields;
    int64_t ret_size = 0;
    for (const auto& field_data : seg_ret.fields_data()) {
        ::milvus::proto::schema::FieldData data;
        if (!schema_util::ReduceFieldByIDs(
                seg_ret.ids(), field_data, &data, &ret_size)) {
            return false;
        }
        reduced_fields.push_back(data);
    }

    if (ret_size > offset_) {
        for (const auto& field_data : reduced_fields) {
            auto new_data = ret->add_fields_data();
            new_data->set_field_id(field_data.field_id());
            new_data->set_type(field_data.type());
            new_data->set_field_name(field_data.field_name());
            new_data->set_is_dynamic(field_data.is_dynamic());
            auto limit = std::min(limit_, ret_size - offset_);
            schema_util::SliceFieldData(
                field_data,
                std::vector<std::tuple<int64_t, int64_t>>{{offset_, limit}},
                new_data);
        }
    } else {
        for (const auto& field_schema : schema_->fields()) {
            if (field_schema.fieldid() >= kStartOfUserFieldId) {
                schema_util::FillEmptyField(field_schema,
                                            ret->add_fields_data());
            }
        }
    }

    FillInFieldInfo(ret);
    for (const auto& name : user_output_fields_) {
        ret->add_output_fields(name);
    }
    return true;
}

void
QueryTask::FillInFieldInfo(::milvus::proto::milvus::QueryResults* result_data) {
    if (output_fields_.size() == 0 || result_data->fields_data_size() == 0) {
        return;
    }
    for (size_t i = 0; i < output_fields_.size(); i++) {
        const std::string& name = output_fields_[i];
        for (const auto& field : schema_->fields()) {
            if (name == field.name()) {
                auto field_id = field.fieldid();
                for (int j = 0; j < result_data->fields_data().size(); j++) {
                    if (field_id == result_data->fields_data(j).field_id()) {
                        result_data->mutable_fields_data(j)->set_field_name(
                            field.name());
                        result_data->mutable_fields_data(j)->set_field_id(
                            field.fieldid());
                        result_data->mutable_fields_data(j)->set_type(
                            field.data_type());
                        result_data->mutable_fields_data(j)->set_is_dynamic(
                            field.is_dynamic());
                    }
                }
            }
        }
    }
}

}  // namespace milvus::local
