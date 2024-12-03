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

#include "insert_task.h"
#include <algorithm>
#include <any>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include "common.h"
#include "log/Log.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "schema_util.h"
#include "status.h"
#include "string_util.hpp"

namespace milvus::local {

using DType = ::milvus::proto::schema::DataType;
int64_t InsertTask::cur_id_ = 0;

InsertTask::InsertTask(::milvus::proto::milvus::InsertRequest* r,
                       const ::milvus::proto::schema::CollectionSchema* schema)
    : insert_request_(r), schema_(schema), num_rows_(0) {
}

bool
InsertTask::AddSystemField() {
    num_rows_ = insert_request_->num_rows();
    if (num_rows_ <= 0) {
        LOG_ERROR("Error rows nums {}", num_rows_);
        return false;
    }

    auto timestamps = GetTimestamps(num_rows_);
    auto row_ids = GetRowIds(timestamps);

    auto row_field = insert_request_->add_fields_data();
    row_field->set_type(DType::Int64);
    row_field->set_field_name(kRowIdFieldName);
    row_field->set_field_id(kRowIdField);
    for (auto id : row_ids) {
        row_field->mutable_scalars()->mutable_long_data()->add_data(id);
    }

    auto time_field = insert_request_->add_fields_data();
    time_field->set_type(DType::Int64);
    time_field->set_field_name(kTimeStampFieldName);
    time_field->set_field_id(kTimeStampField);
    for (auto t : timestamps) {
        time_field->mutable_scalars()->mutable_long_data()->add_data(t);
    }
    return true;
}

bool
InsertTask::GenFieldMap() {
    for (const auto& field : insert_request_->fields_data()) {
        field_data_map_.emplace(field.field_name(), &field);
    }

    for (const auto& field : schema_->fields()) {
        if (field_data_map_.find(field.name()) == field_data_map_.end()) {
            if (field.is_primary_key() && field.autoid()) {
                auto row_id_field = field_data_map_.at(kRowIdFieldName);
                auto pk_field = insert_request_->add_fields_data();
                pk_field->set_field_name(field.name());
                pk_field->set_field_id(field.fieldid());
                pk_field->set_type(field.data_type());
                if (field.data_type() == DType::VarChar) {
                    for (uint32_t i = 0; i < num_rows_; i++) {
                        pk_field->mutable_scalars()
                            ->mutable_string_data()
                            ->add_data(std::to_string(std::any_cast<int64_t>(
                                schema_util::GetField(*row_id_field, i))));
                    }
                } else {
                    for (uint32_t i = 0; i < num_rows_; i++) {
                        pk_field->mutable_scalars()
                            ->mutable_long_data()
                            ->add_data(std::any_cast<int64_t>(
                                schema_util::GetField(*row_id_field, i)));
                    }
                }
                field_data_map_.emplace(field.name(), pk_field);
            } else {
                LOG_ERROR("Lost field {}", field.name());
                return false;
            }
        }
    }

    return true;
}

bool
InsertTask::CheckDynamicFieldData() {
    if (!schema_->enable_dynamic_field()) {
        return true;
    }
    for (int i = 0; i < insert_request_->fields_data_size(); i++) {
        auto field = insert_request_->mutable_fields_data(i);
        if (field->is_dynamic()) {
            field->set_field_name(kMetaFieldName);
            //TODO check json
            return true;
        }
    }
    // no dynamic field found, and default value
    auto dy_field = insert_request_->add_fields_data();
    dy_field->set_field_name(kMetaFieldName);
    dy_field->set_is_dynamic(true);
    for (const auto& schema_field : schema_->fields()) {
        if (schema_field.is_dynamic()) {
            dy_field->set_field_id(schema_field.fieldid());
            break;
        }
    }
    dy_field->set_type(::milvus::proto::schema::DataType::JSON);
    dy_field->mutable_scalars()->mutable_json_data()->add_data()->assign("{}");
    return true;
}

Status
InsertTask::Process(Rows* rows) {
    if (!(AddSystemField() && CheckDynamicFieldData() && GenFieldMap())) {
        return Status::ParameterInvalid();
    }

    CHECK_STATUS(CheckOrSetVectorDim(), "");

    auto pk_field_name = schema_util::GetPkName(*schema_);
    if (!pk_field_name.has_value()) {
        auto err =
            string_util::SFormat("Collection {} has no pk", schema_->name());
        LOG_ERROR(err);
        return Status::ParameterInvalid(err);
    }

    pk_type_ = field_data_map_.at(pk_field_name.value())->type();

    for (uint32_t i = 0; i < num_rows_; i++) {
        ::milvus::proto::segcore::InsertRecord record;
        std::string pk;
        record.set_num_rows(1);
        for (const auto& field : schema_->fields()) {
            auto field_data = record.add_fields_data();
            field_data->set_field_id(field.fieldid());
            field_data->set_field_name(field.name());
            field_data->set_type(field.data_type());
            if (!schema_util::SliceFieldData(
                    *field_data_map_.at(field.name()),
                    std::vector<std::tuple<int64_t, int64_t>>{{i, 1}},
                    field_data)) {
                LOG_ERROR("Parse field data failed");
                return Status::FieldNotFound();
            }
            if (field.name() == pk_field_name.value()) {
                if (field.data_type() == DType::Int64) {
                    pk = std::to_string(
                        field_data->scalars().long_data().data(0));
                } else {
                    pk = field_data->scalars().string_data().data(0);
                }
            }
        }
        rows->push_back(std::make_tuple(pk, record.SerializeAsString()));
    }
    return Status::Ok();
}

Status
InsertTask::CheckOrSetVectorDim() {
    int64_t num_rows = insert_request_->num_rows();
    if (num_rows <= 0) {
        return Status::ParameterInvalid("Err num_rows: {}", num_rows);
    }
    for (const auto& field_schema : schema_->fields()) {
        if (field_schema.data_type() == DType::FloatVector) {
            // int64_t dim = field_data.vectors().dim();
            auto field_data = field_data_map_.at(field_schema.name());
            int64_t dim = schema_util::GetDim(field_schema);
            if (dim <= 0) {
                return Status::ParameterInvalid("Can not found dim info");
            }
            int vect_size = field_data->vectors().float_vector().data_size();
            if (vect_size % dim != 0) {
                return Status::Undefined(
                    "the length({}) of float data should divide the dim({})",
                    vect_size,
                    dim);
            }

            int32_t vec_rows = vect_size / field_data->vectors().dim();

            if (vec_rows != num_rows) {
                return Status::ParameterInvalid(
                    "the num_rows ({}) of field ({}) is not equal to passed "
                    "num_rows ({}): [expected={}][actual={}]",
                    vec_rows,
                    field_data->field_name(),
                    num_rows,
                    num_rows,
                    vec_rows);
            }
        } else if (field_schema.data_type() == DType::SparseFloatVector) {
            // set dim
            auto field_data = field_data_map_.at(field_schema.name());
            for (int i = 0;
                 i <
                 field_data->vectors().sparse_float_vector().contents_size();
                 i++) {
                uint32_t dim = 0;
                const char* pos = field_data->vectors()
                                      .sparse_float_vector()
                                      .contents(i)
                                      .c_str();
                size_t size = field_data->vectors()
                                  .sparse_float_vector()
                                  .contents(i)
                                  .size();
                const char* end = pos + size;

                for (; pos < end; pos += 8) {
                    dim = std::max(dim,
                                   *(reinterpret_cast<const uint32_t*>(pos))) +
                          1;
                }
                const_cast<::milvus::proto::schema::FieldData*>(field_data)
                    ->mutable_vectors()
                    ->mutable_sparse_float_vector()
                    ->set_dim(dim);
            }
        }
    }
    return Status::Ok();
}

std::vector<uint64_t>
InsertTask::GetTimestamps(int64_t size) {
    auto ts = GetTimestamp();
    return std::vector<uint64_t>(size, ts);
}

uint64_t
InsertTask::GetTimestamp() {
    // https://github.com/milvus-io/milvus/blob/master/docs/design_docs/20211214-milvus_hybrid_ts.md
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return (ms << 18) + cur_id_;
}

std::vector<int64_t>
InsertTask::GetRowIds(std::vector<uint64_t>& timestamps) {
    size_t size = timestamps.size();
    std::vector<int64_t> row_ids;
    for (size_t i = 0; i < size; i++) {
        row_ids.push_back(i + cur_id_ + timestamps[i]);
    }
    cur_id_ += size;
    return row_ids;
}

}  // namespace milvus::local
