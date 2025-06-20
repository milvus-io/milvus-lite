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

#include "create_collection_task.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include "common.h"
#include "log/Log.h"
#include "pb/schema.pb.h"
#include "status.h"
#include "string_util.hpp"

namespace milvus::local {

using DType = ::milvus::proto::schema::DataType;
using DCase = ::milvus::proto::schema::ValueField::DataCase;

bool
CreateCollectionTask::HasSystemFields(
    const ::milvus::proto::schema::CollectionSchema& schema) {
    for (const auto& f : schema.fields()) {
        if (f.name() == kRowIdFieldName || f.name() == kTimeStampFieldName ||
            f.name() == kMetaFieldName) {
            return true;
        }
    }
    return false;
}

Status
CheckFieldParams(const ::milvus::proto::schema::FieldSchema& field) {
    if (field.nullable()) {
        return Status::ParameterInvalid("MilvusLite doesn't support nullable");
    }
    return Status::Ok();
}

Status
CreateCollectionTask::GetVarcharFieldMaxLength(
    const ::milvus::proto::schema::FieldSchema& field, uint64_t* max_len) {
    if (field.data_type() != DType::VarChar &&
        field.element_type() != DType::VarChar) {
        return Status::ParameterInvalid("{} is not varchar field",
                                        field.name());
    }

    for (const auto& kv_pair : field.type_params()) {
        if (kv_pair.key() == kMaxLengthKey) {
            try {
                auto length = std::stoll(kv_pair.value());
                if (length <= 0 || length > kMaxLengthLimit) {
                    return Status::ParameterInvalid(
                        "the maximum length specified for a VarChar should "
                        "be "
                        "in (0, 65535])");
                } else {
                    *max_len = static_cast<uint64_t>(length);
                    return Status::Ok();
                }
            } catch (std::exception& e) {
                return Status::ParameterInvalid("Invalid max length {}",
                                                kv_pair.value());
            }
        }
    }

    for (const auto& kv_pair : field.index_params()) {
        if (kv_pair.key() == kMaxLengthKey) {
            try {
                auto length = std::stoll(kv_pair.value());
                if (length <= 0 || length > kMaxLengthLimit) {
                    return Status::ParameterInvalid(
                        "the maximum length specified for a VarChar should "
                        "be "
                        "in (0, 65535])");

                    return Status::Ok();
                } else {
                    *max_len = static_cast<uint64_t>(length);
                    return Status::Ok();
                }
            } catch (std::exception& e) {
                return Status::ParameterInvalid("Invalid max length {}",
                                                kv_pair.value());
            }
        }
    }
    return Status::ParameterInvalid(
        "type param(max_length) should be specified for varChar field of "
        "collection");
}

bool
CreateCollectionTask::CheckDefaultValue(
    const ::milvus::proto::schema::CollectionSchema& schema) {
    for (const auto& f : schema.fields()) {
        if (!f.has_default_value() || !f.has_default_value())
            continue;
        switch (f.default_value().data_case()) {
            case DCase::kBoolData:
                if (f.data_type() != DType::Bool) {
                    LOG_ERROR(
                        "{} field's default value is Bool type, mismatches "
                        "field type",
                        f.name());
                    return false;
                }
                break;
            case DCase::kIntData: {
                if (f.data_type() != DType::Int16 &&
                    f.data_type() != DType::Int32 &&
                    f.data_type() != DType::Int8) {
                    LOG_ERROR(
                        "{} field's default value is Int type, mismatches "
                        "field type",
                        f.name());
                    return false;
                }
                auto default_value = f.default_value().int_data();
                if (f.data_type() == DType::Int16) {
                    if (default_value < std::numeric_limits<int16_t>::min() ||
                        default_value > std::numeric_limits<int16_t>::max()) {
                        LOG_ERROR("{} field's default value out of range.",
                                  f.name());
                        return false;
                    }
                }
                if (f.data_type() == DType::Int8) {
                    if (default_value < std::numeric_limits<int8_t>::min() ||
                        default_value > std::numeric_limits<int8_t>::max()) {
                        LOG_ERROR("{} field's default value out of range.",
                                  f.name());
                        return false;
                    }
                }
            } break;
            case DCase::kLongData:
                if (f.data_type() != DType::Int64) {
                    LOG_ERROR(
                        "{} field's default value is Long type, mismatches "
                        "field type",
                        f.name());
                    return false;
                }
                break;
            case DCase::kFloatData:
                if (f.data_type() != DType::Float) {
                    LOG_ERROR(
                        "{} field's default value is Float type, "
                        "mismatches "
                        "field type",
                        f.name());
                    return false;
                }
                break;
            case DCase::kDoubleData:
                if (f.data_type() != DType::Double) {
                    LOG_ERROR(
                        "{} field's default value is Double type, "
                        "mismatches "
                        "field type",
                        f.name());
                    return false;
                }
                break;
            case DCase::kStringData: {
                if (f.data_type() != DType::VarChar) {
                    LOG_ERROR(
                        "{} field's default value is VarChar type, "
                        "mismatches field type",
                        f.name());
                    return false;
                }
                auto string_len = f.default_value().string_data().size();
                uint64_t max_length = 0;
                auto s = GetVarcharFieldMaxLength(f, &max_length);
                if (s.IsErr()) {
                    LOG_ERROR(s.Detail());
                    return false;
                }

                if (string_len > max_length) {
                    return false;
                }

            } break;
            // case DCase::kBytesData:  // not used
            //     break;
            default:
                return false;
                break;
        }
    }
    return true;
}

void
CreateCollectionTask::AssignFieldId(
    ::milvus::proto::schema::CollectionSchema* schema) {
    for (int i = 0; i < schema->fields_size(); i++) {
        schema->mutable_fields(i)->set_fieldid(kStartOfUserFieldId + i);
    }
}

void
CreateCollectionTask::AppendDynamicField(
    ::milvus::proto::schema::CollectionSchema* schema) {
    if (schema->enable_dynamic_field()) {
        auto dynamice_field = schema->add_fields();
        dynamice_field->set_name(kMetaFieldName);
        dynamice_field->set_description("dynamic schema");
        dynamice_field->set_data_type(DType::JSON);
        dynamice_field->set_is_dynamic(true);
    }
}

void
CreateCollectionTask::AppendSysFields(
    ::milvus::proto::schema::CollectionSchema* schema) {
    auto row_id_field = schema->add_fields();
    row_id_field->set_fieldid(kRowIdField);
    row_id_field->set_name(kRowIdFieldName);
    row_id_field->set_is_primary_key(false);
    row_id_field->set_description("row id");
    row_id_field->set_data_type(DType::Int64);

    auto ts_field = schema->add_fields();
    ts_field->set_fieldid(kTimeStampField);
    ts_field->set_name(kTimeStampFieldName);
    ts_field->set_is_primary_key(false);
    ts_field->set_description("time stamp");
    ts_field->set_data_type(DType::Int64);
}

Status
VaildBM25Functions(const ::milvus::proto::schema::CollectionSchema& schema,
                   const milvus::proto::schema::FunctionSchema& fs) {
    if (fs.type() != milvus::proto::schema::FunctionType::BM25) {
        return Status::ParameterInvalid(
            "MilvusLite only supports BM25 function");
    }
    if (fs.input_field_names_size() != 1 || fs.output_field_names_size() != 1) {
        return Status::ParameterInvalid(
            "The input and output of the BM25 function must be 1");
    }
    bool found_input = false;
    bool found_output = false;
    for (const auto& field : schema.fields()) {
        if (field.name() == fs.input_field_names(0)) {
            found_input = true;
            if (field.is_dynamic()) {
                return Status::ParameterInvalid(
                    "Function's input cannot be dynamic field");
            }
            if (field.data_type() != milvus::proto::schema::DataType::VarChar) {
                return Status::ParameterInvalid(
                    "BM25 funciton's input must be of VARCHAR type");
            }
            if (field.nullable()) {
                return Status::ParameterInvalid(
                    "Function's input cannot be nullable field");
            }
        }

        if (field.name() == fs.output_field_names(0)) {
            found_output = true;
            if (field.is_dynamic()) {
                return Status::ParameterInvalid(
                    "Function's output cannot be dynamic field");
            }
            if (field.data_type() !=
                milvus::proto::schema::DataType::SparseFloatVector) {
                return Status::ParameterInvalid(
                    "BM25 funciton's output must be of SparseFloatVector "
                    "type");
            }
            if (field.nullable()) {
                return Status::ParameterInvalid(
                    "Function's output cannot be nullable field");
            }
        }
    }
    if (!found_input) {
        return Status::ParameterInvalid("function's input {} not found",
                                        fs.input_field_names(0));
    }

    if (!found_output) {
        return Status::ParameterInvalid("function's output {} not found",
                                        fs.output_field_names(0));
    }
    return Status::Ok();
}

Status
CreateCollectionTask::ValidateSchema(
    const ::milvus::proto::schema::CollectionSchema& schema) {
    if (schema.fields_size() > kSchemaFieldLimit)
        return Status::ParameterInvalid(
            "maximum field's number should be limited to {}",
            kSchemaFieldLimit);

    std::set<std::string> field_names;
    std::string pk_name;
    for (const auto& field_schema : schema.fields()) {
        if (field_names.find(field_schema.name()) != field_names.end()) {
            return Status::ParameterInvalid("Duplicated field name: {}",
                                            field_schema.name());
        }
        if (field_schema.is_primary_key()) {
            if (!pk_name.empty()) {
                return Status::ParameterInvalid(
                    "there are more than one primary key, field_name = {}, "
                    "{}",
                    pk_name,
                    field_schema.name());
            } else {
                pk_name = field_schema.name();
            }
        }
        if (field_schema.is_dynamic()) {
            return Status::ParameterInvalid(
                "cannot explicitly set a field as a dynamic field");
        }
        CHECK_STATUS(CheckFieldParams(field_schema), "");
        CHECK_STATUS(CheckFieldName(field_schema.name()), "");
        if (field_schema.data_type() == DType::VarChar) {
            uint64_t max_length = 0;
            CHECK_STATUS(GetVarcharFieldMaxLength(field_schema, &max_length),
                         "");
        }
    }

    // valid functions
    for (const auto& f : schema.functions()) {
        CHECK_STATUS(VaildBM25Functions(schema, f), "");
    }
    return Status::Ok();
}

Status
CreateCollectionTask::Process(
    ::milvus::proto::schema::CollectionSchema* schema) {
    if (!schema->ParseFromString(create_collection_request_->schema())) {
        LOG_ERROR("Failed parse schema");
        return Status::ParameterInvalid("Failed parse schema");
    }

    if (create_collection_request_->collection_name() != schema->name()) {
        auto err = string_util::SFormat(
            "collection name [{}] not matches schema name [{}]",
            create_collection_request_->collection_name(),
            schema->name());
        LOG_ERROR(err);
        return Status::ParameterInvalid(err);
    }

    CHECK_STATUS(ValidateSchema(*schema), "");

    if (HasSystemFields(*schema)) {
        auto err_msg =
            string_util::SFormat("Schema contains system field {}, {}, {}",
                                 kRowIdFieldName,
                                 kTimeStampFieldName,
                                 kMetaFieldName);
        LOG_ERROR(err_msg);
        return Status::ParameterInvalid(err_msg);
    }

    if (!CheckDefaultValue(*schema)) {
        return Status::ParameterInvalid();
    }

    AppendDynamicField(schema);
    AssignFieldId(schema);
    AppendSysFields(schema);
    return Status::Ok();
}

Status
CreateCollectionTask::CheckFieldName(const std::string& field_name) {
    std::string name = string_util::Trim(field_name);
    if (name.empty()) {
        return Status::ParameterInvalid("field {} should not be empty", name);
    }
    std::string invalid_msg =
        string_util::SFormat("Invalid field name {}. ", name);
    if (name.size() > 255) {
        return Status::ParameterInvalid(
            "{}, the length of a field name must "
            "be less than 255 characters",
            invalid_msg);
    }

    char first = name[0];
    if (first != '_' && !string_util::IsAlpha(first)) {
        return Status::ParameterInvalid(
            "{} the first character of a field {} must be an underscore "
            "or letter",
            invalid_msg,
            name);
    }
    std::regex pattern("^[a-zA-Z_][a-zA-Z0-9_]*$");
    if (!std::regex_match(name, pattern)) {
        auto err = string_util::SFormat(
            "{},  field name can only contain "
            "numbers, letters and underscores",
            invalid_msg);
        LOG_ERROR(err);
        return Status::ParameterInvalid(err);
    }
    return Status::Ok();
}

}  // namespace milvus::local
