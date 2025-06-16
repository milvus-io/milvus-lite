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

#include "schema_util.h"

#include <complex>
#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include "common.h"
#include "log/Log.h"
#include "parser/utils.h"
#include "parser/parser.h"

namespace milvus::local {
namespace schema_util {

using DType = ::milvus::proto::schema::DataType;

std::any
GetField(const ::milvus::proto::schema::FieldData& field_data,
         uint32_t field_index) {
    if (field_data.type() == DType::FloatVector) {
        std::vector<float> vec;
        int64_t dim = field_data.vectors().dim();
        auto vd = field_data.vectors().float_vector();
        for (int index = field_index * dim; index < (field_index + 1) * dim;
             index++) {
            vec.push_back(vd.data(index));
        }
        return vec;
    } else if (field_data.type() == DType::BinaryVector) {
        int64_t dim = field_data.vectors().dim();
        int64_t size = dim / 8;
        std::string v = field_data.vectors().binary_vector().substr(
            field_index * size, size);
        return v;
    } else if (field_data.type() == DType::Float16Vector) {
        int64_t dim = field_data.vectors().dim();
        int64_t size = dim * 2;
        std::string v = field_data.vectors().float16_vector().substr(
            field_index * size, size);
        return v;
    } else if (field_data.type() == DType::BFloat16Vector) {
        int64_t dim = field_data.vectors().dim();
        int64_t size = dim * 2;
        std::string v = field_data.vectors().bfloat16_vector().substr(
            field_index * size, size);
        return v;
    } else if (field_data.type() == DType::Bool) {
        return field_data.scalars().bool_data().data(field_index);
    } else if (field_data.type() == DType::Int8 ||
               field_data.type() == DType::Int16 ||
               field_data.type() == DType::Int32) {
        return field_data.scalars().int_data().data(field_index);
    } else if (field_data.type() == DType::Int64) {
        return field_data.scalars().long_data().data(field_index);
    } else if (field_data.type() == DType::Float) {
        return field_data.scalars().float_data().data(field_index);
    } else if (field_data.type() == DType::Double) {
        return field_data.scalars().double_data().data(field_index);
    } else if (field_data.type() == DType::String) {
        return field_data.scalars().string_data().data(field_index);
    } else if (field_data.type() == DType::VarChar) {
        return field_data.scalars().string_data().data(field_index);
    } else if (field_data.type() == DType::Array) {
        auto array_data = field_data.scalars().array_data();
        ::milvus::proto::schema::ArrayArray new_array;
        new_array.set_element_type(array_data.element_type());
        new_array.add_data()->CopyFrom(array_data.data(field_index));
        return new_array;
    } else if (field_data.type() == DType::JSON) {
        return field_data.scalars().json_data().data(field_index);
    } else if (field_data.type() == DType::SparseFloatVector) {
        ::milvus::proto::schema::SparseFloatArray sp;
        sp.set_dim(field_data.vectors().sparse_float_vector().dim());
        auto content =
            field_data.vectors().sparse_float_vector().contents(field_index);
        sp.add_contents(content);
        return sp;
    } else {
        LOG_ERROR("Unkown data type: {}", field_data.type());
        return nullptr;
    }
    return nullptr;
}

bool
IsVectorField(::milvus::proto::schema::DataType dtype) {
    return dtype == ::milvus::proto::schema::DataType::FloatVector ||
           dtype == ::milvus::proto::schema::DataType::BinaryVector ||
           dtype == ::milvus::proto::schema::DataType::Float16Vector ||
           dtype == ::milvus::proto::schema::DataType::BFloat16Vector ||
           dtype == ::milvus::proto::schema::DataType::SparseFloatVector;
}

bool
IsSparseVectorType(::milvus::proto::schema::DataType dtype) {
    return dtype == ::milvus::proto::schema::DataType::SparseFloatVector;
}

bool
FindDimFromFieldParams(const ::milvus::proto::schema::FieldSchema& field,
                       std::string* dim) {
    for (const auto& param : field.type_params()) {
        if (param.key() == kDimKey) {
            dim->assign(param.value());
            return true;
        }
    }

    for (const auto& param : field.index_params()) {
        if (param.key() == kDimKey) {
            dim->assign(param.value());
            return true;
        }
    }
    return false;
}

int64_t
GetDim(const ::milvus::proto::schema::FieldSchema& field) {
    if (!IsVectorField(field.data_type())) {
        LOG_ERROR("{} is not vector type", field.data_type());
        return -1;
    }
    if (IsSparseVectorType(field.data_type())) {
        LOG_ERROR("GetDim should not invoke on sparse vector type");
        return -1;
    }

    std::string dim_str;
    bool succ = FindDimFromFieldParams(field, &dim_str);
    if (!succ) {
        LOG_ERROR("Dim not found");
        return -1;
    }
    try {
        return std::stoll(dim_str);
    } catch (const std::invalid_argument& e) {
        LOG_ERROR("invalid dimension: {}, {}", dim_str, e.what());
    }
    return -1;
}

bool
FillEmptyField(const ::milvus::proto::schema::FieldSchema& field_schema,
               ::milvus::proto::schema::FieldData* field_data) {
    field_data->set_field_name(field_schema.name());
    field_data->set_type(field_schema.data_type());
    field_data->set_field_id(field_schema.fieldid());
    field_data->set_is_dynamic(field_schema.is_dynamic());

    if (field_schema.data_type() == DType::FloatVector) {
        int64_t dim = GetDim(field_schema);
        if (dim < 0) {
            return false;
        }
        auto vec_field = field_data->mutable_vectors();
        vec_field->set_dim(dim);
        vec_field->mutable_float_vector();
    } else if (field_schema.data_type() == DType::BinaryVector) {
        int64_t dim = GetDim(field_schema);
        if (dim < 0) {
            return false;
        }
        auto vec_field = field_data->mutable_vectors();
        vec_field->set_dim(dim);
        vec_field->mutable_binary_vector();
    } else if (field_schema.data_type() == DType::Float16Vector) {
        int64_t dim = GetDim(field_schema);
        if (dim < 0) {
            return false;
        }
        auto vec_field = field_data->mutable_vectors();
        vec_field->set_dim(dim);
        vec_field->mutable_float16_vector();
    } else if (field_schema.data_type() == DType::BFloat16Vector) {
        int64_t dim = GetDim(field_schema);
        if (dim < 0) {
            return false;
        }
        auto vec_field = field_data->mutable_vectors();
        vec_field->set_dim(dim);
        vec_field->mutable_bfloat16_vector();
    } else if (field_schema.data_type() == DType::Bool) {
        field_data->mutable_scalars()->mutable_bool_data();
    } else if (field_schema.data_type() == DType::Int8 ||
               field_schema.data_type() == DType::Int16 ||
               field_schema.data_type() == DType::Int32) {
        field_data->mutable_scalars()->mutable_int_data();
    } else if (field_schema.data_type() == DType::Int64) {
        field_data->mutable_scalars()->mutable_long_data();
    } else if (field_schema.data_type() == DType::Float) {
        field_data->mutable_scalars()->mutable_float_data();
    } else if (field_schema.data_type() == DType::Double) {
        field_data->mutable_scalars()->mutable_double_data();
    } else if (field_schema.data_type() == DType::String) {
        field_data->mutable_scalars()->mutable_string_data();
    } else if (field_schema.data_type() == DType::VarChar) {
        field_data->mutable_scalars()->mutable_string_data();
    } else if (field_schema.data_type() == DType::Array) {
        field_data->mutable_scalars()->mutable_array_data();
    } else if (field_schema.data_type() == DType::JSON) {
        field_data->mutable_scalars()->mutable_json_data();
    } else if (field_schema.data_type() == DType::SparseFloatVector) {
        field_data->mutable_vectors()->mutable_sparse_float_vector();
    } else {
        LOG_ERROR("Unkown data type: {}", field_schema.data_type());
        return false;
    }
    return true;
}

bool
FindDimFromSchema(const ::milvus::proto::schema::CollectionSchema& schema,
                  std::string* dim) {
    for (const auto& field : schema.fields()) {
        if (IsVectorField(field.data_type())) {
            return FindDimFromFieldParams(field, dim);
        }
    }
    return false;
}

std::optional<::milvus::proto::plan::VectorType>
DataTypeToVectorType(::milvus::proto::schema::DataType dtype) {
    if (dtype == ::milvus::proto::schema::DataType::FloatVector) {
        return ::milvus::proto::plan::VectorType::FloatVector;
    } else if (dtype == ::milvus::proto::schema::DataType::BinaryVector) {
        return ::milvus::proto::plan::VectorType::BinaryVector;
    } else if (dtype == ::milvus::proto::schema::DataType::Float16Vector) {
        return ::milvus::proto::plan::VectorType::Float16Vector;
    } else if (dtype == ::milvus::proto::schema::DataType::BFloat16Vector) {
        return ::milvus::proto::plan::VectorType::BFloat16Vector;
    } else if (dtype == ::milvus::proto::schema::DataType::SparseFloatVector) {
        return ::milvus::proto::plan::VectorType::SparseFloatVector;
    } else {
        return std::nullopt;
    }
}

Status
FindVectorField(const ::milvus::proto::schema::CollectionSchema& schema,
                const std::string& ann_field,
                const ::milvus::proto::schema::FieldSchema** field) {
    std::map<const std::string, const ::milvus::proto::schema::FieldSchema*>
        vec_fields;
    for (const auto& field : schema.fields()) {
        if (IsVectorField(field.data_type())) {
            vec_fields[field.name()] = &field;
        }
    }
    if (vec_fields.size() == 0) {
        auto err = string_util::SFormat(
            "Can not found vector field in collection {}", schema.name());
        LOG_ERROR(err);
        return Status::ParameterInvalid(err);
    }
    if (ann_field.empty()) {
        if (vec_fields.size() > 1) {
            auto err = string_util::SFormat(
                "multiple anns_fields exist, please specify a anns_field "
                "insearch_params");
            return Status::ParameterInvalid(err);
        } else {
            *field = vec_fields.begin()->second;
            return Status::Ok();
        }
    } else {
        if (vec_fields.find(ann_field) == vec_fields.end()) {
            auto err =
                string_util::SFormat("fieldName({}) not found", ann_field);
            LOG_ERROR(err);
            return Status::ParameterInvalid(err);
        }
        *field = vec_fields.at(ann_field);
        return Status::Ok();
    }
}

std::string
MergeIndexs(std::vector<std::string>& indexs) {
    ::milvus::proto::segcore::CollectionIndexMeta index_meta;
    index_meta.set_maxindexrowcount(kMaxIndexRow);
    for (size_t i = 0; i < indexs.size(); i++) {
        index_meta.add_index_metas()->ParseFromString(indexs[i]);
    }
    return index_meta.SerializeAsString();
}

std::string
GetIndexMetricType(const milvus::proto::segcore::FieldIndexMeta& field_index) {
    for (int i = 0; i < field_index.type_params_size(); i++) {
        if (field_index.type_params(i).key() == kMetricTypeKey) {
            return field_index.type_params(i).value();
        }
    }

    for (int i = 0; i < field_index.index_params_size(); i++) {
        if (field_index.index_params(i).key() == kMetricTypeKey) {
            return field_index.index_params(i).value();
        }
    }

    for (int i = 0; i < field_index.user_index_params_size(); i++) {
        if (field_index.user_index_params(i).key() == kMetricTypeKey) {
            return field_index.user_index_params(i).value();
        }
    }
    return "";
}

std::optional<int64_t>
GetPkId(const ::milvus::proto::schema::CollectionSchema& schema) {
    for (const auto& field : schema.fields()) {
        if (field.is_primary_key()) {
            return field.fieldid();
        }
    }
    return std::nullopt;
}

std::optional<std::string>
GetPkName(const ::milvus::proto::schema::CollectionSchema& schema) {
    for (const auto& field : schema.fields()) {
        if (field.is_primary_key()) {
            return field.name();
        }
    }
    return std::nullopt;
}

bool
PickFieldDataByIndex(const ::milvus::proto::schema::FieldData& src_data,
                     const std::vector<int64_t>& indexes,
                     ::milvus::proto::schema::FieldData* dst) {
    for (int64_t i : indexes) {
        switch (src_data.type()) {
            case DType::FloatVector: {
                dst->mutable_vectors()->set_dim(src_data.vectors().dim());
                auto vec =
                    std::any_cast<std::vector<float>>(GetField(src_data, i));
                for (const auto& item : vec) {
                    dst->mutable_vectors()->mutable_float_vector()->add_data(
                        item);
                }
            } break;

            case DType::BinaryVector: {
                dst->mutable_vectors()->set_dim(src_data.vectors().dim());
                auto vec = std::any_cast<std::string>(GetField(src_data, i));
                dst->mutable_vectors()->mutable_binary_vector()->assign(
                    dst->mutable_vectors()->binary_vector() + vec);
            } break;

            case DType::Float16Vector: {
                dst->mutable_vectors()->set_dim(src_data.vectors().dim());
                auto vec = std::any_cast<std::string>(GetField(src_data, i));
                dst->mutable_vectors()->mutable_float16_vector()->assign(
                    dst->mutable_vectors()->float16_vector() + vec);
            } break;

            case DType::BFloat16Vector: {
                dst->mutable_vectors()->set_dim(src_data.vectors().dim());
                auto vec = std::any_cast<std::string>(GetField(src_data, i));
                dst->mutable_vectors()->mutable_bfloat16_vector()->assign(
                    dst->mutable_vectors()->bfloat16_vector() + vec);
            } break;

            case DType::Bool: {
                auto data = std::any_cast<bool>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_bool_data()->add_data(data);
            } break;

            case DType::Int8: {
                auto data = std::any_cast<int>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_int_data()->add_data(data);
            } break;

            case DType::Int16: {
                auto data = std::any_cast<int>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_int_data()->add_data(data);
            } break;

            case DType::Int32: {
                auto data = std::any_cast<int32_t>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_int_data()->add_data(data);
            } break;

            case DType::Int64: {
                auto data = std::any_cast<int64_t>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_long_data()->add_data(data);
            } break;

            case DType::Float: {
                auto data = std::any_cast<float>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_float_data()->add_data(data);
            } break;

            case DType::Double: {
                auto data = std::any_cast<double>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_double_data()->add_data(data);
            } break;

            case DType::String: {
                auto data = std::any_cast<std::string>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_string_data()->add_data(data);
            } break;

            case DType::VarChar: {
                auto data = std::any_cast<std::string>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_string_data()->add_data(data);
            } break;

            case DType::Array: {
                auto data = std::any_cast<::milvus::proto::schema::ArrayArray>(
                    GetField(src_data, i));
                auto arr = dst->mutable_scalars()->mutable_array_data();
                arr->set_element_type(data.element_type());
                arr->add_data()->CopyFrom(data.data(0));
            } break;
            case DType::JSON: {
                auto data = std::any_cast<std::string>(GetField(src_data, i));
                dst->mutable_scalars()->mutable_json_data()->add_data(data);
            } break;
            case DType::SparseFloatVector: {
                auto data =
                    std::any_cast<::milvus::proto::schema::SparseFloatArray>(
                        GetField(src_data, i));
                auto mutable_sp =
                    dst->mutable_vectors()->mutable_sparse_float_vector();
                mutable_sp->set_dim(data.dim());
                mutable_sp->add_contents(data.contents(0));
            } break;

            default:
                LOG_ERROR("Field: [{}-{}] unkown data type: {}",
                          src_data.field_name(),
                          src_data.field_id(),
                          src_data.type());
                return false;
        }
    }
    return true;
}

bool
SliceFieldData(const ::milvus::proto::schema::FieldData& src_data,
               const std::vector<std::tuple<int64_t, int64_t>>& ranges,
               ::milvus::proto::schema::FieldData* dst) {
    std::vector<int64_t> indexes;
    for (const auto& range : ranges) {
        int64_t offset = std::get<0>(range);
        int64_t limit = std::get<1>(range);
        for (int64_t i = offset; i < offset + limit; i++) {
            indexes.push_back(i);
        }
    }
    return PickFieldDataByIndex(src_data, indexes, dst);
}

// Support wildcard in output fields:
//
//"*" - all fields
//
// For example, A and B are scalar fields, C and D are vector fields, duplicated fields will automatically be removed.
//
//output_fields=["*"]  ==> [A,B,C,D]
//output_fields=["*",A]  ==> [A,B,C,D]
//output_fields=["*",C]    ==> [A,B,C,D]
bool
TranslateOutputFields(
    const ::google::protobuf::RepeatedPtrField<std::string>& raw_fields,
    const ::milvus::proto::schema::CollectionSchema& schema,
    bool add_primary,
    std::vector<std::string>* result_outputs,
    std::vector<std::string>* user_output_fields) {
    std::string pk_name;

    std::set<std::string> all_fields;

    // when enable dynamic field, result_field store the real field of collection,
    // user_output_field store user-specified name;
    std::set<std::string> result_field;
    std::set<std::string> user_output_field;

    for (const auto& field : schema.fields()) {
        if (field.is_primary_key()) {
            pk_name = field.name();
        }
        if (field.fieldid() >= kStartOfUserFieldId) {
            all_fields.insert(field.name());
        }
    }

    for (const auto& name : raw_fields) {
        auto output_name = string_util::Trim(name);
        if (output_name == "*") {
            for (const std::string& name : all_fields) {
                result_field.insert(name);
                user_output_field.insert(name);
            }
        } else {
            if (all_fields.find(output_name) != all_fields.end()) {
                result_field.insert(output_name);
                user_output_field.insert(output_name);
            } else {
                if (schema.enable_dynamic_field()) {
                    milvus::proto::schema::CollectionSchema schema_;
                    schema_.CopyFrom(schema);
                    auto helper = milvus::local::CreateSchemaHelper(&schema_);
                    auto expr = ParseIdentifier(helper, name);
                    if (expr->column_expr().info().nested_path_size() == 1 &&
                        expr->column_expr().info().nested_path(0) == name) {
                        result_field.insert(kMetaFieldName);
                        user_output_field.insert(name);
                    }
                } else {
                    LOG_ERROR("Field {} not exist", output_name);
                    return false;
                }
            }
        }
    }

    std::set<std::string> bm25_fields_outputs;

    for (const auto& func : schema.functions()) {
        if (func.type() == ::milvus::proto::schema::FunctionType::BM25) {
            for (const auto& output_field : func.output_field_names()) {
                bm25_fields_outputs.insert(output_field);
            }
        }
    }

    if (add_primary) {
        result_field.insert(pk_name);
        user_output_field.insert(pk_name);
    }

    // remove bm25 fields outputs
    for (const auto& fname : bm25_fields_outputs) {
        result_field.erase(fname);
        user_output_field.erase(fname);
    }

    for (const std::string& fname : result_field) {
        result_outputs->push_back(fname);
    }

    for (const std::string& fname : user_output_field) {
        user_output_fields->push_back(fname);
    }

    return true;
}

bool
ReduceFieldByIDs(const ::milvus::proto::schema::IDs& ids,
                 const ::milvus::proto::schema::FieldData& src,
                 ::milvus::proto::schema::FieldData* dst,
                 int64_t* real_size) {
    std::set<std::string> unique_ids;
    std::vector<int64_t> indexes;
    dst->set_type(src.type());
    dst->set_field_id(src.field_id());
    dst->set_field_name(src.field_name());
    dst->set_is_dynamic(src.is_dynamic());
    if (ids.has_int_id()) {
        for (int64_t i = 0; i < ids.int_id().data_size(); ++i) {
            auto cur_id = std::to_string(ids.int_id().data(i));
            if (unique_ids.find(cur_id) != unique_ids.end())
                continue;
            unique_ids.insert(cur_id);
            indexes.push_back(i);
        }
    } else if (ids.has_str_id()) {
        for (int64_t i = 0; i < ids.str_id().data_size(); ++i) {
            auto cur_id = ids.str_id().data(i);
            if (unique_ids.find(cur_id) != unique_ids.end())
                continue;
            unique_ids.insert(cur_id);
            indexes.push_back(i);
        }
    } else {
        // empty data
        return true;
    }
    *real_size = unique_ids.size();
    return PickFieldDataByIndex(src, indexes, dst);
}

Status
ParseExpr(const std::string& expr_str,
          ::milvus::proto::schema::CollectionSchema schema,
          ::milvus::proto::plan::Expr* expr_out) {
    try {
        auto buf = milvus::local::ParserToMessage(schema, expr_str);
        expr_out->ParseFromString(buf);
        return Status::Ok();
    } catch (std::exception& e) {
        LOG_ERROR(e.what());
        return Status::ParameterInvalid(
            string_util::SFormat("Invalid expr: {}", expr_str));
    }
}

bool
SchemaEquals(const std::string& schema_str_l, const std::string& schema_str_r) {
    ::milvus::proto::schema::CollectionSchema schema_l, schema_r;
    if (!schema_l.ParseFromString(schema_str_l) ||
        !schema_r.ParseFromString(schema_str_r)) {
        LOG_ERROR("Parse schema failed");
        return false;
    }
    if (schema_l.name() != schema_r.name() ||
        schema_l.description() != schema_r.description() ||
        schema_l.enable_dynamic_field() != schema_r.enable_dynamic_field() ||
        schema_l.fields_size() != schema_r.fields_size() ||
        !CheckParamsEqual(schema_l.properties(), schema_r.properties())) {
        return false;
    }
    // check field
    for (int i = 0; i < schema_l.fields_size(); i++) {
        if (schema_l.fields(i).fieldid() != schema_r.fields(i).fieldid() ||
            schema_l.fields(i).name() != schema_r.fields(i).name() ||
            schema_l.fields(i).is_primary_key() !=
                schema_r.fields(i).is_primary_key() ||
            schema_l.fields(i).description() !=
                schema_r.fields(i).description() ||
            schema_l.fields(i).data_type() != schema_r.fields(i).data_type() ||
            schema_l.fields(i).autoid() != schema_r.fields(i).autoid() ||
            schema_l.fields(i).is_dynamic() !=
                schema_r.fields(i).is_dynamic() ||
            schema_l.fields(i).element_type() !=
                schema_r.fields(i).element_type() ||
            !CheckValueFieldEqual(schema_l.fields(i).default_value(),
                                  schema_r.fields(i).default_value()) ||
            !CheckParamsEqual(schema_l.fields(i).type_params(),
                              schema_r.fields(i).type_params()) ||
            !CheckParamsEqual(schema_l.fields(i).index_params(),
                              schema_r.fields(i).index_params())

        ) {
            return false;
        }
    }
    return true;
}

bool
CheckParamsEqual(const ::google::protobuf::RepeatedPtrField<
                     ::milvus::proto::common::KeyValuePair>& left,
                 const ::google::protobuf::RepeatedPtrField<
                     ::milvus::proto::common::KeyValuePair>& right) {
    if (left.size() != right.size())
        return false;
    KVMap right_map;
    for (int i = 0; i < right.size(); i++) {
        right_map[right[i].key()] = right[i].value();
    }
    for (int i = 0; i < left.size(); i++) {
        auto it = right_map.find(left[i].key());
        if (it == right_map.end())
            return false;
        if (it->second != left[i].value())
            return false;
    }
    return true;
}

bool
CheckValueFieldEqual(const ::milvus::proto::schema::ValueField& left,
                     const ::milvus::proto::schema::ValueField& right) {
    if (!left.IsInitialized() && !right.IsInitialized())
        return true;

    if (left.data_case() != right.data_case())
        return false;

    if (left.data_case() == 0) {
        return true;
    }

    if (left.has_bool_data() && right.has_bool_data() &&
        left.bool_data() == right.bool_data())
        return true;
    if (left.has_int_data() && right.has_int_data() &&
        left.int_data() == right.int_data())
        return true;
    if (left.has_long_data() && right.has_long_data() &&
        left.long_data() == right.long_data())
        return true;
    if (left.has_float_data() && right.has_float_data() &&
        std::fabs(left.float_data() - right.float_data()) < 0.00001f)
        return true;
    if (left.has_double_data() && right.has_double_data() &&
        std::fabs(left.double_data() - right.double_data()) < 0.0000001)
        return true;
    if (left.has_string_data() && right.has_string_data() &&
        left.string_data() == right.string_data())
        return true;
    if (left.has_bytes_data() && right.has_bytes_data() &&
        left.bytes_data() == right.bytes_data())
        return true;
    return false;
}

void
FillInFieldInfo(
    const std::vector<std::string>& output_fields,
    const ::milvus::proto::schema::CollectionSchema& schema,
    std::variant<::milvus::proto::schema::SearchResultData*,
                 ::milvus::proto::milvus::QueryResults*> result_var) {
    auto process = [](const auto& output_fields,
                      const auto& schema,
                      auto&&... args) {
        return [&output_fields, &schema, args...](auto&& result_data) {
            if (output_fields.size() == 0 ||
                result_data->fields_data_size() == 0) {
                return;
            }
            for (size_t i = 0; i < output_fields.size(); i++) {
                const std::string& name = output_fields[i];
                for (const auto& field : schema.fields()) {
                    if (name == field.name()) {
                        auto field_id = field.fieldid();
                        for (int j = 0; j < result_data->fields_data().size();
                             j++) {
                            if (field_id ==
                                result_data->fields_data(j).field_id()) {
                                result_data->mutable_fields_data(j)
                                    ->set_field_name(field.name());
                                result_data->mutable_fields_data(j)
                                    ->set_field_id(field.fieldid());
                                result_data->mutable_fields_data(j)->set_type(
                                    field.data_type());
                                result_data->mutable_fields_data(j)
                                    ->set_is_dynamic(field.is_dynamic());
                            }
                        }
                    }
                }
            }
        };
    };
    std::visit(process(output_fields, schema), result_var);
}

bool
GetOutputFieldsIds(const std::vector<std::string>& output_fields,
                   const ::milvus::proto::schema::CollectionSchema& schema,
                   std::vector<int64_t>* ids) {
    std::map<std::string, int64_t> name_ids;
    for (const auto& field : schema.fields()) {
        name_ids[field.name()] = field.fieldid();
    }

    for (const auto& output_field : output_fields) {
        auto it = name_ids.find(output_field);
        if (it == name_ids.end()) {
            LOG_ERROR("Can not find output field {} in schema", output_field);
            return false;
        }
        ids->push_back(it->second);
    }
    return true;
}

}  // namespace schema_util

}  // namespace milvus::local
