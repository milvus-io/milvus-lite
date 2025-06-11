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

#include <sys/types.h>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "common.h"
#include "log/Log.h"
#include "pb/schema.pb.h"
#include "pb/common.pb.h"
#include "pb/milvus.pb.h"
#include "status.h"
#include "string_util.hpp"

namespace milvus::local::function {

using milvus::local::Status;

class FunctionMeta : public NonCopyableNonMovable {
 private:
    FunctionMeta(const milvus::proto::schema::CollectionSchema* schema,
                 const milvus::proto::schema::FunctionSchema* func_schema,
                 const std::vector<const milvus::proto::schema::FieldSchema*>
                     input_fields,
                 const std::vector<const milvus::proto::schema::FieldSchema*>
                     output_fields)
        : func_schema_(func_schema),
          input_fields_(input_fields),
          output_fields_(output_fields),
          collection_name_(schema->name()),
          function_type_name_(func_schema->GetTypeName()),
          function_name_(func_schema->name()) {
    }

 public:
    virtual ~FunctionMeta() = default;

 public:
    static std::pair<Status, std::unique_ptr<FunctionMeta>>
    Create(const milvus::proto::schema::CollectionSchema* schema,
           const milvus::proto::schema::FunctionSchema* func_schema) {
        std::map<const std::string, const milvus::proto::schema::FieldSchema*>
            name_field_map;
        for (const auto& field : schema->fields()) {
            name_field_map[field.name()] = &field;
        }

        std::vector<const milvus::proto::schema::FieldSchema*> output_fields;
        for (const auto& name : func_schema->output_field_names()) {
            auto it = name_field_map.find(name);
            if (it == name_field_map.end()) {
                auto err = string_util::SFormat(
                    "Can not find function output {} in schema", name);
                LOG_ERROR(err);
                return std::make_pair(Status::ParameterInvalid(err), nullptr);
            }
            output_fields.push_back(it->second);
        }

        std::vector<const milvus::proto::schema::FieldSchema*> input_fields;
        for (const auto& name : func_schema->input_field_names()) {
            auto it = name_field_map.find(name);
            if (it == name_field_map.end()) {
                auto err = string_util::SFormat(
                    "Can not find function input {} in schema", name);
                LOG_ERROR(err);
                return std::make_pair(Status::ParameterInvalid(err), nullptr);
            }
            input_fields.push_back(it->second);
        }
        return std::make_pair(
            Status::Ok(),
            std::unique_ptr<FunctionMeta>(new FunctionMeta(
                schema, func_schema, input_fields, output_fields)));
    }

 public:
    const milvus::proto::schema::FunctionSchema* func_schema_;
    const std::vector<const milvus::proto::schema::FieldSchema*> input_fields_;
    const std::vector<const milvus::proto::schema::FieldSchema*> output_fields_;

    const std::string collection_name_;
    const std::string function_type_name_;
    const std::string function_name_;
};

class TransformFunctionBase {
 public:
    TransformFunctionBase(std::unique_ptr<FunctionMeta> meta)
        : meta_(std::move(meta)) {
    }
    virtual ~TransformFunctionBase() = default;

 public:
    virtual Status
    ProcessInsert(
        const std::vector<const milvus::proto::schema::FieldData*>& inputs,
        std::vector<milvus::proto::schema::FieldData>* outputs) = 0;
    virtual Status
    ProcessSearch(const milvus::proto::common::PlaceholderGroup& input_plg,
                  milvus::proto::common::PlaceholderGroup* output_plg) = 0;

    const milvus::proto::schema::FieldSchema*
    GetFunctionOutputFieldSchema(uint i) {
        return meta_->output_fields_[i];
    }

    const milvus::proto::schema::FieldSchema*
    GetFunctioninputFieldSchema(uint i) {
        return meta_->input_fields_[i];
    }

    const std::set<std::string>
    GetInputFieldNames() {
        std::set<std::string> field_names;
        for (const auto& f : meta_->input_fields_) {
            field_names.insert(f->name());
        }
        return field_names;
    }

 protected:
    std::unique_ptr<FunctionMeta> meta_;
};

}  // namespace milvus::local::function
