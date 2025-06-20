// Copyright (C) 2019-2025 Zilliz. All rights reserved.
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

#include <memory>
#include <utility>
#include "common.h"
#include "function/function.h"
#include "schema.pb.h"
#include "status.h"
#include "tantivy/tokenizer.h"

namespace milvus::local::function {

class BM25Function : public TransformFunctionBase {
 public:
    BM25Function(std::unique_ptr<FunctionMeta> meta, std::string& params)
        : TransformFunctionBase(std::move(meta)),
          tokenizer_(std::move(params)) {
    }

 public:
    static std::pair<Status, std::unique_ptr<TransformFunctionBase>>
    NewBM25Function(const milvus::proto::schema::CollectionSchema* schema,
                    const milvus::proto::schema::FunctionSchema* func_schema) {
        auto [s, meta] = FunctionMeta::Create(schema, func_schema);
        if (!s.IsOk()) {
            return std::make_pair(s, nullptr);
        }
        if (meta->func_schema_->output_field_names_size() != 1) {
            return std::make_pair(
                Status::ParameterInvalid(
                    "BM25 function only supports one onput, but got {}",
                    meta->output_fields_.size()),
                nullptr);
        }
        if (meta->func_schema_->input_field_names_size() != 1) {
            return std::make_pair(
                Status::ParameterInvalid(
                    "BM25 function only supports one input, but got {}",
                    meta->output_fields_.size()),
                nullptr);
        }
        if (meta->input_fields_[0]->data_type() !=
            milvus::proto::schema::DataType::VarChar) {
            return std::make_pair(
                Status::ParameterInvalid(
                    "BM25 function only supports varchar input, but got {}",
                    milvus::proto::schema::DataType_Name(
                        meta->input_fields_[0]->data_type())),
                nullptr);
        }
        if (meta->output_fields_[0]->data_type() !=
            milvus::proto::schema::DataType::SparseFloatVector) {
            return std::make_pair(
                Status::ParameterInvalid(
                    "BM25 function only supports sparse "
                    "vector output, but got {}",
                    milvus::proto::schema::DataType_Name(
                        meta->output_fields_[0]->data_type())),
                nullptr);
        }

        std::string token_params;
        for (const auto& param : meta->input_fields_[0]->type_params()) {
            if (param.key() == "analyzer_params") {
                token_params = param.value();
                break;
            }
        }
        return std::make_pair(
            Status::Ok(),
            std::unique_ptr<TransformFunctionBase>(
                new BM25Function(std::move(meta), token_params)));
    }

 public:
    Status
    ProcessInsert(
        const std::vector<const milvus::proto::schema::FieldData*>& inputs,
        std::vector<milvus::proto::schema::FieldData>* outputs) override;

    Status
    ProcessSearch(const milvus::proto::common::PlaceholderGroup& input_plg,
                  milvus::proto::common::PlaceholderGroup* output_plg) override;

 private:
    Status
    Embedding(const std::string& doc, SparseVector* embd);

 private:
    milvus::tantivy::Tokenizer tokenizer_;
};

}  // namespace milvus::local::function
