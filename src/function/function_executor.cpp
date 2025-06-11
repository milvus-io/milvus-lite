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

#include "function/function_executor.h"
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include "common.h"
#include "function/bm25_function.h"
#include "function/function_util.h"
#include "log/Log.h"
#include "schema.pb.h"
#include "status.h"

namespace milvus::local::function {

std::pair<Status, std::unique_ptr<FunctionExecutor>>
FunctionExecutor::Create(const milvus::proto::schema::CollectionSchema* schema,
                         std::string field_name) {
    for (const auto& f_schema : schema->functions()) {
        if (f_schema.output_field_names(0) == field_name) {
            auto [s, f] = CreateFunction(schema, &f_schema);
            if (!s.IsOk()) {
                return std::make_pair(s, nullptr);
            }
            std::vector<std::unique_ptr<TransformFunctionBase>> functions_;
            functions_.emplace_back(std::move(f));
            std::unique_ptr<FunctionExecutor> executor(
                new FunctionExecutor(std::move(functions_)));
            return std::make_pair(Status::Ok(), std::move(executor));
        }
    }
    return std::make_pair(
        Status::ParameterInvalid("No function's output is {}", field_name),
        nullptr);
}

std::pair<Status, std::unique_ptr<FunctionExecutor>>
FunctionExecutor::Create(
    const milvus::proto::schema::CollectionSchema* schema) {
    std::vector<std::unique_ptr<TransformFunctionBase>> functions_;
    for (const auto& f_schemn : schema->functions()) {
        auto [s, f] = CreateFunction(schema, &f_schemn);
        if (!s.IsOk()) {
            return std::make_pair(s, nullptr);
        }
        functions_.emplace_back(std::move(f));
    }
    std::unique_ptr<FunctionExecutor> executor(
        new FunctionExecutor(std::move(functions_)));
    return std::make_pair(Status::Ok(), std::move(executor));
}

std::pair<Status, std::unique_ptr<TransformFunctionBase>>
FunctionExecutor::CreateFunction(
    const milvus::proto::schema::CollectionSchema* schema,
    const milvus::proto::schema::FunctionSchema* function_schema) {
    if (function_schema->type() == milvus::proto::schema::FunctionType::BM25) {
        auto [s, f] = BM25Function::NewBM25Function(schema, function_schema);
        if (!s.IsOk()) {
            return std::make_pair(s, nullptr);
        }
        return std::make_pair(Status::Ok(), std::move(f));

    } else {
        return std::make_pair(
            Status::ParameterInvalid("Unsupported function: {}",
                                     milvus::proto::schema::FunctionType_Name(
                                         function_schema->type())),
            nullptr);
    }
}

Status
FunctionExecutor::ProcessInsert(milvus::proto::milvus::InsertRequest* insert) {
    for (const auto& f : functions_) {
        CHECK_STATUS(ProcessSingeFunction(insert, f), "");
    }
    return Status::Ok();
}

Status
FunctionExecutor::ProcessSingeFunction(
    milvus::proto::milvus::InsertRequest* insert,
    const std::unique_ptr<TransformFunctionBase>& f) {
    auto field_names = f->GetInputFieldNames();

    std::vector<const milvus::proto::schema::FieldData*> inputs;
    for (int i = 0; i < insert->fields_data_size(); i++) {
        if (field_names.find(insert->fields_data(i).field_name()) !=
            field_names.end()) {
            inputs.emplace_back(&insert->fields_data(i));
        }
    }
    std::vector<milvus::proto::schema::FieldData> outputs;
    CHECK_STATUS(f->ProcessInsert(inputs, &outputs), "");
    for (auto& field : outputs) {
        insert->mutable_fields_data()->Add(std::move(field));
    }
    return Status::Ok();
}

Status
FunctionExecutor::ProcessSearch(milvus::proto::milvus::SearchRequest* search) {
    milvus::proto::common::PlaceholderGroup ph_group;
    if (!ph_group.ParseFromString(search->placeholder_group())) {
        return Status::ParameterInvalid("Parse placehoder string failed");
    }

    if (ph_group.placeholders_size() != 1) {
        return Status::ParameterInvalid("placeholders size is not equal 1");
    }

    milvus::proto::common::PlaceholderGroup output;
    CHECK_STATUS(functions_[0]->ProcessSearch(ph_group, &output), "");
    search->mutable_placeholder_group()->assign(output.SerializeAsString());
    return Status::Ok();
}

}  // namespace milvus::local::function
