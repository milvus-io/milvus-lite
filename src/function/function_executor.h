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

#include <map>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "common.h"
#include "function/function.h"
#include "status.h"

namespace milvus::local::function {

using milvus::local::Status;

class FunctionExecutor : NonCopyableNonMovable {
 public:
    virtual ~FunctionExecutor() = default;

 private:
    explicit FunctionExecutor(
        std::vector<std::unique_ptr<TransformFunctionBase>> functions)
        : functions_(std::move(functions)) {
    }

 public:
    Status
    ProcessInsert(milvus::proto::milvus::InsertRequest* insert);

    Status
    ProcessSearch(milvus::proto::milvus::SearchRequest* search);

 private:
    Status
    ProcessSingeFunction(milvus::proto::milvus::InsertRequest* insert,
                         const std::unique_ptr<TransformFunctionBase>& f);

 public:
    static std::pair<Status, std::unique_ptr<FunctionExecutor>>
    Create(const milvus::proto::schema::CollectionSchema* schema,
           std::string function_name);

    static std::pair<Status, std::unique_ptr<FunctionExecutor>>
    Create(const milvus::proto::schema::CollectionSchema* schema);

 private:
    static std::pair<Status, std::unique_ptr<TransformFunctionBase>>
    CreateFunction(
        const milvus::proto::schema::CollectionSchema* schema,
        const milvus::proto::schema::FunctionSchema* function_schema);

 private:
    const std::vector<std::unique_ptr<TransformFunctionBase>> functions_;
    //std::unique_ptr<TransformFunctionBase> function_;
};

}  // namespace milvus::local::function
