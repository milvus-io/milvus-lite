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

#include "delete_task.h"
#include "antlr4-runtime.h"
#include "log/Log.h"
#include "parser/parser.h"
#include "parser/utils.h"
#include "schema_util.h"
#include "status.h"
#include "string_util.hpp"

namespace milvus::local {

Status
DeleteTask::Process(::milvus::proto::plan::PlanNode* plan) {
    if (string_util::Trim(delete_request_->expr()) == "") {
        return Status::ParameterInvalid("expr cannot be empty");
    }
    CHECK_STATUS(
        schema_util::ParseExpr(delete_request_->expr(),
                               *schema_,
                               plan->mutable_query()->mutable_predicates()),
        "");

    auto pk_id = schema_util::GetPkId(*schema_);
    if (!pk_id.has_value()) {
        LOG_ERROR("Can not found {}'s primary key", schema_->name());
        return Status::CollectionIllegalSchema();
    }
    plan->add_output_field_ids(*pk_id);
    return Status::Ok();
}

}  // namespace milvus::local
