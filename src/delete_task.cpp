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
