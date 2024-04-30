#pragma once

#include "pb/milvus.pb.h"
#include "pb/schema.pb.h"
#include "pb/plan.pb.h"
#include "common.h"
#include "status.h"

namespace milvus::local {

class DeleteTask : NonCopyableNonMovable {
 public:
    DeleteTask(const ::milvus::proto::milvus::DeleteRequest* delete_request,
               const ::milvus::proto::schema::CollectionSchema* schema)
        : delete_request_(delete_request), schema_(schema) {
    }
    virtual ~DeleteTask() = default;

    Status
    Process(::milvus::proto::plan::PlanNode* plan);

 private:
    const ::milvus::proto::milvus::DeleteRequest* delete_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;
};

}  // namespace milvus::local
