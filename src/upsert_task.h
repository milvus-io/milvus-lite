#pragma once

#include "common.h"
#include "pb/milvus.pb.h"
#include "pb/schema.pb.h"
#include "status.h"

namespace milvus::local {
class UpsertTask : NonCopyableNonMovable {
 public:
    UpsertTask(const ::milvus::proto::milvus::UpsertRequest* r,
               const ::milvus::proto::schema::CollectionSchema* schema)
        : upsert_request_(r), schema_(schema) {
    }
    virtual ~UpsertTask() = default;

 public:
    Status
    Process(::milvus::proto::milvus::InsertRequest* insert_q,
            ::milvus::proto::schema::IDs* delete_ids);

 private:
    const ::milvus::proto::milvus::UpsertRequest* upsert_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;
};

}  // namespace milvus::local
