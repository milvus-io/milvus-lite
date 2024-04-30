#pragma once
#include <any>
#include <cstdint>
#include <vector>
#include "pb/milvus.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "status.h"
#include "type.h"
#include "common.h"

namespace milvus::local {

class InsertTask : NonCopyableNonMovable {
 public:
    InsertTask(::milvus::proto::milvus::InsertRequest* r,
               const ::milvus::proto::schema::CollectionSchema* schema);
    virtual ~InsertTask() = default;

 public:
    Status
    Process(Rows* insert_records);

    ::milvus::proto::schema::DataType
    PkType() {
        return pk_type_;
    }

 private:
    bool
    GenFieldMap();

    Status
    CheckVectorDim();

    std::vector<uint64_t>
    GetTimestamps(int64_t size);

    uint64_t
    GetTimestamp();

    std::vector<int64_t>
    GetRowIds(std::vector<uint64_t>& timestamps);

    bool
    AddSystemField();

    bool
    CheckDynamicFieldData();

 private:
    static int64_t cur_id_;
    ::milvus::proto::milvus::InsertRequest* insert_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;
    std::map<const std::string, const ::milvus::proto::schema::FieldData*>
        field_data_map_;
    uint32_t num_rows_;
    ::milvus::proto::schema::DataType pk_type_;
};

}  // namespace milvus::local
