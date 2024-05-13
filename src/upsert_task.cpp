#include "upsert_task.h"
#include <cstdint>
#include <string>
#include "schema.pb.h"
#include "status.h"

namespace milvus::local {

Status
UpsertTask::Process(::milvus::proto::milvus::InsertRequest* insert_q,
                    ::milvus::proto::schema::IDs* delete_ids) {
    const ::milvus::proto::schema::FieldSchema* primary_field = nullptr;
    for (const auto& field_schema : schema_->fields()) {
        if (field_schema.is_primary_key()) {
            primary_field = &field_schema;
            break;
        }
    }
    if (primary_field == nullptr)
        return Status::ServiceInternal("Can not found primary key");

    for (const auto& field_data : upsert_request_->fields_data()) {
        if (field_data.field_name() == primary_field->name()) {
            if (field_data.type() == ::milvus::proto::schema::DataType::Int64) {
                delete_ids->mutable_int_id()->CopyFrom(
                    field_data.scalars().long_data());
            } else if (field_data.type() ==
                       ::milvus::proto::schema::DataType::VarChar) {
                delete_ids->mutable_str_id()->CopyFrom(
                    field_data.scalars().string_data());
            } else {
                return Status::ParameterInvalid("Primary key error");
            }
            break;
        }
    }

    // upsert request to insert request
    insert_q->mutable_base()->CopyFrom(upsert_request_->base());
    insert_q->set_db_name(upsert_request_->db_name());
    insert_q->set_collection_name(upsert_request_->collection_name());
    insert_q->set_partition_name(upsert_request_->partition_name());
    insert_q->mutable_fields_data()->CopyFrom(upsert_request_->fields_data());
    insert_q->mutable_hash_keys()->CopyFrom(upsert_request_->hash_keys());
    insert_q->set_num_rows(upsert_request_->num_rows());
    return Status::Ok();
}

}  // namespace milvus::local
