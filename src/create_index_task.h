#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "common.h"
#include "pb/common.pb.h"
#include "pb/milvus.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "status.h"

namespace milvus::local {

class CreateIndexTask final : NonCopyableNonMovable {
 public:
    CreateIndexTask(
        const ::milvus::proto::milvus::CreateIndexRequest* create_index_request,
        const ::milvus::proto::schema::CollectionSchema* schema)
        : create_index_request_(create_index_request), schema_(schema) {
        field_id_ = 0;
        collectionid_ = 0;
        is_auto_index_ = false;
    }
    virtual ~CreateIndexTask() = default;

 public:
    Status
    Process(milvus::proto::segcore::FieldIndexMeta* field_meta);

 private:
    Status
    ParseIndexParams();

    void
    WrapUserIndexParams(const std::string& metrics_type);

    bool
    AddAutoIndexParams(size_t numberParams, KVMap* index_params);

    Status
    CheckTrain(const ::milvus::proto::schema::FieldSchema& field,
               KVMap& index_params);

    bool
    ValidateStringIndexType(const std::string& index_type) {
        return index_type == kDefaultStringIndexType ||
               index_type == "marisa-trie" || index_type == kInvertedIndexType;
    }

    bool
    ValidateArithmeticIndexType(const std::string& index_type) {
        return index_type == kDefaultStringIndexType ||
               index_type == "Asceneding" || index_type == kInvertedIndexType;
    }

    bool
    FillDimension(const ::milvus::proto::schema::FieldSchema& field,
                  KVMap* index_params);

    bool
    IsVectorIndex(::milvus::proto::schema::DataType dtype);

 private:
    // string of ::milvus::proto::milvus::CreateIndexRequest;
    const ::milvus::proto::milvus::CreateIndexRequest* create_index_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;

    std::vector<::milvus::proto::common::KeyValuePair> new_index_params_;
    std::vector<::milvus::proto::common::KeyValuePair> new_type_params_;
    std::vector<::milvus::proto::common::KeyValuePair> new_extra_params_;

    std::string index_name_;
    int64_t field_id_;
    int64_t collectionid_;
    bool is_auto_index_;
};

}  // namespace milvus::local
