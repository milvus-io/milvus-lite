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

#include "milvus_local.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <regex>
#include "common.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "schema_util.h"
#include "status.h"
#include "storage.h"
#include "log/Log.h"
#include "string_util.hpp"
#include "bm25_stat.h"
namespace milvus::local {

#define CHECK_COLLECTION_EXIST(collection_name)                               \
    do {                                                                      \
        CHECK_STATUS(CheckCollectionName(string_util::Trim(collection_name)), \
                     "");                                                     \
        if (!storage_.CollectionExist(collection_name)) {                     \
            return Status::CollectionNotFound();                              \
        }                                                                     \
    } while (0)

#define CHECK_COLLECTION_NOT_EXIST(collection_name)                           \
    do {                                                                      \
        CHECK_STATUS(CheckCollectionName(string_util::Trim(collection_name)), \
                     "");                                                     \
        if (storage_.CollectionExist(collection_name)) {                      \
            return Status::CollectionAlreadExist();                           \
        }                                                                     \
    } while (0)

MilvusLocal::MilvusLocal(const char* db_file)
    : db_file_(db_file), storage_(db_file), initialized(false) {
}

MilvusLocal::~MilvusLocal() {
}

Status
MilvusLocal::CheckCollectionName(const std::string& collection_name) {
    if (collection_name.empty()) {
        return Status::ParameterInvalid("collection name should not be empty");
    }
    std::string invalid_msg =
        string_util::SFormat("Invalid collection {}. ", collection_name);
    if (collection_name.size() > 255) {
        return Status::ParameterInvalid(
            "{}, the length of a collection name must "
            "be less than 255 characters",
            invalid_msg);
    }

    char first = collection_name[0];
    if (first != '_' && !string_util::IsAlpha(first)) {
        return Status::ParameterInvalid(
            "{} the first character of a collection {} must be an underscore "
            "or letter",
            invalid_msg,
            collection_name);
    }
    std::regex pattern("^[a-zA-Z_][a-zA-Z0-9_]*$");
    if (!std::regex_match(collection_name, pattern)) {
        auto err = string_util::SFormat(
            "{},  collection name can only contain "
            "numbers, letters and underscores",
            invalid_msg);
        LOG_ERROR(err);
        return Status::ParameterInvalid(err);
    }
    return Status::Ok();
}

bool
MilvusLocal::Init() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized) {
        LOG_WARN("Milvus has already initialized");
        return false;
    }

    if (!storage_.Open()) {
        return false;
    }
    initialized = true;
    return true;
}

Status
MilvusLocal::LoadCollection(const std::string& collection_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    if (index_.HasLoaded(collection_name)) {
        return Status::Ok();
    }

    return DoLoadCollection(collection_name);
}

Status
MilvusLocal::ReleaseCollection(const std::string& collection_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    if (index_.DropCollection(collection_name)) {
        return Status::Ok();
    }
    return Status::SegcoreErr();
}

Status
MilvusLocal::GetLoadState(const std::string& collection_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    if (!index_.HasLoaded(collection_name)) {
        return Status::CollectionNotLoaded();
    }
    return Status::CollectionLoaded();
}

Status
MilvusLocal::CreateCollection(const std::string& collection_name,
                              const std::string& pk_name,
                              const std::string& schema_proto) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_STATUS(CheckCollectionName(string_util::Trim(collection_name)), "");

    if (storage_.CollectionExist(collection_name)) {
        std::string db_schema_proto;
        if (!storage_.GetCollectionSchema(collection_name, &db_schema_proto)) {
            return Status::ServiceInternal();
        }
        if (!schema_util::SchemaEquals(schema_proto, db_schema_proto)) {
            return Status::ParameterInvalid(
                "create duplicate collection with different parameters, "
                "collection {}",
                collection_name);
        }
        return Status::Ok();
    }

    // CHECK_COLLECTION_NOT_EXIST(collection_name);
    CHECK_STATUS(index_.CreateCollection(collection_name, schema_proto, ""),
                 "");
    if (!storage_.CreateCollection(collection_name, pk_name, schema_proto)) {
        return Status::ServiceInternal();
    }
    return Status::Ok();
}

Status
MilvusLocal::GetCollection(const std::string& collection_name,
                           std::string* schema_proto) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    if (!storage_.GetCollectionSchema(collection_name, schema_proto)) {
        return Status::ServiceInternal();
    }
    return Status::Ok();
}

bool
MilvusLocal::DropCollection(const std::string& collection_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!storage_.CollectionExist(collection_name)) {
        LOG_WARN("Collection {} not existed", collection_name);
        return true;
    }
    return index_.DropCollection(collection_name) &&
           storage_.DropCollection(collection_name);
}

void
MilvusLocal::GetAllCollections(std::vector<std::string>* collection_names) {
    std::lock_guard<std::mutex> lock(mutex_);
    storage_.ListCollections(collection_names);
}

Status
MilvusLocal::CreateIndex(const std::string& collection_name,
                         const std::string& index_name,
                         const std::string& index_proto) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    if (storage_.HasIndex(collection_name, index_name)) {
        // TODO add index info check
        LOG_WARN("Collection {}'s index {} alread existed",
                 collection_name,
                 index_name);
        return Status::Ok();
    }
    // get existed index
    // std::vector<std::string> all_index_proto;
    // storage_.GetAllIndex(collection_name, "", &all_index_proto);
    // all_index_proto.push_back(index_proto);
    // auto index_meta_proto = schema_util::MergeIndexs(all_index_proto);

    // // make sure the index info is legal
    // CHECK_STATUS(index_.CreateIndex(collection_name, index_meta_proto), "");
    if (!storage_.CreateIndex(collection_name, index_name, index_proto)) {
        return Status::ServiceInternal();
    }
    // reload collection
    if (!index_.DropCollection(collection_name)) {
        return Status::ServiceInternal();
    }

    return DoLoadCollection(collection_name);
}

Status
MilvusLocal::DropIndex(const std::string& collection_name,
                       const std::string& index_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    if (!storage_.HasIndex(collection_name, index_name)) {
        return Status::Ok();
    }

    std::vector<std::string> all_index_proto;
    storage_.GetAllIndex(collection_name, index_name, &all_index_proto);
    auto index_meta_proto = schema_util::MergeIndexs(all_index_proto);

    CHECK_STATUS(index_.CreateIndex(collection_name, index_meta_proto), "");
    if (!storage_.DropIndex(collection_name, index_name)) {
        return Status::ServiceInternal();
    }
    auto& stat = bm25::StatDict::Instance();
    auto it = stat.stats_dict.find(collection_name);
    if (it != stat.stats_dict.end()) {
        stat.stats_dict.erase(it);
    }

    return Status::Ok();
}

Status
MilvusLocal::Insert(const std::string& collection_name,
                    const Rows& rows,
                    std::vector<std::string>* ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    int64_t count = 0;
    for (const auto& row : rows) {
        if (index_.Insert(collection_name, 1, std::get<1>(row)).IsErr()) {
            break;
        }
        ids->push_back(std::get<0>(row));
        count += 1;
    }
    auto start = rows.begin();
    auto end = rows.begin() + count;
    std::vector<Row> rows_need_insert(start, end);
    storage_.Insert(collection_name, rows_need_insert);
    auto& stat = bm25::StatDict::Instance();

    if (stat.stats_dict.find(collection_name) != stat.stats_dict.end()) {
        auto& coll_stat = stat.stats_dict[collection_name];
        for (auto& row : rows_need_insert) {
            milvus::proto::segcore::InsertRecord record;
            record.ParseFromString(std::get<1>(row));
            for (int i = 0; i < record.fields_data_size(); ++i) {
                auto data = record.fields_data(i);
                if (data.type() ==
                        milvus::proto::schema::DataType::SparseFloatVector &&
                    data.field_name() == coll_stat.output_field_name) {
                    CHECK(
                        data.vectors().sparse_float_vector().contents_size() ==
                        1);
                    auto cont =
                        data.vectors().sparse_float_vector().contents(0);
                    auto pos = cont.c_str();
                    auto end = cont.c_str() + cont.size();
                    for (; pos < end; pos += 8) {
                        const uint32_t key =
                            *(reinterpret_cast<const uint32_t*>(pos));
                        const float freq =
                            *(reinterpret_cast<const float*>(pos + 4));
                        coll_stat.rows_contain_token[key] += 1;
                        coll_stat.token_num += int(freq);
                    }
                    coll_stat.rows_num += 1;
                    coll_stat.contexts[std::get<0>(row)] =
                        std::string(cont.begin(), cont.end());
                }
            }
        }
    }

    return Status::Ok();
}

Status
MilvusLocal::Retrieve(const std::string& collection_name,
                      const std::string& plan,
                      RetrieveResult* result) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    return index_.Retrieve(collection_name, plan, result);
}

Status
MilvusLocal::Search(const std::string& collection_name,
                    const std::string& plan,
                    const std::string& placeholder_group,
                    SearchResult* result) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    return index_.Search(collection_name, plan, placeholder_group, result);
}

Status
MilvusLocal::DeleteByIds(const std::string& collection_name,
                         const std::string& ids,
                         int64_t size,
                         const std::vector<std::string>& storage_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    if (storage_.Delete(collection_name, storage_ids) == -1) {
        return Status::ServiceInternal();
    }
    CHECK_STATUS(index_.DeleteByIds(collection_name, ids, size), "");

    auto& stat = bm25::StatDict::Instance();

    if (stat.stats_dict.find(collection_name) != stat.stats_dict.end()) {
        auto& coll_stat = stat.stats_dict[collection_name];
        for (auto& id : storage_ids) {
            auto it = coll_stat.contexts.find(id);
            if (it != coll_stat.contexts.end()) {
                const auto& cont = it->second;
                auto pos = cont.c_str();
                auto end = cont.c_str() + cont.size();
                for (; pos < end; pos += 8) {
                    const uint32_t key =
                        *(reinterpret_cast<const uint32_t*>(pos));
                    const float freq =
                        *(reinterpret_cast<const float*>(pos + 4));
                    auto key_it = coll_stat.rows_contain_token.find(key);
                    if (key_it != coll_stat.rows_contain_token.end()) {
                        key_it->second -= 1;
                        if (key_it->second == 0)
                            coll_stat.rows_contain_token.erase(key_it);
                    }
                    coll_stat.token_num -= int(freq);
                }
                coll_stat.rows_num -= 1;
                coll_stat.contexts.erase(it);
            }
        }
    }

    /*
    if (stat.stats_dict.find(collection_name) != stat.stats_dict.end()) {
        auto& coll_stat = stat.stats_dict[collection_name];
        for (auto it = coll_stat.rows_contain_token.begin();
             it != coll_stat.rows_contain_token.end();) {
            auto size_bf = it->second.size();
            for (auto& ids : storage_ids) {
                auto del_it = it->second.find(ids);
                if (del_it != it->second.end())
                    it->second.erase(del_it);
            }
            coll_stat.rows_num -= (size_bf - it->second.size());
            if (it->second.empty()) {
                it = coll_stat.rows_contain_token.erase(it);
                coll_stat.token_num -= 1;
            } else {
                ++it;
            }
        }
    }
    */

    return Status::Ok();
}

Status
MilvusLocal::GetIndex(const std::string& collection_name,
                      const std::string& index_name,
                      std::string* index_proto) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    if (!storage_.GetIndex(collection_name, index_name, index_proto)) {
        return Status::IndexNotFound();
    }
    return Status::Ok();
}

Status
MilvusLocal::GetAllIndexs(const std::string& collection_name,
                          std::vector<std::string>* all_index_proto) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    storage_.GetAllIndex(collection_name, "", all_index_proto);
    return Status::Ok();
}

Status
MilvusLocal::GetNumRowsOfCollection(const std::string& collection_name,
                                    int64_t* num) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_COLLECTION_EXIST(collection_name);
    *num = storage_.Count(collection_name);
    if (*num < 0) {
        return Status::ServiceInternal();
    }
    return Status::Ok();
}

Status
MilvusLocal::DoLoadCollection(const std::string& collection_name) {
    std::string schema_proto, index_proto;
    if (!storage_.GetCollectionSchema(collection_name, &schema_proto)) {
        LOG_ERROR("Can not find {}'s schema", collection_name);
        return Status::ServiceInternal("Schema not found");
    }

    // load index info
    std::vector<std::string> all_index_proto;
    storage_.GetAllIndex(collection_name, "", &all_index_proto);
    auto index_meta_proto = schema_util::MergeIndexs(all_index_proto);

    CHECK_STATUS(index_.CreateCollection(
                     collection_name, schema_proto, index_meta_proto),
                 "");
    std::vector<std::string> rows;
    int64_t start = 0;
    while (true) {
        storage_.LoadCollecton(collection_name, start, 200000, &rows);
        if (rows.size() == 0) {
            LOG_INFO("Success load {} rows", start);
            return Status::Ok();
        }
        for (const auto& row : rows) {
            CHECK_STATUS(index_.Insert(collection_name, 1, row),
                         "Load data failed: ");
        }
        start += rows.size();
        rows.clear();
    }
    return Status::Ok();
}

}  // namespace milvus::local
