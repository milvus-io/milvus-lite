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

#include "storage.h"
#include <SQLiteCpp/Database.h>
#include <SQLiteCpp/Transaction.h>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include "collection_data.h"
#include "log/Log.h"
#include "pb/schema.pb.h"
#include "segcore.pb.h"
#include "storage/bm25_stats.h"
#include "storage/collection_meta.h"
#include "type.h"

namespace milvus::local {

Storage::Storage(const char* db_file) : db_file_(db_file) {
}

Storage::~Storage() {
}

bool
Storage::Open() {
    try {
        db_ptr_ = std::make_unique<SQLite::Database>(
            db_file_,
            SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE |
                SQLite::OPEN_FULLMUTEX);

        SQLite::Transaction transaction(*db_ptr_.get());
        if (!cm_.Init(db_ptr_.get())) {
            return false;
        }
        std::vector<std::string> names;
        cm_.CollectionNames(&names);
        for (const auto& name : names) {
            collections_.emplace(
                name, std::make_unique<CollectionData>(name.c_str()));
        }
        bm25_stats_ = std::unique_ptr<BM25Stats>(new BM25Stats());
        if (!bm25_stats_->CreateTables(db_ptr_.get())) {
            return false;
        }
        transaction.commit();
        return true;
    } catch (std::exception& e) {
        LOG_ERROR("Open storage failed, err: {}", e.what());
        return false;
    }
}

bool
Storage::CreateCollection(const std::string& collection_name,
                          const std::string& pk_name,
                          const std::string& schema_proto) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!cm_.CreateCollection(
            db_ptr_.get(), collection_name, pk_name, schema_proto)) {
        return false;
    }
    auto data_ptr = std::make_unique<CollectionData>(collection_name.c_str());
    if (!data_ptr->CreateCollection(db_ptr_.get())) {
        return false;
    }
    collections_[collection_name] = std::move(data_ptr);
    transaction.commit();
    return true;
}

bool
Storage::DropCollection(const std::string& collection_name) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!cm_.DropCollection(db_ptr_.get(), collection_name)) {
        LOG_ERROR("Delete collection: {}'s meta failed", collection_name);
        return false;
    }
    if (!collections_[collection_name]->DropCollection(db_ptr_.get())) {
        LOG_ERROR("Delete collection: {}'s data failed", collection_name);
        return false;
    }
    collections_.erase(collection_name);
    if (!bm25_stats_->DropCollectionStats(db_ptr_.get(), collection_name)) {
        return false;
    }
    transaction.commit();
    return true;
}

bool
Storage::LoadCollecton(const std::string& collection_name,
                       int64_t start,
                       int64_t size,
                       std::vector<std::string>* out_rows) {
    collections_[collection_name]->Load(db_ptr_.get(), start, size, out_rows);
    return static_cast<int64_t>(out_rows->size()) == size;
}

bool
Storage::GetCollectionSchema(const std::string& collection_name,
                             std::string* output_info_str) {
    output_info_str->assign(cm_.GetCollectionSchema(collection_name).c_str());
    return true;
}

bool
Storage::CreateIndex(const std::string& collection_name,
                     const std::string& index_name,
                     const std::string& index_proto) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!cm_.CreateIndex(
            db_ptr_.get(), collection_name, index_name, index_proto)) {
        return false;
    }
    transaction.commit();
    return true;
}

bool
Storage::GetIndex(const std::string& collection_name,
                  const std::string& index_name,
                  std::string* output_index_proto) {
    return cm_.GetCollectionIndex(
        collection_name, index_name, output_index_proto);
}

void
Storage::GetAllIndex(const std::string& collection_name,
                     const std::string& exclude,
                     std::vector<std::string>* index_protos) {
    cm_.GetAllIndex(collection_name, exclude, index_protos);
}

bool
Storage::DropIndex(const std::string& collection_name,
                   const std::string& index_name) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!cm_.DropIndex(db_ptr_.get(), collection_name, index_name)) {
        return false;
    }
    transaction.commit();
    return true;
}

bool
Storage::CollectBM25Stats(const std::vector<Row>& rows,
                          const std::vector<std::string>& bm25_field_names,
                          std::map<std::string, Stats>* stats) {
    for (const auto& field_name : bm25_field_names) {
        (*stats)[field_name] = Stats();
        (*stats)[field_name].rows_num = rows.size();
    }

    for (const auto& row : rows) {
        milvus::proto::segcore::InsertRecord r;
        if (!r.ParseFromString(std::get<1>(row))) {
            LOG_ERROR("Parse insert record failed");
            return false;
        }
        std::map<const std::string, const milvus::proto::schema::FieldData*>
            field_map;
        for (const auto& field : r.fields_data()) {
            if (field.type() ==
                milvus::proto::schema::DataType::SparseFloatVector) {
                field_map[field.field_name()] = &field;
            }
        }
        for (const auto& field_name : bm25_field_names) {
            if (field_map.find(field_name) == field_map.end()) {
                LOG_ERROR("Field {} not found or not sparse vector field",
                          field_name);
                return false;
            }
            // The data has been split by row, so there will only be one sparse vector
            auto vec =
                field_map[field_name]->vectors().sparse_float_vector().contents(
                    0);
            auto pos = vec.c_str();
            auto end = vec.c_str() + vec.size();
            for (; pos < end; pos += 8) {
                const uint32_t token =
                    *(reinterpret_cast<const uint32_t*>(pos));
                const float count = *(reinterpret_cast<const float*>(pos + 4));
                (*stats)[field_name].token_doc_count[token] += 1;
                (*stats)[field_name].token_num += int(count);
            }
        }
    }
    return true;
}

int
Storage::Insert(const std::string collection_name,
                const std::vector<Row>& rows,
                const std::vector<std::string>& bm25_field_names) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!bm25_field_names.empty()) {
        std::map<std::string, Stats> stats;
        if (!CollectBM25Stats(rows, bm25_field_names, &stats)) {
            return -1;
        }
        for (const auto& [field_name, stats] : stats) {
            if (!bm25_stats_->AddBM25Stats(db_ptr_.get(),
                                           collection_name,
                                           field_name,
                                           stats.token_num,
                                           stats.rows_num)) {
                return -1;
            }
            if (!bm25_stats_->AddTokenDoc(db_ptr_.get(),
                                          collection_name,
                                          field_name,
                                          stats.token_doc_count)) {
                return -1;
            }
        }
    }
    for (const auto& row : rows) {
        if (collections_[collection_name]->Insert(db_ptr_.get(),
                                                  std::get<0>(row).c_str(),
                                                  std::get<1>(row)) < 0) {
            return -1;
        }
    }
    transaction.commit();
    return rows.size();
}

int
Storage::Delete(const std::string collection_name,
                const std::vector<std::string>& ids,
                const std::vector<std::string>& bm25_field_names) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!bm25_field_names.empty()) {
        std::vector<Row> rows;
        if (!collections_[collection_name]->GetByIDs(
                db_ptr_.get(), ids, &rows)) {
            return -1;
        }
        std::map<std::string, Stats> stats;
        if (!CollectBM25Stats(rows, bm25_field_names, &stats)) {
            return -1;
        }
        for (const auto& [field_name, stats] : stats) {
            if (!bm25_stats_->DeleteBM25Stats(db_ptr_.get(),
                                              collection_name,
                                              field_name,
                                              stats.token_num,
                                              stats.rows_num)) {
                return -1;
            }
            if (!bm25_stats_->DeleteTokenDoc(db_ptr_.get(),
                                             collection_name,
                                             field_name,
                                             stats.token_doc_count)) {
                return -1;
            }
        }
    }
    int n = collections_[collection_name]->Delete(db_ptr_.get(), ids);
    transaction.commit();
    return n;
}

int64_t
Storage::Count(const std::string& collection_name) {
    SQLite::Transaction transaction(*db_ptr_.get());
    int64_t n = collections_[collection_name]->Count(db_ptr_.get());
    transaction.commit();
    return n;
}

}  // namespace milvus::local
