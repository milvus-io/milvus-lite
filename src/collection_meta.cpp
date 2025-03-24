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

#include "collection_meta.h"
#include <SQLiteCpp/VariadicBind.h>
// #include <sqlite3.h>
#include <exception>
#include <memory>
#include <tuple>
#include "log/Log.h"
#include "string_util.hpp"

namespace milvus::local {

const std::string kSchemaStr = "schema";
const std::string kIndexStr = "index";
const std::string kPartitionStr = "PARTITION";

CollectionMeta::CollectionMeta()
    : table_meta_name_("collection_meta"),
      col_id_("id"),
      col_collection_name_("collection_name"),
      col_meta_type_("meta_type"),
      col_blob_field_("blob_field"),
      col_string_field_("string_field") {
}

CollectionMeta::~CollectionMeta() {
}

bool
CollectionMeta::LoadMeta(SQLite::Database* db) {
    // SELECT * FROM {table_meta_name_}
    std::string load_cmd =
        string_util::SFormat("SELECT * FROM {}", table_meta_name_);
    try {
        SQLite::Statement query(*db, load_cmd);
        while (query.executeStep()) {
            auto collection_name = query.getColumn(1).getString();
            LOG_INFO("Load {}'s meta", collection_name);
            if (collections_.find(collection_name) == collections_.end()) {
                collections_.emplace(collection_name,
                                     std::make_unique<CollectionInfo>());
            }
            auto meta_type = query.getColumn(2).getString();
            if (meta_type == kSchemaStr) {
                auto info =
                    static_cast<const char*>(query.getColumn(3).getBlob());
                auto pk_name = query.getColumn(4).getString();
                collections_[collection_name]->AddSchema(info, pk_name);
            } else if (meta_type == kIndexStr) {
                auto info =
                    static_cast<const char*>(query.getColumn(3).getBlob());
                auto index_name = query.getColumn(4).getString();
                collections_[collection_name]->AddIndex(index_name, info);
            } else {
                LOG_ERROR("Unkown meta data");
                return false;
            }
        }
        return true;
    } catch (std::exception& e) {
        LOG_ERROR("Load meta data failed, err: {}", e.what());
        return false;
    }
}

bool
CollectionMeta::CreateTable(SQLite::Database* db) {
    const std::string table_create_cmd = string_util::SFormat(
        "CREATE TABLE IF NOT EXISTS {} ({} INTEGER PRIMARY KEY, {} "
        "VARCHAR(1024), {} VARCHAR(1024), {} BLOB, {} "
        "VARCHAR(1024))",
        table_meta_name_,
        col_id_,
        col_collection_name_,
        col_meta_type_,
        col_blob_field_,
        col_string_field_);

    if (db->tryExec(table_create_cmd) != 0) {
        const char* err = db->getErrorMsg();
        LOG_ERROR("Create table failed, errs: {}", err);
        return false;
    }
    return true;
}

bool
CollectionMeta::Init(SQLite::Database* db) {
    return CreateTable(db) && LoadMeta(db);
}

bool
CollectionMeta::CreateCollection(SQLite::Database* db,
                                 const std::string& collection_name,
                                 const std::string& pk_name,
                                 const std::string& schema_proto) {
    collections_.emplace(collection_name, std::make_unique<CollectionInfo>());
    collections_[collection_name]->AddSchema(schema_proto.c_str(), pk_name);

    // INSERT INTO {table_name} VALUES (NULL, {collection_name}, "schema", {data}, NULL)
    std::string insert_cmd = string_util::SFormat(
        "INSERT INTO {} VALUES (NULL, ?, ?, ?, ?)", table_meta_name_);
    try {
        SQLite::Statement query(*db, insert_cmd);
        SQLite::bind(query, collection_name, kSchemaStr, schema_proto, pk_name);
        return query.exec() == 1;
    } catch (std::exception& e) {
        LOG_ERROR("Insert data failed, errs: {}", e.what());
        return false;
    }
}

const std::string&
CollectionMeta::GetCollectionSchema(const std::string& collection_name) {
    return collections_[collection_name]->Schema();
}

bool
CollectionMeta::CreateIndex(SQLite::Database* db,
                            const std::string& collection_name,
                            const std::string& index_name,
                            const std::string& index_proto) {
    // INSERT INTO {table_name} VALUES (NULL, {collection_name}, "schema", {data}, NULL)
    collections_[collection_name]->AddIndex(index_name, index_proto.c_str());
    std::string insert_cmd = string_util::SFormat(
        "INSERT INTO {} VALUES (NULL, ?, 'index', ?, ?)", table_meta_name_);
    try {
        SQLite::Statement query(*db, insert_cmd);
        SQLite::bind(query, collection_name, index_proto, index_name);
        return query.exec() == 1;
    } catch (std::exception& e) {
        LOG_ERROR("Add index failed, err: {}", e.what());
        return false;
    }
}

bool
CollectionMeta::GetCollectionIndex(const std::string& collection_name,
                                   const std::string& index_name,
                                   std::string* output_index_info) {
    return collections_[collection_name]->GetIndex(index_name,
                                                   output_index_info);
}

void
CollectionMeta::GetAllIndex(const std::string& collection_name,
                            const std::string& exclude,
                            std::vector<std::string>* all_index) {
    collections_[collection_name]->GetAllIndexs(all_index, exclude);
}

bool
CollectionMeta::DropCollection(SQLite::Database* db,
                               const std::string& collection_name) {
    // DELETE FROM {table_name} WHERE {col_collection_name_}={collection_name};
    std::string delete_cmd = string_util::SFormat(
        "DELETE FROM {} WHERE "
        "{}='{}'",
        table_meta_name_,
        col_collection_name_,
        collection_name);
    try {
        collections_.erase(collection_name);
        db->exec(delete_cmd);
        return true;
    } catch (std::exception& e) {
        LOG_ERROR(
            "Drop collection: {} failed, err: {}", collection_name, e.what());
        return false;
    }
}

bool
CollectionMeta::DropIndex(SQLite::Database* db,
                          const std::string& collection_name,
                          const std::string& index_name) {
    // DELETE FROM {table_name} WHERE {col_collection_name_}={collection_name} and {col_meta_type_}={kIndexStr} and {col_string_field_}={index_name};
    std::string drop_index_cmd = string_util::SFormat(
        "DELETE FROM {} WHERE "
        "{}='{}' and {}='{}' and {}='{}'",
        table_meta_name_,
        col_collection_name_,
        collection_name,
        col_meta_type_,
        kIndexStr,
        col_string_field_,
        index_name);
    try {
        collections_[collection_name]->DropIndex(index_name);
        db->exec(drop_index_cmd);
        return true;
    } catch (std::exception& e) {
        LOG_ERROR("Drop collection {}'s index:{} failed, err: {}",
                  collection_name,
                  index_name,
                  e.what());
        return false;
    }
}

}  // namespace milvus::local
