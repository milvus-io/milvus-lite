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

#include "collection_data.h"
#include <SQLiteCpp/VariadicBind.h>
// #include <sqlite3.h>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>
#include "string_util.hpp"

#include "log/Log.h"

namespace milvus::local {

CollectionData::CollectionData(const char* collection_name)
    : collection_name_(collection_name),
      col_id_("id"),
      col_milvus_id_("milvus_id"),
      col_data_("data") {
}
CollectionData::~CollectionData() {
}

std::string
CollectionData::GetTableCreateSql() {
    return string_util::SFormat(
        "CREATE TABLE IF NOT EXISTS {} ({} INTEGER PRIMARY KEY, {} "
        "VARCHAR(1024), {} BLOB);",
        collection_name_,
        col_id_,
        col_milvus_id_,
        col_data_);
}

bool
CollectionData::CreateCollection(SQLite::Database* db) {
    const std::string table_create_sql = GetTableCreateSql();
    if (db->tryExec(table_create_sql) != 0) {
        const char* err = db->getErrorMsg();
        LOG_ERROR("Create table {} failed, errs: {}", collection_name_, err);
        return false;
    }
    return true;
}

bool
CollectionData::DropCollection(SQLite::Database* db) {
    // DROP TABLE {collection_name_}
    std::string drop_sql =
        string_util::SFormat("DROP TABLE {}", collection_name_);

    if (db->tryExec(drop_sql) != 0) {
        const char* err = db->getErrorMsg();
        LOG_ERROR("Drop collection {} failed, errs: {}", collection_name_, err);
        return false;
    }
    return true;
}

int
CollectionData::Insert(SQLite::Database* db,
                       const std::string& milvus_id,
                       const std::string& data) {
    std::string insert_sql = string_util::SFormat(
        "INSERT INTO {} VALUES (NULL, ?, ?)", collection_name_);
    try {
        SQLite::Statement query(*db, insert_sql);
        SQLite::bind(query, milvus_id, data);
        return query.exec();
    } catch (std::exception& e) {
        LOG_ERROR("Insert data failed, errs: {}", e.what());
        return -1;
    }
}

void
CollectionData::Load(SQLite::Database* db,
                     int64_t start,
                     int64_t limit,
                     std::vector<std::string>* output_rows) {
    // SELECT {col_data_} from {collection_name_} LIMIT {limit} OFFSET {start}
    std::string select_sql =
        string_util::SFormat("SELECT {} from {} LIMIT {} OFFSET {}",
                             col_data_,
                             collection_name_,
                             limit,
                             start);
    try {
        SQLite::Statement query(*db, select_sql);
        while (query.executeStep()) {
            output_rows->push_back(query.getColumn(0).getString());
        }

    } catch (std::exception& e) {
        LOG_ERROR("Load data failed, errs: {}", e.what());
    }
}

int
CollectionData::Delete(SQLite::Database* db,
                       const std::vector<std::string>& milvus_ids) {
    // DELETE FROM {collection_name_} WHERE {col_milvus_id} in ({})
    std::string delete_sql =
        string_util::SFormat("DELETE FROM {} WHERE {} IN (\"{}\")",
                             collection_name_,
                             col_milvus_id_,
                             string_util::Join(",", milvus_ids));
    try {
        SQLite::Statement query(*db, delete_sql);
        return query.exec();
    } catch (std::exception& e) {
        LOG_ERROR("Delete data failed, errs: {}", e.what());
        return -1;
    }
}

int64_t
CollectionData::Count(SQLite::Database* db) {
    // SELECT count(*) FROM {};
    std::string count_sql =
        string_util::SFormat("SELECT count(*) FROM {}", collection_name_);
    try {
        SQLite::Statement query(*db, count_sql);
        query.executeStep();
        return query.getColumn(0).getInt64();
    } catch (std::exception& e) {
        LOG_ERROR("count data failed, errs: {}", e.what());
        return -1;
    }
}

}  // namespace milvus::local
