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

#include "bm25_stats.h"
#include "log/Log.h"
#include "string_util.hpp"

namespace milvus::local {

bool
BM25Stats::AddTokenDoc(SQLite::Database* db,
                       const std::string& collection_name,
                       const std::string& output_field_name,
                       const std::map<uint32_t, int32_t>& token_infos) {
    SQLite::Statement query(
        *db,
        "INSERT INTO token_doc_count (collection_name, "
        "output_field_name, token, doc_count) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(collection_name, output_field_name, token) "
        "DO UPDATE SET doc_count = doc_count + excluded.doc_count");
    try {
        for (const auto& [token, count] : token_infos) {
            query.bind(1, collection_name);
            query.bind(2, output_field_name);
            query.bind(3, token);
            query.bind(4, count);
            query.exec();
            query.reset();
        }
    } catch (std::exception& e) {
        LOG_ERROR("Update bm25 stats failed, err: {}", e.what());
        return false;
    }
    return true;
}

bool
BM25Stats::DeleteTokenDoc(SQLite::Database* db,
                          const std::string& collection_name,
                          const std::string& output_field_name,
                          const std::map<uint32_t, int32_t>& token_infos) {
    SQLite::Statement query(
        *db,
        "UPDATE token_doc_count "
        "SET doc_count = doc_count - ? "
        "WHERE collection_name = ? AND output_field_name = ? AND token = ?");

    try {
        for (const auto& [token, count] : token_infos) {
            query.bind(1, count);
            query.bind(2, collection_name);
            query.bind(3, output_field_name);
            query.bind(4, token);
            query.exec();
            query.reset();
        }
    } catch (std::exception& e) {
        LOG_ERROR("Update bm25 stats failed, err: {}", e.what());
        return false;
    }
    return true;
}

int32_t
BM25Stats::GetTokenDocCount(SQLite::Database* db,
                            const std::string& collection_name,
                            const std::string& output_field_name,
                            uint32_t token) {
    try {
        SQLite::Statement query(
            *db,
            "SELECT doc_count FROM token_doc_count WHERE collection_name = ? "
            "AND output_field_name = ? AND token = ?");

        query.bind(1, collection_name);
        query.bind(2, output_field_name);
        query.bind(3, token);

        if (query.executeStep()) {
            return query.getColumn(0).getInt();
        }
        return 0;
    } catch (const SQLite::Exception& e) {
        LOG_ERROR("Failed to get token doc count, err: {} ", e.what());
        return -1;
    }
    return -1;
}

std::pair<uint32_t, int32_t>
BM25Stats::GetTokenNumAndDocNum(SQLite::Database* db,
                                const std::string& collection_name,
                                const std::string& output_field_name) {
    try {
        SQLite::Statement query(
            *db,
            "SELECT token_num, doc_num FROM bm25_stats "
            "WHERE collection_name = ? AND output_field_name = ?");

        query.bind(1, collection_name);
        query.bind(2, output_field_name);

        if (query.executeStep()) {
            return std::make_pair(query.getColumn(0).getInt(),
                                  query.getColumn(1).getInt());
        }
        return std::make_pair(0, -1);
    } catch (const SQLite::Exception& e) {
        LOG_ERROR("Failed to get token num and doc num, err: {} ", e.what());
        return std::make_pair(0, -1);
    }
}

bool
BM25Stats::DeleteBM25Stats(SQLite::Database* db,
                           const std::string& collection_name,
                           const std::string& output_field_name,
                           uint32_t token_num,
                           int32_t doc_num) {
    try {
        SQLite::Statement query(
            *db,
            "UPDATE bm25_stats "
            "SET token_num = token_num - ?, "
            "doc_num = doc_num - ?"
            "WHERE collection_name = ? AND output_field_name = ?");

        query.bind(1, token_num);
        query.bind(2, doc_num);
        query.bind(3, collection_name);
        query.bind(4, output_field_name);
        query.exec();
        return true;
    } catch (const SQLite::Exception& e) {
        LOG_ERROR("Failed to updata bm25 stats: {}", e.what());
        return false;
    }
}

bool
BM25Stats::AddBM25Stats(SQLite::Database* db,
                        const std::string& collection_name,
                        const std::string& output_field_name,
                        uint32_t token_num,
                        int32_t doc_num) {
    try {
        SQLite::Statement query(
            *db,
            "INSERT INTO bm25_stats (collection_name, "
            "output_field_name, token_num, doc_num) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(collection_name, output_field_name) "
            "DO UPDATE SET "
            "token_num = token_num + excluded.token_num, "
            "doc_num = doc_num + excluded.doc_num");

        query.bind(1, collection_name);
        query.bind(2, output_field_name);
        query.bind(3, static_cast<int64_t>(token_num));
        query.bind(4, doc_num);

        query.exec();
        return true;
    } catch (const SQLite::Exception& e) {
        LOG_ERROR("Failed to updata bm25 stats: {}", e.what());
        return false;
    }
}

bool
BM25Stats::CreateTables(SQLite::Database* db) {
    auto table_create_cmd = R"(
CREATE TABLE IF NOT EXISTS bm25_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_name VARCHAR(1000),
    output_field_name VARCHAR(1000),
    token_num INTEGER,
    doc_num INTEGER,
    UNIQUE (collection_name, output_field_name)
);

CREATE TABLE IF NOT EXISTS token_doc_count (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_name VARCHAR,
    output_field_name VARCHAR,
    token INTEGER,
    doc_count INTEGER,
    UNIQUE (collection_name, output_field_name, token)
);
)";
    if (db->tryExec(table_create_cmd) != 0) {
        const char* err = db->getErrorMsg();
        LOG_ERROR("Create table failed, errs: {}", err);
        return false;
    }
    return true;
}

bool
BM25Stats::DropCollectionStats(SQLite::Database* db,
                               const std::string& collection_name) {
    try {
        {
            SQLite::Statement query(*db,
                                    "DELETE FROM bm25_stats "
                                    "WHERE collection_name=?");
            query.bind(1, collection_name);
            query.exec();
        }
        {
            SQLite::Statement query(*db,
                                    "DELETE FROM token_doc_count "
                                    "WHERE collection_name=?");
            query.bind(1, collection_name);
            query.exec();
        }
        return true;
    } catch (const SQLite::Exception& e) {
        LOG_ERROR("Clear collection {}'s bm25 stats failed, {}",
                  collection_name,
                  e.what());
        return false;
    }
}

}  // namespace milvus::local
