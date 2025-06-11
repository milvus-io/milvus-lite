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

/*
 *  * Table: bm25_stats
 * +---------+----------------+------------------+-----------+---------+
 * |    id   | collection_name| output_field_name| token_num | rows_num|
 * +---------+----------------+------------------+-----------+---------+
 * | INTEGER | VARCHAR (PK)   | VARCHAR (PK)     | INTEGER   | INTEGER |
 * +----------------+------------------+-----------+---------+---------+

  *  * Table: token_doc_count
 * +---------+----------------+------------------+-----------+---------+
 * |    id   | collection_name| output_field_name| token     |doc_count|
 * +---------+----------------+------------------+-----------+---------+
 * | INTEGER | VARCHAR (PK)   | VARCHAR (PK)     | INTEGER   | INTEGER |
 * +----------------+------------------+-----------+---------+---------+
 */

#pragma once

#include <vector>

#include "common.h"
#include "SQLiteCpp/Database.h"

namespace milvus::local {
class BM25Stats {
 public:
    bool
    AddTokenDoc(SQLite::Database* db,
                const std::string& collection_name,
                const std::string& output_field_name,
                const std::map<uint32_t, int32_t>& token_infos);

    bool
    DeleteTokenDoc(SQLite::Database* db,
                   const std::string& collection_name,
                   const std::string& output_field_name,
                   const std::map<uint32_t, int32_t>& token_infos);

    int32_t
    GetTokenDocCount(SQLite::Database* db,
                     const std::string& collection_name,
                     const std::string& output_field_name,
                     uint32_t token);

    std::pair<uint32_t, int32_t>
    GetTokenNumAndDocNum(SQLite::Database* db,
                         const std::string& collection_name,
                         const std::string& output_field_name);
    bool
    AddBM25Stats(SQLite::Database* db,
                 const std::string& collection_name,
                 const std::string& output_field_name,
                 uint32_t token_num,
                 int32_t doc_num);

    bool
    DeleteBM25Stats(SQLite::Database* db,
                    const std::string& collection_name,
                    const std::string& output_field_name,
                    uint32_t token_num,
                    int32_t doc_num);

    bool
    DropCollectionStats(SQLite::Database* db,
                        const std::string& collection_name);

    bool
    CreateTables(SQLite::Database* db);
};

}  // namespace milvus::local
