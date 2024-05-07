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

/*                     collection data table

           ──────┬─────────────┬────────────
             id  │  milvus_id  │   data
                 │             │
                 │             │
           ──────┼─────────────┼────────────
             1   │   1234      │   xxxx
                 │             │
           ──────┼─────────────┼────────────
                 │             │
             2   │   1235      │   xxxx
                 │             │
           ──────┼─────────────┼────────────
                 │             │
             3   │   1236      │   xxxx
                 │             │
          ───────┼─────────────┼────────────
                 │             │
             4   │   1237      │   xxxx
                 │             │
          ───────┼─────────────┼────────────
                 │             │
             5   │   1238      │   xxxx
                 │             │
                 │             │
           ──────┴─────────────┴────────────


*/

#pragma once
#include <SQLiteCpp/Database.h>
#include <cstdint>
#include <string>
#include <vector>
#include "type.h"
#include "common.h"

namespace milvus::local {

class CollectionData final : NonCopyableNonMovable {
 public:
    explicit CollectionData(const char*);
    virtual ~CollectionData();

 public:
    int
    Insert(SQLite::Database* db,
           const std::string& milvus_id,
           const std::string& data);

    int
    Delete(SQLite::Database* db, const std::vector<std::string>& milvus_ids);
    // int
    // upsert(SQLite::Database* db, std::string& milvus_id, const std::string& data);
    // const char*
    // get(SQLite::Database* db, const std::string& milvus_id);

    void
    Load(SQLite::Database* db,
         int64_t start,
         int64_t limit,
         std::vector<std::string>* output_rows);

    bool
    CreateCollection(SQLite::Database* db);

    bool
    DropCollection(SQLite::Database* db);

    const std::string&
    GetTableName() {
        return collection_name_;
    }

    int64_t
    Count(SQLite::Database* db);

 private:
    std::string
    GetTableCreateSql();

 private:
    const std::string collection_name_;

    // table column name
    const std::string col_id_;
    const std::string col_milvus_id_;
    const std::string col_data_;
};
}  // namespace milvus::local
