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

/*                                  meta table

      ──────┬───────────────────┬─────────────┬─────────────┬────────────────
        id  │  collection_name  │  meta_type  │  blob_field │ string_field
            │                   │             │             │
            │                   │             │             │
      ──────┼───────────────────┼─────────────┼─────────────┼────────────────
        1   │    collection1    │  schema     │   xxx       │   pk_name
            │                   │             │             │
      ──────┼───────────────────┼─────────────┼─────────────┼────────────────
            │                   │             │             │
        2   │    collection1    │  index      │   xxx       │   index1
            │                   │             │             │
      ──────┼───────────────────┼─────────────┼─────────────┼────────────────
            │                   │             │             │
        3   │    collection1    │  index      │  xxx        │   index2
            │                   │             │             │
      ──────┼───────────────────┼─────────────┼─────────────┼────────────────
            │                   │             │             │
        4   │    collection1    │  partition  │   null      │    p1
            │                   │             │             │
      ──────┼───────────────────┼─────────────┼─────────────┼────────────────
            │                   │             │             │
        5   │    collction2     │  schema     │   xxx       │    pk_name
            │                   │             │             │
      ──────┴───────────────────┴─────────────┴─────────────┴────────────────
*/

#pragma once
#include <exception>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "SQLiteCpp/Database.h"
#include "log/Log.h"

namespace milvus::local {

using SchemaInfo = std::string;
using IndexInfo = std::map<const std::string, const std::string>;

/*
 * CollectionMeta 存储所有collection的元信息，数据写入sqlite3中，内存中保存副本。
 * Collection 是否存在等检查，都在storage中，CollectionMeta不再进行相关检查。
 */

class CollectionMeta final {
 public:
    class CollectionInfo {
     public:
        CollectionInfo() = default;
        ~CollectionInfo() = default;
        CollectionInfo(const CollectionInfo&) = delete;
        CollectionInfo&
        operator=(const CollectionInfo&) = delete;
        CollectionInfo(const CollectionInfo&&) = delete;
        CollectionInfo&
        operator=(const CollectionInfo&&) = delete;

     public:
        void
        AddSchema(const char* info, const std::string& pk_name) {
            schema_info_.assign(info);
            pk_name_ = pk_name;
        }

        const std::string&
        Schema() {
            return schema_info_;
        }
        bool
        AddIndex(const std::string& index_name, const char* index) {
            if (index_info_.find(index_name) != index_info_.end()) {
                LOG_ERROR("Index: {} already exist", index_name);
                return false;
            }
            index_info_.emplace(index_name, index);
            return true;
        }

        bool
        GetIndex(const std::string& index_name, std::string* index) {
            if (index_info_.find(index_name) == index_info_.end()) {
                return false;
            }
            index->assign(index_info_[index_name].c_str());
            return true;
        }

        void
        GetAllIndexs(std::vector<std::string>* all_index,
                     const std::string& exclude) {
            for (const auto& pair : index_info_) {
                if (pair.first != exclude)
                    all_index->push_back(pair.second);
            }
        }

        bool
        HasIndex(const std::string& index_name) {
            return index_info_.find(index_name) != index_info_.end();
        }

        bool
        DropIndex(const std::string& index_name) {
            if (!HasIndex(index_name)) {
                return true;
            }
            index_info_.erase(index_name);
            return true;
        }

        const std::string&
        GetPkName() {
            return pk_name_;
        }

     private:
        IndexInfo index_info_;
        SchemaInfo schema_info_;
        std::string pk_name_;
    };

 public:
    CollectionMeta();
    ~CollectionMeta();

 public:
    CollectionMeta(const CollectionMeta&) = delete;
    CollectionMeta&
    operator=(const CollectionMeta&) = delete;
    CollectionMeta(const CollectionMeta&&) = delete;
    CollectionMeta&
    operator=(const CollectionMeta&&) = delete;

 public:
    bool
    Init(SQLite::Database* db);

    bool
    CreateCollection(SQLite::Database* db,
                     const std::string& collection_name,
                     const std::string& pk_name,
                     const std::string& schema_proto);

    const std::string&
    GetCollectionSchema(const std::string& collection_name);

    bool
    CreateIndex(SQLite::Database* db,
                const std::string& collection_name,
                const std::string& index_name,
                const std::string& index_proto);

    void
    GetAllIndex(const std::string& collection_name,
                const std::string& exclude,
                std::vector<std::string>* all_index);

    bool
    HasIndex(const std::string& collection_name,
             const std::string& index_name) {
        return collections_[collection_name]->HasIndex(index_name);
    }

    bool
    DropIndex(SQLite::Database* db,
              const std::string& collection_name,
              const std::string& index_name);

    bool
    GetCollectionIndex(const std::string& collection_name,
                       const std::string& index_name,
                       std::string* ouput_index_info);

    std::string
    GetPkName(const std::string& collection_name) {
        return collections_[collection_name]->GetPkName();
    }

    void
    CollectionNames(std::vector<std::string>* collection_names) {
        for (const auto& pair : collections_) {
            collection_names->push_back(pair.first);
        }
    }

    bool
    DropCollection(SQLite::Database* db, const std::string& collection_name);

 private:
    bool
    CreateTable(SQLite::Database* db);

    bool
    LoadMeta(SQLite::Database* db);

    // std::string
    // schema_info(const std::string& collection_name);

 private:
    // collection meta
    std::map<std::string, std::unique_ptr<CollectionInfo>> collections_;

 private:
    // sqlite3 table info
    const std::string table_meta_name_;

    // table column name
    const std::string col_id_;
    const std::string col_collection_name_;
    const std::string col_meta_type_;
    const std::string col_blob_field_;
    const std::string col_string_field_;
};

}  // namespace milvus::local
