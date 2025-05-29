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

#pragma once
#include <SQLiteCpp/Database.h>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include "collection_data.h"
#include "collection_meta.h"

namespace milvus::local {

class CollectionMeta;
class CollectionData;

class Storage final {
 public:
    explicit Storage(const char* db_file);
    ~Storage();
    Storage(const Storage&) = delete;
    Storage&
    operator=(const Storage&) = delete;
    Storage(const Storage&&) = delete;
    Storage&
    operator=(const Storage&&) = delete;

 public:
    bool
    Open();

    bool
    CreateCollection(const std::string& collection_name,
                     const std::string& pk_name,
                     const std::string& schema_proto);
    bool
    DropCollection(const std::string& collection_name);

    void
    ListCollections(std::vector<std::string>* collection_names) {
        cm_.CollectionNames(collection_names);
    }

    /*
     * @brief 读取collection数据
     *
     * @collection collection 名字
     * @size 一次读取的数量
     * @out_rows 输出, 如果out_rows的size小于参数的size，说明已经读取完
     */
    bool
    LoadCollecton(const std::string& collection_name,
                  int64_t start,
                  int64_t size,
                  std::vector<std::string>* out_rows);

    bool
    CreateIndex(const std::string& collection_name,
                const std::string& index_name,
                const std::string& index_proto);

    bool
    GetIndex(const std::string& collection_name,
             const std::string& index_name,
             std::string* output_index_proto);

    bool
    DropIndex(const std::string& collection_name,
              const std::string& index_name);

    void
    GetAllIndex(const std::string& collection_name,
                const std::string& exclude,
                std::vector<std::string>* index_protos);

    bool
    HasIndex(const std::string& collection_name,
             const std::string& index_name) {
        return cm_.HasIndex(collection_name, index_name);
    }

    std::string
    GetPrimaryKey(const std::string& collection_name) {
        return cm_.GetPkName(collection_name);
    }

    bool
    GetCollectionSchema(const std::string& collection_name,
                        std::string* output_info_str);

    // data interface
    int
    Insert(const std::string collection_name, const std::vector<Row>& rows);

    int
    Delete(const std::string collection_name,
           const std::vector<std::string>& ids);

    bool
    CollectionExist(const std::string& collection_name) {
        return collections_.find(collection_name) != collections_.end();
    }

    int64_t
    Count(const std::string& collection_name);

 private:
    CollectionMeta cm_;
    std::map<const std::string, std::unique_ptr<CollectionData>> collections_;

 private:
    std::unique_ptr<SQLite::Database> db_ptr_;
    const char* db_file_;
};

}  // namespace milvus::local
