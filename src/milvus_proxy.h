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

#include <vector>
#include "common.h"
#include "milvus_local.h"
#include "pb/schema.pb.h"
#include "pb/milvus.pb.h"
#include "status.h"

namespace milvus::local {

class MilvusProxy : NonCopyableNonMovable {
 public:
    explicit MilvusProxy(const char* work_dir);
    virtual ~MilvusProxy();

 public:
    bool
    Init();

    Status
    LoadCollection(const std::string& collection_name);

    Status
    ReleaseCollection(const std::string& collection_name);

    Status
    CreateCollection(
        const ::milvus::proto::milvus::CreateCollectionRequest* request);

    Status
    HasCollection(const std::string& collection_name,
                  ::milvus::proto::milvus::BoolResponse* response);

    Status
    GetLoadState(const std::string& collection_name,
                 ::milvus::proto::milvus::GetLoadStateResponse* response);

    bool
    DropCollection(const std::string& collection_name);

    void
    ListCollection(std::vector<std::string>* collections) {
        milvus_local_.GetAllCollections(collections);
    }

    Status
    CreateIndex(const ::milvus::proto::milvus::CreateIndexRequest* request);

    Status
    GetIndex(const std::string& collection_name,
             const std::string& index_name,
             ::milvus::proto::milvus::DescribeIndexResponse* response);

    Status
    DropIndex(const std::string& collection_name,
              const std::string& index_name);

    Status
    Insert(const ::milvus::proto::milvus::InsertRequest* request,
           ::milvus::proto::schema::IDs* ids);

    Status
    Upsert(const ::milvus::proto::milvus::UpsertRequest* request,
           ::milvus::proto::schema::IDs* ids);

    Status
    Search(const ::milvus::proto::milvus::SearchRequest* request,
           ::milvus::proto::milvus::SearchResults* search_result);

    Status
    HybridSearch(const ::milvus::proto::milvus::HybridSearchRequest* request,
                 ::milvus::proto::milvus::SearchResults* search_result);

    Status
    Query(const ::milvus::proto::milvus::QueryRequest* request,
          ::milvus::proto::milvus::QueryResults* response);

    Status
    Delete(const ::milvus::proto::milvus::DeleteRequest* request,
           ::milvus::proto::milvus::MutationResult* response);

    Status
    DescribeCollection(
        const ::milvus::proto::milvus::DescribeCollectionRequest* request,
        ::milvus::proto::milvus::DescribeCollectionResponse* response);

    Status
    GetCollectionStatistics(
        const std::string& collection_name,
        ::milvus::proto::milvus::GetCollectionStatisticsResponse* r);

 private:
    Status
    GetSchemaInfo(const std::string& collection_name,
                  ::milvus::proto::schema::CollectionSchema* schema);

    Status
    ParseIndex(const std::string& index_proto,
               const ::milvus::proto::schema::CollectionSchema& schema,
               ::milvus::proto::milvus::IndexDescription* index);

    Status
    DoSearch(const ::milvus::proto::milvus::SearchRequest* r,
             const ::milvus::proto::schema::CollectionSchema& schema,
             const std::vector<std::string>& all_index,
             ::milvus::proto::milvus::SearchResults* search_result);

 private:
    MilvusLocal milvus_local_;
};

}  // namespace milvus::local
