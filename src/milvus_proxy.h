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
    HasCollection(const std::string& collection_name);

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
    Search(const ::milvus::proto::milvus::SearchRequest* request,
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

 private:
    MilvusLocal milvus_local_;
};

}  // namespace milvus::local
