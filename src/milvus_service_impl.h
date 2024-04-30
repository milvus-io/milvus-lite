#pragma once

#include <grpcpp/support/status.h>
#include "pb/milvus.grpc.pb.h"
#include "milvus_proxy.h"

namespace milvus::local {
class MilvusServiceImpl final
    : public milvus::proto::milvus::MilvusService::Service {
 public:
    MilvusServiceImpl(const std::string& work_dir) : proxy_(work_dir.c_str()) {
    }
    virtual ~MilvusServiceImpl() = default;

 public:
    bool
    Init() {
        return proxy_.Init();
    }

 public:
    ::grpc::Status
    CreateCollection(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::CreateCollectionRequest* request,
        ::milvus::proto::common::Status* response) override;

    ::grpc::Status
    CreateIndex(::grpc::ServerContext* context,
                const ::milvus::proto::milvus::CreateIndexRequest* request,
                ::milvus::proto::common::Status* response) override;

    ::grpc::Status
    LoadCollection(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::LoadCollectionRequest* request,
        ::milvus::proto::common::Status* response) override;

    ::grpc::Status
    Insert(::grpc::ServerContext* context,
           const ::milvus::proto::milvus::InsertRequest* request,
           ::milvus::proto::milvus::MutationResult* response) override;

    ::grpc::Status
    Search(::grpc::ServerContext* context,
           const ::milvus::proto::milvus::SearchRequest* request,
           ::milvus::proto::milvus::SearchResults* response) override;

    ::grpc::Status
    Query(::grpc::ServerContext* context,
          const ::milvus::proto::milvus::QueryRequest* request,
          ::milvus::proto::milvus::QueryResults* response) override;

    ::grpc::Status
    Delete(::grpc::ServerContext* context,
           const ::milvus::proto::milvus::DeleteRequest* request,
           ::milvus::proto::milvus::MutationResult* response) override;

    ::grpc::Status
    DescribeCollection(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::DescribeCollectionRequest* request,
        ::milvus::proto::milvus::DescribeCollectionResponse* response) override;

    ::grpc::Status
    Connect(::grpc::ServerContext* context,
            const ::milvus::proto::milvus::ConnectRequest* request,
            ::milvus::proto::milvus::ConnectResponse* response) override;

    ::grpc::Status
    DescribeIndex(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::DescribeIndexRequest* request,
        ::milvus::proto::milvus::DescribeIndexResponse* response) override;

    ::grpc::Status
    AllocTimestamp(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::AllocTimestampRequest* request,
        ::milvus::proto::milvus::AllocTimestampResponse* response) override;

    ::grpc::Status
    GetLoadingProgress(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::GetLoadingProgressRequest* request,
        ::milvus::proto::milvus::GetLoadingProgressResponse* response) override;

    ::grpc::Status
    DropCollection(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::DropCollectionRequest* request,
        ::milvus::proto::common::Status* response) override;

    ::grpc::Status
    ReleaseCollection(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::ReleaseCollectionRequest* request,
        ::milvus::proto::common::Status* response) override;

    ::grpc::Status
    ShowCollections(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::ShowCollectionsRequest* request,
        ::milvus::proto::milvus::ShowCollectionsResponse* response) override;

    ::grpc::Status
    HasCollection(::grpc::ServerContext* context,
                  const ::milvus::proto::milvus::HasCollectionRequest* request,
                  ::milvus::proto::milvus::BoolResponse* response) override;

    ::grpc::Status
    DropIndex(::grpc::ServerContext* context,
              const ::milvus::proto::milvus::DropIndexRequest* request,
              ::milvus::proto::common::Status* response) override;

    ::grpc::Status
    Flush(::grpc::ServerContext* context,
          const ::milvus::proto::milvus::FlushRequest* request,
          ::milvus::proto::milvus::FlushResponse* response) override;

    ::grpc::Status
    GetFlushState(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::GetFlushStateRequest* request,
        ::milvus::proto::milvus::GetFlushStateResponse* response) override;

    // for now only row count is returned
    ::grpc::Status
    GetCollectionStatistics(
        ::grpc::ServerContext* context,
        const ::milvus::proto::milvus::GetCollectionStatisticsRequest* request,
        ::milvus::proto::milvus::GetCollectionStatisticsResponse* response)
        override;

 private:
    ::milvus::local::MilvusProxy proxy_;
};

}  // namespace milvus::local
