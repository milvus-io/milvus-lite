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

#include "milvus_service_impl.h"
#include <grpcpp/support/status.h>
#include <cstdint>
#include "status.h"

namespace milvus::local {

void
Status2Response(Status& s, ::milvus::proto::common::Status* response) {
    response->set_code(s.Code());
    response->set_retriable(false);
    response->set_detail(s.Detail());
    response->set_reason(s.Detail() + ": " + s.Msg());
}

::grpc::Status
MilvusServiceImpl::CreateCollection(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::CreateCollectionRequest* request,
    ::milvus::proto::common::Status* response) {
    Status s = proxy_.CreateCollection(request);
    Status2Response(s, response);
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::LoadCollection(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::LoadCollectionRequest* request,
    ::milvus::proto::common::Status* response) {
    Status s = proxy_.LoadCollection(request->collection_name());
    Status2Response(s, response);
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::HasCollection(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::HasCollectionRequest* request,
    ::milvus::proto::milvus::BoolResponse* response) {
    Status s = proxy_.HasCollection(request->collection_name());
    Status2Response(s, response->mutable_status());
    response->set_value(s.IsOk());
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::CreateIndex(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::CreateIndexRequest* request,
    ::milvus::proto::common::Status* response) {
    Status s = proxy_.CreateIndex(request);
    Status2Response(s, response);
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::Insert(::grpc::ServerContext* context,
                          const ::milvus::proto::milvus::InsertRequest* request,
                          ::milvus::proto::milvus::MutationResult* response) {
    Status s = proxy_.Insert(request, response->mutable_ids());
    Status2Response(s, response->mutable_status());
    auto num_rows = request->num_rows();
    auto succ_size = std::max(response->ids().int_id().data_size(),
                              response->ids().str_id().data_size());
    response->set_insert_cnt(succ_size);
    for (int64_t i = 0; i < succ_size; ++i) {
        response->mutable_succ_index()->Add(i);
    }

    for (int64_t i = succ_size; i < num_rows; ++i) {
        response->mutable_err_index()->Add(i);
    }
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::Upsert(::grpc::ServerContext* context,
                          const ::milvus::proto::milvus::UpsertRequest* request,
                          ::milvus::proto::milvus::MutationResult* response) {
    Status s = proxy_.Upsert(request, response->mutable_ids());
    Status2Response(s, response->mutable_status());
    auto num_rows = request->num_rows();
    auto succ_size = std::max(response->ids().int_id().data_size(),
                              response->ids().str_id().data_size());
    response->set_insert_cnt(succ_size);
    response->set_upsert_cnt(succ_size);
    response->set_delete_cnt(num_rows);
    for (int64_t i = 0; i < succ_size; ++i) {
        response->mutable_succ_index()->Add(i);
    }

    for (int64_t i = succ_size; i < num_rows; ++i) {
        response->mutable_err_index()->Add(i);
    }
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::Search(::grpc::ServerContext* context,
                          const ::milvus::proto::milvus::SearchRequest* request,
                          ::milvus::proto::milvus::SearchResults* response) {
    Status s = proxy_.Search(request, response);
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::Query(::grpc::ServerContext* context,
                         const ::milvus::proto::milvus::QueryRequest* request,
                         ::milvus::proto::milvus::QueryResults* response) {
    Status s = proxy_.Query(request, response);
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::Delete(::grpc::ServerContext* context,
                          const ::milvus::proto::milvus::DeleteRequest* request,
                          ::milvus::proto::milvus::MutationResult* response) {
    Status s = proxy_.Delete(request, response);
    Status2Response(s, response->mutable_status());
    if (s.IsErr()) {
        response->clear_delete_cnt();
        response->clear_ids();
    }
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::DescribeCollection(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::DescribeCollectionRequest* request,
    ::milvus::proto::milvus::DescribeCollectionResponse* response) {
    Status s = proxy_.DescribeCollection(request, response);
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

/*
 * Useless interface, just to make the process run.
 */
::grpc::Status
MilvusServiceImpl::Connect(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::ConnectRequest* request,
    ::milvus::proto::milvus::ConnectResponse* response) {
    Status s = Status::Ok();
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::DescribeIndex(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::DescribeIndexRequest* request,
    ::milvus::proto::milvus::DescribeIndexResponse* response) {
    auto s = proxy_.GetIndex(
        request->collection_name(), request->index_name(), response);
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

/*
 * Useless interface, just to make the process run.
 */
::grpc::Status
MilvusServiceImpl::AllocTimestamp(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::AllocTimestampRequest* request,
    ::milvus::proto::milvus::AllocTimestampResponse* response) {
    Status s = Status::Ok();
    Status2Response(s, response->mutable_status());
    response->set_timestamp(0);
    return ::grpc::Status::OK;
}

/*
 * Useless interface, just to make the process run.
 */
::grpc::Status
MilvusServiceImpl::GetLoadingProgress(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::GetLoadingProgressRequest* request,
    ::milvus::proto::milvus::GetLoadingProgressResponse* response) {
    response->set_progress(100);
    response->set_refresh_progress(100);
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::DropCollection(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::DropCollectionRequest* request,
    ::milvus::proto::common::Status* response) {
    if (proxy_.DropCollection(request->collection_name())) {
        Status s = Status::Ok();
        Status2Response(s, response);
    } else {
        Status s = Status::ServiceInternal();
        Status2Response(s, response);
    }
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::ReleaseCollection(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::ReleaseCollectionRequest* request,
    ::milvus::proto::common::Status* response) {
    auto s = proxy_.ReleaseCollection(request->collection_name());
    Status2Response(s, response);
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::ShowCollections(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::ShowCollectionsRequest* request,
    ::milvus::proto::milvus::ShowCollectionsResponse* response) {
    std::vector<std::string> collections;
    proxy_.ListCollection(&collections);
    for (const auto& name : collections) {
        response->add_collection_names(name);
    }
    Status s = Status::Ok();
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::DropIndex(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::DropIndexRequest* request,
    ::milvus::proto::common::Status* response) {
    auto s =
        proxy_.DropIndex(request->collection_name(), request->index_name());
    Status2Response(s, response);
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::Flush(::grpc::ServerContext* context,
                         const ::milvus::proto::milvus::FlushRequest* request,
                         ::milvus::proto::milvus::FlushResponse* response) {
    // do nothing
    Status s = Status::Ok();
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::GetFlushState(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::GetFlushStateRequest* request,
    ::milvus::proto::milvus::GetFlushStateResponse* response) {
    Status s = Status::Ok();
    Status2Response(s, response->mutable_status());
    response->set_flushed(true);
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::GetCollectionStatistics(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::GetCollectionStatisticsRequest* request,
    ::milvus::proto::milvus::GetCollectionStatisticsResponse* response) {
    Status s =
        proxy_.GetCollectionStatistics(request->collection_name(), response);
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

::grpc::Status
MilvusServiceImpl::GetLoadState(
    ::grpc::ServerContext* context,
    const ::milvus::proto::milvus::GetLoadStateRequest* request,
    ::milvus::proto::milvus::GetLoadStateResponse* response) {
    auto s = proxy_.GetLoadState(request->collection_name(), response);
    Status2Response(s, response->mutable_status());
    return ::grpc::Status::OK;
}

}  // namespace milvus::local
