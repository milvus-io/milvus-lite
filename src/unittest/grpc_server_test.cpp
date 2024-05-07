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
#include "pb/milvus.pb.h"
#include "test_util.h"
#include <grpcpp/server_context.h>
#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
#include "status.h"

namespace milvus::local {
namespace test {

const char* tmp_db_name = "server_test.db";

TEST(MilvusServiceImplTest, create_collection) {
    const char* collection_name = "test_collection";
    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status response;
        // drop
        auto drop_r = GetDropCollectionRequest(collection_name);
        service.DropCollection(&server_context, &drop_r, &response);

        auto r = GetCreateCollectionRequestProto(collection_name);
        service.CreateCollection(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);
        service.CreateCollection(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);
    }

    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status response;

        // collection alread exists
        auto r = GetCreateCollectionRequestProto(collection_name);
        service.CreateCollection(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);

        auto lr = GetLoadCollectionRequestProto(collection_name);
        service.LoadCollection(&server_context, &lr, &response);
        EXPECT_EQ(response.code(), 0);

        auto new_lr = GetLoadCollectionRequestProto("not_exist");
        service.LoadCollection(&server_context, &new_lr, &response);
        EXPECT_EQ(response.code(), ErrCollectionNotFound);
    }
}

TEST(MilvusServiceImplTest, CreateIndex) {
    const char* collection_name = "test_collection";
    ::milvus::local::MilvusServiceImpl service(tmp_db_name);
    EXPECT_TRUE(service.Init());
    ::grpc::ServerContext server_context;
    ::milvus::proto::common::Status response;
    auto drop_r = GetDropCollectionRequest(collection_name);
    service.DropCollection(&server_context, &drop_r, &response);
    auto r = GetCreateCollectionRequestProto(collection_name);
    service.CreateCollection(&server_context, &r, &response);
    EXPECT_EQ(response.code(), 0);

    {
        auto r =
            GetCreateIndexRequestProto(collection_name, "test_index", VEC_NAME);
        service.CreateIndex(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);

        service.CreateIndex(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);
    }

    {
        auto r =
            GetCreateIndexRequestProto("not_exist", "test_index", VEC_NAME);
        service.CreateIndex(&server_context, &r, &response);
        EXPECT_EQ(response.code(), ErrCollectionNotFound);
    }
}

TEST(MilvusServiceImplTest, Insert) {
    const char* collection_name = "test_collection";
    ::milvus::local::MilvusServiceImpl service(tmp_db_name);
    EXPECT_TRUE(service.Init());
    ::grpc::ServerContext server_context;
    ::milvus::proto::common::Status response;
    auto drop_r = GetDropCollectionRequest(collection_name);
    service.DropCollection(&server_context, &drop_r, &response);
    auto r = GetCreateCollectionRequestProto(collection_name);
    service.CreateCollection(&server_context, &r, &response);
    EXPECT_EQ(response.code(), 0);

    {
        auto insert_requst = GetInsertRequestProto(collection_name, 3);
        ::milvus::proto::milvus::MutationResult insert_response;
        service.Insert(&server_context, &insert_requst, &insert_response);
        EXPECT_EQ(insert_response.insert_cnt(), 3);
        EXPECT_EQ(insert_response.status().code(), 0);
    }
}

TEST(MilvusServiceImplTest, Search) {
    const char* collection_name = "test_collection";
    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status response;
        auto drop_r = GetDropCollectionRequest(collection_name);
        service.DropCollection(&server_context, &drop_r, &response);
        auto r = GetCreateCollectionRequestProto(collection_name);
        service.CreateCollection(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);
        auto insert_requst = GetInsertRequestProto(collection_name, 3);
        ::milvus::proto::milvus::MutationResult insert_response;
        service.Insert(&server_context, &insert_requst, &insert_response);
        EXPECT_EQ(insert_response.insert_cnt(), 3);
        EXPECT_EQ(insert_response.status().code(), 0);

        auto index_req =
            GetCreateIndexRequestProto(collection_name, "test_index", VEC_NAME);
        service.CreateIndex(&server_context, &index_req, &response);
        EXPECT_EQ(response.code(), 0);

        auto search_req = GetSearchRequestProto(
            collection_name,
            "id in [1, 2, 3]",
            std::vector<std::vector<float>>{{0.1, 0.3, 0.6}, {0.3, 0.3, 0.4}},
            "2",
            "COSINE",
            "1");
        ::milvus::proto::milvus::SearchResults search_result;
        service.Search(&server_context, &search_req, &search_result);
    }
    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status response;
        auto lr = GetLoadCollectionRequestProto(collection_name);
        service.LoadCollection(&server_context, &lr, &response);
        EXPECT_EQ(response.code(), 0);
        auto search_req = GetSearchRequestProto(
            collection_name,
            "id in [1, 2, 3]",
            std::vector<std::vector<float>>{{0.1, 0.3, 0.6}, {0.3, 0.3, 0.4}},
            "2",
            "COSINE",
            "1");
        ::milvus::proto::milvus::SearchResults search_result;
        service.Search(&server_context, &search_req, &search_result);
    }
}

TEST(MilvusServiceImplTest, Query) {
    const char* collection_name = "test_collection";
    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status response;
        auto drop_r = GetDropCollectionRequest(collection_name);
        service.DropCollection(&server_context, &drop_r, &response);
        auto r = GetCreateCollectionRequestProto(collection_name);
        service.CreateCollection(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);
        auto insert_requst = GetInsertRequestProto(collection_name, 3);
        ::milvus::proto::milvus::MutationResult insert_response;
        service.Insert(&server_context, &insert_requst, &insert_response);
        EXPECT_EQ(insert_response.insert_cnt(), 3);
        EXPECT_EQ(insert_response.status().code(), 0);

        auto query_req = GetQueryRequestProto(collection_name,
                                              "id in [1, 2, 3]",
                                              "2",
                                              "0",
                                              std::vector<std::string>{"id"});
        ::milvus::proto::milvus::QueryResults query_result;
        service.Query(&server_context, &query_req, &query_result);
        EXPECT_EQ(
            query_result.fields_data()[0].scalars().long_data().data_size(), 2);
    }
}

TEST(MilvusServiceImplTest, Delete) {
    const char* collection_name = "test_collection";
    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status response;
        auto drop_r = GetDropCollectionRequest(collection_name);
        service.DropCollection(&server_context, &drop_r, &response);
        auto r = GetCreateCollectionRequestProto(collection_name);
        service.CreateCollection(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);
        auto insert_requst = GetInsertRequestProto(collection_name, 10);
        ::milvus::proto::milvus::MutationResult insert_response;
        service.Insert(&server_context, &insert_requst, &insert_response);
        EXPECT_EQ(insert_response.insert_cnt(), 10);
        EXPECT_EQ(insert_response.status().code(), 0);

        auto delete_req =
            GetDeleteRequestProto(collection_name, "id in [1, 2]");
        ::milvus::proto::milvus::MutationResult delete_result;
        service.Delete(&server_context, &delete_req, &delete_result);
        EXPECT_EQ(delete_result.delete_cnt(), 2);

        auto query_req = GetQueryRequestProto(collection_name,
                                              "id in [1, 2, 3]",
                                              "3",
                                              "0",
                                              std::vector<std::string>{"id"});
        ::milvus::proto::milvus::QueryResults query_result;
        service.Query(&server_context, &query_req, &query_result);
        EXPECT_EQ(
            query_result.fields_data()[0].scalars().long_data().data_size(), 1);
    }
    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status response;
        auto lr = GetLoadCollectionRequestProto(collection_name);
        service.LoadCollection(&server_context, &lr, &response);
        EXPECT_EQ(response.code(), 0);
        auto query_req = GetQueryRequestProto(collection_name,
                                              "id in [1, 2, 3]",
                                              "3",
                                              "0",
                                              std::vector<std::string>{"id"});
        ::milvus::proto::milvus::QueryResults query_result;
        service.Query(&server_context, &query_req, &query_result);
        EXPECT_EQ(
            query_result.fields_data()[0].scalars().long_data().data_size(), 1);
    }
}

TEST(MilvusServiceImplTest, describe_collection) {
    const char* collection_name = "test_collection";

    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status drop_res;
        auto drop_r = GetDropCollectionRequest(collection_name);
        service.DropCollection(&server_context, &drop_r, &drop_res);
        ::milvus::proto::milvus::DescribeCollectionResponse res;
        auto r = GetDescribeCollectionRequest(collection_name);
        service.DescribeCollection(&server_context, &r, &res);
        EXPECT_EQ(res.status().code(), ErrCollectionNotFound);
    }

    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::common::Status response;
        auto r = GetCreateCollectionRequestProto(collection_name);
        service.CreateCollection(&server_context, &r, &response);
        EXPECT_EQ(response.code(), 0);

        ::milvus::proto::milvus::DescribeCollectionResponse res;
        auto dr = GetDescribeCollectionRequest(collection_name);
        service.DescribeCollection(&server_context, &dr, &res);
        EXPECT_EQ(res.status().code(), 0);
    }

    {
        ::milvus::local::MilvusServiceImpl service(tmp_db_name);
        EXPECT_TRUE(service.Init());
        ::grpc::ServerContext server_context;
        ::milvus::proto::milvus::DescribeCollectionResponse res;
        auto dr = GetDescribeCollectionRequest(collection_name);
        service.DescribeCollection(&server_context, &dr, &res);
        EXPECT_EQ(res.status().code(), 0);
    }
}

}  // namespace test
}  // namespace milvus::local
