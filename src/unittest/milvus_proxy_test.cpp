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

#include "milvus_proxy.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "log/Log.h"
#include "pb/milvus.pb.h"
#include "pb/schema.pb.h"
#include "test_util.h"
#include <glog/logging.h>

namespace milvus::local {
namespace test {

const char* tmp_db_name = "test.db";

TEST(MilvusProxyTest, CreateCollection) {
    const char* collection_name = "test_collection";
    {
        // create new collection
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        EXPECT_FALSE(proxy.Init());
        proxy.DropCollection(collection_name);
        auto cr = GetCreateCollectionRequestProto(collection_name);
        EXPECT_TRUE(proxy.CreateCollection(&cr).IsOk());
    }

    {
        // load collection
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        EXPECT_TRUE(proxy.LoadCollection(collection_name).IsOk());
        // reload is ok
        EXPECT_TRUE(proxy.LoadCollection(collection_name).IsOk());

        EXPECT_TRUE(proxy.ReleaseCollection(collection_name).IsOk());
        EXPECT_TRUE(proxy.ReleaseCollection(collection_name).IsOk());

        EXPECT_FALSE(proxy.LoadCollection("not_existed").IsOk());
    }
}

TEST(MilvusProxyTest, CreateIndex) {
    const char* collection_name = "test_collection";
    const char* index_name = "test_index";

    ::milvus::local::MilvusProxy proxy(tmp_db_name);
    EXPECT_TRUE(proxy.Init());
    proxy.DropCollection(collection_name);
    auto cr = GetCreateCollectionRequestProto(collection_name);
    EXPECT_TRUE(proxy.CreateCollection(&cr).IsOk());

    {
        // create new index
        auto index_req =
            GetCreateIndexRequestProto(collection_name, index_name, VEC_NAME);
        EXPECT_TRUE(proxy.CreateIndex(&index_req).IsOk());

        EXPECT_TRUE(proxy.CreateIndex(&index_req).IsOk());
    }
}

TEST(MilvusProxyTest, Insert) {
    const char* collection_name = "test_collection";
    ::milvus::local::MilvusProxy proxy(tmp_db_name);
    EXPECT_TRUE(proxy.Init());
    proxy.DropCollection(collection_name);
    auto cr = GetCreateCollectionRequestProto(collection_name);
    EXPECT_TRUE(proxy.CreateCollection(&cr).IsOk());

    {
        auto data = GetInsertRequestProto(collection_name, 3);
        ::milvus::proto::schema::IDs ids;
        proxy.Insert(&data, &ids);
        EXPECT_EQ(3, ids.int_id().data_size());
    }
}

TEST(MilvusProxyTest, search) {
    const char* collection_name = "test_collection";
    {
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        proxy.DropCollection(collection_name);
        auto cr = GetCreateCollectionRequestProto(collection_name);
        EXPECT_TRUE(proxy.CreateCollection(&cr).IsOk());
        auto data = GetInsertRequestProto(collection_name, 10);
        ::milvus::proto::schema::IDs ids;
        proxy.Insert(&data, &ids);
        EXPECT_EQ(10, ids.int_id().data_size());
        auto index_req =
            GetCreateIndexRequestProto(collection_name, "vindex", VEC_NAME);
        EXPECT_TRUE(proxy.CreateIndex(&index_req).IsOk());

        auto search_req = GetSearchRequestProto(
            collection_name,
            "id in [1, 2, 3]",
            std::vector<std::vector<float>>{{-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}},
            "2",
            "IP",
            "1");
        ::milvus::proto::milvus::SearchResults search_result;
        EXPECT_TRUE(proxy.Search(&search_req, &search_result).IsOk());
        // return ids: 2, 3, 2, 1
        EXPECT_EQ(search_result.results().ids().int_id().data(0), 2);
        EXPECT_EQ(search_result.results().ids().int_id().data(1), 3);
        EXPECT_EQ(search_result.results().ids().int_id().data(2), 2);
        EXPECT_EQ(search_result.results().ids().int_id().data(3), 1);
    }

    {
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        EXPECT_TRUE(proxy.LoadCollection(collection_name).IsOk());
        auto search_req = GetSearchRequestProto(
            collection_name,
            "id in [1, 2, 3, 4]",
            std::vector<std::vector<float>>{{-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}},
            "2",
            "IP",
            "0");
        ::milvus::proto::milvus::SearchResults search_result;
        EXPECT_TRUE(proxy.Search(&search_req, &search_result).IsOk());
        // return ids: 1,2,4,3
        EXPECT_EQ(search_result.results().ids().int_id().data(0), 1);
        EXPECT_EQ(search_result.results().ids().int_id().data(1), 2);
        EXPECT_EQ(search_result.results().ids().int_id().data(2), 4);
        EXPECT_EQ(search_result.results().ids().int_id().data(3), 3);
    }
    {
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        EXPECT_TRUE(proxy.LoadCollection(collection_name).IsOk());
        auto search_req = GetSearchRequestProto(
            collection_name,
            "id in [1, 2, 3, 4]",
            std::vector<std::vector<float>>{{-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}},
            "2",
            "IP",
            "20");
        ::milvus::proto::milvus::SearchResults search_result;
        EXPECT_TRUE(proxy.Search(&search_req, &search_result).IsOk());
        EXPECT_FALSE(search_result.results().has_ids());
    }
}

TEST(MilvusProxyTest, query) {
    const char* collection_name = "test_collection";
    {
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        proxy.DropCollection(collection_name);
        auto cr = GetCreateCollectionRequestProto(collection_name);
        EXPECT_TRUE(proxy.CreateCollection(&cr).IsOk());
        auto data = GetInsertRequestProto(collection_name, 10);
        ::milvus::proto::schema::IDs ids;
        proxy.Insert(&data, &ids);
        EXPECT_EQ(10, ids.int_id().data_size());
        auto query_req = GetQueryRequestProto(collection_name,
                                              "id in [1, 2, 3]",
                                              "2",
                                              "0",
                                              std::vector<std::string>{"id"});
        ::milvus::proto::milvus::QueryResults query_result;
        EXPECT_TRUE(proxy.Query(&query_req, &query_result).IsOk());

        EXPECT_EQ(
            query_result.fields_data()[0].scalars().long_data().data_size(), 2);
    }

    {
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        EXPECT_TRUE(proxy.LoadCollection(collection_name).IsOk());
        auto query_req = GetQueryRequestProto(
            collection_name, "id==1", "2", "0", std::vector<std::string>{"id"});
        ::milvus::proto::milvus::QueryResults query_result;
        EXPECT_TRUE(proxy.Query(&query_req, &query_result).IsOk());
        query_result.PrintDebugString();
    }
}

TEST(MilvusProxyTest, delete) {
    const char* collection_name = "test_collection";
    {
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        proxy.DropCollection(collection_name);
        auto cr = GetCreateCollectionRequestProto(collection_name);
        EXPECT_TRUE(proxy.CreateCollection(&cr).IsOk());
        auto data = GetInsertRequestProto(collection_name, 10);
        ::milvus::proto::schema::IDs ids;
        proxy.Insert(&data, &ids);
        EXPECT_EQ(10, ids.int_id().data_size());

        auto delete_req =
            GetDeleteRequestProto(collection_name, "id in [1, 2]");
        ::milvus::proto::milvus::MutationResult response;
        EXPECT_TRUE(proxy.Delete(&delete_req, &response).IsOk());

        auto query_req = GetQueryRequestProto(collection_name,
                                              "id in [1, 2, 3]",
                                              "3",
                                              "0",
                                              std::vector<std::string>{"id"});
        ::milvus::proto::milvus::QueryResults query_result;
        EXPECT_TRUE(proxy.Query(&query_req, &query_result).IsOk());
        EXPECT_EQ(
            query_result.fields_data()[0].scalars().long_data().data_size(), 1);
    }
    {
        ::milvus::local::MilvusProxy proxy(tmp_db_name);
        EXPECT_TRUE(proxy.Init());
        proxy.LoadCollection(collection_name);
        auto query_req = GetQueryRequestProto(collection_name,
                                              "id in [1, 2, 3]",
                                              "3",
                                              "0",
                                              std::vector<std::string>{"id"});
        ::milvus::proto::milvus::QueryResults query_result;
        EXPECT_TRUE(proxy.Query(&query_req, &query_result).IsOk());
        EXPECT_EQ(
            query_result.fields_data()[0].scalars().long_data().data_size(), 1);
    }
}

}  // namespace test
}  // namespace milvus::local
