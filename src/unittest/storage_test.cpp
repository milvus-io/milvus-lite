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

#include "storage/storage.h"
#include "storage/bm25_stats.h"
#include <gtest/gtest.h>
#include <cstdint>

namespace milvus::local {

TEST(Storage, h) {
    const char* db_path = "test.db";
    auto db_ptr = std::make_unique<SQLite::Database>(
        db_path,
        SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE | SQLite::OPEN_FULLMUTEX);
    BM25Stats bm25_stats;
    EXPECT_TRUE(bm25_stats.CreateTables(db_ptr.get()));
    std::map<uint32_t, int32_t> tokens{};
    EXPECT_TRUE(bm25_stats.AddTokenDoc(db_ptr.get(), "test", "sparse", tokens));
    tokens[1] = 1;
    tokens[2] = 2;
    tokens[3] = 3;
    EXPECT_TRUE(bm25_stats.AddTokenDoc(db_ptr.get(), "test", "sparse", tokens));

    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 1),
              1);
    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 2),
              2);
    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 3),
              3);

    tokens.clear();
    tokens[1] = 1;
    tokens[2] = 2;
    EXPECT_TRUE(bm25_stats.AddTokenDoc(db_ptr.get(), "test", "sparse", tokens));

    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 1),
              2);
    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 2),
              4);
    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 3),
              3);

    EXPECT_TRUE(
        bm25_stats.DeleteTokenDoc(db_ptr.get(), "test", "sparse", tokens));
    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 1),
              1);
    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 2),
              2);
    EXPECT_EQ(bm25_stats.GetTokenDocCount(db_ptr.get(), "test", "sparse", 3),
              3);

    EXPECT_TRUE(
        bm25_stats.AddBM25Stats(db_ptr.get(), "test", "sparse", 100, 20));
    {
        auto [token, doc] =
            bm25_stats.GetTokenNumAndDocNum(db_ptr.get(), "test", "sparse");
        EXPECT_EQ(token, 100);
        EXPECT_EQ(doc, 20);
    }
    EXPECT_TRUE(
        bm25_stats.DeleteBM25Stats(db_ptr.get(), "test", "sparse", 20, 10));

    {
        auto [token, doc] =
            bm25_stats.GetTokenNumAndDocNum(db_ptr.get(), "test", "sparse");
        EXPECT_EQ(token, 80);
        EXPECT_EQ(doc, 10);
    }

    std::remove(db_path);
}

}  // namespace milvus::local
