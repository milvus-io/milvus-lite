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

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>

#include "pb/schema.pb.h"
#include "schema_util.h"

namespace milvus::local {

using DType = ::milvus::proto::schema::DataType;

// ---------------------------------------------------------------------------
// DecompactFieldData
// ---------------------------------------------------------------------------

TEST(DecompactFieldData, NoopWhenNoValidData) {
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::Int64);
    fd.mutable_scalars()->mutable_long_data()->add_data(10);
    fd.mutable_scalars()->mutable_long_data()->add_data(20);

    ASSERT_TRUE(schema_util::DecompactFieldData(&fd));
    EXPECT_EQ(fd.scalars().long_data().data_size(), 2);
    EXPECT_EQ(fd.scalars().long_data().data(0), 10);
    EXPECT_EQ(fd.scalars().long_data().data(1), 20);
}

TEST(DecompactFieldData, NoopWhenAlreadyUncompacted) {
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::Int64);
    fd.mutable_scalars()->mutable_long_data()->add_data(10);
    fd.mutable_scalars()->mutable_long_data()->add_data(0);
    fd.mutable_scalars()->mutable_long_data()->add_data(30);
    fd.add_valid_data(true);
    fd.add_valid_data(false);
    fd.add_valid_data(true);

    ASSERT_TRUE(schema_util::DecompactFieldData(&fd));
    EXPECT_EQ(fd.scalars().long_data().data_size(), 3);
}

TEST(DecompactFieldData, ExpandsCompactedInt64) {
    // Compacted: valid_data=[T,F,T], data=[10,30] -> expanded data=[10,0,30]
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::Int64);
    fd.mutable_scalars()->mutable_long_data()->add_data(10);
    fd.mutable_scalars()->mutable_long_data()->add_data(30);
    fd.add_valid_data(true);
    fd.add_valid_data(false);
    fd.add_valid_data(true);

    ASSERT_TRUE(schema_util::DecompactFieldData(&fd));
    EXPECT_EQ(fd.scalars().long_data().data_size(), 3);
    EXPECT_EQ(fd.scalars().long_data().data(0), 10);
    EXPECT_EQ(fd.scalars().long_data().data(1), 0);
    EXPECT_EQ(fd.scalars().long_data().data(2), 30);
    EXPECT_EQ(fd.valid_data_size(), 3);
}

TEST(DecompactFieldData, ExpandsCompactedVarChar) {
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::VarChar);
    fd.mutable_scalars()->mutable_string_data()->add_data("hello");
    fd.add_valid_data(true);
    fd.add_valid_data(false);
    fd.add_valid_data(false);

    ASSERT_TRUE(schema_util::DecompactFieldData(&fd));
    EXPECT_EQ(fd.scalars().string_data().data_size(), 3);
    EXPECT_EQ(fd.scalars().string_data().data(0), "hello");
    EXPECT_EQ(fd.scalars().string_data().data(1), "");
    EXPECT_EQ(fd.scalars().string_data().data(2), "");
}

TEST(DecompactFieldData, ExpandsCompactedFloat) {
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::Float);
    fd.mutable_scalars()->mutable_float_data()->add_data(1.5f);
    fd.add_valid_data(false);
    fd.add_valid_data(true);

    ASSERT_TRUE(schema_util::DecompactFieldData(&fd));
    EXPECT_EQ(fd.scalars().float_data().data_size(), 2);
    EXPECT_FLOAT_EQ(fd.scalars().float_data().data(0), 0.0f);
    EXPECT_FLOAT_EQ(fd.scalars().float_data().data(1), 1.5f);
}

TEST(DecompactFieldData, ExpandsCompactedBool) {
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::Bool);
    fd.mutable_scalars()->mutable_bool_data()->add_data(true);
    fd.add_valid_data(true);
    fd.add_valid_data(false);

    ASSERT_TRUE(schema_util::DecompactFieldData(&fd));
    EXPECT_EQ(fd.scalars().bool_data().data_size(), 2);
    EXPECT_TRUE(fd.scalars().bool_data().data(0));
    EXPECT_FALSE(fd.scalars().bool_data().data(1));
}

TEST(DecompactFieldData, ExpandsCompactedFloatVector) {
    // dim=2, valid_data=[T,F,T], compacted data=[v0_0,v0_1,v2_0,v2_1]
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::FloatVector);
    fd.mutable_vectors()->set_dim(2);
    auto* vd = fd.mutable_vectors()->mutable_float_vector();
    vd->add_data(1.0f);
    vd->add_data(2.0f);
    vd->add_data(5.0f);
    vd->add_data(6.0f);
    fd.add_valid_data(true);
    fd.add_valid_data(false);
    fd.add_valid_data(true);

    ASSERT_TRUE(schema_util::DecompactFieldData(&fd));
    EXPECT_EQ(fd.vectors().float_vector().data_size(), 6);
    EXPECT_FLOAT_EQ(fd.vectors().float_vector().data(0), 1.0f);
    EXPECT_FLOAT_EQ(fd.vectors().float_vector().data(1), 2.0f);
    EXPECT_FLOAT_EQ(fd.vectors().float_vector().data(2), 0.0f);
    EXPECT_FLOAT_EQ(fd.vectors().float_vector().data(3), 0.0f);
    EXPECT_FLOAT_EQ(fd.vectors().float_vector().data(4), 5.0f);
    EXPECT_FLOAT_EQ(fd.vectors().float_vector().data(5), 6.0f);
}

TEST(DecompactFieldData, AllNulls) {
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::Int64);
    // No data entries, all null
    fd.add_valid_data(false);
    fd.add_valid_data(false);

    ASSERT_TRUE(schema_util::DecompactFieldData(&fd));
    EXPECT_EQ(fd.scalars().long_data().data_size(), 2);
    EXPECT_EQ(fd.scalars().long_data().data(0), 0);
    EXPECT_EQ(fd.scalars().long_data().data(1), 0);
}

// ---------------------------------------------------------------------------
// GetFieldDataCount
// ---------------------------------------------------------------------------

TEST(GetFieldDataCount, BFloat16VectorUsesCorrectAccessor) {
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::BFloat16Vector);
    fd.mutable_vectors()->set_dim(4);
    // 2 rows of dim=4 bfloat16 => 2 * 4 * 2 = 16 bytes
    std::string data(16, '\0');
    fd.mutable_vectors()->set_bfloat16_vector(data);

    EXPECT_EQ(schema_util::GetFieldDataCount(fd), 2);
}

TEST(GetFieldDataCount, Float16VectorUsesCorrectAccessor) {
    ::milvus::proto::schema::FieldData fd;
    fd.set_type(DType::Float16Vector);
    fd.mutable_vectors()->set_dim(4);
    std::string data(16, '\0');
    fd.mutable_vectors()->set_float16_vector(data);

    EXPECT_EQ(schema_util::GetFieldDataCount(fd), 2);
}

// ---------------------------------------------------------------------------
// PickFieldDataByIndex - valid_data propagation
// ---------------------------------------------------------------------------

TEST(PickFieldDataByIndex, PropagatesValidData) {
    // Source: Int64, uncompacted, valid_data=[T,F,T], data=[10,0,30]
    ::milvus::proto::schema::FieldData src;
    src.set_type(DType::Int64);
    src.mutable_scalars()->mutable_long_data()->add_data(10);
    src.mutable_scalars()->mutable_long_data()->add_data(0);
    src.mutable_scalars()->mutable_long_data()->add_data(30);
    src.add_valid_data(true);
    src.add_valid_data(false);
    src.add_valid_data(true);

    ::milvus::proto::schema::FieldData dst;
    dst.set_type(DType::Int64);
    std::vector<int64_t> indexes = {2, 0, 1};
    ASSERT_TRUE(schema_util::PickFieldDataByIndex(src, indexes, &dst));

    EXPECT_EQ(dst.scalars().long_data().data_size(), 3);
    EXPECT_EQ(dst.scalars().long_data().data(0), 30);
    EXPECT_EQ(dst.scalars().long_data().data(1), 10);
    EXPECT_EQ(dst.scalars().long_data().data(2), 0);

    EXPECT_EQ(dst.valid_data_size(), 3);
    EXPECT_TRUE(dst.valid_data(0));
    EXPECT_TRUE(dst.valid_data(1));
    EXPECT_FALSE(dst.valid_data(2));
}

TEST(PickFieldDataByIndex, NoValidDataMeansNoValidDataInOutput) {
    ::milvus::proto::schema::FieldData src;
    src.set_type(DType::Int64);
    src.mutable_scalars()->mutable_long_data()->add_data(10);
    src.mutable_scalars()->mutable_long_data()->add_data(20);

    ::milvus::proto::schema::FieldData dst;
    dst.set_type(DType::Int64);
    std::vector<int64_t> indexes = {1, 0};
    ASSERT_TRUE(schema_util::PickFieldDataByIndex(src, indexes, &dst));

    EXPECT_EQ(dst.scalars().long_data().data_size(), 2);
    EXPECT_EQ(dst.scalars().long_data().data(0), 20);
    EXPECT_EQ(dst.scalars().long_data().data(1), 10);
    EXPECT_EQ(dst.valid_data_size(), 0);
}

TEST(PickFieldDataByIndex, VarCharWithValidData) {
    ::milvus::proto::schema::FieldData src;
    src.set_type(DType::VarChar);
    src.mutable_scalars()->mutable_string_data()->add_data("a");
    src.mutable_scalars()->mutable_string_data()->add_data("");
    src.mutable_scalars()->mutable_string_data()->add_data("c");
    src.add_valid_data(true);
    src.add_valid_data(false);
    src.add_valid_data(true);

    ::milvus::proto::schema::FieldData dst;
    dst.set_type(DType::VarChar);
    std::vector<int64_t> indexes = {1};
    ASSERT_TRUE(schema_util::PickFieldDataByIndex(src, indexes, &dst));

    EXPECT_EQ(dst.scalars().string_data().data_size(), 1);
    EXPECT_EQ(dst.scalars().string_data().data(0), "");
    EXPECT_EQ(dst.valid_data_size(), 1);
    EXPECT_FALSE(dst.valid_data(0));
}

// ---------------------------------------------------------------------------
// SliceFieldData with valid_data
// ---------------------------------------------------------------------------

TEST(SliceFieldData, PropagatesValidData) {
    ::milvus::proto::schema::FieldData src;
    src.set_type(DType::Int64);
    src.mutable_scalars()->mutable_long_data()->add_data(10);
    src.mutable_scalars()->mutable_long_data()->add_data(0);
    src.mutable_scalars()->mutable_long_data()->add_data(30);
    src.add_valid_data(true);
    src.add_valid_data(false);
    src.add_valid_data(true);

    ::milvus::proto::schema::FieldData dst;
    dst.set_type(DType::Int64);
    std::vector<std::tuple<int64_t, int64_t>> ranges = {{1, 2}};
    ASSERT_TRUE(schema_util::SliceFieldData(src, ranges, &dst));

    EXPECT_EQ(dst.scalars().long_data().data_size(), 2);
    EXPECT_EQ(dst.scalars().long_data().data(0), 0);
    EXPECT_EQ(dst.scalars().long_data().data(1), 30);
    EXPECT_EQ(dst.valid_data_size(), 2);
    EXPECT_FALSE(dst.valid_data(0));
    EXPECT_TRUE(dst.valid_data(1));
}

}  // namespace milvus::local
