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
#include "function/bm25_function.h"
#include "test_util.h"

namespace milvus::local::function {
namespace test {

using namespace milvus::local::test;

using function = milvus::local::function::BM25Function;

TEST(BM25FunctionTest, normal) {
    auto coll_str = CreateCollection("test", "sparse_vector");
    milvus::proto::schema::CollectionSchema schema;
    schema.ParseFromString(coll_str);

    ::milvus::proto::schema::FunctionSchema fschema;
    fschema.ParseFromString(
        CreateBM25Function("bm25", VARCHAR_SCALAR_NAME, SPARSE_VEC));
    auto [s, f] = BM25Function::NewBM25Function(&schema, &fschema);
    EXPECT_TRUE(s.IsOk());
    {
        milvus::proto::common::PlaceholderGroup input_plg;
        auto plg = input_plg.add_placeholders();
        plg->set_type(::milvus::proto::common::PlaceholderType::VarChar);
        for (int i = 0; i < 10; i++) {
            auto v = plg->add_values();
            v->assign("doc");
        }

        milvus::proto::common::PlaceholderGroup output_plg;
        f->ProcessSearch(input_plg, &output_plg);
        EXPECT_EQ(output_plg.placeholders(0).values_size(), 10);
        EXPECT_EQ(output_plg.placeholders(0).type(),
                  ::milvus::proto::common::PlaceholderType::SparseFloatVector);
    }
    {
        std::vector<milvus::proto::schema::FieldData*> inputs;
        milvus::proto::schema::FieldData text_field;
        text_field.set_field_name(VARCHAR_SCALAR_NAME);
        text_field.set_is_dynamic(false);
        text_field.set_type(::milvus::proto::schema::DataType::VarChar);
        for (int i = 0; i < 10; i++) {
            text_field.mutable_scalars()->mutable_string_data()->add_data(
                "doc");
        }
        inputs.push_back(&text_field);
        std::vector<milvus::proto::schema::FieldData> outputs;
        f->ProcessInsert(inputs, &outputs);
        EXPECT_EQ(outputs.size(), 1);
        EXPECT_EQ(outputs[0].type(),
                  ::milvus::proto::common::PlaceholderType::SparseFloatVector);
        EXPECT_EQ(outputs[0].vectors().sparse_float_vector().dim(), 1);
    }
}
}  // namespace test
}  // namespace milvus::local::function
