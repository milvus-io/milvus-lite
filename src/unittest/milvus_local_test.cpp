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

#include "milvus_local.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "antlr4-runtime.h"
#include "parser/parser.h"
#include "parser/utils.h"
#include "pb/plan.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "test_util.h"
#include "type.h"

namespace milvus::local {

namespace test {

TEST(MilvusLocal, h) {
    std::string collection_name("test_schema");
    milvus::local::Rows data = CreateData(10);
    auto schema_str = CreateCollection();
    auto index_str = CreateVectorIndex();

    std::remove("milvus_data.db");
    {
        MilvusLocal ms("./");
        ms.Init();
        ms.CreateCollection(collection_name, PK_NAME, schema_str);
        ms.CreateIndex(collection_name, "test_index", index_str);
        auto rows = CreateData(20);
        std::vector<std::string> ids;
        ms.Insert(collection_name, rows, &ids);
        milvus::proto::schema::CollectionSchema schema;
        schema.ParseFromString(schema_str);

        // std::cout << schema.DebugString() << std::endl;
        std::string exprstr("id in [1, 2, 6, 5, 8, 9]");
        antlr4::ANTLRInputStream input(exprstr);
        PlanLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        PlanParser parser(&tokens);
        PlanParser::ExprContext* tree = parser.expr();
        auto helper = milvus::CreateSchemaHelper(&schema);
        milvus::PlanCCVisitor visitor(&helper);
        auto res = std::any_cast<milvus::ExprWithDtype>(visitor.visit(tree));
        {
            ::milvus::proto::plan::PlanNode plan;
            plan.mutable_query()->set_is_count(false);
            plan.mutable_query()->set_limit(5);
            plan.mutable_query()->set_allocated_predicates(res.expr);
            plan.add_output_field_ids(200);
            plan.add_output_field_ids(202);
            std::cout << plan.DebugString() << std::endl;
            RetrieveResult result;
            ms.Retrieve(collection_name, plan.SerializeAsString(), &result);
            milvus::proto::segcore::RetrieveResults rs;
            rs.ParseFromArray(result.retrieve_result_.proto_blob,
                              result.retrieve_result_.proto_size);
            std::cout << rs.DebugString() << std::endl;
        }

        {
            std::cout << "===============================" << std::endl;
            ::milvus::proto::plan::PlanNode plan;
            plan.mutable_vector_anns()->set_field_id(201);
            plan.mutable_vector_anns()->set_allocated_predicates(res.expr);
            plan.mutable_vector_anns()->set_placeholder_tag("$0");
            plan.mutable_vector_anns()->set_vector_type(
                ::milvus::proto::plan::VectorType::FloatVector);
            plan.mutable_vector_anns()->mutable_query_info()->set_topk(3);
            plan.mutable_vector_anns()->mutable_query_info()->set_metric_type(
                "IP");
            plan.mutable_vector_anns()->mutable_query_info()->set_search_params(
                "{\"nprobe\": 10}");
            plan.mutable_vector_anns()->mutable_query_info()->set_round_decimal(
                -1);
            plan.add_output_field_ids(200);
            plan.add_output_field_ids(202);
            std::cout << plan.DebugString() << std::endl;

            milvus::proto::common::PlaceholderGroup raw_group;
            auto value = raw_group.add_placeholders();
            value->set_tag("$0");
            value->set_type(
                milvus::proto::common::PlaceholderType::FloatVector);
            std::vector<float> vec{0.3, 0.5, 0.2};
            value->add_values(vec.data(), vec.size() * sizeof(float));

            auto slice_nqs = std::vector<int64_t>{1};
            auto slice_topKs = std::vector<int64_t>{3};
            SearchResult result(slice_nqs, slice_topKs);
            ms.Search(collection_name,
                      plan.SerializeAsString(),
                      raw_group.SerializeAsString(),
                      &result);
            milvus::proto::schema::SearchResultData rz;
            rz.ParseFromArray(result.result_[0].proto_blob,
                              result.result_[0].proto_size);
            // std::cout << rz.DebugString() << std::endl;

            milvus::proto::schema::IDs ids;
            ids.mutable_int_id()->add_data(0);
            ids.mutable_int_id()->add_data(1);
            ids.mutable_int_id()->add_data(2);
            ms.DeleteByIds(collection_name,
                           ids.SerializeAsString(),
                           3,
                           std::vector<std::string>{"0", "1", "2"});
            ms.Search(collection_name,
                      plan.SerializeAsString(),
                      raw_group.SerializeAsString(),
                      &result);
            rz.ParseFromArray(result.result_[0].proto_blob,
                              result.result_[0].proto_size);
            // std::cout << rz.DebugString() << std::endl;

            ms.ReleaseCollection(collection_name);
        }
    }

    {
        MilvusLocal ms("./");
        ms.Init();
        ms.LoadCollection(collection_name);
        ms.ReleaseCollection(collection_name);
    }

    std::remove("milvus_data.db");
}

TEST(MilvusLocal, parser) {
    auto schema_str = CreateCollection();
    milvus::proto::schema::CollectionSchema schema;
    schema.ParseFromString(schema_str);

    // std::string exprstr("sc in [1, 2, 3, 4]");
    // antlr4::ANTLRInputStream input(exprstr);
    // PlanLexer lexer(&input);
    // antlr4::CommonTokenStream tokens(&lexer);
    // PlanParser parser(&tokens);

    // PlanParser::ExprContext* tree = parser.expr();

    // auto helper = milvus::CreateSchemaHelper(&schema);
    // milvus::PlanCCVisitor visitor(&helper);
    // auto res = std::any_cast<milvus::ExprWithDtype>(visitor.visit(tree));
    // std::cout << res.expr->DebugString() << std::endl;
    // std::cout << "---------------" << std::endl;
    // std::cout << res.expr->term_expr().DebugString() << std::endl;
}
}  // namespace test
}  // namespace milvus::local
