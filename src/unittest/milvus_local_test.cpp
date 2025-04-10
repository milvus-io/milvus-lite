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

TEST(MilvusLocal, BM25) {
    std::string collection_name("test_bm25");
    ::milvus::proto::schema::CollectionSchema schema;
    schema.set_name("bm25");
    {
        auto field = schema.add_fields();
        field->set_name("id");
        field->set_fieldid(101);
        field->set_data_type(milvus::proto::schema::DataType::Int64);
        field->set_is_primary_key(true);
        field->set_autoid(false);
    }
    {
        auto field = schema.add_fields();
        field->set_name("document_content");
        field->set_data_type(milvus::proto::schema::DataType::VarChar);
        {
            auto param = field->add_type_params();
            param->set_key("max_length");
            param->set_value("9000");
        }
        {
            auto param = field->add_type_params();
            param->set_key("enable_analyzer");
            param->set_value("true");
        }
    }
    {
        auto field = schema.add_fields();
        field->set_name("sparse_vector");
        field->set_autoid(102);
        field->set_data_type(
            milvus::proto::schema::DataType::SparseFloatVector);
    }
    {
        auto func = schema.add_functions();
        func->set_name("bm25_fn");
        func->add_input_field_names();
        func->set_input_field_names(0, "document_content");
        func->add_output_field_names();
        func->set_output_field_names(0, "sparse_vector");
        func->set_type(milvus::proto::schema::FunctionType::BM25);
    }
    MilvusLocal ms("bm.db");
    ms.Init();
    ms.CreateCollection("doc_in_doc_out", "id", schema.SerializeAsString());
    milvus::proto::segcore::FieldIndexMeta meta;
    {
        meta.set_index_name("sparse_inverted_index");
        meta.set_fieldid(102);
        auto index_type = meta.add_type_params();
        index_type->set_key("index_type");
        index_type->set_value("SPARSE_INVERTED_INDEX");
        auto metric_type = meta.add_type_params();
        metric_type->set_key("metric_type");
        metric_type->set_value("BM25");

        auto bm25_k1 = meta.add_index_params();
        bm25_k1->set_key("bm25_k1");
        bm25_k1->set_value(std::to_string(1.2f));

        auto bm25_b = meta.add_index_params();
        bm25_b->set_key("bm25_b");
        bm25_b->set_value(std::to_string(0.75f));
    }
    std::string meta_proto;
    meta.SerializeToString(&meta_proto);
    std::cout << "Create Index" << std::endl;
    std::cout << meta.DebugString() << std::endl;
    ms.CreateIndex("doc_in_doc_out", "sparse_inverted_index", meta_proto);

    milvus::proto::segcore::InsertRecord record;
    record.set_num_rows(1);
    {
        auto data = record.add_fields_data();
        data->set_field_name("id");
        data->set_field_id(101);
        data->set_type(milvus::proto::schema::DataType::Int64);
        data->mutable_scalars()->mutable_long_data()->add_data(1);
    }
    {
        auto data = record.add_fields_data();
        data->set_field_name("document_content");
        data->set_field_id(102);
        data->set_type(milvus::proto::schema::DataType::String);
        data->mutable_scalars()->mutable_string_data()->add_data("hello world");
    }
    {
        auto data = record.add_fields_data();
        data->set_field_id(0);
        data->set_type(milvus::proto::schema::DataType::Int64);
        data->mutable_scalars()->mutable_long_data()->add_data(0);
    }

    {
        auto data = record.add_fields_data();
        data->set_field_id(1);
        data->set_type(milvus::proto::schema::DataType::Int64);
        data->mutable_scalars()->mutable_long_data()->add_data(10000000);
    }

    std::vector<std::string> ids;
    auto status = ms.Insert(
        "doc_in_doc_out",
        {std::make_tuple(std::to_string(1), record.SerializeAsString())},
        &ids);
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
