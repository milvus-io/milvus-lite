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

#include "parser.h"

namespace milvus::local {

std::string
ParserToMessage(milvus::proto::schema::CollectionSchema& schema,
                const std::string& exprstr) {
    antlr4::ANTLRInputStream input(exprstr);
    PlanLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    PlanParser parser(&tokens);

    PlanParser::ExprContext* tree = parser.expr();

    auto helper = milvus::local::CreateSchemaHelper(&schema);

    milvus::local::PlanCCVisitor visitor(&helper);
    auto res = std::any_cast<milvus::local::ExprWithDtype>(visitor.visit(tree));
    return res.expr->SerializeAsString();
}

std::shared_ptr<milvus::proto::plan::Expr>
ParseIdentifier(milvus::local::SchemaHelper helper,
                const std::string& identifier) {
    auto expr =
        google::protobuf::Arena::CreateMessage<milvus::proto::plan::Expr>(NULL);

    expr->ParseFromString(ParserToMessage(*(helper.schema), identifier));

    auto ret = std::make_shared<milvus::proto::plan::Expr>();
    ret.reset(expr);
    return ret;
};

}  // namespace milvus::local
