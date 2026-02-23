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

#include <algorithm>
#include <regex>
#include <vector>

namespace milvus::local {

namespace {

std::string
Trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos)
        return "";
    return s.substr(start, s.find_last_not_of(" \t\n\r") - start + 1);
}

struct NullCheckInfo {
    std::string field_name;
    bool is_not_null;
};

bool
ContainsNullCheck(const std::string& expr) {
    std::string lower = expr;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower.find("is null") != std::string::npos ||
           lower.find("is not null") != std::string::npos;
}

std::optional<NullCheckInfo>
TryParseNullCheck(const std::string& fragment) {
    std::string input = Trim(fragment);
    while (input.size() >= 2 && input.front() == '(' && input.back() == ')') {
        input = Trim(input.substr(1, input.size() - 2));
    }

    static const std::regex kNotNull(
        R"(^\s*(\w+)\s+[iI][sS]\s+[nN][oO][tT]\s+[nN][uU][lL][lL]\s*$)");
    static const std::regex kNull(
        R"(^\s*(\w+)\s+[iI][sS]\s+[nN][uU][lL][lL]\s*$)");

    std::smatch m;
    if (std::regex_match(input, m, kNotNull))
        return NullCheckInfo{m[1].str(), true};
    if (std::regex_match(input, m, kNull))
        return NullCheckInfo{m[1].str(), false};
    return std::nullopt;
}

std::string
BuildNullExprSerialized(const proto::schema::FieldSchema& field,
                        proto::plan::NullExpr_NullOp op) {
    proto::plan::Expr expr;
    auto* null_expr = expr.mutable_null_expr();
    auto* info = null_expr->mutable_column_info();
    info->set_field_id(field.fieldid());
    info->set_data_type(field.data_type());
    info->set_is_primary_key(field.is_primary_key());
    info->set_is_autoid(field.autoid());
    info->set_nullable(field.nullable());
    info->set_element_type(field.element_type());
    null_expr->set_op(op);
    return expr.SerializeAsString();
}

enum class LogicalConn { NONE, AND, OR };

struct Fragment {
    std::string text;
    LogicalConn conn;
};

std::vector<Fragment>
SplitTopLevelLogical(const std::string& expr) {
    std::vector<Fragment> frags;
    std::string lower = expr;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    int paren = 0;
    bool in_str = false;
    char str_char = 0;
    size_t last = 0;
    auto pending = LogicalConn::NONE;

    auto emit = [&](size_t end, LogicalConn next_conn, size_t skip) {
        frags.push_back({Trim(expr.substr(last, end - last)), pending});
        pending = next_conn;
        last = end + skip;
    };

    for (size_t i = 0; i < expr.size(); i++) {
        char c = expr[i];
        if (in_str) {
            if (c == str_char && (i == 0 || expr[i - 1] != '\\'))
                in_str = false;
            continue;
        }
        if (c == '\'' || c == '"') {
            in_str = true;
            str_char = c;
            continue;
        }
        if (c == '(') {
            paren++;
            continue;
        }
        if (c == ')') {
            paren--;
            continue;
        }
        if (paren > 0)
            continue;

        if (i + 2 <= expr.size() && expr.substr(i, 2) == "&&") {
            emit(i, LogicalConn::AND, 2);
            i = last - 1;
            continue;
        }
        if (i + 2 <= expr.size() && expr.substr(i, 2) == "||") {
            emit(i, LogicalConn::OR, 2);
            i = last - 1;
            continue;
        }
        bool boundary_left =
            (i == 0 || (!std::isalnum(expr[i - 1]) && expr[i - 1] != '_'));
        if (boundary_left && i + 3 <= lower.size() &&
            lower.substr(i, 3) == "and" &&
            (i + 3 >= expr.size() ||
             (!std::isalnum(expr[i + 3]) && expr[i + 3] != '_'))) {
            emit(i, LogicalConn::AND, 3);
            i = last - 1;
            continue;
        }
        if (boundary_left && i + 2 <= lower.size() &&
            lower.substr(i, 2) == "or" &&
            (i + 2 >= expr.size() ||
             (!std::isalnum(expr[i + 2]) && expr[i + 2] != '_'))) {
            emit(i, LogicalConn::OR, 2);
            i = last - 1;
            continue;
        }
    }
    frags.push_back({Trim(expr.substr(last)), pending});
    return frags;
}

std::string
ParseFragmentWithAntlr(milvus::proto::schema::CollectionSchema& schema,
                       SchemaHelper& helper,
                       const std::string& fragment) {
    antlr4::ANTLRInputStream input(fragment);
    PlanLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    PlanParser parser(&tokens);
    auto* tree = parser.expr();
    PlanCCVisitor visitor(&helper);
    auto res = std::any_cast<ExprWithDtype>(visitor.visit(tree));
    return res.expr->SerializeAsString();
}

std::string
CombineWithBinaryOp(const std::string& left_ser,
                    const std::string& right_ser,
                    LogicalConn conn) {
    proto::plan::Expr left, right, combined;
    left.ParseFromString(left_ser);
    right.ParseFromString(right_ser);
    auto* bin = combined.mutable_binary_expr();
    bin->set_op(conn == LogicalConn::AND
                    ? proto::plan::BinaryExpr_BinaryOp_LogicalAnd
                    : proto::plan::BinaryExpr_BinaryOp_LogicalOr);
    *bin->mutable_left() = left;
    *bin->mutable_right() = right;
    return combined.SerializeAsString();
}

}  // namespace

std::string
ParserToMessage(milvus::proto::schema::CollectionSchema& schema,
                const std::string& exprstr) {
    if (exprstr.empty()) {
        google::protobuf::Arena arena;
        auto alway_true_expr =
            google::protobuf::Arena::CreateMessage<proto::plan::AlwaysTrueExpr>(
                &arena);
        auto expr =
            google::protobuf::Arena::CreateMessage<proto::plan::Expr>(&arena);
        expr->unsafe_arena_set_allocated_always_true_expr(alway_true_expr);
        return expr->SerializeAsString();
    }

    if (!ContainsNullCheck(exprstr)) {
        antlr4::ANTLRInputStream input(exprstr);
        PlanLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        PlanParser parser(&tokens);
        PlanParser::ExprContext* tree = parser.expr();
        auto helper = milvus::local::CreateSchemaHelper(&schema);
        milvus::local::PlanCCVisitor visitor(&helper);
        auto res =
            std::any_cast<milvus::local::ExprWithDtype>(visitor.visit(tree));
        return res.expr->SerializeAsString();
    }

    auto helper = CreateSchemaHelper(&schema);
    auto frags = SplitTopLevelLogical(exprstr);

    std::vector<std::string> parsed;
    for (auto& frag : frags) {
        auto null_info = TryParseNullCheck(frag.text);
        if (null_info) {
            auto& field = helper.GetFieldFromName(null_info->field_name);
            auto op = null_info->is_not_null
                          ? proto::plan::NullExpr_NullOp_IsNotNull
                          : proto::plan::NullExpr_NullOp_IsNull;
            parsed.push_back(BuildNullExprSerialized(field, op));
        } else {
            parsed.push_back(ParseFragmentWithAntlr(schema, helper, frag.text));
        }
    }

    // AND binds tighter than OR: collapse AND-connected fragments first
    std::vector<std::string> or_operands;
    std::string current = parsed[0];
    for (size_t i = 1; i < frags.size(); i++) {
        if (frags[i].conn == LogicalConn::AND) {
            current = CombineWithBinaryOp(current, parsed[i], LogicalConn::AND);
        } else {
            or_operands.push_back(std::move(current));
            current = parsed[i];
        }
    }
    or_operands.push_back(std::move(current));

    std::string result = or_operands[0];
    for (size_t i = 1; i < or_operands.size(); i++) {
        result = CombineWithBinaryOp(result, or_operands[i], LogicalConn::OR);
    }
    return result;
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
