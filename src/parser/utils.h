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

#pragma once
#include <iomanip>
#include <any>
#include <iostream>
#include <optional>
#include <string>
#include "antlr/PlanBaseVisitor.h"
#include "antlr/PlanLexer.h"
#include "antlr/PlanParser.h"
#include "pb/plan.pb.h"

namespace milvus::local {

class Exception : public std::exception {
 public:
    explicit Exception(const std::string& msg) : msg_(msg) {
    }

    const char*
    what() const noexcept override {
        return msg_.c_str();
    }

 private:
    std::string msg_;
};

#define TRY_WITH_EXCEPTION(call)    \
    do {                            \
        if (!(call)) {              \
            throw Exception(#call); \
        }                           \
    } while (0)

template <typename T>
inline T*
CreateMessageWithCopy(google::protobuf::Arena* arena, const T& val) {
    T* ret = google::protobuf::Arena::CreateMessage<T>(arena);
    ret->CopyFrom(val);
    return ret;
}

struct ExprWithDtype {
    proto::plan::Expr* expr;
    proto::schema::DataType dtype;
    bool dependent;
    ExprWithDtype(proto::plan::Expr* const expr,
                  proto::schema::DataType dtype,
                  bool dependent)
        : expr(expr), dtype(dtype), dependent(dependent) {
    }
};

inline std::any
extractValue(proto::plan::Expr* expr) {
    if (!expr->has_value_expr())
        return nullptr;

    if (!expr->value_expr().has_value())
        return nullptr;
    if (expr->value_expr().value().has_bool_val())
        return expr->value_expr().value().bool_val();
    if (expr->value_expr().value().has_float_val())
        return expr->value_expr().value().float_val();
    if (expr->value_expr().value().has_string_val())
        return expr->value_expr().value().string_val();
    if (expr->value_expr().value().has_int64_val())
        return expr->value_expr().value().int64_val();

    return nullptr;
}

template <typename T>
inline proto::plan::Expr*
createValueExpr(const T val, google::protobuf::Arena* arena = nullptr) {
    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena);
    auto val_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::ValueExpr>(arena);
    auto value =
        google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
            arena);
    if constexpr (std::is_same_v<T, int64_t>)
        value->set_int64_val(val);
    else if constexpr (std::is_same_v<T, int8_t>)
        value->set_int64_val(int64_t(val));
    else if constexpr (std::is_same_v<T, int32_t>)
        value->set_int64_val(int64_t(val));
    else if constexpr (std::is_same_v<T, int16_t>)
        value->set_int64_val(int16_t(val));
    else if constexpr (std::is_same_v<T, std::string>)
        value->set_string_val(val);
    else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
        value->set_float_val(val);
    else if constexpr (std::is_same_v<T, bool>)
        value->set_bool_val(val);
    else
        TRY_WITH_EXCEPTION(false);

    val_expr->unsafe_arena_set_allocated_value(value);
    expr->unsafe_arena_set_allocated_value_expr(val_expr);
    return expr;
}

inline std::vector<std::string>
tokenize(std::string s, std::string del = " ") {
    std::vector<std::string> results;
    int start, end = -1 * del.size();
    do {
        start = end + del.size();
        end = s.find(del, start);
        results.push_back(s.substr(start, end - start));
    } while (end != -1);
    return results;
}

template <proto::plan::BinaryExpr_BinaryOp T>
inline proto::plan::Expr*
createBinExpr(proto::plan::Expr* left,
              proto::plan::Expr* right,
              google::protobuf::Arena* arena = nullptr) {
    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena);
    auto bin_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::BinaryExpr>(arena);
    bin_expr->set_op(T);
    bin_expr->unsafe_arena_set_allocated_left(left);
    bin_expr->unsafe_arena_set_allocated_right(right);
    expr->unsafe_arena_set_allocated_binary_expr(bin_expr);
    return expr;
}

template <proto::plan::ArithOpType T>
inline proto::plan::Expr*
createBinArithExpr(proto::plan::Expr* left,
                   proto::plan::Expr* right,
                   google::protobuf::Arena* arena = nullptr) {
    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena);
    auto bin_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::BinaryArithExpr>(
            arena);
    bin_expr->set_op(T);
    bin_expr->unsafe_arena_set_allocated_left(left);
    bin_expr->unsafe_arena_set_allocated_right(right);
    expr->unsafe_arena_set_allocated_binary_arith_expr(bin_expr);
    return expr;
}

inline bool
arithmeticDtype(proto::schema::DataType type) {
    switch (type) {
        case proto::schema::DataType::Float:
            return true;
        case proto::schema::DataType::Double:
            return true;
        case proto::schema::DataType::Int8:
            return true;
        case proto::schema::DataType::Int16:
            return true;
        case proto::schema::DataType::Int32:
            return true;
        case proto::schema::DataType::Int64:
            return true;
        default:
            return false;
    }
}

inline proto::schema::DataType
getArrayElementType(proto::plan::Expr* expr) {
    if (expr->has_column_expr()) {
        return expr->column_expr().info().element_type();
    }
    if (expr->has_value_expr() && expr->value_expr().has_value() &&
        expr->value_expr().value().has_array_val()) {
        return expr->value_expr().value().array_val().element_type();
    }

    return proto::schema::DataType::None;
}

inline bool
canArithmeticDtype(proto::schema::DataType left_type,
                   proto::schema::DataType right_type) {
    if (left_type == proto::schema::DataType::JSON &&
        right_type == proto::schema::DataType::JSON)
        return false;
    if (left_type == proto::schema::DataType::JSON &&
        arithmeticDtype(right_type))
        return true;
    if (arithmeticDtype(left_type) &&
        right_type == proto::schema::DataType::JSON)
        return true;
    if (arithmeticDtype(left_type) && arithmeticDtype(right_type))
        return true;
    return false;
}

inline proto::schema::DataType
calDataType(ExprWithDtype* a, ExprWithDtype* b) {
    auto a_dtype = a->dtype;
    auto b_dtype = b->dtype;
    if (a->dtype == proto::schema::DataType::Array) {
        a_dtype = getArrayElementType(a->expr);
    }
    if (b->dtype == proto::schema::DataType::Array) {
        b_dtype = getArrayElementType(b->expr);
    }
    if (a_dtype == proto::schema::DataType::JSON) {
        if (b_dtype == proto::schema::DataType::Float ||
            b_dtype == proto::schema::DataType::Double)
            return proto::schema::DataType::Float;
        if (b_dtype == proto::schema::DataType::Int8 ||
            b_dtype == proto::schema::DataType::Int16 ||
            b_dtype == proto::schema::DataType::Int32 ||
            b_dtype == proto::schema::DataType::Int64)
            return proto::schema::DataType::Int64;
        if (b_dtype == proto::schema::DataType::JSON)
            return proto::schema::DataType::JSON;
    }

    if (a_dtype == proto::schema::DataType::Float ||
        a_dtype == proto::schema::DataType::Double) {
        if (b_dtype == proto::schema::DataType::JSON)
            return proto::schema::DataType::Double;
        if (arithmeticDtype(b_dtype))
            return proto::schema::DataType::Double;
    }

    if (a_dtype == proto::schema::DataType::Int8 ||
        a_dtype == proto::schema::DataType::Int16 ||
        a_dtype == proto::schema::DataType::Int32 ||
        a_dtype == proto::schema::DataType::Int64) {
        if (b_dtype == proto::schema::DataType::Float ||
            b_dtype == proto::schema::DataType::Double)
            return proto::schema::DataType::Double;
        if (b_dtype == proto::schema::DataType::Int8 ||
            b_dtype == proto::schema::DataType::Int16 ||
            b_dtype == proto::schema::DataType::Int32 ||
            b_dtype == proto::schema::DataType::Int64 ||
            b_dtype == proto::schema::DataType::JSON)
            return proto::schema::DataType::Int64;
    }
    TRY_WITH_EXCEPTION(false);
}

struct SchemaHelper {
    SchemaHelper() = default;

    proto::schema::CollectionSchema* schema = nullptr;
    std::map<std::string, int> name_offset;
    std::map<int64_t, int> id_offset;

    int primary_key_offset = -1;
    int partition_key_offset = -1;

    const proto::schema::FieldSchema&
    GetPrimaryKeyField() {
        TRY_WITH_EXCEPTION(primary_key_offset != -1);
        return schema->fields(primary_key_offset);
    }

    const proto::schema::FieldSchema&
    GetPartitionKeyField() {
        TRY_WITH_EXCEPTION(partition_key_offset != -1);
        return schema->fields(partition_key_offset);
    }

    const proto::schema::FieldSchema&
    GetFieldFromName(const std::string& name) {
        auto it = name_offset.find(name);
        TRY_WITH_EXCEPTION(it != name_offset.end());
        return schema->fields(it->second);
    }

    const proto::schema::FieldSchema&
    GetFieldFromNameDefaultJSON(const std::string& name) {
        auto it = name_offset.find(name);
        if (it == name_offset.end()) {
            return GetDefaultJSONField();
        }
        return schema->fields(it->second);
    }

    const proto::schema::FieldSchema&
    GetDefaultJSONField() {
        for (int i = 0; i < schema->fields_size(); ++i) {
            auto& field = schema->fields(i);
            if (field.data_type() == proto::schema::DataType::JSON &&
                field.is_dynamic())
                return field;
        }

        TRY_WITH_EXCEPTION(false);
    }

    const proto::schema::FieldSchema&
    GetFieldFromID(int64_t id) {
        auto it = id_offset.find(id);
        TRY_WITH_EXCEPTION(it != id_offset.end());
        return schema->fields(it->second);
    }

    int
    GetVectorDimFromID(int64_t id) {
        auto& field = GetFieldFromID(id);
        if (field.data_type() != proto::schema::DataType::FloatVector &&
            field.data_type() != proto::schema::DataType::Float16Vector &&
            field.data_type() != proto::schema::DataType::BinaryVector &&
            field.data_type() != proto::schema::DataType::BFloat16Vector) {
            TRY_WITH_EXCEPTION(false);
        }
        for (int i = 0; i < field.type_params_size(); ++i) {
            if (field.type_params(i).key() == "dim")
                return std::stoi(
                    field.type_params(i).value().c_str(), NULL, 10);
        }
        TRY_WITH_EXCEPTION(false);
    }
};

inline SchemaHelper
CreateSchemaHelper(proto::schema::CollectionSchema* schema) {
    TRY_WITH_EXCEPTION(schema);
    SchemaHelper schema_helper;
    schema_helper.schema = schema;
    for (int i = 0; i < schema->fields_size(); ++i) {
        auto field = schema->fields(i);
        auto it = schema_helper.name_offset.find(field.name());
        if (it != schema_helper.name_offset.end())
            TRY_WITH_EXCEPTION(false);
        schema_helper.name_offset[field.name()] = i;
        schema_helper.id_offset[field.fieldid()] = i;
        if (field.is_primary_key()) {
            TRY_WITH_EXCEPTION(schema_helper.primary_key_offset == -1);
            schema_helper.primary_key_offset = i;
        }
        if (field.is_partition_key()) {
            TRY_WITH_EXCEPTION(schema_helper.primary_key_offset == -1);
            schema_helper.partition_key_offset = i;
        }
    }
    return schema_helper;
}

inline std::string
convertEscapeSingle(const std::string& in) {
    std::vector<size_t> need_replace_index;
    size_t escape_ch_count = 0;
    size_t in_string_lenth = in.length();
    for (size_t i = 1; i < in_string_lenth - 1; ++i) {
        if (in[i] == '\\') {
            escape_ch_count++;
            continue;
        }
        if (in[i] == '"' && escape_ch_count % 2 == 0) {
            need_replace_index.push_back(i);
        }

        if (in[i] == '\'' && escape_ch_count % 2 != 0) {
            need_replace_index.push_back(i);
        }

        escape_ch_count = 0;
    }

    std::string in_;
    in_ += '"';
    size_t start = 1;
    for (auto end : need_replace_index) {
        if (in[end] == '"') {
            in_ += in.substr(start, end - start);
            in_ += "\\\"";
        } else {
            in_ += in.substr(start, end - start - 1);
            in_ += '\'';
        }
        start = end + 1;
    }

    in_ += in.substr(start, in.length() - start - 1);

    in_ += '"';
    std::stringstream ss;
    ss << in_;
    std::string out;
    ss >> std::quoted(out);

    return out;
}

inline bool
hasWildcards(std::string pattern) {
    int64_t l = pattern.length();
    int64_t i = l - 1;
    for (; i >= 0; i--) {
        if (pattern[i] == '%' || pattern[i] == '_') {
            if (i > 0 && pattern[i - 1] == '\\') {
                i--;
                continue;
            }
            return true;
        }
    }
    return false;
}

inline int
findLastNotOfWildcards(std::string pattern) {
    int loc = pattern.length() - 1;
    for (; loc >= 0; loc--) {
        if (pattern[loc] == '%' || pattern[loc] == '_') {
            if (loc > 0 && pattern[loc - 1] == '\\') {
                break;
            }
        } else {
            break;
        }
    }
    return loc;
}

inline std::pair<proto::plan::OpType, std::string>
translatePatternMatch(const std::string& pattern) {
    size_t l = pattern.length();
    size_t loc = findLastNotOfWildcards(pattern);
    if (loc < 0) {
        return std::make_pair(proto::plan::OpType::PrefixMatch, "");
    }
    bool exist = hasWildcards(pattern.substr(0, loc + 1));

    if (loc >= l - 1 && !exist) {
        return std::make_pair(proto::plan::OpType::Equal, pattern);
    }

    if (!exist) {
        return std::make_pair(proto::plan::OpType::PrefixMatch,
                              pattern.substr(0, loc + 1));
    }

    return std::make_pair(proto::plan::OpType::Match, pattern);
}

inline bool
canBeComparedDataType(proto::schema::DataType a, proto::schema::DataType b) {
    switch (a) {
        case proto::schema::DataType::Bool:
            return (b == proto::schema::DataType::Bool) ||
                   (b == proto::schema::DataType::JSON);
        case proto::schema::DataType::Int8:
            return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
        case proto::schema::DataType::Int16:
            return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
        case proto::schema::DataType::Int32:
            return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
        case proto::schema::DataType::Int64:
            return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
        case proto::schema::DataType::Float:
            return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
        case proto::schema::DataType::Double:
            return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
        case proto::schema::DataType::VarChar:
            return b == proto::schema::DataType::String ||
                   b == proto::schema::DataType::VarChar ||
                   b == proto::schema::DataType::JSON;
        case proto::schema::DataType::String:
            return b == proto::schema::DataType::String ||
                   b == proto::schema::DataType::VarChar ||
                   b == proto::schema::DataType::JSON;
        case proto::schema::DataType::JSON:
            return true;
        default:
            return false;
    }
}

inline bool
canBeCompared(ExprWithDtype a, ExprWithDtype b) {
    if (a.dtype != proto::schema::DataType::Array &&
        b.dtype != proto::schema::DataType::Array) {
        return canBeComparedDataType(a.dtype, b.dtype);
    }

    if (a.dtype == proto::schema::DataType::Array &&
        b.dtype == proto::schema::DataType::Array) {
        return canBeComparedDataType(getArrayElementType(a.expr),
                                     getArrayElementType(b.expr));
    }

    if (a.dtype == proto::schema::DataType::Array) {
        return canBeComparedDataType(getArrayElementType(a.expr), b.dtype);
    }

    return canBeComparedDataType(b.dtype, getArrayElementType(b.expr));
}

inline ExprWithDtype
toValueExpr(proto::plan::GenericValue* value,
            google::protobuf::Arena* arena = nullptr) {
    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena);

    auto value_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::ValueExpr>(arena);
    value_expr->unsafe_arena_set_allocated_value(value);

    expr->unsafe_arena_set_allocated_value_expr(value_expr);
    if (value->has_bool_val()) {
        return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
    }
    if (value->has_int64_val()) {
        return ExprWithDtype(expr, proto::schema::DataType::Int64, false);
    }
    if (value->has_float_val()) {
        return ExprWithDtype(expr, proto::schema::DataType::Float, false);
    }
    if (value->has_string_val()) {
        return ExprWithDtype(expr, proto::schema::DataType::VarChar, false);
    }
    if (value->has_array_val()) {
        return ExprWithDtype(expr, proto::schema::DataType::Array, false);
    }
    TRY_WITH_EXCEPTION(false);
}

inline proto::plan::GenericValue*
castValue(proto::schema::DataType dtype,
          proto::plan::GenericValue* value,
          google::protobuf::Arena* arena = nullptr) {
    if (dtype == proto::schema::DataType::JSON)
        return CreateMessageWithCopy<proto::plan::GenericValue>(arena, *value);
    if (dtype == proto::schema::DataType::Array && value->has_array_val())
        return CreateMessageWithCopy<proto::plan::GenericValue>(arena, *value);
    if (dtype == proto::schema::DataType::VarChar && value->has_string_val())
        return CreateMessageWithCopy<proto::plan::GenericValue>(arena, *value);

    if (dtype == proto::schema::DataType::Bool && value->has_bool_val())
        return CreateMessageWithCopy<proto::plan::GenericValue>(arena, *value);

    if (dtype == proto::schema::DataType::Float ||
        dtype == proto::schema::DataType::Double) {
        if (value->has_float_val())
            return CreateMessageWithCopy<proto::plan::GenericValue>(arena,
                                                                    *value);
        ;
        if (value->has_int64_val()) {
            auto value_tmp = google::protobuf::Arena::CreateMessage<
                proto::plan::GenericValue>(arena);
            value_tmp->set_float_val(double(value->int64_val()));
            return value_tmp;
        }
    }

    if (dtype == proto::schema::DataType::Int8 ||
        dtype == proto::schema::DataType::Int16 ||
        dtype == proto::schema::DataType::Int32 ||
        dtype == proto::schema::DataType::Int64) {
        if (value->has_int64_val())
            return CreateMessageWithCopy<proto::plan::GenericValue>(arena,
                                                                    *value);
    }

    TRY_WITH_EXCEPTION(false);
}

inline proto::plan::Expr*
combineArrayLengthExpr(proto::plan::OpType op,
                       proto::plan::ArithOpType arith_op,
                       const proto::plan::ColumnInfo& info,
                       const proto::plan::GenericValue& value,
                       google::protobuf::Arena* arena = nullptr) {
    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena);
    auto range_expr = google::protobuf::Arena::CreateMessage<
        proto::plan::BinaryArithOpEvalRangeExpr>(arena);
    expr->unsafe_arena_set_allocated_binary_arith_op_eval_range_expr(
        range_expr);
    range_expr->set_op(op);
    range_expr->set_arith_op(arith_op);
    range_expr->unsafe_arena_set_allocated_value(
        CreateMessageWithCopy<proto::plan::GenericValue>(arena, value));
    range_expr->unsafe_arena_set_allocated_column_info(
        CreateMessageWithCopy<proto::plan::ColumnInfo>(arena, info));
    return expr;
}

inline proto::plan::Expr*
combineBinaryArithExpr(proto::plan::OpType op,
                       proto::plan::ArithOpType arith_op,
                       const proto::plan::ColumnInfo& info,
                       const proto::plan::GenericValue& operand,
                       const proto::plan::GenericValue& value,
                       google::protobuf::Arena* arena = nullptr) {
    auto data_type = info.data_type();
    if (data_type != proto::schema::DataType::Array &&
        info.nested_path_size() != 0) {
        data_type = info.element_type();
    }
    auto casted_value = castValue(
        data_type,
        CreateMessageWithCopy<proto::plan::GenericValue>(arena, operand));
    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena);
    auto range_expr = google::protobuf::Arena::CreateMessage<
        proto::plan::BinaryArithOpEvalRangeExpr>(arena);
    expr->unsafe_arena_set_allocated_binary_arith_op_eval_range_expr(
        range_expr);
    range_expr->unsafe_arena_set_allocated_column_info(
        CreateMessageWithCopy<proto::plan::ColumnInfo>(arena, info)

    );
    range_expr->set_arith_op(arith_op);
    range_expr->unsafe_arena_set_allocated_right_operand(casted_value);
    range_expr->unsafe_arena_set_allocated_value(
        CreateMessageWithCopy<proto::plan::GenericValue>(arena, value));
    range_expr->set_op(op);

    return expr;
}

inline proto::plan::Expr*
handleBinaryArithExpr(proto::plan::OpType op,
                      proto::plan::BinaryArithExpr* arith_expr,
                      proto::plan::ValueExpr* value_expr,
                      google::protobuf::Arena* arena = nullptr) {
    switch (op) {
        case proto::plan::OpType::Equal:
            break;
        case proto::plan::OpType::NotEqual:
            break;
        default:
            TRY_WITH_EXCEPTION(false);
    }

    auto left_expr = arith_expr->left().column_expr();
    auto left_value = arith_expr->left().value_expr();
    auto right_expr = arith_expr->right().column_expr();
    auto right_value = arith_expr->right().value_expr();

    auto arith_op = arith_expr->op();

    if (arith_op == proto::plan::ArithOpType::ArrayLength) {
        return combineArrayLengthExpr(
            op, arith_op, left_expr.info(), value_expr->value(), arena);
    }
    if (arith_expr->left().has_column_expr() &&
        arith_expr->right().has_column_expr()) {
        TRY_WITH_EXCEPTION(false);
    }
    if (arith_expr->left().has_value_expr() &&
        arith_expr->right().has_value_expr()) {
        TRY_WITH_EXCEPTION(false);
    }
    if (arith_expr->left().has_column_expr() &&
        arith_expr->right().has_value_expr()) {
        return combineBinaryArithExpr(op,
                                      arith_op,
                                      left_expr.info(),
                                      right_value.value(),
                                      value_expr->value(),
                                      arena);
    }
    if (arith_expr->right().has_column_expr() &&
        arith_expr->left().has_value_expr()) {
        switch (arith_expr->op()) {
            case proto::plan::ArithOpType::Add:
                return combineBinaryArithExpr(op,
                                              arith_op,
                                              right_expr.info(),
                                              left_value.value(),
                                              value_expr->value(),
                                              arena);
            case proto::plan::ArithOpType::Mul:
                return combineBinaryArithExpr(op,
                                              arith_op,
                                              right_expr.info(),
                                              left_value.value(),
                                              value_expr->value(),
                                              arena);
            default:
                TRY_WITH_EXCEPTION(false);
        }
    }
    TRY_WITH_EXCEPTION(false);
}

inline proto::plan::Expr*
handleCompareRightValue(proto::plan::OpType op,
                        ExprWithDtype a,
                        ExprWithDtype b,
                        google::protobuf::Arena* arena = nullptr) {
    auto data_type = a.dtype;
    if (data_type == proto::schema::DataType::Array &&
        a.expr->column_expr().info().nested_path_size() != 0) {
        data_type = a.expr->column_expr().info().element_type();
    }
    auto value = b.expr->value_expr().value();
    auto castedvalue = castValue(data_type, &value, arena);
    if (a.expr->has_binary_arith_expr()) {
        auto value_expr =
            google::protobuf::Arena::CreateMessage<proto::plan::ValueExpr>(
                arena);
        value_expr->unsafe_arena_set_allocated_value(castedvalue);
        return handleBinaryArithExpr(
            op,
            CreateMessageWithCopy<proto::plan::BinaryArithExpr>(
                arena, a.expr->binary_arith_expr()),
            value_expr,
            arena);
    }

    TRY_WITH_EXCEPTION(a.expr->has_column_expr());
    auto info = a.expr->column_expr().info();

    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena);
    auto unary_range_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::UnaryRangeExpr>(
            arena);

    unary_range_expr->set_op(op);
    unary_range_expr->unsafe_arena_set_allocated_column_info(
        CreateMessageWithCopy<proto::plan::ColumnInfo>(arena, info));

    unary_range_expr->unsafe_arena_set_allocated_value(castedvalue);

    expr->unsafe_arena_set_allocated_unary_range_expr(unary_range_expr);

    return expr;
}

inline proto::plan::Expr*
HandleCompare(proto::plan::OpType op,
              ExprWithDtype a,
              ExprWithDtype b,
              google::protobuf::Arena* arena = nullptr) {
    TRY_WITH_EXCEPTION(a.expr->has_column_expr() && b.expr->has_column_expr());

    auto a_info = a.expr->column_expr().info();

    auto b_info = b.expr->column_expr().info();

    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena);
    auto compare_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::CompareExpr>(arena);
    compare_expr->unsafe_arena_set_allocated_left_column_info(
        CreateMessageWithCopy<proto::plan::ColumnInfo>(arena, a_info));
    compare_expr->unsafe_arena_set_allocated_right_column_info(
        CreateMessageWithCopy<proto::plan::ColumnInfo>(arena, b_info));
    compare_expr->set_op(op);
    expr->unsafe_arena_set_allocated_compare_expr(compare_expr);

    return expr;
}

inline proto::plan::OpType
reverseOrder(proto::plan::OpType op) {
    switch (op) {
        case proto::plan::OpType::LessThan:
            return proto::plan::OpType::GreaterThan;
        case proto::plan::OpType::LessEqual:
            return proto::plan::OpType::GreaterEqual;
        case proto::plan::OpType::GreaterThan:
            return proto::plan::OpType::LessThan;
        case proto::plan::OpType::GreaterEqual:
            return proto::plan::OpType::LessEqual;
        case proto::plan::OpType::Equal:
            return proto::plan::OpType::NotEqual;
        case proto::plan::OpType::NotEqual:
            return proto::plan::OpType::Equal;
        default:
            return proto::plan::OpType::Invalid;
    }
}

inline proto::plan::Expr*
HandleCompare(int op,
              ExprWithDtype a,
              ExprWithDtype b,
              google::protobuf::Arena* arena = nullptr) {
    TRY_WITH_EXCEPTION(canBeCompared(a, b));
    std::map<int, proto::plan::OpType> cmpOpMap{
        {PlanParser::LT, proto::plan::OpType::LessThan},
        {PlanParser::LE, proto::plan::OpType::LessEqual},
        {PlanParser::GT, proto::plan::OpType::GreaterThan},
        {PlanParser::GE, proto::plan::OpType::GreaterEqual},
        {PlanParser::EQ, proto::plan::OpType::Equal},
        {PlanParser::NE, proto::plan::OpType::NotEqual}};
    auto cmpop = cmpOpMap[op];
    if (a.expr->has_value_expr()) {
        auto op = reverseOrder(cmpop);
        return handleCompareRightValue(op, b, a, arena);
    } else if (b.expr->has_value_expr()) {
        return handleCompareRightValue(cmpop, a, b, arena);
    }

    return HandleCompare(cmpop, a, b, arena);
}

inline bool
checkDirectComparisonBinaryField(proto::plan::ColumnInfo* info) {
    if (info->data_type() == proto::schema::DataType::Array &&
        info->nested_path_size() == 0) {
        return false;
    }
    return true;
}
}  // namespace milvus::local
