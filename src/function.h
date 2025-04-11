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

#include <memory>
#include "pb/schema.pb.h"
#include "pb/common.pb.h"
#include "pb/milvus.pb.h"
#include "tantivy/tokenizer.h"
#include <zlib.h>
#include <limits.h>
namespace milvus {
namespace local {

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

class FuncRunner {
 public:
    virtual void
    InsertConvert(milvus::proto::milvus::InsertRequest* r) = 0;
    virtual std::string
    SearchConvert(const std::string& doc) = 0;
};

constexpr int substringLengthForCRC = 100;
class BM25FuncRunner : public FuncRunner {
 public:
    BM25FuncRunner(std::string&& params,
                   const milvus::proto::schema::FieldSchema in_field,
                   const milvus::proto::schema::FieldSchema out_field)
        : tokenizer_(std::move(params)),
          in_field_(in_field),
          out_field_(out_field) {
    }

    std::string
    SearchConvert(const std::string& doc) override {
        auto tk = this->tokenizer_.Clone();
        auto tokenstream = tk->CreateTokenStream(std::string(doc));
        std::map<uint32_t, float> ebedding;
        while (tokenstream->advance()) {
            auto token = tokenstream->get_token();
            if (token.size() > substringLengthForCRC) {
                token = token.substr(0, substringLengthForCRC);
            };
            uint32_t crc =
                crc32(0,
                      reinterpret_cast<const unsigned char*>(token.c_str()),
                      token.length());
            ebedding[crc % UINT32_MAX] += 1;
        }

        std::string buf;
        for (auto&& k_v : ebedding) {
            uint32_t k = k_v.first;
            float v = k_v.second;
            buf.append(reinterpret_cast<const char*>(&k), sizeof(k));
            buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
        }
        return buf;
    }
    void
    InsertConvert(milvus::proto::milvus::InsertRequest* r) override {
        const milvus::proto::schema::StringArray* in_str_array = NULL;
        for (int i = 0; i < r->fields_data_size(); ++i) {
            if (r->fields_data(i).field_name() == in_field_.name()) {
                TRY_WITH_EXCEPTION(
                    r->fields_data(i).scalars().has_string_data());
                in_str_array = &r->fields_data(i).scalars().string_data();
                break;
            }
        }
        TRY_WITH_EXCEPTION(in_str_array != NULL);
        auto tk = this->tokenizer_.Clone();
        auto add_field = r->add_fields_data();
        add_field->set_field_name(out_field_.name());
        add_field->set_type(milvus::proto::schema::DataType::SparseFloatVector);
        milvus::proto::schema::SparseFloatArray* sparr =
            new milvus::proto::schema::SparseFloatArray();
        add_field->mutable_vectors()->set_allocated_sparse_float_vector(sparr);
        sparr->set_dim(0);

        for (int i = 0; i < in_str_array->data_size(); i++) {
            auto&& doc = in_str_array->data(i);
            auto tokenstream = tk->CreateTokenStream(std::string(doc));

            std::map<uint32_t, float> ebedding;
            while (tokenstream->advance()) {
                auto token = tokenstream->get_token();
                if (token.size() > substringLengthForCRC) {
                    token = token.substr(0, substringLengthForCRC);
                };
                uint32_t crc =
                    crc32(0,
                          reinterpret_cast<const unsigned char*>(token.c_str()),
                          token.length());
                ebedding[crc % UINT32_MAX] += 1;
            }
            //default little-endian macOS\Linux
            auto buf = sparr->add_contents();
            int dim = 0;
            for (auto&& k_v : ebedding) {
                uint32_t k = k_v.first;
                float v = k_v.second;
                buf->append(reinterpret_cast<const char*>(&k), sizeof(k));
                buf->append(reinterpret_cast<const char*>(&v), sizeof(v));
                dim++;
            }
            if (sparr->dim() < dim)
                sparr->set_dim(dim);
        }
    }

 private:
    milvus::tantivy::Tokenizer tokenizer_;
    milvus::proto::schema::FieldSchema in_field_;
    milvus::proto::schema::FieldSchema out_field_;
};

#define ANALYZER_PARAMS "analyzer_params"
std::shared_ptr<FuncRunner>
CreateRunner(const milvus::proto::schema::CollectionSchema& schema) {
    TRY_WITH_EXCEPTION(schema.functions_size() == 1 &&
                       schema.functions(0).input_field_names_size() == 1 &&
                       schema.functions(0).output_field_names_size() == 1);
    milvus::proto::schema::FunctionSchema func = schema.functions(0);
    milvus::proto::schema::FieldSchema in_field;
    milvus::proto::schema::FieldSchema out_field;
    std::string params;
    for (int i = 0; i < schema.fields_size(); ++i) {
        if (schema.fields(i).name() == func.output_field_names(0))
            out_field = schema.fields(i);
        if (schema.fields(i).name() == func.input_field_names(0)) {
            in_field = schema.fields(i);
            for (int j = 0; j < in_field.type_params_size(); ++j) {
                if (in_field.type_params(i).key() == ANALYZER_PARAMS) {
                    params = in_field.type_params(i).value();
                    break;
                }
            }
        }
    }
    if (func.type() == milvus::proto::schema::FunctionType::BM25)
        return std::make_shared<BM25FuncRunner>(
            std::move(params), in_field, out_field);
    return nullptr;
};

}  // namespace local

}  // namespace milvus
