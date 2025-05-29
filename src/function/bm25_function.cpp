// Copyright (C) 2019-2025 Zilliz. All rights reserved.
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

#include "function/bm25_function.h"
#include "common.h"
#include "schema_util.h"
#include "common.pb.h"
#include "schema.pb.h"
#include "status.h"

#include <zlib.h>
#include <utility>
#include <vector>

namespace milvus::local::function {

using milvus::local::schema_util::SparseVectorToByte;

constexpr int substringLengthForCRC = 100;

Status
BM25Function::ProcessInsert(
    const std::vector<milvus::proto::schema::FieldData*>& inputs,
    std::vector<milvus::proto::schema::FieldData>* outputs) {
    if (inputs.size() != 1) {
        return Status::ParameterInvalid("BM25Function inputs size must be 1");
    }

    if (inputs[0]->type() != milvus::proto::schema::DataType::VarChar) {
        return Status::ParameterInvalid(
            "BM25Function input type must be VarChar");
    }

    if (!inputs[0]->scalars().has_string_data()) {
        return Status::ParameterInvalid(
            "BM25Function input data is not string");
    }

    auto docs = inputs[0]->scalars().string_data();

    auto tk = this->tokenizer_.Clone();
    milvus::proto::schema::FieldData sparse_field;
    sparse_field.set_type(::milvus::proto::schema::DataType::SparseFloatVector);
    sparse_field.set_field_name(meta_->output_fields_[0]->name());
    auto sparse_array =
        sparse_field.mutable_vectors()->mutable_sparse_float_vector();
    sparse_array->set_dim(0);
    for (int i = 0; i < docs.data_size(); i++) {
        auto&& doc = docs.data(i);
        SparseVector embd;
        Embedding(doc, &embd);
        //default little-endian macOS\Linux
        auto buf = sparse_array->add_contents();
        int dim = embd.size();
        SparseVectorToByte(embd, buf);
        if (sparse_array->dim() < dim)
            sparse_array->set_dim(dim);
    }
    outputs->push_back(sparse_field);
    return Status::Ok();
}

Status
BM25Function::ProcessSearch(
    const milvus::proto::common::PlaceholderGroup& input_plg,
    milvus::proto::common::PlaceholderGroup* output_plg) {
    auto& ph = input_plg.placeholders(0);
    if (ph.type() != milvus::proto::common::PlaceholderType::VarChar) {
        return Status::ParameterInvalid(
            "BM25 function error, the input is not varch");
    }

    std::vector<SparseVector> embds;
    for (const auto& doc : ph.values()) {
        SparseVector embd;
        Embedding(doc, &embd);
        embds.emplace_back(std::move(embd));
    }

    auto plg_value = output_plg->add_placeholders();
    plg_value->set_tag(kPlaceholderTag);
    plg_value->set_type(
        ::milvus::proto::common::PlaceholderType::SparseFloatVector);

    for (auto&& embd : embds) {
        std::string buf;
        SparseVectorToByte(embd, &buf);
        plg_value->add_values(buf);
    }
    return Status::Ok();
}

Status
BM25Function::Embedding(const std::string& doc, SparseVector* embd) {
    auto tk = this->tokenizer_.Clone();
    auto tokenstream = tk->CreateTokenStream(std::string(doc));
    while (tokenstream->advance()) {
        auto token = tokenstream->get_token();
        if (token.size() > substringLengthForCRC) {
            token = token.substr(0, substringLengthForCRC);
        };
        uint32_t crc =
            crc32(0,
                  reinterpret_cast<const unsigned char*>(token.c_str()),
                  token.length());
        (*embd)[crc % UINT32_MAX] += 1;
    }
    return Status::Ok();
}

}  // namespace milvus::local::function
