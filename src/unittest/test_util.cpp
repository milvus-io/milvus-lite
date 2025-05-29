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

#include "test_util.h"
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common.h"
#include "log/Log.h"
#include "type.h"
#include "pb/milvus.pb.h"
#include "pb/msg.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"

namespace milvus::local {
namespace test {

std::string
CreateCollection(const std::string& collection_name,
                 const std::string& vec_type) {
    ::milvus::proto::schema::CollectionSchema schema;

    schema.set_name(collection_name);
    schema.set_enable_dynamic_field(false);
    auto field1 = schema.add_fields();
    field1->set_fieldid(PK_ID);
    field1->set_name(PK_NAME);
    field1->set_is_primary_key(true);
    field1->set_data_type(::milvus::proto::schema::DataType::Int64);

    if (vec_type == "float_vector") {
        auto field2 = schema.add_fields();
        field2->set_fieldid(VEC_ID);
        field2->set_name(VEC_NAME);

        field2->set_data_type(::milvus::proto::schema::DataType::FloatVector);
        auto params = field2->add_type_params();
        params->set_key(VEC_DIM_NAME);
        params->set_value(std::to_string(VEC_DIM));
        auto field3 = schema.add_fields();
        field3->set_fieldid(SCALAR_ID);
        field3->set_name(SCALAR_NAME);
        field3->set_data_type(::milvus::proto::schema::DataType::Int32);
        return schema.SerializeAsString();

    } else {
        auto field2 = schema.add_fields();
        field2->set_fieldid(VEC_ID);
        field2->set_name(SPARSE_VEC);
        field2->set_data_type(
            ::milvus::proto::schema::DataType::SparseFloatVector);
        auto field3 = schema.add_fields();
        field3->set_fieldid(SCALAR_ID);
        field3->set_name(VARCHAR_SCALAR_NAME);
        field3->set_data_type(::milvus::proto::schema::DataType::VarChar);
        auto p1 = field3->add_type_params();
        p1->set_key("enable_analyzer");
        p1->set_value("true");
        auto p2 = field3->add_type_params();
        p2->set_key("max_length");
        p2->set_value("1000");
        return schema.SerializeAsString();
    }
}

std::string
CreateBM25Function(const std::string& name,
                   const std::string& input_name,
                   const std::string& output_name) {
    ::milvus::proto::schema::FunctionSchema fschema;
    fschema.set_name(name);
    fschema.add_input_field_names(input_name);
    fschema.add_output_field_names(output_name);
    fschema.set_type(::milvus::proto::schema::FunctionType::BM25);
    return fschema.SerializeAsString();
}

std::string
CreateVectorIndex() {
    milvus::proto::segcore::CollectionIndexMeta index_meta;
    index_meta.set_maxindexrowcount(1000000);
    auto field_meta = index_meta.add_index_metas();
    field_meta->set_index_name("vec_index");
    field_meta->set_fieldid(VEC_ID);
    field_meta->set_collectionid(0);
    field_meta->set_is_auto_index(true);

    auto pair = field_meta->add_index_params();
    pair->set_key("metric_type");
    pair->set_value("IP");

    return index_meta.SerializeAsString();
}

milvus::local::Rows
CreateData(int32_t count) {
    milvus::local::Rows rs;
    for (int64_t i = 0; i < count; i++) {
        ::milvus::proto::segcore::InsertRecord r;
        r.set_num_rows(1);

        // set pk
        ::milvus::proto::schema::FieldData* pk = r.add_fields_data();
        pk->set_field_id(PK_ID);
        pk->set_field_name(PK_NAME);
        pk->set_type(::milvus::proto::schema::Int64);
        pk->mutable_scalars()->mutable_long_data()->add_data(i);

        // set vec
        ::milvus::proto::schema::FieldData* vec = r.add_fields_data();
        vec->set_field_id(VEC_ID);
        vec->set_field_name(VEC_NAME);
        vec->set_type(::milvus::proto::schema::FloatVector);
        auto v = vec->mutable_vectors();
        v->set_dim(VEC_DIM);
        auto vd = v->mutable_float_vector();
        vd->add_data(0.1);
        vd->add_data(0.5);
        vd->add_data(0.4);

        // set scalar
        ::milvus::proto::schema::FieldData* sc = r.add_fields_data();
        sc->set_field_id(SCALAR_ID);
        sc->set_field_name(SCALAR_NAME);
        sc->set_type(::milvus::proto::schema::Int32);
        sc->mutable_scalars()->mutable_int_data()->add_data(i);
        rs.push_back(std::make_tuple(std::to_string(i), r.SerializeAsString()));
    }
    return rs;
}

::milvus::proto::milvus::CreateCollectionRequest
GetCreateCollectionRequestProto(const std::string& collection_name) {
    ::milvus::proto::milvus::CreateCollectionRequest r;
    r.set_collection_name(collection_name);
    auto schema_str = CreateCollection(collection_name);
    r.set_schema(schema_str.data());
    r.set_consistency_level(::milvus::proto::common::ConsistencyLevel::Strong);
    return r;
}

::milvus::proto::milvus::LoadCollectionRequest
GetLoadCollectionRequestProto(const std::string& collection_name) {
    ::milvus::proto::milvus::LoadCollectionRequest r;
    r.set_collection_name(collection_name);
    return r;
}

::milvus::proto::milvus::CreateIndexRequest
GetCreateIndexRequestProto(const std::string& collection_name,
                           const std::string& index_name,
                           const std::string& field_name) {
    ::milvus::proto::milvus::CreateIndexRequest r;
    r.set_index_name(index_name);
    r.set_collection_name(collection_name);
    r.set_field_name(field_name);

    auto p1 = r.add_extra_params();
    p1->set_key("params");
    p1->set_value("{}");
    auto p2 = r.add_extra_params();
    p2->set_key("metric_type");
    p2->set_value("IP");

    auto p3 = r.add_extra_params();
    p3->set_key("index_type");
    p3->set_value("AUTOINDEX");

    return r;
}

::milvus::proto::milvus::InsertRequest
GetInsertRequestProto(const std::string& collection_name, int64_t row_num) {
    ::milvus::proto::milvus::InsertRequest r;
    r.set_collection_name(collection_name);
    r.set_num_rows(row_num);

    // set pk
    ::milvus::proto::schema::FieldData* pk = r.add_fields_data();
    pk->set_field_id(PK_ID);
    pk->set_field_name(PK_NAME);
    pk->set_type(::milvus::proto::schema::Int64);
    for (int64_t i = 0; i < row_num; ++i) {
        pk->mutable_scalars()->mutable_long_data()->add_data(i);
    }

    // set vec
    ::milvus::proto::schema::FieldData* vec = r.add_fields_data();
    vec->set_field_id(VEC_ID);
    vec->set_field_name(VEC_NAME);
    vec->set_type(::milvus::proto::schema::FloatVector);
    auto v = vec->mutable_vectors();
    v->set_dim(VEC_DIM);
    auto vd = v->mutable_float_vector();
    for (int64_t i = 0; i < row_num; ++i) {
        vd->add_data(0.1 * i);
        vd->add_data(0.5 * i);
        vd->add_data(0.4 * i);
    }

    // set scalar
    ::milvus::proto::schema::FieldData* sc = r.add_fields_data();
    sc->set_field_id(SCALAR_ID);
    sc->set_field_name(SCALAR_NAME);
    sc->set_type(::milvus::proto::schema::Int32);
    for (int64_t i = 0; i < row_num; ++i) {
        sc->mutable_scalars()->mutable_int_data()->add_data(i);
    }

    return r;
}

::milvus::proto::milvus::UpsertRequest
GetUpsertRequestProto(const std::string& collection_name, int64_t row_num) {
    ::milvus::proto::milvus::UpsertRequest r;
    r.set_collection_name(collection_name);
    r.set_num_rows(row_num);

    // set pk
    ::milvus::proto::schema::FieldData* pk = r.add_fields_data();
    pk->set_field_id(PK_ID);
    pk->set_field_name(PK_NAME);
    pk->set_type(::milvus::proto::schema::Int64);
    for (int64_t i = 0; i < row_num; ++i) {
        pk->mutable_scalars()->mutable_long_data()->add_data(i);
    }

    // set vec
    ::milvus::proto::schema::FieldData* vec = r.add_fields_data();
    vec->set_field_id(VEC_ID);
    vec->set_field_name(VEC_NAME);
    vec->set_type(::milvus::proto::schema::FloatVector);
    auto v = vec->mutable_vectors();
    v->set_dim(VEC_DIM);
    auto vd = v->mutable_float_vector();
    for (int64_t i = 0; i < row_num; ++i) {
        vd->add_data(0.1 * i);
        vd->add_data(0.5 * i);
        vd->add_data(0.4 * i);
    }

    // set scalar
    ::milvus::proto::schema::FieldData* sc = r.add_fields_data();
    sc->set_field_id(SCALAR_ID);
    sc->set_field_name(SCALAR_NAME);
    sc->set_type(::milvus::proto::schema::Int32);
    for (int64_t i = 0; i < row_num; ++i) {
        sc->mutable_scalars()->mutable_int_data()->add_data(i);
    }

    return r;
}

::milvus::proto::milvus::SearchRequest
GetSearchRequestProto(const std::string& collection_name,
                      const std::string& expr,
                      const std::vector<std::vector<float>>& vecs,
                      const std::string& topk,
                      const std::string& metric_type,
                      const std::string& offset) {
    ::milvus::proto::milvus::SearchRequest r;
    r.set_collection_name(collection_name);
    r.set_dsl(expr);
    r.set_dsl_type(::milvus::proto::common::DslType::BoolExprV1);
    r.mutable_output_fields()->Add(SCALAR_NAME);
    r.mutable_output_fields()->Add(VEC_NAME);

    milvus::proto::common::PlaceholderGroup raw_group;
    auto value = raw_group.add_placeholders();
    value->set_tag(milvus::local::kPlaceholderTag);
    value->set_type(milvus::proto::common::PlaceholderType::FloatVector);
    for (const auto& vec : vecs) {
        value->add_values(vec.data(), vec.size() * sizeof(float));
    }

    r.set_placeholder_group(raw_group.SerializeAsString());
    r.set_nq(vecs.size());

    auto p1 = r.mutable_search_params()->Add();
    // p1->set_key("search_param");
    // p1->set_value("{\"nprobe\":10}");
    p1->set_key("params");
    p1->set_value("{\"nprobe\":10}");

    auto p2 = r.mutable_search_params()->Add();
    p2->set_key("round_decimal");
    p2->set_value("-1");

    auto p3 = r.mutable_search_params()->Add();
    p3->set_key("ignore_growing");
    p3->set_value("False");

    auto p4 = r.mutable_search_params()->Add();
    p4->set_key("topk");
    p4->set_value(topk);

    auto p5 = r.mutable_search_params()->Add();
    p5->set_key("metric_type");
    p5->set_value(metric_type);

    auto p6 = r.mutable_search_params()->Add();
    p6->set_key("offset");
    p6->set_value(offset);
    return r;
}

::milvus::proto::milvus::QueryRequest
GetQueryRequestProto(const std::string& collection_name,
                     const std::string& expr,
                     const std::string& limit,
                     const std::string& offset,
                     const std::vector<std::string>& output_fields) {
    ::milvus::proto::milvus::QueryRequest r;
    r.set_collection_name(collection_name);
    r.set_expr(expr);
    auto p1 = r.mutable_query_params()->Add();
    p1->set_key("limit");
    p1->set_value(limit);

    auto p2 = r.mutable_query_params()->Add();
    p2->set_key("reduce_stop_for_best");
    p2->set_value("False");

    auto p3 = r.mutable_query_params()->Add();
    p3->set_key("ignore_growing");
    p3->set_value("False");

    auto p4 = r.mutable_query_params()->Add();
    p4->set_key("offset");
    p4->set_value(offset);
    r.set_guarantee_timestamp(1);
    r.set_use_default_consistency(true);
    for (const auto& f : output_fields) {
        r.add_output_fields(f);
    }
    return r;
}

::milvus::proto::milvus::DeleteRequest
GetDeleteRequestProto(const std::string& collection_name,
                      const std::string& expr) {
    ::milvus::proto::milvus::DeleteRequest r;
    r.set_collection_name(collection_name);
    r.set_expr(expr);
    return r;
}

::milvus::proto::milvus::DescribeCollectionRequest
GetDescribeCollectionRequest(const std::string& collection_name) {
    ::milvus::proto::milvus::DescribeCollectionRequest r;
    r.set_collection_name(collection_name);
    return r;
}

}  // namespace test
}  // namespace milvus::local
