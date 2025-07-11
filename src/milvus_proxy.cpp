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

#include "milvus_proxy.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "common.h"
#include "common/Types.h"
#include "log/Log.h"
#include "create_collection_task.h"
#include "create_index_task.h"
#include "delete_task.h"
#include "insert_task.h"
#include "milvus_id.hpp"
#include "milvus_local.h"
#include "pb/common.pb.h"
#include "pb/milvus.pb.h"
#include "pb/plan.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "query_task.h"
#include "retrieve_result.h"
#include "hybrid_search_task.h"
#include "schema_util.h"
#include "search_result.h"
#include "search_task.h"
#include "status.h"
#include "string_util.hpp"
#include "timer.h"
#include "type.h"
#include "upsert_task.h"
#include <cmath>
#include "nlohmann/json.hpp"

namespace milvus::local {

MilvusProxy::MilvusProxy(const char* work_dir) : milvus_local_(work_dir) {
}

MilvusProxy::~MilvusProxy() {
}

bool
MilvusProxy::Init() {
    return milvus_local_.Init();
}

Status
MilvusProxy::LoadCollection(const std::string& collection_name) {
    return milvus_local_.LoadCollection(collection_name);
}

Status
MilvusProxy::GetLoadState(
    const std::string& collection_name,
    ::milvus::proto::milvus::GetLoadStateResponse* response) {
    auto s = milvus_local_.GetLoadState(collection_name);
    if (s.Code() == ErrCollectionNotFound) {
        response->set_state(
            ::milvus::proto::common::LoadState::LoadStateNotExist);
        return Status::Ok();
    }

    if (s.Code() == ErrCollectionNotLoaded) {
        response->set_state(
            ::milvus::proto::common::LoadState::LoadStateNotLoad);
        return Status::Ok();
    }

    if (s.Code() == ErrCollectionLoaded) {
        response->set_state(
            ::milvus::proto::common::LoadState::LoadStateLoaded);
        return Status::Ok();
    }
    return s;
}

Status
MilvusProxy::ReleaseCollection(const std::string& collection_name) {
    // Alignment error code with milvus
    auto s = milvus_local_.ReleaseCollection(collection_name);
    if (s.Code() == ErrCollectionNotFound) {
        return Status::ParameterInvalid("collection not found[collection={}]",
                                        collection_name);
    }
    return s;
}

Status
MilvusProxy::HasCollection(const std::string& collection_name,
                           ::milvus::proto::milvus::BoolResponse* response) {
    std::string tmp;
    Status status = milvus_local_.GetCollection(collection_name, &tmp);
    response->set_value(status.IsOk());
    // If the collection does not exist, change Status code to ok, and the response bool value determines if it noexists.
    if (status.Code() == ErrCode::ErrCollectionNotFound) {
        LOG_INFO("HasCollection change error code from collection-not-found to ok.");
        status = Status::Ok();
    }
    return status;
}

Status
MilvusProxy::CreateCollection(
    const ::milvus::proto::milvus::CreateCollectionRequest* r) {
    ::milvus::proto::schema::CollectionSchema schema;

    CHECK_STATUS(CreateCollectionTask(r).Process(&schema), "");

    for (const auto& field : schema.fields()) {
        if (field.is_primary_key()) {
            return milvus_local_.CreateCollection(
                schema.name(), field.name(), schema.SerializeAsString());
        }
    }
    return Status::FieldNotFound("Lost primary key field");
}

Status
MilvusProxy::CreateIndex(const ::milvus::proto::milvus::CreateIndexRequest* r) {
    ::milvus::proto::schema::CollectionSchema schema;
    if (!GetSchemaInfo(r->collection_name(), &schema).IsOk()) {
        auto err = string_util::SFormat("Can not find collection {}",
                                        r->collection_name());
        LOG_ERROR(err);
        return Status::CollectionNotFound(err);
    }
    // get all index
    milvus::proto::segcore::FieldIndexMeta field_meta;
    CHECK_STATUS(CreateIndexTask(r, &schema).Process(&field_meta), "");
    return milvus_local_.CreateIndex(r->collection_name(),
                                     field_meta.index_name(),
                                     field_meta.SerializeAsString());
}

Status
MilvusProxy::Insert(const ::milvus::proto::milvus::InsertRequest* r,
                    ::milvus::proto::schema::IDs* ids) {
    ::milvus::proto::schema::CollectionSchema schema;
    if (!GetSchemaInfo(r->collection_name(), &schema).IsOk()) {
        auto err = string_util::SFormat("Collection {} not found",
                                        r->collection_name());
        return Status::CollectionNotFound();
    }

    CHECK_STATUS(milvus_local_.LoadCollection(r->collection_name()), "");
    Rows rows;
    auto insert_task = InsertTask(
        const_cast<::milvus::proto::milvus::InsertRequest*>(r), &schema);
    CHECK_STATUS(insert_task.Process(&rows), "");
    auto bm25_fields = schema_util::GetBM25SparseField(schema);
    std::vector<std::string> insert_ids;
    milvus_local_.Insert(r->collection_name(), rows, &insert_ids, bm25_fields);

    if (insert_task.PkType() == ::milvus::proto::schema::DataType::Int64) {
        for (const auto& id : insert_ids) {
            ids->mutable_int_id()->add_data(std::stoll(id));
        }
    } else {
        for (const auto& id : insert_ids) {
            ids->mutable_str_id()->add_data(id);
        }
    }
    return Status::Ok();
}

Status
MilvusProxy::Upsert(const ::milvus::proto::milvus::UpsertRequest* r,
                    ::milvus::proto::schema::IDs* ids) {
    ::milvus::proto::schema::CollectionSchema schema;
    if (!GetSchemaInfo(r->collection_name(), &schema).IsOk()) {
        auto err = string_util::SFormat("Collection {} not found",
                                        r->collection_name());
        return Status::CollectionNotFound();
    }
    CHECK_STATUS(milvus_local_.LoadCollection(r->collection_name()), "");
    auto task = UpsertTask(r, &schema);
    ::milvus::proto::milvus::InsertRequest insert_q;
    ::milvus::proto::schema::IDs delete_ids;
    CHECK_STATUS(task.Process(&insert_q, &delete_ids), "");
    std::vector<std::string> storage_ids;
    if (delete_ids.has_int_id()) {
        for (const auto& id : delete_ids.int_id().data()) {
            storage_ids.push_back(std::to_string(id));
        }

    } else {
        for (const auto& id : delete_ids.str_id().data()) {
            storage_ids.push_back(id);
        }
    }
    auto bm25_fields = schema_util::GetBM25SparseField(schema);
    CHECK_STATUS(milvus_local_.DeleteByIds(r->collection_name(),
                                           delete_ids.SerializeAsString(),
                                           storage_ids.size(),
                                           storage_ids,
                                           bm25_fields),
                 "");
    return Insert(&insert_q, ids);
}

Status
MilvusProxy::Search(const ::milvus::proto::milvus::SearchRequest* r,
                    ::milvus::proto::milvus::SearchResults* search_result) {
    ::milvus::proto::schema::CollectionSchema schema;
    if (!GetSchemaInfo(r->collection_name(), &schema).IsOk()) {
        auto err = string_util::SFormat("Can not find {}'s schema",
                                        r->collection_name());
        LOG_ERROR(err);
        return Status::CollectionNotFound(err);
    }

    CHECK_STATUS(milvus_local_.LoadCollection(r->collection_name()), "");

    // get index
    std::vector<std::string> all_index;
    CHECK_STATUS(milvus_local_.GetAllIndexs(r->collection_name(), &all_index),
                 "");
    return DoSearch(r, schema, all_index, search_result);
}

Status
MilvusProxy::DoSearch(const ::milvus::proto::milvus::SearchRequest* r,
                      const ::milvus::proto::schema::CollectionSchema& schema,
                      const std::vector<std::string>& all_index,
                      ::milvus::proto::milvus::SearchResults* search_result) {
    std::string placeholder_group;
    ::milvus::proto::plan::PlanNode plan;
    std::vector<int64_t> nqs, topks;

    SearchTask task(const_cast<::milvus::proto::milvus::SearchRequest*>(r),
                    &schema,
                    &all_index);
    CHECK_STATUS(task.Process(&plan, &placeholder_group, &nqs, &topks), "");
    if (task.GetMetric() == kMetricsBM25Name) {
        auto [token_count, doc_count] = milvus_local_.GetBM25TokenAndDocCount(
            schema.name(), task.GetAnnFieldName());
        milvus::proto::common::PlaceholderGroup ph_group;
        if (!ph_group.ParseFromString(placeholder_group)) {
            return Status::ServiceInternal("Parse placeholder failed");
        }
        for (int i = 0; i < ph_group.placeholders(0).values_size(); i++) {
            auto vec = ph_group.mutable_placeholders(0)->mutable_values(i);
            auto pos = vec->data();
            auto end = vec->data() + vec->size();
            for (; pos < end; pos += 8) {
                const uint32_t token = *(reinterpret_cast<uint32_t*>(pos));
                const float freq = *(reinterpret_cast<float*>(pos + 4));
                auto nq = milvus_local_.GetTokenNQ(
                    schema.name(), task.GetAnnFieldName(), token);
                *(reinterpret_cast<float*>(pos + 4)) =
                    freq * log(1 + (float(doc_count) - float(nq) + 0.5) /
                                       (float(nq) + 0.5));
            }
        }
        placeholder_group = ph_group.SerializeAsString();
        auto params_str = plan.mutable_vector_anns()
                              ->mutable_query_info()
                              ->mutable_search_params();

        auto params = nlohmann::json::parse(params_str->c_str());
        if (token_count == 0 || doc_count == 0) {
            params[kBM25AvgName] = 0;
        } else {
            params[kBM25AvgName] = float(token_count) / float(doc_count);
        }
        params_str->assign(params.dump());
    }
    RecordEvent("ParseProto");
    SearchResult result(nqs, topks);
    CHECK_STATUS(milvus_local_.Search(r->collection_name(),
                                      plan.SerializeAsString(),
                                      placeholder_group,
                                      &result),
                 "");
    search_result->set_collection_name(r->collection_name());
    RecordEvent("DoSearch");
    task.PostProcess(result, search_result);
    RecordEvent("PostProcess");
    // search_result->PrintDebugString();
    if (search_result->results().has_ids()) {
        return Status::Ok();
    } else {
        return Status::Ok("search result is empty");
    }
}

Status
MilvusProxy::HybridSearch(
    const ::milvus::proto::milvus::HybridSearchRequest* r,
    ::milvus::proto::milvus::SearchResults* search_result) {
    ::milvus::proto::schema::CollectionSchema schema;
    if (!GetSchemaInfo(r->collection_name(), &schema).IsOk()) {
        auto err = string_util::SFormat("Can not find {}'s schema",
                                        r->collection_name());
        LOG_ERROR(err);
        return Status::CollectionNotFound(err);
    }
    CHECK_STATUS(milvus_local_.LoadCollection(r->collection_name()), "");

    // get index
    std::vector<std::string> all_index;
    CHECK_STATUS(milvus_local_.GetAllIndexs(r->collection_name(), &all_index),
                 "");
    HyBridSearchTask task(&schema, &all_index);
    std::vector<::milvus::proto::milvus::SearchRequest> search_requests;
    CHECK_STATUS(task.ProcessSearch(r, &search_requests), "");

    std::vector<::milvus::proto::milvus::SearchResults> search_results;
    for (const auto& req : search_requests) {
        ::milvus::proto::milvus::SearchResults ret;
        CHECK_STATUS(DoSearch(&req, schema, all_index, &ret), "");
        search_results.push_back(ret);
    }
    CHECK_STATUS(task.PostProcessSearch(search_results, search_result), "");
    if (schema_util::IDsSize(search_result->results().ids()) == 0) {
        return Status::Ok();
    }
    // requery, get output field data
    ::milvus::proto::milvus::QueryRequest query_req;
    task.ProcessQuery(r, search_result, &query_req);
    ::milvus::proto::plan::PlanNode plan;
    QueryTask qtask(&query_req, &schema);
    CHECK_STATUS(qtask.Process(&plan), "");

    RetrieveResult query_result;
    CHECK_STATUS(
        milvus_local_.Retrieve(
            r->collection_name(), plan.SerializeAsString(), &query_result),
        "");

    return task.PostProcess(query_result, search_result);
}

Status
MilvusProxy::Query(const ::milvus::proto::milvus::QueryRequest* r,
                   ::milvus::proto::milvus::QueryResults* query_result) {
    ::milvus::proto::schema::CollectionSchema schema;
    if (!GetSchemaInfo(r->collection_name(), &schema).IsOk()) {
        return Status::CollectionNotFound();
    }
    CHECK_STATUS(milvus_local_.LoadCollection(r->collection_name()), "");
    ::milvus::proto::plan::PlanNode plan;
    QueryTask task(r, &schema);
    CHECK_STATUS(task.Process(&plan), "");
    RecordEvent("ParseProto");
    RetrieveResult result;
    CHECK_STATUS(milvus_local_.Retrieve(
                     r->collection_name(), plan.SerializeAsString(), &result),
                 "");

    query_result->set_collection_name(r->collection_name());
    RecordEvent("DoQuery");
    task.PostProcess(result, query_result);
    RecordEvent("PostProcess");
    return Status::Ok();
}

Status
MilvusProxy::Delete(const ::milvus::proto::milvus::DeleteRequest* r,
                    ::milvus::proto::milvus::MutationResult* response) {
    ::milvus::proto::schema::CollectionSchema schema;
    if (!GetSchemaInfo(r->collection_name(), &schema).IsOk()) {
        return Status::CollectionNotFound();
    }
    CHECK_STATUS(milvus_local_.LoadCollection(r->collection_name()), "");

    ::milvus::proto::plan::PlanNode plan;
    CHECK_STATUS(DeleteTask(r, &schema).Process(&plan), "");

    RetrieveResult result;
    CHECK_STATUS(milvus_local_.Retrieve(
                     r->collection_name(), plan.SerializeAsString(), &result),
                 "");

    ::milvus::proto::segcore::RetrieveResults seg_result;
    seg_result.ParseFromArray(result.retrieve_result_.proto_blob,
                              result.retrieve_result_.proto_size);
    auto ids_str = seg_result.ids().SerializeAsString();
    std::vector<std::string> storage_ids;
    if (seg_result.ids().has_int_id()) {
        for (const auto& id : seg_result.ids().int_id().data()) {
            storage_ids.push_back(std::to_string(id));
            response->mutable_ids()->mutable_int_id()->add_data(id);
        }

    } else {
        for (const auto& id : seg_result.ids().str_id().data()) {
            storage_ids.push_back(id);
            response->mutable_ids()->mutable_str_id()->add_data(id);
        }
    }
    if (storage_ids.size() != 0) {
        auto bm25_fields = schema_util::GetBM25SparseField(schema);
        CHECK_STATUS(milvus_local_.DeleteByIds(r->collection_name(),
                                               ids_str,
                                               storage_ids.size(),
                                               storage_ids,
                                               bm25_fields),
                     "");
    }
    response->set_delete_cnt(storage_ids.size());
    return Status::Ok();
}

Status
MilvusProxy::GetSchemaInfo(const std::string& collection_name,
                           ::milvus::proto::schema::CollectionSchema* schema) {
    std::string schema_proto;
    CHECK_STATUS(milvus_local_.GetCollection(collection_name, &schema_proto),
                 "");
    if (!schema->ParseFromString(schema_proto)) {
        LOG_ERROR("Failed to parse schema info: {}", schema_proto);
        return Status::ServiceInternal();
        ;
    }
    return Status::Ok();
}

Status
MilvusProxy::DescribeCollection(
    const ::milvus::proto::milvus::DescribeCollectionRequest* request,
    ::milvus::proto::milvus::DescribeCollectionResponse* response) {
    ::milvus::proto::schema::CollectionSchema schema;
    CHECK_STATUS(GetSchemaInfo(request->collection_name(), &schema), "");
    auto mutable_schema = response->mutable_schema();
    mutable_schema->set_name(schema.name());
    mutable_schema->set_description(schema.description());
    mutable_schema->set_enable_dynamic_field(schema.enable_dynamic_field());
    mutable_schema->mutable_properties()->CopyFrom(schema.properties());
    for (const auto& field : schema.fields()) {
        if (field.name() == kTimeStampFieldName ||
            field.name() == kRowIdFieldName || field.is_dynamic())
            continue;
        mutable_schema->add_fields()->CopyFrom(field);
    }
    return Status::Ok();
}

Status
MilvusProxy::GetIndex(
    const std::string& collection_name,
    const std::string& index_name,
    ::milvus::proto::milvus::DescribeIndexResponse* response) {
    ::milvus::proto::schema::CollectionSchema schema;
    if (!GetSchemaInfo(collection_name, &schema).IsOk()) {
        auto err =
            string_util::SFormat("Can not find collection {}", collection_name);
        LOG_ERROR(err);
        return Status::CollectionNotFound(err);
    }

    if (index_name.empty()) {
        std::vector<std::string> all_index;
        CHECK_STATUS(milvus_local_.GetAllIndexs(collection_name, &all_index),
                     "");
        for (const auto& index : all_index) {
            CHECK_STATUS(
                ParseIndex(index, schema, response->add_index_descriptions()),
                "");
        }
        return Status::Ok();
    } else {
        std::string index_proto;
        CHECK_STATUS(
            milvus_local_.GetIndex(collection_name, index_name, &index_proto),
            "");
        return ParseIndex(
            index_proto, schema, response->add_index_descriptions());
    }
}

Status
MilvusProxy::DropIndex(const std::string& collection_name,
                       const std::string& index_name) {
    CHECK_STATUS(milvus_local_.LoadCollection(collection_name), "");
    return milvus_local_.DropIndex(collection_name, index_name);
}

Status
MilvusProxy::ParseIndex(const std::string& index_proto,
                        const ::milvus::proto::schema::CollectionSchema& schema,
                        ::milvus::proto::milvus::IndexDescription* index) {
    milvus::proto::segcore::FieldIndexMeta field_index;
    if (!field_index.ParseFromString(index_proto)) {
        return Status::ServiceInternal("Error index info in db");
    }

    auto field_id = field_index.fieldid();
    for (const auto& field : schema.fields()) {
        if (field.fieldid() == field_id) {
            index->set_field_name(field.name());
        }
    }
    index->set_index_name(field_index.index_name());
    index->set_indexid(GetIndexId(field_index.index_name()));
    index->set_state(::milvus::proto::common::IndexState::Finished);
    for (const auto& param : field_index.type_params()) {
        auto new_param = index->add_params();
        new_param->set_key(param.key());
        new_param->set_value(param.value());
    }

    for (const auto& param : field_index.index_params()) {
        auto new_param = index->add_params();
        new_param->set_key(param.key());
        new_param->set_value(param.value());
    }

    for (const auto& param : field_index.user_index_params()) {
        auto new_param = index->add_params();
        new_param->set_key(param.key());
        new_param->set_value(param.value());
    }
    return Status::Ok();
}

bool
MilvusProxy::DropCollection(const std::string& collection_name) {
    return milvus_local_.DropCollection(collection_name);
}

Status
MilvusProxy::GetCollectionStatistics(
    const std::string& collection_name,
    ::milvus::proto::milvus::GetCollectionStatisticsResponse* r) {
    int64_t num = -1;
    CHECK_STATUS(milvus_local_.GetNumRowsOfCollection(collection_name, &num),
                 "");
    auto s = r->add_stats();
    s->set_key("row_count");
    s->set_value(std::to_string(num));
    return Status::Ok();
}

}  // namespace milvus::local
