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

#include "segcore_wrapper.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>
#include "common.h"
#include "common/type_c.h"
#include "log/Log.h"
#include "pb/segcore.pb.h"
#include "retrieve_result.h"
#include "segcore/collection_c.h"
#include "segcore/reduce_c.h"
#include "segcore/segment_c.h"
#include "pb/schema.pb.h"
#include "status.h"

namespace milvus::local {

const int64_t DEFAULT_MAX_OUTPUT_SIZE = 67108864;

class RetrievePlanWrapper final : private NonCopyableNonMovable {
 public:
    RetrievePlanWrapper() : plan_(nullptr) {
    }
    virtual ~RetrievePlanWrapper() {
        DELETE_AND_SET_NULL(plan_, DeleteRetrievePlan);
    }

 public:
    CRetrievePlan plan_;
};

class SearchPlanWrapper final : private NonCopyableNonMovable {
 public:
    SearchPlanWrapper() : plan_(nullptr) {
    }
    virtual ~SearchPlanWrapper() {
        DELETE_AND_SET_NULL(plan_, DeleteSearchPlan);
    }

 public:
    CSearchPlan plan_;
};

class PlaceholderGroupWrapper final : private NonCopyableNonMovable {
 public:
    PlaceholderGroupWrapper() : group_(nullptr) {
    }
    virtual ~PlaceholderGroupWrapper() {
        DELETE_AND_SET_NULL(group_, DeletePlaceholderGroup);
    }

 public:
    CPlaceholderGroup group_;
};

class SearchResultWrapper final : private NonCopyableNonMovable {
 public:
    SearchResultWrapper() : ret_(nullptr) {
    }
    virtual ~SearchResultWrapper() {
        DELETE_AND_SET_NULL(ret_, DeleteSearchResult);
    }

 public:
    CSearchResult ret_;
};

SegcoreWrapper::~SegcoreWrapper() {
    if (collection_ != nullptr) {
        try {
            DELETE_AND_SET_NULL(collection_, DeleteCollection);
        } catch (std::exception& e) {
            LOG_ERROR("Release collection {} failed", collection_name_);
        }
    }
    if (segment_ != nullptr) {
        try {
            DELETE_AND_SET_NULL(segment_, DeleteSegment);
        } catch (std::exception& e) {
            LOG_ERROR("Release segment {} failed", collection_name_);
        }
    }
}

Status
SegcoreWrapper::SetCollectionInfo(const std::string& collection_name,
                                  const std::string& collection_info,
                                  const std::string& index_meta) {
    assert(collection_ == nullptr);
    auto new_collection_info = NewCollectionInfo(collection_info);
    try {
        collection_ = ::NewCollection(new_collection_info.c_str(),
                                      new_collection_info.size());
        // set index info if has
        if (!index_meta.empty()) {
            ::SetIndexMeta(collection_, index_meta.c_str(), index_meta.size());
        }
        CHECK_STATUS(
            Status(::NewSegment(collection_, Growing, 0, &segment_, false)),
            "Init segcore failed");
        collection_name_ = collection_name;
        return Status::Ok();
    } catch (std::exception& e) {
        return Status::SegcoreErr(e.what());
    }
}

std::string
SegcoreWrapper::NewCollectionInfo(const std::string& info) {
    ::milvus::proto::schema::CollectionSchema schema;
    schema.ParseFromString(info);
    for (auto it = schema.fields().begin(); it != schema.fields().end();) {
        if (it->fieldid() < kStartOfUserFieldId) {
            schema.mutable_fields()->erase(it);
        } else {
            ++it;
        }
    }
    return schema.SerializeAsString();
}

Status
SegcoreWrapper::SetIndexMeta(const std::string& meta_info) {
    try {
        ::SetIndexMeta(collection_, meta_info.c_str(), meta_info.size());
        return Status::Ok();
    } catch (std::exception& e) {
        LOG_ERROR("Set Index meta failed, err: {}", e.what());
        return Status::SegcoreErr(e.what());
    }
}

Status
SegcoreWrapper::Insert(int64_t size, const std::string& insert_record_proto) {
    try {
        int64_t offset = 0;
        CHECK_STATUS(Status(::PreInsert(segment_, size, &offset)),
                     "Pre insert failed, err:");

        ::milvus::proto::segcore::InsertRecord r;
        r.ParseFromString(insert_record_proto);

        std::vector<int64_t> row_ids;
        std::vector<uint64_t> timestamps;
        for (const auto& field_data : r.fields_data()) {
            if (field_data.field_id() == kRowIdField) {
                for (int64_t rowid : field_data.scalars().long_data().data()) {
                    row_ids.push_back(rowid);
                }
            }
            if (field_data.field_id() == kTimeStampField) {
                for (int64_t ts : field_data.scalars().long_data().data()) {
                    timestamps.push_back(ts);
                }
            }
        }

        CHECK_STATUS(Status(::Insert(segment_,
                                     offset,
                                     size,
                                     row_ids.data(),
                                     timestamps.data(),
                                     reinterpret_cast<const uint8_t*>(
                                         insert_record_proto.data()),
                                     insert_record_proto.size())),
                     "Insert failed:");
        return Status::Ok();
    } catch (std::exception& e) {
        LOG_ERROR("Insert failed, err: {}", e.what());
        return Status::SegcoreErr(e.what());
    }
}

Status
SegcoreWrapper::Retrieve(const std::string& plan, RetrieveResult* result) {
    try {
        RetrievePlanWrapper retrieve_plan;
        auto status = Status(::CreateRetrievePlanByExpr(
            collection_, plan.c_str(), plan.size(), &retrieve_plan.plan_));
        CHECK_STATUS(status, "Create retrieve plan failed, invalid expr");
        auto job = ::AsyncRetrieve({},
                                   segment_,
                                   retrieve_plan.plan_,
                                   GetTimestamp(),
                                   DEFAULT_MAX_OUTPUT_SIZE,
                                   false);
        std::mutex mu;
        mu.lock();
        future_register_ready_callback(
            job,
            [](CLockedGoMutex* mutex) { ((std::mutex*)(mutex))->unlock(); },
            (CLockedGoMutex*)(&mu));
        mu.lock();
        auto rs = Status(
            future_leak_and_get(job, (void**)&(result->retrieve_result_)));
        CHECK_STATUS(rs, "Retrieve failed, errs:");

        return Status::Ok();
    } catch (std::exception& e) {
        LOG_ERROR("Retrieve failed, err: {}", e.what());
        return Status::SegcoreErr(e.what());
    }
}

Status
SegcoreWrapper::Search(const std::string& plan,
                       const std::string& placeholder_group,
                       SearchResult* result) {
    try {
        SearchPlanWrapper search_plan;
        CHECK_STATUS(
            Status(::CreateSearchPlanByExpr(
                collection_, plan.c_str(), plan.size(), &search_plan.plan_)),
            "Create search plan failed, err:");

        PlaceholderGroupWrapper group;
        CHECK_STATUS(
            Status(::ParsePlaceholderGroup(search_plan.plan_,
                                           (void*)placeholder_group.c_str(),
                                           placeholder_group.size(),
                                           &group.group_)),
            "Parse placeholder group failed");
        SearchResultWrapper search_result;
        auto job = ::AsyncSearch(
            {}, segment_, search_plan.plan_, group.group_, GetTimestamp());
        std::mutex mu;
        mu.lock();
        future_register_ready_callback(
            job,
            [](CLockedGoMutex* mutex) { ((std::mutex*)(mutex))->unlock(); },
            (CLockedGoMutex*)(&mu));
        mu.lock();
        auto rs =
            Status(future_leak_and_get(job, (void**)&(search_result.ret_)));
        CHECK_STATUS(rs, "Search failed");

        CHECK_STATUS(
            Status(::ReduceSearchResultsAndFillData({},
                                                    &(result->blob_),
                                                    search_plan.plan_,
                                                    &(search_result.ret_),
                                                    1,
                                                    result->slice_nqs_.data(),
                                                    result->slice_topKs_.data(),
                                                    result->slice_nqs_.size())),
            "Reduce search result failed");
        result->result_.resize(result->slice_nqs_.size());
        for (size_t i = 0; i < result->slice_nqs_.size(); i++) {
            CHECK_STATUS(Status(::GetSearchResultDataBlob(
                             &(result->result_[i]), result->blob_, i)),
                         "Get search reault blob failed");
        }

        return Status::Ok();
    } catch (std::exception& e) {
        LOG_ERROR("Search failed, err: {}", e.what());
        return Status::SegcoreErr(e.what());
    }
}

Status
SegcoreWrapper::DeleteByIds(const std::string& ids, int64_t size) {
    CHECK_STATUS(Status(::Delete(segment_,
                                 0,
                                 size,
                                 reinterpret_cast<const uint8_t*>(ids.data()),
                                 ids.size(),
                                 GetTimestamps(size).data())),
                 "Detete failed");
    return Status::Ok();
}

std::vector<uint64_t>
SegcoreWrapper::GetTimestamps(int64_t size) {
    auto ts = GetTimestamp();
    return std::vector<uint64_t>(size, ts);
}

uint64_t
SegcoreWrapper::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return (ms << 18) + cur_id_;
}

}  // namespace milvus::local
