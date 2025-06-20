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

#include <cstdint>
#include <map>
#include <string>

namespace milvus::local {

#define CHECK_STATUS(status, err) \
    do {                          \
        Status _s = status;       \
        if (!_s.IsOk()) {         \
            return _s;            \
        }                         \
    } while (0)

#define DELETE_AND_SET_NULL(ptr, deleter) \
    do {                                  \
        if ((ptr) != nullptr) {           \
            deleter(ptr);                 \
            (ptr) = nullptr;              \
        }                                 \
    } while (0)

using KVMap = std::map<std::string, std::string>;
using SparseVector = std::map<uint32_t, float>;

// system field id:
// 0: unique row id
// 1: timestamp
// 100: first user field id
// 101: second user field id
// 102: ...
const int64_t kStartOfUserFieldId = 100;

const int64_t kRowIdField = 0;

const int64_t kTimeStampField = 1;

const std::string kRowIdFieldName("RowID");

const std::string kTimeStampFieldName("Timestamp");

const std::string kMetaFieldName("$meta");

const std::string kPlaceholderTag("$0");

const int64_t kTopkLimit = 16384;
const int64_t kSchemaFieldLimit = 64;
const int64_t kMaxLengthLimit = 65535;

// scalar index type
const std::string kDefaultStringIndexType("Trie");
const std::string kInvertedIndexType("INVERTED");
const std::string kDefaultArithmeticIndexType = ("STL_SORT");

const int64_t kMaxIndexRow = 1000000;

// Search, Index parameter keys
const std::string kTopkKey("topk");
// const std::string kSearchParamKey("search_param");
const std::string kSearchParamKey("params");
const std::string kOffsetKey("offset");
const std::string kRoundDecimalKey("round_decimal");
const std::string kGroupByFieldKey("group_by_field");
const std::string kAnnFieldKey("anns_field");
const std::string kSegmentNumKey("segment_num");
const std::string kWithFilterKey("with_filter");
const std::string kWithOptimizeKey("with_optimize");
const std::string kCollectionKey("collection");
const std::string kIndexParamsKey("params");
const std::string kIndexTypeKey("index_type");
const std::string kMetricTypeKey("metric_type");
const std::string kDimKey("dim");
const std::string kMaxLengthKey("max_length");
const std::string kMaxCapacityKey("max_capacity");
const std::string kReduceStopForBestKey("reduce_stop_for_best");
const std::string kLimitKey("limit");
const std::string KMetricsIPName("IP");
const std::string kMetricsCosineName("COSINE");
const std::string kMetricsBM25Name("BM25");
const std::string kMetricsL2Name("L2");

const std::string kBM25AvgName("bm25_avgdl");

const std::string kCountStr("count(*)");

inline int64_t
GetCollectionId(const std::string& collection_name) {
    std::hash<std::string> hasher;
    size_t hash_value = hasher(collection_name);
    return static_cast<int64_t>(hash_value);
}

inline int64_t
GetIndexId(const std::string& index_name) {
    std::hash<std::string> hasher;
    size_t hash_value = hasher(index_name);
    return static_cast<int64_t>(hash_value);
}

struct NonCopyableNonMovable {
    constexpr NonCopyableNonMovable() noexcept = default;
    virtual ~NonCopyableNonMovable() noexcept = default;

    NonCopyableNonMovable(NonCopyableNonMovable&&) = delete;
    NonCopyableNonMovable&
    operator=(NonCopyableNonMovable&&) = delete;
    NonCopyableNonMovable(const NonCopyableNonMovable&) = delete;
    NonCopyableNonMovable&
    operator=(const NonCopyableNonMovable&) = delete;
};

}  // namespace milvus::local
