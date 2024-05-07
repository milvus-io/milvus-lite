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

#include <cassert>
#include <cstdint>
#include <vector>
#include "common.h"
#include "common/type_c.h"
#include "segcore/reduce_c.h"

namespace milvus::local {

class SearchResult final : NonCopyableNonMovable {
 public:
    SearchResult(const std::vector<int64_t>& slice_nqs,
                 const std::vector<int64_t>& slice_topKs)
        : slice_nqs_(slice_nqs), slice_topKs_(slice_topKs) {
        blob_ = nullptr;
    }
    ~SearchResult() {
        if (blob_ != nullptr) {
            DELETE_AND_SET_NULL(blob_, DeleteSearchResultDataBlobs);
            result_.clear();
        }
    }

 public:
    // std::vector<std::vector<char>>
    CSearchResultDataBlobs blob_;

    // milvus::proto::schema::SearchResultData
    // ptr to blob_
    std::vector<CProto> result_;

 public:
    std::vector<int64_t> slice_nqs_;
    std::vector<int64_t> slice_topKs_;
};

}  // namespace milvus::local
