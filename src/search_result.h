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
