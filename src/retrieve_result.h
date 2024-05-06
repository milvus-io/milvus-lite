#pragma once

#include "common.h"
#include "segcore/segment_c.h"

namespace milvus::local {

class RetrieveResult final : NonCopyableNonMovable {
 public:
    RetrieveResult() {
        retrieve_result_.proto_blob = nullptr;
        retrieve_result_.proto_size = 0;
    }
    ~RetrieveResult() {
        if (retrieve_result_.proto_blob != nullptr) {
            DeleteRetrieveResult(&retrieve_result_);
            retrieve_result_.proto_blob = nullptr;
            retrieve_result_.proto_size = 0;
        }
    };

 public:
    // milvus::proto::segcore::RetrieveResults
    CRetrieveResult retrieve_result_;
};

}  // namespace milvus::local
