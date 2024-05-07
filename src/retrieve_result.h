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
