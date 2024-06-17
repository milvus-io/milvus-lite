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
// #include "common.h"
#include "schema.pb.h"

namespace milvus::local {

class MilvusID {
    enum IDType { INT, STR };

 public:
    explicit MilvusID(int64_t id) {
        int_id_ = id;
        type_ = INT;
    }

    explicit MilvusID(const std::string& id) {
        str_id_ = id;
        type_ = STR;
    }

    explicit MilvusID(const ::milvus::proto::schema::IDs& ids, int64_t idx) {
        if (ids.has_int_id()) {
            int_id_ = ids.int_id().data(idx);
            type_ = INT;
        } else {
            str_id_ = ids.str_id().data(idx);
            type_ = STR;
        }
    }

    virtual ~MilvusID() = default;

 public:
    bool
    operator==(const MilvusID& right) const {
        if (type_ == INT) {
            return int_id_ == right.int_id_;
        } else {
            return str_id_ == right.str_id_;
        }
    }

    bool
    operator<(const MilvusID& right) const {
        if (type_ == INT) {
            return int_id_ < right.int_id_;
        } else {
            return str_id_ < right.str_id_;
        }
    }

    void
    AppendToIDs(::milvus::proto::schema::IDs* ids) const {
        if (type_ == INT) {
            ids->mutable_int_id()->add_data(int_id_);
        } else {
            ids->mutable_str_id()->add_data(str_id_);
        }
    }

 private:
    int64_t int_id_;
    std::string str_id_;
    IDType type_;
};

}  // namespace milvus::local
