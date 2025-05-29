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
#include <set>
#include <map>
#include <cstdint>
#include <string>

#include <iostream>
#include <vector>
#include "common.h"

namespace milvus::local::function {

class Stats {
 public:
    Stats() = default;
    ~Stats() = default;
    Stats(const std::string& name) : output_field_name(name) {
    }

 public:
    void
    Add(const std::string& embd) {
        auto pos = embd.c_str();
        auto end = embd.c_str() + embd.size();
        for (; pos < end; pos += 8) {
            const uint32_t key = *(reinterpret_cast<const uint32_t*>(pos));
            const float freq = *(reinterpret_cast<const float*>(pos + 4));
            rows_contain_token[key] += 1;
            token_num += int(freq);
        }
        rows_num += 1;
    }

    void
    Remove(const std::string& embd) {
        auto pos = embd.c_str();
        auto end = embd.c_str() + embd.size();
        for (; pos < end; pos += 8) {
            const uint32_t key = *(reinterpret_cast<const uint32_t*>(pos));
            const float freq = *(reinterpret_cast<const float*>(pos + 4));
            rows_contain_token[key] -= 1;
            token_num -= int(freq);
        }
        rows_num += 1;
    }

 private:
    std::string output_field_name;
    std::map<uint32_t, int32_t> rows_contain_token;
    int32_t token_num = 0;
    int32_t rows_num = 0;
};

class BM25Stats : NonCopyableNonMovable {
 public:
    ~BM25Stats() = default;

 private:
    BM25Stats() = default;

 public:
    static BM25Stats&
    Instance() {
        static BM25Stats instance;
        return instance;
    }

 public:
    void
    Add(std::vector<std::string>) {
    }

 private:
    std::map<std::string, std::map<std::string, Stats>> bm25_stats_;
};

}  // namespace milvus::local::function
