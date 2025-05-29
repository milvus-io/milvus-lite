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

namespace milvus::local {

namespace bm25 {

struct Stats {
    Stats(){};
    Stats(const std::string& name) : output_field_name(name) {
    }
    std::string output_field_name;
    std::map<uint32_t, int32_t> rows_contain_token;
    std::map<std::string, std::string> contexts;
    int32_t token_num = 0;
    int32_t rows_num = 0;
};

class StatDict {
 public:
    static StatDict&
    Instance() {
        static StatDict dict;
        return dict;
    }
    StatDict(const StatDict&) = delete;
    StatDict&
    operator=(const StatDict&) = delete;

    std::map<std::string, Stats> stats_dict;

 private:
    StatDict(){};
};

inline void
DebugPrint(const Stats& stat) {
    std::cout << "name: " << stat.output_field_name << std::endl;
    std::cout << "token num: " << stat.token_num << std::endl;
    std::cout << "rows num: " << stat.rows_num << std::endl;
    std::cout << "token dict" << std::endl;
    for (auto kv : stat.rows_contain_token) {
        std::cout << kv.first << "/t" << kv.second << std::endl;
    }
}

}  // namespace bm25
//
}  // namespace milvus::local
