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

#include <folly/Format.h>
#include <folly/String.h>
#include <string>
#include <regex>
#include <algorithm>
#include <vector>
#include <sstream>
#include <random>
#include <chrono>

namespace milvus::local {

namespace string_util {

inline std::string
ToLower(const std::string& str) {
    std::string lower_str;
    std::transform(str.begin(),
                   str.end(),
                   std::back_inserter(lower_str),
                   [](unsigned char c) { return std::tolower(c); });
    return lower_str;
}

inline std::string
ToUpper(const std::string& str) {
    std::string upper_str;
    std::transform(str.begin(),
                   str.end(),
                   std::back_inserter(upper_str),
                   [](unsigned char c) { return std::toupper(c); });
    return upper_str;
}

inline std::string
Trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (std::string::npos == first)
        return "";
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, (last - first + 1));
}

template <class... Args>
inline std::string
SFormat(const std::string& str, Args&&... args) {
    return folly::sformat(str, args...);
}

template <class Delim, class Container>
inline std::string
Join(const Delim& delimiter, const Container& container) {
    return folly::join(delimiter, container);
}

inline bool
IsAlpha(char c) {
    if ((c < 'A' || c > 'Z') && (c < 'a' || c > 'z')) {
        return false;
    }
    return true;
}

inline std::string
GenRandomString(const char* prefix) {
    const std::string CHARACTERS =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    std::uniform_int_distribution<int> distribution(0, CHARACTERS.size() - 1);

    std::ostringstream oss;
    oss << prefix << '-';

    for (size_t i = 0; i < 8; ++i) {
        int random_number = distribution(generator);
        oss << CHARACTERS[random_number];
    }

    return oss.str();
}
}  // namespace string_util

}  // namespace milvus::local
