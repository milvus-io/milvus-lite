#pragma once

#include <folly/Format.h>
#include <folly/String.h>
#include <string>
#include <regex>
#include <algorithm>
#include <vector>

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

}  // namespace string_util

}  // namespace milvus::local
