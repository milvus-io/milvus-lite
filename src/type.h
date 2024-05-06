#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace milvus::local {

using Row = std::tuple<const std::string, const std::string>;
using Rows = std::vector<Row>;

}  // namespace milvus::local
