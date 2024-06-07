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

#include "re_scorer.h"

#include <cmath>

#include "common.h"
#include "string_util.hpp"

namespace milvus::local {

void
RRFScorer::ReScore(::milvus::proto::milvus::SearchResults* search_results) {
    for (int i = 0; i < search_results->results().scores_size(); i++) {
        search_results->mutable_results()->set_scores(i, 1.0 / (k_ + (i + 1)));
    }
}

std::function<float(float)>
WeightedScorer::GetActivateFunc() {
    auto metric_type = string_util::ToUpper(GetMetricType());
    if (metric_type == kMetricsCosineName) {
        return [](float distance) -> float { return (1 + distance) * 0.5; };
    } else if (metric_type == KMetricsIPName) {
        return [](float distance) -> float {
            return 0.5 + static_cast<float>(
                             std::atan(static_cast<double>(distance))) /
                             M_PI;
        };
    } else {
        return [](float distance) -> float {
            return 1.0 - 2 *
                             static_cast<float>(
                                 std::atan(static_cast<double>(distance))) /
                             M_PI;
        };
    }
}

void
WeightedScorer::ReScore(
    ::milvus::proto::milvus::SearchResults* search_results) {
    auto activate_func = GetActivateFunc();
    for (int i = 0; i < search_results->results().scores_size(); i++) {
        float distance = search_results->results().scores(i);
        search_results->mutable_results()->set_scores(
            i, weight_ * activate_func(distance));
    }
}

}  // namespace milvus::local
