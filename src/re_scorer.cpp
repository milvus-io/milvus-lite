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
#include <memory>
#include "nlohmann/json.hpp"
#include "common.h"
#include "log/Log.h"
#include "status.h"
#include "string_util.hpp"

namespace milvus::local {

const int kDefaultRRFParamsValue = 60;

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

Status
NewReScorers(int req_cnt,
             const ::google::protobuf::RepeatedPtrField<
                 ::milvus::proto::common::KeyValuePair>& rank_params,
             std::vector<std::unique_ptr<ReScorer>>* re_scorers) {
    if (req_cnt == 0) {
        return Status::Ok();
    }

    std::string rank_type, param_str;
    for (const auto& pair : rank_params) {
        if (pair.key() == kRankTypeKey) {
            rank_type = string_util::Trim(pair.value());
        } else if (pair.key() == kRankParamsKey) {
            param_str = pair.value();
        }
    }
    if (rank_type.empty()) {
        LOG_INFO("rank strategy not specified, use rrf instead");
        for (int i = 0; i < req_cnt; i++) {
            re_scorers->emplace_back(
                std::make_unique<RRFScorer>("rrf", kDefaultRRFParamsValue));
        }
    }

    if (rank_type != kRRFRankTypeName && rank_type != kWeightedRankTypeName) {
        return Status::ParameterInvalid("unsupported rank type {}", rank_type);
    }

    if (param_str.empty()) {
        return Status::ParameterInvalid("{} not found in rank_params",
                                        kRankParamsKey);
    }

    try {
        nlohmann::json data = nlohmann::json::parse(param_str);
        if (rank_type == kRRFRankTypeName) {
            if (!data.contains(kRRFParamsKey)) {
                return Status::ParameterInvalid("{} not found in rank_params",
                                                kRRFParamsKey);
            }
            if (!data[kRRFParamsKey].is_number()) {
                return Status::ParameterInvalid(
                    "The type of rank param k should be float");
            }
            double k = data[kRRFParamsKey];
            if (k <= 0 || k >= 16384) {
                return Status::ParameterInvalid(
                    "The rank params k should be in range (0, 16384)");
            }
            for (int i = 0; i < req_cnt; i++) {
                re_scorers->emplace_back(
                    std::make_unique<RRFScorer>("rrf", float(k)));
            }
        } else if (rank_type == kWeightedRankTypeName) {
            if (!data.contains(kWeightsParamsKey)) {
                return Status::ParameterInvalid("{} not found in rank_params",
                                                kWeightsParamsKey);
            }
            std::vector<float> weights;
            if (data[kWeightsParamsKey].is_array()) {
                for (const auto& weight : data[kWeightsParamsKey]) {
                    if (weight.is_number()) {
                        float w = weight;
                        if (w < 0 || w > 1) {
                            return Status::ParameterInvalid(
                                "rank param weight should be in range[0, 1]");
                        }
                        weights.push_back(w);
                    } else {
                        return Status::ParameterInvalid(
                            "The type of rank param weight should be float");
                    }
                }
            } else {
                return Status::ParameterInvalid(
                    "The weights param should be an array");
            }
            if (req_cnt != weights.size()) {
                return Status::ParameterInvalid(
                    "the length of weights param mismatch with ann search "
                    "requests: invalid parameter[expected={}][actual={}]",
                    req_cnt,
                    weights.size());
            }

            for (int i = 0; i < req_cnt; i++) {
                re_scorers->emplace_back(
                    std::make_unique<WeightedScorer>("rrf", weights[i]));
            }
        } else {
            return Status::ParameterInvalid("unsupported rank type {}",
                                            rank_type);
        }
    } catch (nlohmann::json::parse_error& e) {
        auto err = string_util::SFormat("Index params err: {}", e.what());
        LOG_ERROR(err);
        return Status::ParameterInvalid(err);
    }

    return Status::Ok();
}

}  // namespace milvus::local
