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
#include "common.pb.h"
#include "milvus.pb.h"
#include "status.h"
#include <functional>
#include <memory>
#include <vector>

namespace milvus::local {

using RankType = int;

const RankType invalidRankType = 0;
const RankType rrfRankType = 1;
const RankType weightedRankType = 2;
const RankType udfExprRankType = 3;

const std::string kRRFRankTypeName("rrf");
const std::string kWeightedRankTypeName("weighted");

const std::string kRankTypeKey("strategy");
const std::string kRankParamsKey("params");
const std::string kRRFParamsKey("k");
const std::string kWeightsParamsKey("weights");

class ReScorer : NonCopyableNonMovable {
 public:
    ReScorer(const std::string& name) : name_(name) {
    }
    virtual ~ReScorer() = default;

 public:
    inline const std::string
    Name() {
        return name_;
    }

    virtual void
    ReScore(::milvus::proto::milvus::SearchResults*) = 0;

    virtual RankType
    ScorerType() = 0;

    inline void
    SetMetricType(const std::string& metric_type) {
        metric_type_ = metric_type;
    }

    inline const std::string
    GetMetricType() {
        return metric_type_;
    }

 private:
    std::string name_;
    std::string metric_type_;
};

class RRFScorer : public ReScorer {
 public:
    RRFScorer(const std::string& name, float k) : ReScorer(name), k_(k) {
    }
    virtual ~RRFScorer() = default;

 public:
    RankType
    ScorerType() override {
        return rrfRankType;
    }
    void
    ReScore(::milvus::proto::milvus::SearchResults*) override;

 private:
    float k_;
};

class WeightedScorer : public ReScorer {
 public:
    WeightedScorer(const std::string& name, float weight)
        : ReScorer(name), weight_(weight) {
    }
    virtual ~WeightedScorer() = default;

 public:
    RankType
    ScorerType() override {
        return weightedRankType;
    }

    void
    ReScore(::milvus::proto::milvus::SearchResults*) override;

 private:
    std::function<float(float)>
    GetActivateFunc();

 private:
    float weight_;
};

Status
NewReScorers(int req_cnt,
             const ::google::protobuf::RepeatedPtrField<
                 ::milvus::proto::common::KeyValuePair>& rank_params,
             std::vector<std::unique_ptr<ReScorer>>* re_scorers);

}  // namespace milvus::local
