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
#include "milvus.pb.h"
#include <functional>

namespace milvus::local {

using RankType = int;

const RankType invalidRankType = 0;
const RankType rrfRankType = 1;
const RankType weightedRankType = 2;
const RankType udfExprRankType = 3;

class ReScorer : NonCopyableNonMovable {
 public:
    ReScorer(const std::string& name, const std::string& metric_type)
        : name_(name), metric_type_(metric_type) {
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
    RRFScorer(const std::string& name, const std::string& metric_type, float k)
        : ReScorer(name, metric_type), k_(k) {
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
    WeightedScorer(const std::string& name,
                   const std::string& metric_type,
                   float weight)
        : ReScorer(name, metric_type), weight_(weight) {
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

}  // namespace milvus::local
