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

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <tuple>
#include <cassert>

#include "log/Log.h"

namespace milvus::local {

using Record =
    std::tuple<std::string, std::chrono::high_resolution_clock::time_point>;

class Timer {
 public:
    Timer() : start_(false) {
    }
    void
    Start(const std::string& uid) {
        assert(!start_);
        start_ = true;
        start_time_ = std::chrono::high_resolution_clock::now();
        uid_ = uid;
    }

    void
    DoRecord(const std::string& label) {
        assert(start_);
        auto now = std::chrono::high_resolution_clock::now();
        time_record_.emplace_back(label, now);
    }

    void
    Print() const {
        assert(start_ && time_record_.size() != 0);
        auto pre_time = start_time_;
        const auto& [_, time_point] = time_record_.back();
        LOG_INFO("uid: {} TOTAL duration:{} microseconds ",
                 uid_,
                 std::chrono::duration_cast<std::chrono::microseconds>(
                     time_point - start_time_)
                     .count());
        for (const auto& record : time_record_) {
            const auto& [label, time_point] = record;
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    time_point - pre_time)
                    .count();
            LOG_INFO(
                "uid: {} {} Duration:{} microseconds ", uid_, label, duration);
            pre_time = time_point;
        }
    }

    void
    Stop() {
        assert(start_);
        time_record_.clear();
        uid_ = "";
        start_ = false;
    }

 private:
    bool start_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<Record> time_record_;
    std::string uid_;
};

void
InitializeTimer(const std::string& uid);
void
RecordEvent(const std::string& label);
void
PrintTimerRecords();
void
StopTimer();
}  // namespace milvus::local
