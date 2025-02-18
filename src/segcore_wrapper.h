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

#include <cstdint>
#include <string>
#include <vector>
#include "status.h"
#include "retrieve_result.h"
#include "search_result.h"
#include "segcore/collection_c.h"
#include "segcore/segcore_init_c.h"
#include "segcore/segment_c.h"
#include "storage/storage_c.h"

namespace milvus::local {

class SegcoreWrapper final : NonCopyableNonMovable {
 public:
    SegcoreWrapper() : collection_(nullptr), cur_id_(0), segment_(nullptr) {
        SegcoreSetEnableTempSegmentIndex(true);
        InitLocalChunkManagerSingleton("/tmp/mydata");
        CMmapConfig conf;
        conf.growing_enable_mmap = false;
        conf.scalar_index_enable_mmap = false;
        conf.cache_read_ahead_policy = "willneed";
        conf.mmap_path = "/tmp/mydata";
        conf.disk_limit = 1024;
        // uint64_t(2) * uint64_t(1024) * uint64_t(1024) * uint64_t(1024);
        // conf.fix_file_size = uint64_t(4) * uint64_t(1024) * uint64_t(1024);
        conf.fix_file_size = 1024;
        InitMmapManager(conf);
    }
    virtual ~SegcoreWrapper();

 public:
    Status
    SetCollectionInfo(const std::string& collection_name_,
                      const std::string& collection_info,
                      const std::string& index_meta);

    Status
    SetIndexMeta(const std::string& meta_info);

    Status
    CreateIndex(const std::string& meta_info);

    Status
    Insert(int64_t size, const std::string& insert_record_proto);

    Status
    Retrieve(const std::string& plan, RetrieveResult* result);

    Status
    Search(const std::string& plan,
           const std::string& placeholder_group,
           SearchResult* result);

    Status
    DeleteByIds(const std::string& ids, int64_t size);

 private:
    std::vector<uint64_t>
    GetTimestamps(int64_t size);

    uint64_t
    GetTimestamp();

    std::string
    NewCollectionInfo(const std::string& info);

 private:
    CCollection collection_;
    int64_t cur_id_;
    CSegmentInterface segment_;
    std::string collection_name_;
};

}  // namespace milvus::local
