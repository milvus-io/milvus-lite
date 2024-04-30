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

namespace milvus::local {

class SegcoreWrapper final : NonCopyableNonMovable {
 public:
    SegcoreWrapper() : collection_(nullptr), cur_id_(0), segment_(nullptr) {
        SegcoreSetEnableTempSegmentIndex(true);
    }
    virtual ~SegcoreWrapper();

 public:
    Status
    SetCollectionInfo(const std::string& collection_name_,
                      const std::string& collection_info);

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
