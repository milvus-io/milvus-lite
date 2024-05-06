#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include "segcore_wrapper.h"
#include "common.h"
#include "status.h"

namespace milvus::local {

class Index final : NonCopyableNonMovable {
 public:
    Index() = default;
    virtual ~Index() = default;

 public:
    // meta interface
    Status
    CreateCollection(const std::string& collection_name,
                     const std::string& schema_proto);
    bool
    DropCollection(const std::string& collection_name);

    bool
    HasLoaded(const std::string& collection_name) {
        return collections_.find(collection_name) != collections_.end();
    }

    Status
    CreateIndex(const std::string& collection_name,
                const std::string& index_proto);

    Status
    Insert(const std::string& collection_name,
           int64_t size,
           const std::string& insert_record_proto);

    Status
    Retrieve(const std::string& collection_name,
             const std::string& expr,
             RetrieveResult* result);

    Status
    Search(const std::string& collection_name,
           const std::string& plan,
           const std::string& placeholder_group,
           SearchResult* result);

    Status
    DeleteByIds(const std::string& collection_name,
                const std::string& ids,
                int64_t size);

 private:
    std::map<std::string, std::unique_ptr<SegcoreWrapper>> collections_;
};

}  // namespace milvus::local
