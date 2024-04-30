#include "index.h"
#include "common.h"
#include "log/Log.h"
#include "status.h"
#include <exception>

namespace milvus::local {

Status
Index::CreateCollection(const std::string& collection_name,
                        const std::string& schema_proto) {
    if (HasLoaded(collection_name)) {
        LOG_INFO("Collection {} alread load", collection_name);
        return Status::Ok();
    }

    auto c = std::make_unique<SegcoreWrapper>();
    CHECK_STATUS(c->SetCollectionInfo(collection_name, schema_proto),
                 "Create collection failed: ");
    collections_[collection_name] = std::move(c);
    return Status::Ok();
}

bool
Index::DropCollection(const std::string& collection_name) {
    if (collections_.find(collection_name) != collections_.end()) {
        collections_.erase(collection_name);
    }
    return true;
}

Status
Index::CreateIndex(const std::string& collection_name,
                   const std::string& index_proto) {
    if (collections_.find(collection_name) == collections_.end()) {
        LOG_ERROR("Collecton {} not existed", collection_name);
        return Status::CollectionNotFound();
    }
    CHECK_STATUS(collections_[collection_name]->SetIndexMeta(index_proto),
                 "Create index failed:");
    return Status::Ok();
}

Status
Index::Insert(const std::string& collection_name,
              int64_t size,
              const std::string& insert_record_prot) {
    if (collections_.find(collection_name) == collections_.end()) {
        LOG_ERROR("Collecton {} not existed", collection_name);
        return Status::CollectionNotFound();
    }
    return collections_[collection_name]->Insert(size, insert_record_prot);
}

Status
Index::Retrieve(const std::string& collection_name,
                const std::string& plan,
                RetrieveResult* result) {
    if (collections_.find(collection_name) == collections_.end()) {
        LOG_ERROR("Collecton {} not existed", collection_name);
        return Status::CollectionNotFound();
    }
    return collections_[collection_name]->Retrieve(plan, result);
}

Status
Index::Search(const std::string& collection_name,
              const std::string& plan,
              const std::string& placeholder_group,
              SearchResult* result) {
    if (collections_.find(collection_name) == collections_.end()) {
        LOG_ERROR("Collecton {} not existed", collection_name);
        return Status::CollectionNotFound();
    }
    return collections_[collection_name]->Search(
        plan, placeholder_group, result);
}

Status
Index::DeleteByIds(const std::string& collection_name,
                   const std::string& ids,
                   int64_t size) {
    if (collections_.find(collection_name) == collections_.end()) {
        LOG_ERROR("Collecton {} not existed", collection_name);
        return Status::CollectionNotFound();
    }
    return collections_[collection_name]->DeleteByIds(ids, size);
}

}  // namespace milvus::local
