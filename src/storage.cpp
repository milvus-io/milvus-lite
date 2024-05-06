#include "storage.h"
#include <SQLiteCpp/Database.h>
#include <SQLiteCpp/Transaction.h>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>
#include "collection_data.h"
#include "log/Log.h"
#include "pb/schema.pb.h"

namespace milvus::local {

Storage::Storage(const char* db_file) : db_file_(db_file) {
}

Storage::~Storage() {
}

bool
Storage::Open() {
    try {
        db_ptr_ = std::make_unique<SQLite::Database>(
            db_file_,
            SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE |
                SQLite::OPEN_FULLMUTEX);
        if (!cm_.Init(db_ptr_.get())) {
            return false;
        }
        std::vector<std::string> names;
        cm_.CollectionNames(&names);
        for (const auto& name : names) {
            collections_.emplace(
                name, std::make_unique<CollectionData>(name.c_str()));
        }
        return true;
    } catch (std::exception& e) {
        LOG_ERROR("Open storage failed, err: {}", e.what());
        return false;
    }
}

bool
Storage::CreateCollection(const std::string& collection_name,
                          const std::string& pk_name,
                          const std::string& schema_proto) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!cm_.CreateCollection(
            db_ptr_.get(), collection_name, pk_name, schema_proto)) {
        return false;
    }
    auto data_ptr = std::make_unique<CollectionData>(collection_name.c_str());
    if (!data_ptr->CreateCollection(db_ptr_.get())) {
        return false;
    }
    collections_[collection_name] = std::move(data_ptr);
    transaction.commit();
    return true;
}

bool
Storage::DropCollection(const std::string& collection_name) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!cm_.DropCollection(db_ptr_.get(), collection_name)) {
        LOG_ERROR("Delete collection: {}'s meta failed", collection_name);
        return false;
    }
    if (!collections_[collection_name]->DropCollection(db_ptr_.get())) {
        LOG_ERROR("Delete collection: {}'s data failed", collection_name);
        return false;
    }
    collections_.erase(collection_name);
    transaction.commit();
    return true;
}

bool
Storage::LoadCollecton(const std::string& collection_name,
                       int64_t start,
                       int64_t size,
                       std::vector<std::string>* out_rows) {
    collections_[collection_name]->Load(db_ptr_.get(), start, size, out_rows);
    return static_cast<int64_t>(out_rows->size()) == size;
}

bool
Storage::GetCollectionSchema(const std::string& collection_name,
                             std::string* output_info_str) {
    output_info_str->assign(cm_.GetCollectionSchema(collection_name).c_str());
    return true;
}

bool
Storage::CreateIndex(const std::string& collection_name,
                     const std::string& index_name,
                     const std::string& index_proto) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!cm_.CreateIndex(
            db_ptr_.get(), collection_name, index_name, index_proto)) {
        return false;
    }
    transaction.commit();
    return true;
}

bool
Storage::GetIndex(const std::string& collection_name,
                  const std::string& index_name,
                  std::string* output_index_proto) {
    return cm_.GetCollectionIndex(
        collection_name, index_name, output_index_proto);
}

void
Storage::GetAllIndex(const std::string& collection_name,
                     const std::string& exclude,
                     std::vector<std::string>* index_protos) {
    cm_.GetAllIndex(collection_name, exclude, index_protos);
}

bool
Storage::DropIndex(const std::string& collection_name,
                   const std::string& index_name) {
    SQLite::Transaction transaction(*db_ptr_.get());
    if (!cm_.DropIndex(db_ptr_.get(), collection_name, index_name)) {
        return false;
    }
    transaction.commit();
    return true;
}

int
Storage::Insert(const std::string collection_name,
                const std::vector<Row>& rows) {
    SQLite::Transaction transaction(*db_ptr_.get());
    for (const auto& row : rows) {
        if (collections_[collection_name]->Insert(db_ptr_.get(),
                                                  std::get<0>(row).c_str(),
                                                  std::get<1>(row)) < 0) {
            return -1;
        }
    }
    transaction.commit();
    return rows.size();
}

int
Storage::Delete(const std::string collection_name,
                const std::vector<std::string>& ids) {
    SQLite::Transaction transaction(*db_ptr_.get());
    int n = collections_[collection_name]->Delete(db_ptr_.get(), ids);
    transaction.commit();
    return n;
}

int64_t
Storage::Count(const std::string& collection_name) {
    SQLite::Transaction transaction(*db_ptr_.get());
    int64_t n = collections_[collection_name]->Count(db_ptr_.get());
    transaction.commit();
    return n;
}

}  // namespace milvus::local
