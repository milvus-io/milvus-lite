#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include "pb/milvus.pb.h"
#include "pb/plan.pb.h"
#include "schema.pb.h"
#include "search_result.h"
#include "status.h"

namespace milvus::local {

class SearchTask final : NonCopyableNonMovable {
 public:
    SearchTask(::milvus::proto::milvus::SearchRequest* search_reques,
               const ::milvus::proto::schema::CollectionSchema* schema);
    virtual ~SearchTask();

 public:
    Status
    Process(::milvus::proto::plan::PlanNode* plan,
            std::string* placeholder_group,
            std::vector<int64_t>* nqs,
            std::vector<int64_t>* topks);

    bool
    PostProcess(const SearchResult& segcore_reaul,
                ::milvus::proto::milvus::SearchResults* search_results);

 private:
    bool
    ParseSearchInfo(::milvus::proto::plan::QueryInfo* info);

    bool
    GetOutputFieldsIds(std::vector<int64_t>* ids);

    std::optional<std::tuple<std::string, int64_t>>
    GetVectorField();

    void
    FillInFieldInfo(::milvus::proto::schema::SearchResultData* result_data);

 private:
    ::milvus::proto::milvus::SearchRequest* search_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;

    std::vector<std::string> output_fields_;
    std::vector<std::string> user_output_fields_;
    int64_t topk_, offset_;
    std::string ann_field_;
    std::string metric_;
    std::string groupby_field_name_;
};

}  // namespace milvus::local
