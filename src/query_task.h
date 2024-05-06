#pragma once

#include "pb/milvus.pb.h"
#include "pb/plan.pb.h"
#include <cstdint>
#include <vector>

#include "retrieve_result.h"
#include "status.h"

namespace milvus::local {

class QueryTask : NonCopyableNonMovable {
 public:
    QueryTask(const ::milvus::proto::milvus::QueryRequest* query_request,
              const ::milvus::proto::schema::CollectionSchema*);
    virtual ~QueryTask();

    Status
    Process(::milvus::proto::plan::PlanNode* plan);

    bool
    PostProcess(const RetrieveResult& rt,
                ::milvus::proto::milvus::QueryResults* ret);

 private:
    bool
    GetOutputFieldIds(std::vector<int64_t>* ids);

    void
    FilterSystemField();

    void
    FillInFieldInfo(::milvus::proto::milvus::QueryResults* result_data);

    Status
    ParseQueryParams(::milvus::proto::plan::PlanNode* plan, bool expr_empty);

 private:
    const ::milvus::proto::milvus::QueryRequest* query_request_;
    const ::milvus::proto::schema::CollectionSchema* schema_;

    std::vector<std::string> output_fields_;
    std::vector<std::string> user_output_fields_;

    int64_t limit_;
    int64_t offset_;
    bool is_count_;
};

}  // namespace milvus::local
