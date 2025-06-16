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

#include "create_index_task.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "common.h"
#include "nlohmann/json.hpp"
#include "pb/common.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "schema_util.h"
#include "log/Log.h"
#include "status.h"
#include "string_util.hpp"

namespace milvus::local {

/*      dtype               index_type                                     metric_type   
                                                                                     
 *    FloatVector:         FLAT, HNWS                                      L2, IP, COSINE, AUTOINDEX
 *    BinaryVector:        BIN_FLAT                                        HAMMING, JACCARD, SUBSTRUCTURE, SUPERSTRUCTURE
 *    Float16Vector:       FLAT                                            L2, IP, COSINE
 *    BFloat16Vector:      FLAT                                            L2, IP, COSINE
 *    SparseFloatVector:   SPARSE_INVERTED_INDEX, SPARSE_WAND              IP, BM25 
 */

// metrics type
const char* kL2 = "L2";
const char* kIP = "IP";
const char* kCosine = "COSINE";
const char* kHamming = "HAMMING";
const char* kJaccard = "JACCARD";
const char* kSubStructure = "SUBSTRUCTURE";
const char* kSuperStructure = "SUPERSTRUCTURE";

// index_type
const char* kAutoIndex = "AUTOINDEX";
const char* kFlat = "FLAT";
const char* kBin_Flat = "BIN_FLAT";
// const char* kHNSW = "HNSW";
const char* kIvf_Flat = "IVF_FLAT";
const char* kSparseInvertedIndex = "SPARSE_INVERTED_INDEX";
const char* kSparseWand = "SPARSE_WAND";

// default metric
const char* kFloatVectorDefaultMetricType = kIP;
const char* kSparseFloatVectorDefaultMetricType = kIP;
const char* kBinaryVectorDefaultMetricType = kJaccard;

//bm25 params
const char* kBm25K1 = "bm25_k1";
const char* kBm25B = "bm25_b";

class AutoIndexConfig final : NonCopyableNonMovable {
 public:
    AutoIndexConfig()
        : index_param({{"M", "18"},
                       {"efConstruction", "240"},
                       {"index_type", kAutoIndex},
                       {"metric_type", kIP}}) {
    }
    ~AutoIndexConfig() = default;

 public:
    const KVMap index_param;
};

using DType = ::milvus::proto::schema::DataType;

static const AutoIndexConfig kAutoIndexConfig;

class IndexChecker : NonCopyableNonMovable {
 public:
    IndexChecker(const std::string& index_type,
                 const std::string& metric,
                 int64_t dim)
        : index_type_(index_type),
          metric_(metric),
          dim_(dim),
          need_check_dim_(true) {
    }
    virtual ~IndexChecker() = default;

    Status
    Check();

 protected:
    std::string index_type_;
    std::string metric_;
    int64_t dim_;

    bool need_check_dim_;
    int64_t min_dim_;
    int64_t max_dim_;
    std::vector<std::string> supported_index_;
    std::vector<std::string> supported_metric_;
};

Status
IndexChecker::Check() {
    if (need_check_dim_ && (dim_ < min_dim_ || dim_ > max_dim_)) {
        auto err = string_util::SFormat(
            "invalid dimension: {}. should be in range {} ~ {}",
            dim_,
            min_dim_,
            max_dim_);
        return Status::Undefined(err);
    }
    if (std::find(supported_index_.begin(),
                  supported_index_.end(),
                  index_type_) == supported_index_.end()) {
        auto err = string_util::SFormat(
            "invalid index type: {}, local mode only support {}",
            index_type_,
            string_util::Join(" ", supported_index_));
        return Status::Undefined(err);
    }
    if (std::find(supported_metric_.begin(),
                  supported_metric_.end(),
                  metric_) == supported_metric_.end()) {
        auto err = string_util::SFormat(
            "metric type {} not found or not supported, supported: {}",
            metric_,
            string_util::Join(" ", supported_metric_));
        return Status::Undefined(err);
    }
    return Status::Ok();
}

class FloatVectorIndexChecker : public virtual IndexChecker {
 public:
    FloatVectorIndexChecker(const std::string& index_type,
                            const std::string& metric,
                            int64_t dim)
        : IndexChecker(index_type, metric, dim) {
        min_dim_ = 2;
        max_dim_ = 32768;
        supported_index_ = {kFlat, kIvf_Flat, kAutoIndex};
        supported_metric_ = {kL2, kIP, kCosine};
    }

    virtual ~FloatVectorIndexChecker() = default;
};

class BinaryVectorChecker : public virtual IndexChecker {
 public:
    BinaryVectorChecker(const std::string& index_type,
                        const std::string& metric,
                        int64_t dim)
        : IndexChecker(index_type, metric, dim) {
        min_dim_ = 2;
        max_dim_ = 32768;
        supported_index_ = {kBin_Flat};
        supported_metric_ = {
            kHamming, kJaccard, kSubStructure, kSuperStructure};
    }

    virtual ~BinaryVectorChecker() = default;
};

class Float16VectorChecker : public virtual IndexChecker {
 public:
    Float16VectorChecker(const std::string& index_type,
                         const std::string& metric,
                         int64_t dim)
        : IndexChecker(index_type, metric, dim) {
        min_dim_ = 2;
        max_dim_ = 32768;
        supported_index_ = {kFlat};
        supported_metric_ = {kL2, kIP, kCosine};
    }

    virtual ~Float16VectorChecker() = default;
};

class BFloat16VectorChecker : public virtual IndexChecker {
 public:
    BFloat16VectorChecker(const std::string& index_type,
                          const std::string& metric,
                          int64_t dim)
        : IndexChecker(index_type, metric, dim) {
        min_dim_ = 2;
        max_dim_ = 32768;
        supported_index_ = {kFlat};
        supported_metric_ = {kL2, kIP, kCosine};
    }

    virtual ~BFloat16VectorChecker() = default;
};

class SparseFloatVectorChecker : public virtual IndexChecker {
 public:
    SparseFloatVectorChecker(const std::string& index_type,
                             const std::string& metric,
                             int64_t dim)
        : IndexChecker(index_type, metric, dim) {
        min_dim_ = -1;
        max_dim_ = -1;
        need_check_dim_ = false;
        supported_index_ = {kSparseInvertedIndex, kSparseWand};
        supported_metric_ = {kIP, kMetricsBM25Name};
    }

    virtual ~SparseFloatVectorChecker() = default;
};

Status
Check(DType field_type,
      const std::string& index_type,
      const std::string& metric,
      int64_t dim) {
    if (field_type == DType::FloatVector) {
        return FloatVectorIndexChecker(index_type, metric, dim).Check();
    } else if (field_type == DType::Float16Vector) {
        return Float16VectorChecker(index_type, metric, dim).Check();
    } else if (field_type == DType::BFloat16Vector) {
        return BFloat16VectorChecker(index_type, metric, dim).Check();
    } else if (field_type == DType::BinaryVector) {
        return BinaryVectorChecker(index_type, metric, dim).Check();
    } else if (field_type == DType::SparseFloatVector) {
        return SparseFloatVectorChecker(index_type, metric, dim).Check();
    } else {
        return Status::ParameterInvalid("Unknow data type");
    }
}

void
CreateIndexTask::WrapUserIndexParams(const std::string& metrics_type) {
    ::milvus::proto::common::KeyValuePair p1;
    p1.set_key(kIndexTypeKey);
    p1.set_value(kAutoIndex);
    new_extra_params_.push_back(p1);

    ::milvus::proto::common::KeyValuePair p2;
    p2.set_key(kMetricTypeKey);
    p2.set_value(metrics_type);
    new_extra_params_.push_back(p2);
}

bool
CreateIndexTask::AddAutoIndexParams(KVMap* index_params) {
    is_auto_index_ = true;
    if (index_params->find(kMetricTypeKey) != index_params->end()) {
        const auto metrics_type = index_params->at(kMetricTypeKey);
        WrapUserIndexParams(metrics_type);
        index_params->insert(kAutoIndexConfig.index_param.begin(),
                             kAutoIndexConfig.index_param.end());
        (*index_params)[kMetricTypeKey] = metrics_type;
        return true;
    } else {
        auto it = kAutoIndexConfig.index_param.find(kMetricTypeKey);
        WrapUserIndexParams(it->second);
        index_params->insert(kAutoIndexConfig.index_param.begin(),
                             kAutoIndexConfig.index_param.end());
        (*index_params)[kMetricTypeKey] = it->second;
        return true;
    }
    return true;
}

Status
CreateIndexTask::CheckTrain(const ::milvus::proto::schema::FieldSchema& field,
                            KVMap& index_params) {
    auto index_type = index_params.at(kIndexTypeKey);
    if (!IsVectorIndex(field.data_type())) {
        return Status::Ok();
    }
    if (schema_util::IsSparseVectorType(field.data_type())) {
        if (index_params[kMetricTypeKey] == kMetricsBM25Name) {
            {
                auto [succ, num] =
                    string_util::ToNumber(index_params[kBM25AvgName]);
                if (!succ) {
                    return Status::ParameterInvalid("{}: {} is not a number",
                                                    kBM25AvgName,
                                                    index_params[kBM25AvgName]);
                }
            }
            {
                auto [succ, num] = string_util::ToNumber(index_params[kBm25K1]);
                if (!succ) {
                    return Status::ParameterInvalid("{}: {} is not a number",
                                                    kBm25K1,
                                                    index_params[kBm25K1]);
                }
                if (num < 0 || num > 3.0) {
                    return Status::ParameterInvalid(
                        "{}: {} should be in range [0.0, 3.0]",
                        kBm25K1,
                        index_params[kBm25K1]);
                }
            }
            {
                auto [succ, num] = string_util::ToNumber(index_params[kBm25B]);
                if (!succ) {
                    return Status::ParameterInvalid(
                        "{}: {} is not a number", kBm25B, index_params[kBm25B]);
                }
                if (num < 0.0 || num > 1.0) {
                    return Status::ParameterInvalid(
                        "{}: {} should be in range [0.0, 1.0]",
                        kBm25B,
                        index_params[kBm25B]);
                }
            }
        }
    }
    if (!schema_util::IsSparseVectorType(field.data_type())) {
        if (!FillDimension(field, &index_params)) {
            return Status::ParameterInvalid();
        }
    }
    int64_t dim = -1;
    if (!schema_util::IsSparseVectorType(field.data_type())) {
        dim = std::stoll(index_params.at(kDimKey));
    }

    std::string metric = index_params.at(kMetricTypeKey);
    return Check(field.data_type(), index_type, metric, dim);
}

Status
CreateIndexTask::ParseIndexParams() {
    const milvus::proto::schema::FieldSchema* field_ptr = nullptr;
    for (const auto& field : schema_->fields()) {
        if (field.name() == create_index_request_->field_name()) {
            field_ptr = &field;
        }
    }
    if (field_ptr == nullptr) {
        auto err = string_util::SFormat("Can not found field {}",
                                        create_index_request_->field_name());
        LOG_ERROR(err);
        return Status::ParameterInvalid(err);
    }

    if (!create_index_request_->index_name().empty()) {
        index_name_ = create_index_request_->index_name();
    } else {
        index_name_ = field_ptr->name();
    }

    field_id_ = field_ptr->fieldid();
    collectionid_ = GetCollectionId(schema_->name());

    KVMap index_params;
    for (const auto& param : create_index_request_->extra_params()) {
        if (param.key() == kIndexParamsKey) {
            try {
                nlohmann::json data = nlohmann::json::parse(param.value());
                for (auto& [key, value] : data.items()) {
                    if (!value.is_string()) {
                        index_params[key] = value.dump();
                    }
                }
            } catch (nlohmann::json::parse_error& e) {
                auto err =
                    string_util::SFormat("Index params err: {}", e.what());
                LOG_ERROR(err);
                return Status::ParameterInvalid(err);
            }

        } else {
            index_params[param.key()] = param.value();
        }
    }

    if (IsVectorIndex(field_ptr->data_type())) {
        auto it = index_params.find(kIndexTypeKey);
        if (it == index_params.end() || it->second == kAutoIndex) {
            // default index and auto index only support floatvector type
            if (!AddAutoIndexParams(&index_params))
                return Status::SegcoreErr();
        }

        auto metric_it = index_params.find(kMetricTypeKey);
        if (metric_it == index_params.end()) {
            if (field_ptr->data_type() == DType::FloatVector ||
                field_ptr->data_type() == DType::BFloat16Vector ||
                field_ptr->data_type() == DType::Float16Vector) {
                index_params[kMetricTypeKey] = kFloatVectorDefaultMetricType;
            } else if (field_ptr->data_type() == DType::BinaryVector) {
                index_params[kMetricTypeKey] = kBinaryVectorDefaultMetricType;
            } else if (field_ptr->data_type() == DType::SparseFloatVector) {
                index_params[kMetricTypeKey] =
                    kSparseFloatVectorDefaultMetricType;
            } else {
                LOG_ERROR("Unkwon index data type: {}", field_ptr->data_type());
                return Status::ParameterInvalid();
            }
        }

        if (index_params[kMetricTypeKey] == kMetricsBM25Name) {
            if (index_params.find(kBm25K1) == index_params.end()) {
                index_params[kBm25K1] = "1.2";
            }
            if (index_params.find(kBm25B) == index_params.end()) {
                index_params[kBm25B] = "0.75";
            }
            if (index_params.find(kBM25AvgName) == index_params.end()) {
                index_params[kBM25AvgName] = "1000";
            }
        }

    } else {
        // scalar index
        auto it = index_params.find(kIndexTypeKey);
        if (field_ptr->data_type() == DType::VarChar) {
            if (it == index_params.end()) {
                index_params[kIndexTypeKey] = kDefaultStringIndexType;
            } else if (!ValidateStringIndexType(it->second)) {
                auto err =
                    string_util::SFormat("Unkown index type {}", it->second);
                LOG_ERROR(err);
                return Status::ParameterInvalid(err);
            }
        } else if (field_ptr->data_type() == DType::Float ||
                   field_ptr->data_type() == DType::Double ||
                   field_ptr->data_type() == DType::Int16 ||
                   field_ptr->data_type() == DType::Int8 ||
                   field_ptr->data_type() == DType::Int32 ||
                   field_ptr->data_type() == DType::Int64) {
            if (it == index_params.end()) {
                index_params[kIndexTypeKey] = kDefaultArithmeticIndexType;
            } else if (!ValidateArithmeticIndexType(it->second)) {
                auto err =
                    string_util::SFormat("Unkown index type {}", it->second);
                LOG_ERROR(err);
                return Status::ParameterInvalid(err);
            }
        } else if (field_ptr->data_type() == DType::Bool) {
            if (it == index_params.end()) {
                LOG_ERROR("no index type specified");
                return Status::ParameterInvalid("no index type specified");
            }
            if (it->second != kInvertedIndexType) {
                auto err = string_util::SFormat(
                    "index type {} not supported for boolean, supported: {}",
                    it->second,
                    kInvertedIndexType);
                LOG_ERROR(err);
                return Status::ParameterInvalid(err);
            }
        } else {
            LOG_ERROR(
                "Only int, varchar, float, double and bool fields support "
                "scalar index.");
            return Status::ParameterInvalid();
        }
    }
    auto it = index_params.find(kIndexTypeKey);
    if (it == index_params.end()) {
        LOG_ERROR("IndexType not specified");
        return Status::ParameterInvalid();
    }

    CHECK_STATUS(CheckTrain(*field_ptr, index_params), "");

    index_params.erase(kDimKey);
    index_params.erase(kMaxLengthKey);

    for (const auto& param : index_params) {
        ::milvus::proto::common::KeyValuePair p;
        p.set_key(param.first);
        p.set_value(param.second);
        new_index_params_.push_back(p);
    }

    auto type_params = field_ptr->type_params();
    KVMap type_params_map;
    for (const auto& param : type_params) {
        ::milvus::proto::common::KeyValuePair p;
        p.set_key(param.key());
        p.set_value(param.value());
        new_type_params_.push_back(p);
    }
    return Status::Ok();
}

bool
CreateIndexTask::IsVectorIndex(::milvus::proto::schema::DataType dtype) {
    return schema_util::IsVectorField(dtype);
}

bool
CreateIndexTask::FillDimension(
    const ::milvus::proto::schema::FieldSchema& field, KVMap* index_params) {
    if (!IsVectorIndex(field.data_type())) {
        return true;
    }

    std::string dim;
    if (!schema_util::FindDimFromFieldParams(field, &dim)) {
        LOG_ERROR("Dimension not found in schema");
        return false;
    }

    auto it = index_params->find(kDimKey);
    if (it != index_params->end() && it->second != dim) {
        LOG_ERROR("dimension mismatch, dimension in schema: {}, dimension: {}",
                  dim,
                  it->second);
        return false;
    } else {
        (*index_params)[kDimKey] = dim;
    }
    return true;
}

Status
CreateIndexTask::Process(milvus::proto::segcore::FieldIndexMeta* field_meta) {
    CHECK_STATUS(ParseIndexParams(), "");

    field_meta->set_index_name(index_name_);
    field_meta->set_fieldid(field_id_);
    field_meta->set_collectionid(collectionid_);
    field_meta->set_is_auto_index(is_auto_index_);

    std::set<std::string> kset;

    for (const auto& param : new_index_params_) {
        if (kset.find(param.key()) == kset.end()) {
            auto pair = field_meta->add_index_params();
            pair->set_key(param.key());
            pair->set_value(param.value());
            kset.insert(param.key());
        }
    }

    for (const auto& param : new_type_params_) {
        if (kset.find(param.key()) == kset.end()) {
            auto pair = field_meta->add_index_params();
            pair->set_key(param.key());
            pair->set_value(param.value());
            kset.insert(param.key());
        }
    }

    for (const auto& param : new_extra_params_) {
        if (kset.find(param.key()) == kset.end()) {
            auto pair = field_meta->add_index_params();
            pair->set_key(param.key());
            pair->set_value(param.value());
            kset.insert(param.key());
        }
    }

    return Status::Ok();
}

}  // namespace milvus::local
