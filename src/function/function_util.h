// Copyright (C) 2019-2025 Zilliz. All rights reserved.
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

#pragma once

#include <string>

#include "pb/common.pb.h"
#include "pb/milvus.pb.h"
#include "status.h"

namespace milvus::local::function {

template <typename T>
class FunctionUtil {
 public:
    static Status
    ParseFieldData(milvus::proto::schema::FieldData* field_data,
                   const std::string& field_name,
                   std::vector<T>* values);
};

template <typename T>
Status
FunctionUtil<T>::ParseFieldData(milvus::proto::schema::FieldData* field_data,
                                const std::string& field_name,
                                std::vector<T>* values) {
    return Status::Ok();
}

}  // namespace milvus::local::function