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

#include "storage.h"
#include <gtest/gtest.h>
#include "test_util.h"

namespace milvus::local {

TEST(Storage, h) {
    // auto schema_str = create_test_collection();
    // auto index_str = create_test_index();

    // ::milvus::proto::msg::InsertRequest insert;
    // auto row_data = insert.add_row_data();

    // const char* db_path = "test.db";
    // {
    //     Storage s(db_path);
    //     s.open();
    //     s.create_collection("test", schema_str);
    //     s.create_index("test", "test_index", index_str);
    // }

    // {
    //     Storage s(db_path);
    //     s.open();
    //     std::string schema;
    //     std::string index;
    //     s.get_collection_schema("test", schema);
    //     s.get_index("test", "test_index", &index);
    //     ::milvus::proto::schema::CollectionSchema sc;
    //     sc.ParseFromString(schema);

    //     milvus::proto::segcore::FieldIndexMeta index_mt;
    //     index_mt.ParseFromString(index);
    //     std::cout << index_mt.index_name();
    // }
    // std::remove(db_path);
}

}  // namespace milvus::local
