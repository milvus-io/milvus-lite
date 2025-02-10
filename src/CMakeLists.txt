# Copyright (C) 2019-2024 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License

include_directories(${CMAKE_CURRENT_LIST_DIR})
add_library(milite ${LITE_BUILD_TYPE}
  ${CMAKE_CURRENT_LIST_DIR}/collection_meta.cpp
  ${CMAKE_CURRENT_LIST_DIR}/collection_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
  ${CMAKE_CURRENT_LIST_DIR}/timer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/index.cpp
  ${CMAKE_CURRENT_LIST_DIR}/milvus_local.cpp
  ${CMAKE_CURRENT_LIST_DIR}/segcore_wrapper.cpp
  ${CMAKE_CURRENT_LIST_DIR}/milvus_proxy.cpp
  ${CMAKE_CURRENT_LIST_DIR}/create_collection_task.cpp
  ${CMAKE_CURRENT_LIST_DIR}/create_index_task.cpp
  ${CMAKE_CURRENT_LIST_DIR}/search_task.cpp
  ${CMAKE_CURRENT_LIST_DIR}/insert_task.cpp
  ${CMAKE_CURRENT_LIST_DIR}/query_task.cpp
  ${CMAKE_CURRENT_LIST_DIR}/delete_task.cpp
  ${CMAKE_CURRENT_LIST_DIR}/upsert_task.cpp  
  ${CMAKE_CURRENT_LIST_DIR}/schema_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/re_scorer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/hybrid_search_task.cpp  
)

target_link_libraries(
  milite
  PUBLIC
  parser
  milvus_core
  milvus_log
  yaml-cpp
  roaring::roaring
  SQLiteCpp
  ${antlr4-cppruntime_LIBRARIES}
  marisa::marisa
  TBB::tbb
  protobuf::protobuf
  arrow::arrow
  simdjson
  tantivy_binding
)

add_library( 
  milvus_service
  STATIC
  "${CMAKE_SOURCE_DIR}/src/milvus_service_impl.cpp"
)

target_link_libraries(
  milvus_service
  PUBLIC
  milvus_grpc_service
)

add_executable(milvus server.cpp)

target_link_libraries(
  milvus
  milvus_service
  milite
)

if(ENABLE_UNIT_TESTS)
  add_subdirectory(unittest)
endif()
