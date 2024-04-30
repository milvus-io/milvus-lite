include_directories(${CMAKE_CURRENT_LIST_DIR})

add_library(milite STATIC
  ${CMAKE_CURRENT_LIST_DIR}/collection_meta.cpp
  ${CMAKE_CURRENT_LIST_DIR}/collection_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
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
  ${CMAKE_CURRENT_LIST_DIR}/schema_util.cpp    
)


target_link_libraries(
  milite
  PUBLIC
  parser
  milvus_proto
  milvus_segcore
  ${CONAN_LIBS}
  SQLiteCpp
  ${antlr4-cppruntime_LIBRARIES}
  marisa::marisa
  TBB::tbb  
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
