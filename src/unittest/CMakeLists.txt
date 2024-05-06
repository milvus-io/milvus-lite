find_package(GTest REQUIRED)
include_directories(${GTest_INCLUDES})

find_program(PYTHON_EXECUTABLE NAMES python python3)

include_directories(${CMAKE_CURRENT_LIST_DIR})


set(MILVUS_LOCAL_TEST_DEPS
  milite
  GTest::gtest
  GTest::gtest_main  
)

add_executable(
  milvus_proxy_test
  milvus_proxy_test.cpp
  test_util.cpp  
)
target_link_libraries(milvus_proxy_test
  PRIVATE
  ${MILVUS_LOCAL_TEST_DEPS}  
)

add_test(
  NAME milvus_proxy_test
  COMMAND $<TARGET_FILE:milvus_proxy_test>
)

add_executable(
  server_test
  grpc_server_test.cpp
  test_util.cpp
)
target_link_libraries(
  server_test
  PRIVATE
  milvus_service
  ${MILVUS_LOCAL_TEST_DEPS}
)


add_test(
  NAME server_test
  COMMAND $<TARGET_FILE:server_test>
)

add_test(
  NAME run_examples
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_examples.py
)

set_tests_properties(run_examples
  PROPERTIES
  ENVIRONMENT "BIN_PATH=${CMAKE_BINARY_DIR}/lib;PYTHONPATH=${CMAKE_SOURCE_DIR}/python/src"
)
