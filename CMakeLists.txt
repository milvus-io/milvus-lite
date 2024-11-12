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

cmake_minimum_required(VERSION 3.20)
project(milvus-lite)

option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(COMPILER_SUPPORTS_CXX17 True)
if(APPLE)
    SET(CMAKE_OSX_DEPLOYMENT_TARGET 11.0)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-g  -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-O3 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

if(APPLE)
    if(DEFINED ENV{HOMEBREW_PREFIX})
        message(STATUS "Homebrew prefix from environment: $ENV{HOMEBREW_PREFIX}")
        set(OMP_INCLUDE $ENV{HOMEBREW_PREFIX}/opt/libomp/include)
        include_directories($ENV{HOMEBREW_PREFIX}/opt/libomp/include)
        link_directories($ENV{HOMEBREW_PREFIX}/opt/libomp/lib)
        
    else()
        message(STATUS "Homebrew prefix from environment not found, use default")
        set(OMP_INCLUDE /opt/homebrew/opt/libomp/include)
        include_directories(/opt/homebrew/opt/libomp/include)
        link_directories(/opt/homebrew/opt/libomp/lib)
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lomp")
    set(OpenMP_C "${CMAKE_C_COMPILER}")
 	set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${OMP_INCLUDE}")
 	set(OpenMP_C_LIB_NAMES "libomp")
 	set(OpenMP_libomp_LIBRARY "omp")

 	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
 	set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${OMP_INCLUDE}")
    set(OpenMP_CXX_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "omp")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(MILVUS_LITE True)

include(FetchContent)

set(CONAN_LIBS "${CONAN_LIBS};marisa::marisa;")

find_package(SQLiteCpp REQUIRED)
include_directories(${SQLiteCpp_INCLUDE_DIRS})

find_package(antlr4-runtime REQUIRED)
include_directories(${antlr4-cppruntime_INCLUDES})

find_package(Protobuf REQUIRED)
include_directories(${protobuf_INCLUDE_DIRS})

find_package(gRPC REQUIRED)
include_directories(${gRPC_INCLUDE_DIRS})

find_package(TBB REQUIRED)
include_directories(${TBB_tbb_INCLUDE_DIRS})

find_package(nlohmann_json REQUIRED)
include_directories(${nlohmann_json_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(folly REQUIRED)
include_directories(${Folly_INCLUDE_DIRS})

find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})

find_package(glog REQUIRED)
include_directories(${glog_INCLUDE_DIRS})

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIRS})

find_package(Arrow REQUIRED)
include_directories(${arrow_INCLUDE_DIRS})

find_package(re2 REQUIRED)
include_directories(${re2_INCLUDE_DIRS})
link_directories(${re2_LIB_DIRS})

find_package(double-conversion REQUIRED)
include_directories(${double-conversion_INCLUDE_DIRS})

find_package(prometheus-cpp REQUIRED)
include_directories(${prometheus-cpp_INCLUDE_DIRS})

find_package(marisa REQUIRED)
include_directories(${marisa_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})

find_package(absl REQUIRED)
include_directories(${absl_type_traits_INCLUDE_DIRS})

add_definitions(-DANTLR4CPP_STATIC)
add_definitions(-DHAVE_CPP_STDLIB)
add_definitions(-DMILVUS_LITE)

IF(APPLE)
    add_definitions("-D_GNU_SOURCE")
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/pb)

execute_process(
  COMMAND git diff --quiet
  WORKING_DIRECTORY  ${CMAKE_SOURCE_DIR}/thirdparty/milvus
  RESULT_VARIABLE CHECK_RESULT
)

if(${CHECK_RESULT} EQUAL 0)
  message("Apply milvus patch...")
  execute_process(COMMAND git apply ${CMAKE_SOURCE_DIR}/thirdparty/milvus.patch
    WORKING_DIRECTORY  ${CMAKE_SOURCE_DIR}/thirdparty/milvus
    OUTPUT_VARIABLE result
  )
else ()
  message("Milvus not need applying patch...")
endif()

add_library(milvus_proto STATIC "${CMAKE_SOURCE_DIR}/src/proto/plan.proto"
    "${CMAKE_SOURCE_DIR}/src/proto/schema.proto"
    "${CMAKE_SOURCE_DIR}/src/proto/common.proto"
    "${CMAKE_SOURCE_DIR}/src/proto/segcore.proto"
    "${CMAKE_SOURCE_DIR}/src/proto/milvus.proto"
    "${CMAKE_SOURCE_DIR}/src/proto/msg.proto"
    "${CMAKE_SOURCE_DIR}/src/proto/feder.proto"
    "${CMAKE_SOURCE_DIR}/src/proto/rg.proto"
)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/pb")

protobuf_generate(
    TARGET milvus_proto
    IMPORT_DIRS "${CMAKE_SOURCE_DIR}/src/proto"
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_library(milvus_grpc_service STATIC
  "${CMAKE_SOURCE_DIR}/src/proto/milvus.proto"
)

target_link_libraries(milvus_grpc_service
  gRPC::grpc++
  milvus_proto
)

protobuf_generate(
  TARGET milvus_grpc_service
  # OUT_VAR PROTO_GENERATED_FILES
  LANGUAGE grpc
  GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
  PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
  IMPORT_DIRS  "${CMAKE_SOURCE_DIR}/src/proto"
  PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

include_directories("${CMAKE_SOURCE_DIR}/src/parser/")
include_directories("${CMAKE_SOURCE_DIR}/src/parser/antlr/")
include_directories("${CMAKE_BINARY_DIR}/pb")
include_directories("${CMAKE_BINARY_DIR}")

add_library(parser STATIC "${CMAKE_SOURCE_DIR}/src/parser/parser.cc"
    "${CMAKE_SOURCE_DIR}/src/parser/antlr/PlanBaseVisitor.cpp"
    "${CMAKE_SOURCE_DIR}/src/parser/antlr/PlanLexer.cpp"
    "${CMAKE_SOURCE_DIR}/src/parser/antlr/PlanParser.cpp"
    "${CMAKE_SOURCE_DIR}/src/parser/antlr/PlanVisitor.cpp"
  )


  target_link_libraries(parser milvus_proto ${antlr4-cppruntime_LIBRARIES})

add_subdirectory(thirdparty/milvus/internal/core/thirdparty/knowhere)

include_directories("${CMAKE_SOURCE_DIR}/thirdparty/milvus/internal/core/thirdparty/")
include_directories("${CMAKE_SOURCE_DIR}/thirdparty/milvus/internal/core/thirdparty/tantivy/tantivy-binding/include/")
include_directories("${CMAKE_SOURCE_DIR}/thirdparty/milvus/internal/core/thirdparty/tantivy/")

include_directories("${knowhere_SOURCE_DIR}/include")
function(MILVUS_ADD_PKG_CONFIG MODULE)
    configure_file(${MODULE}.pc.in "${CMAKE_CURRENT_BINARY_DIR}/${MODULE}.pc" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${MODULE}.pc"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
endfunction()


include(cmake/milvus-storage.cmake)

include_directories("${CMAKE_SOURCE_DIR}/thirdparty/milvus/internal/core/src/")
add_library(boost_bitset_ext 
 thirdparty/milvus/internal/core/thirdparty/boost_ext/dynamic_bitset_ext.cpp)


FetchContent_Declare(
    simdjson
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git
    GIT_TAG        v3.1.7
)
FetchContent_MakeAvailable(simdjson)


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build)
else ()
    set(CARGO_CMD cargo build --release)
endif ()

set(HOME_VAR $ENV{HOME})

add_custom_command(OUTPUT ls_cargo
        COMMENT "ls cargo"
        COMMAND ls ${HOME_VAR}/.cargo/bin/
        )

add_custom_target(ls_cargo_target DEPENDS ls_cargo)

add_custom_command(OUTPUT compile_tantivy
        COMMENT "Compiling tantivy binding"
        COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} MACOSX_DEPLOYMENT_TARGET=11.0 ${CARGO_CMD}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/thirdparty/milvus/internal/core/thirdparty/tantivy/tantivy-binding
        DEPENDS ls_cargo_target
    )

add_custom_target(tantivy_binding_target DEPENDS compile_tantivy)

add_library(tantivy_binding STATIC IMPORTED)
set_target_properties(tantivy_binding 
    PROPERTIES
    IMPORTED_GLOBAL TRUE
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/release/libtantivy_binding.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/thirdparty/milvus/internal/core/thirdparty/tantivy/tantivy-binding/include/"
)

add_dependencies(tantivy_binding tantivy_binding_target)

set(MILVUS_ENGINE_SRC
    "${CMAKE_SOURCE_DIR}/thirdparty/milvus/internal/core/src")
add_subdirectory(thirdparty/milvus/internal/core/src/log)
add_subdirectory(thirdparty/milvus/internal/core/src/config)
add_subdirectory(thirdparty/milvus/internal/core/src/common)
add_subdirectory(thirdparty/milvus/internal/core/src/storage)
add_subdirectory(thirdparty/milvus/internal/core/src/query)
add_subdirectory(thirdparty/milvus/internal/core/src/exec)
add_subdirectory(thirdparty/milvus/internal/core/src/index)
add_subdirectory(thirdparty/milvus/internal/core/src/segcore)
add_subdirectory(thirdparty/milvus/internal/core/src/bitset)

if(ENABLE_UNIT_TESTS)
  include(CTest)
  enable_testing()
endif()

add_subdirectory(src)

find_program(MEMORYCHECK_COMMAND NAMES valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --track-origins=yes --leak-check=full --show-leak-kinds=all")
