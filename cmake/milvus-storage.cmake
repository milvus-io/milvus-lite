
file(GLOB_RECURSE SRC_FILES thirdparty/milvus-storage/cpp/src/*.cpp thirdparty/milvus-storage/cpp/src/*.cc)
add_library(milvus-storage ${SRC_FILES})
target_include_directories(milvus-storage PUBLIC BEFORE thirdparty/milvus-storage/cpp/include/milvus-storage thirdparty/milvus-storage/cpp/src)
target_link_libraries(milvus-storage PUBLIC
    arrow::arrow
    Boost::boost
    protobuf::protobuf
    glog::glog
    opendal
)
