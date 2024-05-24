from conans import ConanFile
from conan.tools.cmake import CMake


class MilvusLiteConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = (
        # gtest
        "gtest/1.13.0",
        # glog
        "xz_utils/5.4.5",
        "zlib/1.2.13",
        "glog/0.6.0",
        # protobuf
        "protobuf/3.21.4",
        # folly
        "fmt/9.1.0",
        "folly/2023.10.30.05@milvus/dev",
        # antlr
        "antlr4-cppruntime/4.13.1",
        # sqlite
        "sqlitecpp/3.3.1",
        "onetbb/2021.9.0",
        "nlohmann_json/3.11.2",
        "boost/1.82.0",
        "fmt/9.1.0",
        "openssl/1.1.1t",
        "libcurl/7.86.0",
        "opentelemetry-cpp/1.8.1.1@milvus/dev",
        "prometheus-cpp/1.1.0",
        "re2/20230301",
        "simdjson/3.7.0",
        "arrow/12.0.1",
        "double-conversion/3.2.1",
        "marisa/0.2.6",
        "zstd/1.5.4",
        "yaml-cpp/0.7.0",
        "libdwarf/0.9.1",
        "google-cloud-cpp/2.5.0@milvus/dev",
    )

    generators = {"cmake", "cmake_find_package"}

    default_options = {
        "glog:with_gflags": True,
        "glog:shared": True,
        "gtest:build_gmock": False,
        "onetbb:tbbmalloc": False,
        "onetbb:tbbproxy": False,
        "boost:without_locale": False,
        "boost:without_test": True,
        "fmt:header_only": True,
        "prometheus-cpp:with_pull": False,
        "double-conversion:shared": True,
        "arrow:filesystem_layer": True,
        "arrow:parquet": True,
        "arrow:compute": True,
        "arrow:with_re2": True,
        "arrow:with_zstd": True,
        "arrow:with_boost": True,
        "arrow:with_thrift": True,
        "arrow:with_jemalloc": True,
        "arrow:shared": False,
        "arrow:with_s3": True,
        "aws-sdk-cpp:config": True,
        "aws-sdk-cpp:text-to-speech": False,
        "aws-sdk-cpp:transfer": False,
    }

    def configure(self):
        if self.settings.os == "Macos":
            self.options["arrow"].with_jemalloc = False

        if self.settings.compiler == "gcc":
            if self.settings.compiler.libcxx == "libstdc++":
                raise Exception("This package is only compatible with libstdc++11")

    def requirements(self):
        if self.settings.os != "Macos":
            self.requires("libunwind/1.7.2")

    def imports(self):
        self.copy("*.so*", "./lib", "lib")
        self.copy("*.dylib", "./lib", "lib")

    def build(self):
        target = "11.0"
        self.run("export MACOSX_DEPLOYMENT_TARGET={}".format(target))

