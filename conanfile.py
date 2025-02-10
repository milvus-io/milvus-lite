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
        "protobuf/3.21.12",
        # folly
        "fmt/9.1.0",
        "folly/2023.10.30.09@milvus/dev",
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
        "grpc/1.50.1",
        "prometheus-cpp/1.1.0",
        "re2/20230301",
        "simdjson/3.7.0",
        "arrow/12.0.1",
        "double-conversion/3.2.1",
        "marisa/0.2.6",
        "zstd/1.5.4#308b8b048f9a3823ce248f9c150cc889",
        "yaml-cpp/0.7.0",
        "libdwarf/0.9.1",
        "rapidjson/cci.20230929#624c0094d741e6a3749d2e44d834b96c",
        "roaring/3.0.0#25a703f80eda0764a31ef939229e202d",
    )

    generators = {"cmake", "cmake_find_package"}

    default_options = {
        "glog:with_gflags": True,
        "glog:shared": True,
        "gtest:build_gmock": False,
        "onetbb:tbbmalloc": False,
        "onetbb:tbbproxy": False,
        "boost:without_locale": True,
        "boost:without_test": True,
        "boost:without_stacktrace": True,
        "fmt:header_only": True,
        "prometheus-cpp:with_pull": False,
        "double-conversion:shared": True,
        "arrow:filesystem_layer": True,
        "arrow:parquet": True,
        "arrow:compute": True,
        "arrow:with_re2": True,
        "arrow:with_zstd": False,
        "arrow:with_boost": True,
        "arrow:with_thrift": True,
        "arrow:with_jemalloc": True,
        "arrow:shared": False,
        "arrow:with_s3": False,
        "libcurl:with_ssl": False,
    }

    def configure(self):
        if self.settings.os in ["Macos", "Android"]:
            self.options["arrow"].with_jemalloc = False

        if self.settings.compiler == "gcc":
            if self.settings.compiler.libcxx == "libstdc++":
                raise Exception("This package is only compatible with libstdc++11")

    def config_options(self):
        if self.settings.os != "Macos":
            self.options["onetbb"].tbbbind = False

    def requirements(self):
        if self.settings.os not in ["Macos", "Android"]:
            self.requires("libunwind/1.7.2")
        if self.settings.os == "Android":
            self.requires("openblas/0.3.27")

    def imports(self):
        self.copy("*.so*", "./lib", "lib")
        self.copy("*.dylib", "./lib", "lib")

    def build(self):
        target = "11.0"
        self.run("export MACOSX_DEPLOYMENT_TARGET={}".format(target))
