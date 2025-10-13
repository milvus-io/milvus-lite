from conans import ConanFile
from conan.tools.cmake import CMake


class MilvusLiteConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = (
        # gtest
        "gtest/1.13.0",
        # glog
        "xz_utils/5.4.0#a6d90890193dc851fa0d470163271c7a",
        "zlib/1.2.13#df233e6bed99052f285331b9f54d9070",
        "glog/0.6.0#d22ebf9111fed68de86b0fa6bf6f9c3f",
        # protobuf
        "protobuf/3.21.4#fd372371d994b8585742ca42c12337f9",
        # folly
        "fmt/9.1.0#95259249fb7ef8c6b5674a40b00abba3",
        "folly/2023.10.30.08@milvus/dev#81d7729cd4013a1b708af3340a3b04d9",
        # antlr
        "antlr4-cppruntime/4.13.1",
        # sqlite
        "sqlitecpp/3.3.1",
        "onetbb/2021.9.0#4a223ff1b4025d02f31b65aedf5e7f4a",
        "nlohmann_json/3.11.2#ffb9e9236619f1c883e36662f944345d",
        "boost/1.82.0#744a17160ebb5838e9115eab4d6d0c06",
        "openssl/3.1.2#02594c4c0a6e2b4feb3cd15119993597",
        "libcurl/7.86.0#bbc887fae3341b3cb776c601f814df05",
        "grpc/1.50.1@milvus/dev#75103960d1cac300cf425ccfccceac08",
        "prometheus-cpp/1.1.0#ea9b101cb785943adb40ad82eda7856c",
        "re2/20230301#f8efaf45f98d0193cd0b2ea08b6b4060",
        # "simdjson/3.7.0",
        "arrow/15.0.0#0456d916ff25d509e0724c5b219b4c45",
        "double-conversion/3.2.1#640e35791a4bac95b0545e2f54b7aceb",
        "marisa/0.2.6#68446854f5a420672d21f21191f8e5af",
        "zstd/1.5.4#308b8b048f9a3823ce248f9c150cc889",
        "yaml-cpp/0.7.0#9c87b3998de893cf2e5a08ad09a7a6e0",
        "libdwarf/20191104#7f56c6c7ccda5fadf5f28351d35d7c01",
        "rapidjson/cci.20230929#624c0094d741e6a3749d2e44d834b96c",
        "roaring/3.0.0#25a703f80eda0764a31ef939229e202d",
        "libevent/2.1.12#4fd19d10d3bed63b3a8952c923454bc0",
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
