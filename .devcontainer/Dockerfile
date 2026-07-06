FROM almalinux:8

RUN dnf install -y epel-release \
    && dnf config-manager --set-enabled powertools \
    && dnf install -y \
        gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ gcc-toolset-12-libatomic-devel \
        cmake git openblas-devel python3.11 python3.11-pip python3.11-devel \
        perl-IPC-Cmd perl-Digest-SHA perl-Thread-Queue perl-Thread-Queue \
    && dnf clean all \
    && ln -sf /usr/include/openblas/*.h /usr/include/ \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/pip3.11 /usr/local/bin/pip3

# Enable GCC 12 for all shells
ENV PATH="/opt/rh/gcc-toolset-12/root/usr/bin:${PATH}"

# Build toolchain
RUN pip3 install conan==1.63.0 wheel build
RUN curl https://sh.rustup.rs -sSf | bash -s -- --default-toolchain=1.89 -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN conan remote add default-conan-local \
    https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local

# Test dependencies
RUN pip3 install pymilvus==2.5.4 pytest scikit-learn pandas numpy scipy \
    timeout_decorator==0.5.0 delayed-assert==0.3.5 ujson==5.5.0 \
    pytest-html==3.1.1 'git+https://github.com/Projectplace/pytest-tags'

WORKDIR /workspace
