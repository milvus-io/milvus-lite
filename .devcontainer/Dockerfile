FROM quay.io/pypa/manylinux2014_x86_64

COPY setup/centos7/install_deps.sh /tmp
RUN bash /tmp/install_deps.sh

COPY setup/centos7/install_tbb.sh /tmp
RUN bash /tmp/install_tbb.sh

COPY setup/centos7/install_openblas.sh /tmp
RUN bash /tmp/install_openblas.sh

ENV BOOST_INCLUDEDIR=/usr/include/boost169
ENV BOOST_LIBRARYDIR=/usr/lib64/boost169

# force using devtoolset
ENV PATH=/usr/lib64/ccache:/opt/rh/devtoolset-7/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/opt/rh/devtoolset-7/root/usr/lib64/dyninst:/opt/rh/devtoolset-7/root/usr/lib/dyninst:/usr/local/lib64
ENV DEVTOOLSET_ROOTPATH=/opt/rh/devtoolset-7/root
