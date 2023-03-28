#!/bin/bash

export LANG=en_US.utf-8
set -e

build_dir=$(cd $(dirname $0); pwd)
cd ${build_dir}

# load envs
. env.sh

# getopts
while getopts "fr:b:p:" arg; do
    case $arg in
        f)
            BUILD_FORCE=YES
        ;;
        r)
            MILVUS_REPO=$OPTARG
        ;;
        b)
            MILVUS_VERSION=$OPTARG
        ;;
        p)
            BUILD_PROXY=$OPTARG
        ;;
        *)
        ;;
    esac
done

# proxy if needed
if [ ! -z "${BUILD_PROXY}" ] ; then
    echo using proxy during build: $BUILD_PROXY
    export http_proxy=${BUILD_PROXY}
    export https_proxy=${BUILD_PROXY}
fi

# remove milvus source if build force
if [[ ${BUILD_FORCE} == "YES" ]] ; then
    rm -fr milvus
fi

# clone milvus
if [[ ! -d milvus ]] ; then
    git clone ${MILVUS_REPO} milvus
    cd milvus
    git checkout ${MILVUS_VERSION}
    # apply milvus patch later if needed
    # patch -p1 < ../milvus_patches/${MILVUS_PATCH_NAME}.patch
    cd -
fi

# get host
OS=$(uname -s)
ARCH=$(uname -m)


# patch Makefile
if [[ "${OS}" == "Darwin" ]] ; then
    sed -i '' 's/-ldflags="/-ldflags="-s -w /' milvus/Makefile
    sed -i '' 's/-ldflags="-s -w -s -w /-ldflags="-s -w /' milvus/Makefile
else
    sed 's/-ldflags="/-ldflags="-s -w /' -i milvus/Makefile
    sed 's/-ldflags="-s -w -s -w /-ldflags="-s -w /' -i milvus/Makefile
fi

# build for linux x86_64
function build_linux_x86_64() {
    cd milvus
    # conan after 2.3
    # pip3 install "conan<2.0"
    make -j $(nproc) milvus
    cd bin
    rm -fr lib*

    for x in $(ldd milvus | awk '{print $1}') ; do
        if [[ $x =~ libc.so.* ]] ; then
            :
        elif [[ $x =~ libdl.so.* ]] ; then
            :
        elif [[ $x =~ libm.so.* ]] ; then
            :
        elif [[ $x =~ librt.so.* ]] ; then
            :
        elif [[ $x =~ libpthread.so.* ]] ; then
            :
        elif test -f $x ; then
            :
        else
            echo $x
            for p in ../internal/core/output/lib ../internal/core/output/lib64 /lib64 /usr/lib64 /usr/local/lib64 /usr/local/lib /usr/lib64/boost169 ; do
                if test -f $p/$x && ! test -f $x ; then
                    file=$p/$x
                    while test -L $file ; do
                        file=$(dirname $file)/$(readlink $file)
                    done
                    cp -frv $file $x
                fi
            done
        fi
    done
}

function install_deps_for_macosx() {
    brew install boost libomp ninja tbb openblas ccache pkg-config
    if [[ ! -d "/usr/local/opt/llvm" ]]; then
        ln -s /usr/local/opt/llvm@14 /usr/local/opt/llvm
    fi
}

# build for macos arm64/x86_64
build_macosx_common() {
    cd milvus
    make -j $(sysctl -n hw.physicalcpu) milvus

    # resolve dependencies for milvus
    cd bin
    rm -fr lib*
    files=("milvus")
    while true ; do
        new_files=()
        for file in ${files[@]} ; do
            for line in $(otool -L $file | grep -v ${file}: | grep -v /usr/lib | grep -v /System/Library | awk '{print $1}') ; do
                filename=$(basename $line)
                if [[ -f ${filename} ]] ; then
                    continue
                fi
                find_in_build_dir=$(find ../cmake_build -name $filename)
                if [[ ! -z "$find_in_build_dir" ]] ; then
                    cp -frv ${find_in_build_dir} ${filename}
                    new_files+=( "${filename}" )
                    continue
                fi
                if [[ -f $line ]] ; then
                    cp -frv $line $filename
                    new_files+=( "${filename}" )
                    continue
                fi
            done
        done
        if [[ ${#new_files[@]} -eq 0 ]] ; then
            break
        fi
        for file in ${new_files[@]} ; do
            files+=( ${file} )
        done
    done
}


function build_macosx_x86_64() {
    install_deps_for_macosx
    build_macosx_common
}

function build_macosx_arm64() {
    build_macosx_common
}


function build_milvus() {
    set -e
    if [ -f ${build_dir}/milvus/build.ok ] ; then
        echo already build success, if you need rebuild it use -f flag or remove file: ${build_dir}/milvus/build.ok
    else
        # build for os
        case $OS in
            Linux)
                build_linux_${ARCH}
                ;;
            MINGW*)
                build_msys
                ;;
            Darwin)
                build_macosx_${ARCH}
                ;;
            *)
                ;;
        esac
        touch ${build_dir}/milvus/build.ok
    fi
}

build_milvus