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


# get host
OS=$(uname -s)
ARCH=$(uname -m)

case $OS in
    Linux)
        osname=linux
        ;;
    MINGW*)
        osname=msys
        ;;
    Darwin)
        osname=macosx
        ;;
    *)
        osname=none
        ;;
esac

# clone milvus
if [[ ! -d milvus ]] ; then
    git clone ${MILVUS_REPO} milvus
    cd milvus
    git checkout ${MILVUS_VERSION}
    # apply milvus patch later if needed
    if [ -f ../patches/milvus-${MILVUS_VERSION}.patch ] ; then
        patch -p1 < ../patches/milvus-${MILVUS_VERSION}.patch
    fi
    if [ -d ../patches/milvus-${MILVUS_VERSION} ] ; then
        for pf in ../patches/milvus-${MILVUS_VERSION}/*.patch ; do
            git apply ${pf}
        done
    fi
    cd -
fi

# patch Makefile
if [[ "${osname}" == "macosx" ]] ; then
    sed -i '' 's/-ldflags="/-ldflags="-s -w /' milvus/Makefile
    sed -i '' 's/-ldflags="-s -w -s -w /-ldflags="-s -w /' milvus/Makefile
    sed -i '' 's/="-dev"/="-lite"/' milvus/Makefile
else
    sed 's/-ldflags="/-ldflags="-s -w /' -i milvus/Makefile
    sed 's/-ldflags="-s -w -s -w /-ldflags="-s -w /' -i milvus/Makefile
    sed 's/="-dev"/="-lite"/' -i milvus/Makefile
fi

# build for linux x86_64
function build_linux_x86_64() {
    cd milvus
    # conan need for milvus 2.3
    pip3 install --user "conan<2.0"
    export PATH=${HOME}/.local/bin:${PATH}
    make -j $(nproc) milvus
    cd bin
    rm -fr lib*

    has_new_file=true
    while ${has_new_file} ; do
        has_new_file=false
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
                for p in ../internal/core/output/lib ../internal/core/output/lib64 /lib64 /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib ; do
                    if test -f $p/$x && ! test -f $x ; then
                        file=$p/$x
                        while test -L $file ; do
                            filelink=$(readlink $file)
                            if [[ $filelink =~ ^/ ]] ; then
                                file=$filelink
                            else
                                file=$(dirname $file)/$filelink
                            fi
                        done
                        cp -frv $file $x.normal
                        echo remove rpath for $x}
                        patchelf --remove-rpath $x.normal
                        mv -fv $x.normal $x
                        strip $x
                        has_new_file=true
                    fi
                done
            fi
        done
    done
    patchelf --remove-rpath milvus
    strip milvus
}

function install_deps_for_macosx() {
    bash milvus/scripts/install_deps.sh
    # need this for cache binary
    brew install coreutils
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
    install_deps_for_macosx
    build_macosx_common
}

function build_msys() {
    cd milvus
    bash scripts/install_deps_msys.sh
    source scripts/setenv.sh

    export GOROOT=/mingw64/lib/go
    go version

    make -j $(nproc) milvus

    cd bin
    mv milvus milvus.exe
    rm -fr *.log *.dll

    has_new_file=true
    while ${has_new_file} ; do
        has_new_file=false
        for binary in milvus.exe *.dll ; do
            for x in $(ldd ${binary} | grep -vi /windows/ | awk '{print $1}') ; do
                if test -f $x ; then
                    :
                else
                    echo $x
                    for p in ../internal/core/output/lib ../internal/core/output/bin /mingw64/bin ; do
                        if test -f $p/$x && ! test -f $x ; then
                            file=$p/$x
                            while test -L $file ; do
                                file=$(dirname $file)/$(readlink $file)
                            done
                            cp -frv $file $x
                            has_new_file=true
                        fi
                    done
                fi
            done
        done
    done
}

function build_milvus() {
    set -e
    # prepare output
    cd ${build_dir}
    # check if prev build ok
    if [ -f output/build.txt ] ; then
        cp -fr env.sh output/env.sh.txt
        cp -fr build.sh output/build.sh.txt
        if md5sum -c output/build.txt ; then
            echo already build success, if you need rebuild it use -f flag or remove file: ${build_dir}/output/build.txt
            exit 0
        fi
    fi
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

    cd ${build_dir}
    rm -fr output && mkdir output
    cp -fr env.sh output/env.sh.txt
    cp -fr build.sh output/build.sh.txt
    cp -fr milvus/bin/* output
    md5sum output/* | grep -v build.txt > output/build.txt
}

build_milvus
