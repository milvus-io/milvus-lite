#!/bin/bash

this_dir=$(dirname $0)

echo -n > ${this_dir}/env

echo DEV_USER=$(id -un) >> ${this_dir}/env
echo DEV_UID=$(id -u) >> ${this_dir}/env
echo DEV_GROUP=$(id -gn) >> ${this_dir}/env
echo DEV_GID=$(id -g) >> ${this_dir}/env
echo DEV_HOME=$HOME >> ${this_dir}/env
echo DEV_SHELL=$SHELL >> ${this_dir}/env

mkdir -p ${this_dir}/../../.conan
