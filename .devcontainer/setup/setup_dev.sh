#!/bin/bash

set -e

project_dir=$(dirname $(dirname $(cd $(dirname $0); pwd)))

python3 -m pip install --user -U pip
python3 -m pip install --user -r ${project_dir}/requirements.txt
