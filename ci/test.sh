#!/bin/bash



# Function to display usage
usage() {
    echo "Usage: $0 -m {nightly|e2e}"
    exit 1
}

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    usage
fi

# Initialize variables
MODE=""

# Parse the command-line arguments
while getopts ":m:" opt; do
    case ${opt} in
        m )
            MODE=$OPTARG
            ;;
        \? )
            usage
            ;;
    esac
done

# Validate the MODE value
if [ "$MODE" != "nightly" ] && [ "$MODE" != "e2e" ]; then
    usage
fi


# Install dependencies
export PIP_TRUSTED_HOST="nexus-ci.zilliz.cc"
export PIP_INDEX_URL="https://nexus-ci.zilliz.cc/repository/pypi-all/simple"
export PIP_INDEX="https://nexus-ci.zilliz.cc/repository/pypi-all/pypi"
export PIP_FIND_LINKS="https://nexus-ci.zilliz.cc/repository/pypi-all/pypi"
python3 -m pip install --no-cache-dir -r tests/requirements.txt --timeout 300 --retries 6

cd tests/milvus_lite

# Main logic based on the MODE
if [ "$MODE" == "nightly" ]; then
    echo "Running in nightly mode"
    pytest -v -m 'not L3' --enable_milvus_local_api ./lite-nighlty.db   # nighlty
elif [ "$MODE" == "e2e" ]; then
    echo "Running in e2e mode"
    pytest -s  -v  --tags  L0  --enable_milvus_local_api  lite-e2e.db    # pr 合并时
fi

