#!/bin/bash

# Run Python tests against a locally built milvus-lite wheel using Docker.
# Requires a wheel in dist/ (built by build_local.sh).
#
# Usage:
#   ./scripts/test_local.sh                          # run default test set
#   ./scripts/test_local.sh tests/test_nullable.py   # run specific file(s)
#   ./scripts/test_local.sh tests/test_query.py tests/test_search.py
#
# By default this runs the test files in tests/ that don't depend on the shared
# test harness (conftest.py / common_func.py), which unconditionally imports
# jax, tensorflow, kubernetes, and pymilvus 2.4.4 at module load time.
# The tests in tests/milvus_lite/ and tests/testcases/ also test milvus-lite
# but can't run here due to those transitive imports. They require the full
# CI environment — see tests/requirements.txt.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WHEEL=$(ls -t "$REPO_ROOT"/dist/milvus_lite-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "Error: no wheel found in dist/. Run ./scripts/build_local.sh first."
    exit 1
fi
WHEEL_NAME=$(basename "$WHEEL")

DEFAULT_TESTS=(
    tests/test_nullable.py
    tests/test_query.py
    tests/test_search.py
    tests/test_delete.py
    tests/test_schema.py
    tests/test_bm25.py
)

if [ $# -gt 0 ]; then
    TEST_PATHS=("$@")
else
    TEST_PATHS=("${DEFAULT_TESTS[@]}")
fi

CONTAINER_PATHS=""
for t in "${TEST_PATHS[@]}"; do
    CONTAINER_PATHS="$CONTAINER_PATHS /workspace/milvus-lite/$t"
done

echo "==> Testing with wheel: $WHEEL_NAME"
echo "==> Test files: ${TEST_PATHS[*]}"

docker run --rm \
    -v "$REPO_ROOT:/workspace/milvus-lite:ro" \
    -v "$REPO_ROOT/dist:/workspace/dist:ro" \
    -w /tmp/testrun \
    --platform linux/$(uname -m | sed 's/x86_64/amd64/;s/arm64/arm64/') \
    python:3.11-slim \
    bash -c "
set -e
pip install -q pytest pymilvus pandas numpy scipy \
    /workspace/dist/$WHEEL_NAME 2>&1 | tail -1
pytest $CONTAINER_PATHS -v --tb=short --noconftest -o 'addopts='
"
