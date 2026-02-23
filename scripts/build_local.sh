#!/bin/bash

# Build milvus-lite wheel from local source using Docker.
# Mounts your working tree into the manylinux container so you can iterate
# on C++ / Python changes without a native toolchain. A Docker named volume
# caches Conan packages between runs (first build ~13 min, subsequent ~30s).
#
# Usage:
#   ./scripts/build_local.sh            # build wheel (Conan cache enabled)
#   ./scripts/build_local.sh --no-cache # fresh build, discard Conan cache
#
# Output: dist/milvus_lite-*.whl, dist/build.log
#
# To run tests against the built wheel: ./scripts/test_local.sh
#
# If the Conan cache gets corrupted: docker volume rm milvus_lite_conan_cache

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_TAG="latest"
VOLUME_NAME="milvus_lite_conan_cache"

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    DOCKERFILE="$SCRIPT_DIR/Dockerfile.manylinux.aarch64"
else
    DOCKERFILE="$SCRIPT_DIR/Dockerfile.manylinux.x86_64"
fi

USE_CACHE=true
for arg in "$@"; do
    case "$arg" in
        --no-cache) USE_CACHE=false ;;
        --cache)    USE_CACHE=true ;;
    esac
done

CACHE_ARGS=""
if $USE_CACHE; then
    echo "==> Using Docker volume '$VOLUME_NAME' for Conan cache"
    CACHE_ARGS="-e CONAN_USER_HOME=/workspace/conan -v $VOLUME_NAME:/workspace/conan"
fi

echo "==> Building Docker image from $DOCKERFILE..."
docker build -t build_milvus_lite:$IMAGE_TAG -f "$DOCKERFILE" "$SCRIPT_DIR"

echo "==> Building milvus-lite from local source..."
echo "==> Build log will be at: $REPO_ROOT/dist/build.log"
mkdir -p "$REPO_ROOT/dist"

INNER_SCRIPT=$(mktemp)
trap "rm -f $INNER_SCRIPT" EXIT
cat > "$INNER_SCRIPT" << 'BUILDEOF'
set -e
LOG=/workspace/out/build.log
echo "Build started at $(date)" > $LOG

pip3 install wheel build >> $LOG 2>&1

# Ensure Milvus JFrog remote is configured when using a custom CONAN_USER_HOME
conan remote add default-conan-local \
    https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local \
    --insert 0 >> $LOG 2>&1 || true

# Perl modules needed if openssl must be built from source
yum install -y perl-IPC-Cmd perl-Digest-SHA >> $LOG 2>&1 || true

cd /workspace/milvus-lite/python
echo "==> Starting wheel build..." | tee -a $LOG
python3 -m build --wheel --no-isolation >> $LOG 2>&1 \
    || { echo "BUILD FAILED. Tail of log:"; tail -80 $LOG; exit 1; }

cp dist/*.whl /workspace/out/
echo "==> Done! Wheel(s) copied to dist/" | tee -a $LOG
BUILDEOF

docker run --rm \
    -v "$REPO_ROOT:/workspace/milvus-lite" \
    -v "$REPO_ROOT/dist:/workspace/out" \
    -v "$INNER_SCRIPT:/workspace/build_inner.sh:ro" \
    $CACHE_ARGS \
    build_milvus_lite:$IMAGE_TAG \
    bash /workspace/build_inner.sh

echo "==> Wheel(s) available in $REPO_ROOT/dist/"
ls -lh "$REPO_ROOT/dist/"*.whl 2>/dev/null
