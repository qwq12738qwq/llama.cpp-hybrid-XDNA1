#!/bin/bash
# build_gpu_bench.sh -- Build gpu_bench and gpu_burn inside npu-dev container
#
# Usage:
#   podman exec npu-dev bash /experiments/npu-heterogeneous/build_gpu_bench.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Copy from gpu-npu-prefill if not present locally
for f in gpu_bench.cpp gemm.spv gemm.comp; do
    if [ ! -f "$f" ] && [ -f "/experiments/gpu-npu-prefill/$f" ]; then
        cp "/experiments/gpu-npu-prefill/$f" .
    fi
done

# Copy from gpu-npu-overlap if not present locally
for f in gpu_burn.cpp; do
    if [ ! -f "$f" ] && [ -f "/experiments/gpu-npu-overlap/$f" ]; then
        cp "/experiments/gpu-npu-overlap/$f" .
    fi
done

if [ ! -f "gpu_bench.cpp" ]; then
    echo "ERROR: gpu_bench.cpp not found. Copy from experiments/gpu-npu-prefill/" >&2
    exit 1
fi

echo "Building gpu_bench..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include/vulkan \
    gpu_bench.cpp \
    -o gpu_bench \
    -lvulkan -lpthread

if [ -f "gpu_burn.cpp" ]; then
    echo "Building gpu_burn..."
    g++ -std=c++17 -O2 -Wall -Wextra \
        -I/usr/include/vulkan \
        gpu_burn.cpp \
        -o gpu_burn \
        -lvulkan -lpthread
fi

echo "Build complete:"
ls -lh gpu_bench gpu_burn 2>/dev/null
