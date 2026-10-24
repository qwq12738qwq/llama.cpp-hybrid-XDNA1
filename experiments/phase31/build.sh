#!/bin/bash
# Phase 31 microbenchmark build helper.
#
# Usage:
#   bash build.sh            # rebuild libggml-vulkan if needed, then compile the bench
#   bash build.sh norlib     # only compile the bench
set -e
cd "$(dirname "$0")"
ROOT=../..
BUILD="$ROOT/build-new"

if [ "$1" != "norlib" ]; then
    make -C "$BUILD" ggml-vulkan -j"$(nproc)"
fi

g++ -O2 -std=c++17 attention_partition_bench.cpp -o attention_partition_bench \
    -I "$ROOT/ggml/include" -I "$ROOT/ggml/src" \
    -L "$BUILD/bin" -lggml -lggml-base -lggml-cpu -lggml-vulkan \
    -Wl,-rpath,"$(cd "$BUILD/bin" && pwd)"
echo "built $PWD/attention_partition_bench"
