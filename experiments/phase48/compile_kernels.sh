#!/usr/bin/env bash
# Phase 48: pre-build the Q8SCALEV-C4 mmul-v2 kernels for every PP point.
#
# The XDNA kernel registry keys its cache by M (= physical ubatch), so a kernel
# exists only for the M values that were compiled earlier. A missing artifact
# makes supports_op() return false and the FFN silently falls back to CPU, so
# every M used by the matrix must be present before the NPU runs.
#
# Compilation is aiecc only: it does not open /dev/accel/accel0.
#
# usage: compile_kernels.sh [M_LIST] [K,N ...]
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CACHE=${XDNA_KERNEL_CACHE:-$ROOT/experiments/npu-int8-gemm/kernel_cache}
COMPILER=$ROOT/experiments/phase40/compile_q8_c4_mmul_v2.py

M_LIST=${1:-"768 1024 1536 2048 3072 4096 5120"}
shift || true
SHAPES=${*:-"1024,3584 3584,1024"}

export XDNA_KERNEL_CACHE=$CACHE
mkdir -p "$CACHE"

for shape in $SHAPES; do
  K=${shape%%,*}
  N=${shape##*,}
  for M in $M_LIST; do
    dir="$CACHE/M${M}_K${K}_N${N}_C4_Q8SCALEV_MMUL"
    if [ -f "$dir/design.xclbin" ] && [ -f "$dir/design.insts.bin" ]; then
      echo "skip M=$M K=$K N=$N (cached)"
      continue
    fi
    echo "compiling M=$M K=$K N=$N -> $dir"
    if ! python3 "$COMPILER" --M "$M" --K "$K" --N "$N" --cols 4 --work-dir "$dir" 2>&1 | tail -3; then
      echo "FAILED M=$M K=$K N=$N" >&2
      exit 1
    fi
  done
done

echo "kernel cache:"
ls -d "$CACHE"/*Q8SCALEV_MMUL 2>/dev/null