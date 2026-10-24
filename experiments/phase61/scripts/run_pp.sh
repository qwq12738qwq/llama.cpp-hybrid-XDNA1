#!/usr/bin/env bash
# Phase 61: PP benchmark with the NPU guard.
#
# usage: run_pp.sh <tag> <engine|off> <route_ops> <pp> <reps> [extra env assignments]
set -u
ROOT=/workspace/llama.cpp
LOG=$ROOT/experiments/phase61/logs
mkdir -p "$LOG"

TAG=$1; ENGINE=$2; ROUTE=$3; PP=$4; REPS=$5; shift 5

ENVS=""
if [ "$ENGINE" != "off" ]; then
  # GGML_XDNA_QWEN_PP=1 enables the NPU routing; without it the XDNA backend
  # stays registered but executes 0 ops, which is the Vulkan-only reference.
  ENVS="$ENVS GGML_XDNA_QWEN_PP=1"
  ENVS="$ENVS XDNA_KERNEL_CACHE=$ROOT/experiments/npu-int8-gemm/kernel_cache"
  ENVS="$ENVS XDNA_Q8_ENGINE=$ENGINE"
fi
if [ "$ROUTE" != "none" ]; then
  ENVS="$ENVS XDNA_ROUTE_OPS=$ROUTE"
fi
ENVS="$ENVS $*"

OUT=$LOG/${TAG}.txt
echo "=== $TAG :: $ENVS :: pp=$PP r=$REPS" | tee "$OUT"
# shellcheck disable=SC2086
env $ENVS "$ROOT/experiments/phase48/npu_guard.sh" -- \
  "$ROOT/build/bin/llama-bench" -m "$ROOT/models/Qwen3.5-0.8B-Q8_0.gguf" \
  -ngl 99 -p "$PP" -n 0 -r "$REPS" 2>&1 | tee -a "$OUT"
echo "=== done $TAG" | tee -a "$OUT"
