#!/usr/bin/env bash
# Phase 61: perplexity with the NPU guard.
#
# usage: run_ppl.sh <tag> <engine|off> <route_ops> [extra env assignments]
set -u
ROOT=/workspace/llama.cpp
LOG=$ROOT/experiments/phase61/logs
mkdir -p "$LOG"

TAG=$1; ENGINE=$2; ROUTE=$3; shift 3

ENVS=""
if [ "$ENGINE" != "off" ]; then
  ENVS="$ENVS GGML_XDNA_QWEN_PP=1"
  ENVS="$ENVS XDNA_KERNEL_CACHE=$ROOT/experiments/npu-int8-gemm/kernel_cache"
  ENVS="$ENVS XDNA_Q8_ENGINE=$ENGINE"
fi
if [ "$ROUTE" != "none" ]; then
  ENVS="$ENVS XDNA_ROUTE_OPS=$ROUTE"
fi
ENVS="$ENVS $*"

OUT=$LOG/ppl_${TAG}.txt
echo "=== $TAG :: $ENVS" | tee "$OUT"
# shellcheck disable=SC2086
env $ENVS "$ROOT/experiments/phase48/npu_guard.sh" -- \
  "$ROOT/build/bin/llama-perplexity" -m "$ROOT/models/Qwen3.5-0.8B-Q8_0.gguf" \
  -f "$ROOT/experiments/phase61/ppl/wiki.test.raw" \
  -c 512 --chunks 6 2>&1 | tee -a "$OUT"
echo "=== done $TAG" | tee -a "$OUT"
