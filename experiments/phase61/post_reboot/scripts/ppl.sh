#!/usr/bin/env bash
# Phase 61 post-reboot: perplexity correctness gate.
# usage: ppl.sh <tag> <engine|off>
set -u
ROOT=/workspace/llama.cpp
LOG=$ROOT/experiments/phase61/post_reboot/logs
CACHE=$ROOT/experiments/npu-int8-gemm/kernel_cache
MODEL=$ROOT/models/Qwen3.5-0.8B-Q8_0.gguf
CORPUS=$ROOT/experiments/phase61/ppl/wiki.test.raw
GUARD=$ROOT/experiments/phase48/npu_guard.sh
TAG=$1; ENGINE=$2

OUT=$LOG/ppl_${TAG}.txt
if [ "$ENGINE" = "off" ]; then
  env GGML_XDNA_DISABLE=1 build/bin/llama-perplexity -m "$MODEL" -f "$CORPUS" \
      -c 512 --chunks 6 2>&1 | tee "$OUT" | grep -iE "final estimate|perplexity:.*pass|error|FAIL|abort"
else
  route=all
  [ "$ENGINE" = "c4_mmul_v4" ] && route=gate,up,down
  env GGML_XDNA_QWEN_PP=1 XDNA_KERNEL_CACHE="$CACHE" XDNA_Q8_ENGINE="$ENGINE" \
      XDNA_ROUTE_OPS="$route" \
    "$GUARD" --timeout 120 -- build/bin/llama-perplexity -m "$MODEL" -f "$CORPUS" \
      -c 512 --chunks 6 2>&1 | tee "$OUT" | grep -iE "final estimate|missing_kernel|error|FAIL|abort|WHITELIST|TIMEOUT"
fi
echo "=== done $TAG engine=$ENGINE"
