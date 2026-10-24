#!/usr/bin/env bash
# Phase 62: per-operator profile of the Vulkan path at a given PP.
#
# usage: profile_vk.sh <pp> <reps>
set -u
ROOT=/workspace/llama.cpp
LOG=$ROOT/experiments/phase62/logs
PP=${1:-512}
REPS=${2:-1}

GGML_XDNA_DISABLE=1 GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
  "$ROOT/build/bin/llama-bench" -m "$ROOT/models/Qwen3.5-0.8B-Q8_0.gguf" \
  -ngl 99 -p "$PP" -n 0 -r "$REPS" > "$LOG/vkprof_pp${PP}.txt" 2>&1
echo "wrote $LOG/vkprof_pp${PP}.txt"
