#!/usr/bin/env bash
# Qwen3.5-4B: GPU long-context degradation curve, sandwich style.
# pp512 anchors before/after each point to flag thermal drift (documented on
# this machine: sustained long tests throttle the 780M).
set -u
ROOT=/workspace/llama.cpp
M=$ROOT/models/Qwen3.5-4B-Q4_K_M.gguf
LOG=$ROOT/experiments/phase61/post_reboot/logs/curve_4b.txt
: > "$LOG"
for pp in 512 2048 512 4096 512 8192 512 16384 512; do
  r=$(GGML_XDNA_DISABLE=1 timeout 900 "$ROOT/build/bin/llama-bench" -m "$M" -ngl 99 -p "$pp" -n 0 -r 2 2>/dev/null | grep -oE "[0-9.]+ ± [0-9.]+")
  echo "pp$pp: $r  $(date +%H:%M:%S)" | tee -a "$LOG"
  sleep 15
done
echo "=== profile FA share at pp8192 and pp16384 ===" | tee -a "$LOG"
for pp in 8192 16384; do
  GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 GGML_XDNA_DISABLE=1 \
    timeout 900 "$ROOT/build/bin/llama-bench" -m "$M" -ngl 99 -p "$pp" -n 0 -r 1 \
    > "$ROOT/experiments/phase61/post_reboot/logs/vkprof4b_pp$pp.txt" 2>&1
  echo "wrote vkprof4b_pp$pp.txt" | tee -a "$LOG"
  sleep 15
done
