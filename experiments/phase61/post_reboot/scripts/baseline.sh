#!/usr/bin/env bash
# Phase 61 post-reboot: re-measure the current binaries on the recovered NPU.
# Vulkan and hybrid runs are interleaved so machine-level drift hits both.
#
# Engines and routing:
#   c4_mmul_v2 : phase 60 fidelity baseline  (route all)
#   c4_mmul_v3 : phase 61 fidelity production (route all)
#   c4_mmul_v4 : phase 62 per-chunk scale     (route gate,up,down: only FFN
#                kernels are compiled; attention N=4096 has no c256 kernel)
set -u
ROOT=/workspace/llama.cpp
LOG=$ROOT/experiments/phase61/post_reboot/logs
mkdir -p "$LOG"
CACHE=$ROOT/experiments/npu-int8-gemm/kernel_cache
MODEL=$ROOT/models/Qwen3.5-0.8B-Q8_0.gguf
GUARD=$ROOT/experiments/phase48/npu_guard.sh
REPS=${1:-5}
PP=${2:-512}

route_for () {  # engine -> route ops
  case "$1" in
    c4_mmul_v4) echo "gate,up,down" ;;
    *)          echo "all" ;;
  esac
}

vulkan () {  # tag pp reps
  local tag=$1 pp=$2 reps=$3
  env GGML_XDNA_DISABLE=1 build/bin/llama-bench -m "$MODEL" -ngl 99 -p "$pp" -n 0 -r "$reps" \
    2>&1 | tee "$LOG/${tag}.txt" | grep -E "pp$pp|error|fail"
}

hybrid () {  # tag engine pp reps [extra env]
  local tag=$1 engine=$2 pp=$3 reps=$4; shift 4
  local route; route=$(route_for "$engine")
  env GGML_XDNA_QWEN_PP=1 XDNA_KERNEL_CACHE="$CACHE" XDNA_Q8_ENGINE="$engine" \
      XDNA_ROUTE_OPS="$route" "$@" \
    "$GUARD" --timeout 120 -- build/bin/llama-bench -m "$MODEL" -ngl 99 -p "$pp" -n 0 -r "$reps" \
    2>&1 | tee "$LOG/${tag}.txt" | grep -E "pp$pp|missing_kernel|error|fail|abort|TIMEOUT" | tail -6
}

echo "########## interleaved baseline pp=$pp reps=$reps $(date) ##########"
vulkan        vulkan_pp${pp}_a  "$pp" "$REPS"
hybrid v2  c4_mmul_v2 "$pp" "$REPS"
hybrid v3  c4_mmul_v3 "$pp" "$REPS"
hybrid v4  c4_mmul_v4 "$pp" "$REPS"
vulkan        vulkan_pp${pp}_b  "$pp" "$REPS"
hybrid v4b c4_mmul_v4 "$pp" "$REPS"
echo "########## DONE $(date) ##########"
