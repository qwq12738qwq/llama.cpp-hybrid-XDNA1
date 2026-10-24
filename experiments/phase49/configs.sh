#!/bin/bash
# Phase 49: the four flag combinations, run to settle semantics before any work.
#
#   A  QWEN_PP=0 WEIGHTS=0   Vulkan-only control
#   B  QWEN_PP=1 WEIGHTS=0   FFN routing enabled, weights stay on Vulkan
#   C  QWEN_PP=1 WEIGHTS=1   FFN routing + NPU weight placement (the real path)
#   D  QWEN_PP=0 WEIGHTS=1   weight placement without routing (expect: no NPU ops)
cd /workspace/llama.cpp/build
export LD_LIBRARY_PATH=/workspace/llama.cpp/build/bin
export XDNA_KERNEL_CACHE=/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache
M=../models/Qwen3.5-0.8B-Q8_0.gguf
OUT=/workspace/llama.cpp/experiments/phase49/results
TRACE=${TRACE:-1}

run() { # $1 cfg  $2 pp  $3 warmup  $4 measure  $5 tag
  local cfg=$1 pp=$2 w=$3 n=$4 tag=$5
  local dev="Vulkan0" envs="" guard=""
  case "$cfg" in
    A) dev="Vulkan0";              envs="GGML_XDNA_DISABLE=1" ;;
    B) dev="Vulkan0,AMD Phoenix XDNA1 NPU"; envs="GGML_XDNA_QWEN_PP=1" ;;
    C) dev="Vulkan0,AMD Phoenix XDNA1 NPU"; envs="GGML_XDNA_QWEN_PP=1 GGML_XDNA_WEIGHTS=1" ;;
    D) dev="Vulkan0,AMD Phoenix XDNA1 NPU"; envs="GGML_XDNA_WEIGHTS=1" ;;
  esac
  if [ "$cfg" = A ]; then guard=""; else guard="/workspace/llama.cpp/experiments/phase48/npu_guard.sh --timeout 900 --"; fi
  echo "=== cfg$cfg $tag PP=$pp env=[$envs] ==="
  # shellcheck disable=SC2086
  env $envs XDNA_Q8_ENGINE=c4_mmul_v2 GGML_SCHED_TRACE_STATS=1 GGML_SCHED_TRACE_COPY=1 \
    $guard ./bin/bench_pp_scaling --model $M --dev "$dev" --pp "$pp" --batch "$pp" --ubatch 512 \
      --warmup "$w" --measure "$n" --tag "cfg${cfg}_$tag" > "$OUT/cfg${cfg}_pp${pp}.bench.log" 2> "$OUT/cfg${cfg}_pp${pp}.bench.err"
  grep -oE "pp_tps=[0-9.]+|pp_ms=[0-9.]+" "$OUT/cfg${cfg}_pp${pp}.bench.log" | tail -2 | tr '\n' ' '; echo
  grep "SCHED-SUMMARY" "$OUT/cfg${cfg}_pp${pp}.bench.err" | tail -1 | \
    grep -oE "n_mm_gpu=[0-9]+|n_mm_xdna=[0-9]+|n_mm_cpu=[0-9]+|n_splits=[0-9]+|n_transitions=[0-9]+|n_copies=[0-9]+|copy_bytes=[0-9]+|v2x_n=[0-9]+|v2x_bytes=[0-9]+|v2x_ms=[0-9.]+|x2v_n=[0-9]+|x2v_bytes=[0-9]+|x2v_ms=[0-9.]+|n_nodes_cpu=[0-9]+" | tr '\n' ' '; echo
  echo "  npu_gemm=$(grep -c 'XDNA-SUMMARY' "$OUT/cfg${cfg}_pp${pp}.bench.err")  gpu_ms=$(grep 'VK-SUMMARY' "$OUT/cfg${cfg}_pp${pp}.bench.err" | tail -1 | grep -oE 'busy_ms=[0-9.]+')"
}

PP=${PP:-512}
run A "$PP" 1 1 bench
run B "$PP" 1 1 bench
run D "$PP" 1 1 bench
run C "$PP" 1 1 bench
