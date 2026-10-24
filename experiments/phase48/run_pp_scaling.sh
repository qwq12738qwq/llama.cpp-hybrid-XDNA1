#!/usr/bin/env bash
# Phase 48: PP scaling matrix for Vulkan-only and Vulkan+XDNA heterogeneous PP.
#
# For every PP point this runs three separate processes so that the headline
# throughput is never measured with instrumentation enabled:
#
#   bench   : warmup + measured prompt phase -> pp t/s, total ms, peak RSS
#   trace   : same, with GGML_SCHED_TRACE_STATS -> splits/copies/nodes/NPU timings
#   verify  : greedy decode of 32 tokens -> md5 of the token sequence
#
# Nothing here is allowed to run two NPU processes at once: every NPU invocation
# goes through npu_guard.sh, which waits until /dev/accel/accel0 has no holders.
#
# MODE:
#   vk         Vulkan only, XDNA backend not even registered
#   hybrid     Vulkan + XDNA, GGML_XDNA_QWEN_PP=1. No NPU weight placement, so
#              the scheduler keeps 2 splits and the NPU stays idle. This is the
#              PHASE47B frozen baseline.
#   hybrid_npu adds GGML_XDNA_WEIGHTS=1: ~6 of 24 layers land in NPU buffers and
#              their FFN goes through the NPU kernel. Slow by construction.
#   ab         vk and hybrid interleaved per PP, ROUNDS times each, so the
#              Hybrid/Vulkan ratio is not polluted by machine drift. This is the
#              number to quote.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BUILD=${BUILD:-$ROOT/build}
BIN=$BUILD/bin/bench_pp_scaling
MODEL=${MODEL:-$ROOT/models/Qwen3.5-0.8B-Q8_0.gguf}
OUT=${OUT:-$HERE/results}
PP_LIST=${PP_LIST:-"512 768 1024 1536 2048 3072 4096 5120"}
# The Vulkan compute buffer scales with the physical ubatch: on this host
# (7 GB, UMA iGPU) ubatch=PP=3072 needs a 3.32 GB buffer and fails to
# allocate, while ubatch=512 stays at ~175 MB RSS. PP grows through n_batch
# and the physical ubatch stays at the frozen PHASE47B baseline, so the NPU
# GEMM shape (M=ubatch) is identical at every PP point.
UBATCH=${UBATCH:-512}
WARMUP=${WARMUP:-2}
MEASURE=${MEASURE:-5}
NPRED=${NPRED:-32}
MODE=${MODE:-ab}            # ab | full | vk | hybrid | hybrid_npu
ROUNDS=${ROUNDS:-2}
GUARD_TIMEOUT=${GUARD_TIMEOUT:-1800}

mkdir -p "$OUT"

export LD_LIBRARY_PATH="$BUILD/bin:${LD_LIBRARY_PATH:-}"
export XDNA_KERNEL_CACHE=${XDNA_KERNEL_CACHE:-$ROOT/experiments/npu-int8-gemm/kernel_cache}
export XDNA_Q8_ENGINE=${XDNA_Q8_ENGINE:-c4_mmul_v2}

if [ ! -x "$BIN" ]; then
  echo "missing $BIN - build bench_pp_scaling first" >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "missing model $MODEL" >&2
  exit 1
fi

VK_DEV="Vulkan0"
NPU_DEV="Vulkan0,AMD Phoenix XDNA1 NPU"

# $1 tag  $2 dev  $3 env string  -> one bench + trace + verify triple
run_point() {
  local tag=$1 dev=$2 envs=$3
  local pp ub
  pp=$PP_CUR
  ub=$UBATCH
  [ "$ub" = "pp" ] && ub=$pp
  local base="$OUT/${tag}_pp${pp}"

  echo "=== $tag PP=$pp ubatch=$ub dev=[$dev] env=[$envs] ==="

  # shellcheck disable=SC2086
  env $envs "$BIN" --model "$MODEL" --dev "$dev" --pp "$pp" --batch "$pp" \
      --ubatch "$ub" --warmup "$WARMUP" --measure "$MEASURE" --tag "$tag" \
      > "${base}.bench.log" 2> "${base}.bench.err"
  grep -E '^(RESULT|INFO)' "${base}.bench.log" || echo "BENCH FAILED for ${tag} PP=${pp}"

  # shellcheck disable=SC2086
  env $envs GGML_SCHED_TRACE_STATS=1 "$BIN" --model "$MODEL" --dev "$dev" --pp "$pp" \
      --batch "$pp" --ubatch "$ub" --warmup 1 --measure 1 --mode bench \
      --tag "${tag}_trace" > "${base}.trace.log" 2> "${base}.trace.err"

  # shellcheck disable=SC2086
  env $envs "$HERE/npu_guard.sh" --timeout "$GUARD_TIMEOUT" -- \
      "$BIN" --model "$MODEL" --dev "$dev" --pp "$pp" --batch "$pp" --ubatch "$ub" \
      --mode verify --n-predict "$NPRED" --tag "$tag" \
      > "${base}.verify.log" 2> "${base}.verify.err"
  grep -E '^VERIFY' "${base}.verify.log" | cut -c1-60 || echo "VERIFY FAILED for ${tag} PP=${pp}"
}

for pp in $PP_LIST; do
  PP_CUR=$pp
  case "$MODE" in
    ab)
      for r in $(seq 1 "$ROUNDS"); do
        run_point "vk_r$r"     "$VK_DEV" "GGML_XDNA_DISABLE=1"
        run_point "hybrid_r$r" "$NPU_DEV" "GGML_XDNA_QWEN_PP=1"
      done
      ;;
    full|hybrid_npu)
      run_point hybrid_npu "$NPU_DEV" "GGML_XDNA_QWEN_PP=1 GGML_XDNA_WEIGHTS=1"
      ;;
    full|hybrid)
      run_point hybrid "$NPU_DEV" "GGML_XDNA_QWEN_PP=1"
      ;;
    full|vk)
      run_point vk "$VK_DEV" "GGML_XDNA_DISABLE=1"
      ;;
    *)
      echo "unknown MODE=$MODE" >&2; exit 1 ;;
  esac
done

echo
echo "logs in $OUT"