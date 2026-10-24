#!/bin/bash
# Phase 20G step 10: native XRT int8 GEMM layout correctness sweep.
# M in {64,128,256,512} x K=1024 x N in {1024,2048}, vs CPU float64 reference.
set -u
KC=/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache
BIN=/workspace/llama.cpp/experiments/phase20g/xdna_smoke
fail=0
while read -r M K N D; do
  out=$("$BIN" "$KC/$D/design.xclbin" "$KC/$D/design.insts.bin" "$M" "$K" "$N" 2>/dev/null | grep '^SMOKE')
  echo "$out"
  case "$out" in
    *PASS*) ;;
    *) fail=1 ;;
  esac
done <<'EOF'
64 1024 1024 M64_K1024_N1024
64 1024 2048 M64_K1024_N2048_C1_Tm64_Tk64_Tn64
128 1024 1024 M128_K1024_N1024
128 1024 2048 M128_K1024_N2048_C1_Tm64_Tk64_Tn64
256 1024 1024 M256_K1024_N1024_C1_Tm64_Tk64_Tn64
256 1024 2048 M256_K1024_N2048_C1_Tm64_Tk64_Tn64
512 1024 1024 M512_K1024_N1024_C1
512 1024 2048 M512_K1024_N2048_C1
EOF
echo "SWEEP_DONE fail=$fail"
exit $fail
