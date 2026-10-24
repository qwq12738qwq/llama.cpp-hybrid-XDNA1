#!/bin/bash
# Phase 21 native whole-array GEMM roofline sweep.
# Compiles gen_gemm.py designs on demand and runs benchmark_gemm, appending CSV.
set -u
HERE=/workspace/llama.cpp/experiments/phase21
KC=/Storage/Cache/xdna_kernels/phase21
IRON_PY=/Storage/Cache/ironenv-v143/bin/python
CSV=${CSV:-$HERE/results/roofline.csv}
PEAK_PER_TILE=${PEAK_PER_TILE:-1024}    # GOPS/tile @1.0GHz, i8 8x8x8 (512 MAC*2)
REPEAT=${REPEAT:-10}

mkdir -p "$(dirname "$CSV")" "$KC"
echo "tag,M,K,N,in_b,out_b,cols,ms_med,ms_min,gops,peak_gops,util_pct,max_rel,status" > "$CSV"

inb() { case "$1" in i8) echo 1;; i16|bf16) echo 2;; esac; }
outb() { case "$1" in i8) echo 1;; i16|bf16) echo 2;; i32|f32) echo 4;; esac; }

run_one() {  # M K N cols din dout
  local M=$1 K=$2 N=$3 C=$4 DI=$5 DO=$6
  local tag="${M}x${K}x${N}_c${C}_${DI}${DO}"
  local wd="$KC/$tag"
  mkdir -p "$wd"
  if [ ! -f "$wd/design.xclbin" ]; then
    if ! "$IRON_PY" "$HERE/gen_gemm.py" --M "$M" --K "$K" --N "$N" -m 64 -k 64 -n 32 \
         --n-aie-cols "$C" --dtype-in "$DI" --dtype-out "$DO" --work-dir "$wd" >"$wd.genlog" 2>&1; then
      echo "GENFAIL $tag"; return
    fi
  fi
  local peak
  peak=$(awk -v c="$C" -v p="$PEAK_PER_TILE" 'BEGIN{printf "%.1f", c*4*p}')
  "$HERE/benchmark_gemm" "$wd/design.xclbin" "$wd/design.insts.bin" \
      "$M" "$K" "$N" "$(inb "$DI")" "$(outb "$DO")" "$C" "$peak" "$REPEAT" "$tag" "$CSV"
}

# --- column scaling @ fixed shape (cols=1 design is incomplete; see notes) ---
for C in 2 4; do run_one 512 4096 4096 "$C" i8 i32; done
# --- shape scaling @ 4 cols ---
run_one 1024 4096 4096 4 i8 i32
run_one 2048 4096 4096 4 i8 i32
run_one 4096 4096 4096 4 i8 i32
# --- wide N (FFN-like) ---
run_one 512  4096 8192 4 i8 i32
run_one 2048 4096 8192 4 i8 i32
# --- K scaling ---
run_one 512 2048 4096 4 i8 i32
run_one 512 8192 4096 4 i8 i32
# --- i16 native ---
run_one 512 4096 4096 4 i16 i32
echo "SWEEP_DONE csv=$CSV"
