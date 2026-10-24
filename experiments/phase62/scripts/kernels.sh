#!/usr/bin/env bash
# Phase 62: standalone kernel campaign.
#
# Every candidate is compiled with the real aiecc, run on the real NPU and
# checked against a float64 numpy reference.  The results land in
# experiments/phase62/kbench/results.jsonl.
set -u
ROOT=/workspace/llama.cpp
. $ROOT/experiments/phase62/env.sh
KB=$ROOT/experiments/phase62/kbench
KD=$ROOT/experiments/phase62/kernels
LOG=$ROOT/experiments/phase62/logs
mkdir -p "$KD" "$LOG"

run () {  # tag M K N wcols m_tile n_tile k_chunk variant layout [extra]
  tag=$1; M=$2; K=$3; N=$4; W=$5; mt=$6; nt=$7; kc=$8; var=$9; lay=${10}; shift 10
  wd=$KD/$tag
  rm -rf "$wd"
  echo "=== $tag" | tee -a "$LOG/kernels.log"
  timeout 900 $PY $KB/kbench62.py --M $M --K $K --N $N --wcols $W \
      --m-tile $mt --n-tile $nt --k-chunk $kc --variant $var --layout $lay \
      --repeat 5 --work-dir "$wd" --recompile --tag "$tag" "$@" \
      2>&1 | grep -E "^compile |ms_median|gop_s|rel_rmse|max_abs_err|error" | tee -a "$LOG/kernels.log"
}

FFN="512 1024 3584"
DOWN="512 3584 1024"

# A: the per-chunk scale kernel, worker count and data layout
run A_c256_c4_sep   $FFN 4 64 64 256 c256 sep
run A_c256_c4_mrg   $FFN 4 64 64 256 c256 mrg
run A_c256_c7_mrg   $FFN 7 64 64 256 c256 mrg
run A_c256_c8_mrg   $FFN 8 64 64 256 c256 mrg
run A_c256_c14_mrg  $FFN 14 64 64 256 c256 mrg

# B: the two-chain variant of the same kernel
run B_c256x2_c4     $FFN 4 64 64 256 c256x2 sep
run B_c256x2_c8     $FFN 8 64 64 256 c256x2 mrg

# C: tile shape sweep on the best structure
run C_c256_mt32     $FFN 4 32 64 256 c256 sep
run C_c256_mt128    $FFN 4 128 64 256 c256 sep
run C_c256_nt32     $FFN 4 64 32 256 c256 sep
run C_c256_kc128    $FFN 4 64 64 128 c256 sep

# D: the down projection shape
run D_c256_down_c4  $DOWN 4 64 64 256 c256 sep
run D_c256_down_c7  $DOWN 7 64 64 256 c256 mrg

echo "########## KERNEL CAMPAIGN DONE ##########" | tee -a "$LOG/kernels.log"
