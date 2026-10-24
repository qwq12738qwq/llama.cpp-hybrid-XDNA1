#!/usr/bin/env bash
# Phase 61: run a list of skbench variants, one NPU process at a time.
#
# usage: sweep.sh <sweep-name> "<variant>:<mmul>:<kchunk>:<m_tile>:<n_tile>:<tag>" ...
set -u
ROOT=/workspace/llama.cpp
KB=$ROOT/experiments/phase61/kbench
KERNELS=$ROOT/experiments/phase61/kernels
. $ROOT/experiments/phase61/env.sh

SWEEP=$1; shift
mkdir -p "$KERNELS" "$KB"

for spec in "$@"; do
  IFS=: read -r VAR MMUL KC MT NT TAG <<EOF
$spec
EOF
  WD=$KERNELS/${SWEEP}_${TAG}
  echo "############ $SWEEP / $spec"
  # the iron/aiecc build cache lives in the work dir: a stale one silently
  # serves the previous kernel source, so always start from a clean dir
  rm -rf "$WD"
  $PY $KB/skbench.py \
    --M ${M:-512} --K ${K:-1024} --N ${N:-3584} \
    --variant "$VAR" --mmul "$MMUL" --k-chunk "$KC" \
    --m-tile "$MT" --n-tile "$NT" \
    --kernel-reps ${REPS:-1} --repeat ${REPEAT:-3} \
    --pattern ${PATTERN:-realistic} \
    --work-dir "$WD" --tag "$SWEEP/$TAG" --recompile 2>&1 | tail -30
done
