#!/bin/bash
# Phase 20G: environment for the IRON/Peano XDNA kernel toolchain.
#
# The toolchain lives inside the ironenv-v143 venv:
#   - IRON python API : aie.iron  (mlir_aie package)
#   - aiecc           : site-packages/mlir_aie/bin/aiecc
#   - Peano (LLVM)    : site-packages/llvm-aie/bin/clang++
#
# Source this before running any generator, e.g.:
#   source experiments/phase20g/iron_env.sh
#   python experiments/npu-int8-gemm/gen_xdna_gemm.py --M 64 --K 1024 --N 64 --work-dir /tmp/out
#
IRONENV=/Storage/Cache/ironenv-v143
export IRON_PYTHON="$IRONENV/bin/python"
export PATH="$IRONENV/lib/python3.14/site-packages/llvm-aie/bin:$IRONENV/lib/python3.14/site-packages/mlir_aie/bin:$PATH"
echo "IRON_PYTHON=$IRON_PYTHON"
echo "clang++=$(command -v clang++)"
echo "aiecc=$(command -v aiecc)"
