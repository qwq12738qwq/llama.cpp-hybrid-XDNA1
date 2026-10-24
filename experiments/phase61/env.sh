# Phase 61: MLIR-AIE / IRON toolchain environment.
# Source this before running any kernel generator.
IRONENV=/Storage/Cache/ironenv-v143
export PATH="$IRONENV/lib/python3.14/site-packages/mlir_aie/bin:$IRONENV/lib/python3.14/site-packages/llvm-aie/bin:$IRONENV/bin:$PATH"
export PYTHONPATH="$IRONENV/lib/python3.14/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$IRONENV/lib/python3.14/site-packages/mlir_aie/lib:$IRONENV/lib/python3.14/site-packages/llvm-aie/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PY=$IRONENV/bin/python3
