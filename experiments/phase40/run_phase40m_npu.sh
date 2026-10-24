#!/bin/bash
# Phase 40M: Compile and test Q8SCALEV-C4 mmul v2 kernel on NPU hardware.
#
# Run INSIDE the npu-dev container:
#   bash experiments/phase40/run_phase40m_npu.sh [M] [K] [N] [cols]
#
# Default: M=512 K=1024 N=3584 cols=4

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="/Storage/Cache/llama.cpp"
CACHE_DIR="$WORKSPACE/experiments/npu-int8-gemm/kernel_cache"
WORK_DIR="/tmp/q8c4mmul_npu"

M=${1:-512}
K=${2:-1024}
N=${3:-3584}
COLS=${4:-4}

echo "=== Phase 40M: Q8SCALEV-C4 mmul v2 NPU test ==="
echo "  M=$M K=$K N=$N cols=$COLS"
echo "  Work dir: $WORK_DIR"
echo "  Cache:    $CACHE_DIR"
echo ""

# Check aie.iron is available
python3 -c "import aie.iron; print('aie.iron OK')" || {
    echo "ERROR: aie.iron not available. Run inside npu-dev container."
    exit 1
}

# Step 1: Compile kernel
echo "[1/3] Compiling kernel..."
mkdir -p "$WORK_DIR"
python3 "$SCRIPT_DIR/compile_q8_c4_mmul_v2.py" \
    --M "$M" --K "$K" --N "$N" --cols "$COLS" \
    --work-dir "$WORK_DIR"
echo ""

# Step 2: Run hardware benchmark
echo "[2/3] Running hardware benchmark..."
python3 "$SCRIPT_DIR/bench_q8_c4_mmul_v2.py" \
    --M "$M" --K "$K" --N "$N" --cols "$COLS" --iters 5 \
    --work-dir "$WORK_DIR" --pattern geometric
echo ""

# Step 3: Copy to kernel cache
echo "[3/3] Installing to kernel cache..."
CACHE_NAME="M${M}_K${K}_N${N}_C${COLS}_Q8SCALEV_MMUL"
mkdir -p "$CACHE_DIR/$CACHE_NAME"
cp "$WORK_DIR/design.xclbin" "$CACHE_DIR/$CACHE_NAME/"
cp "$WORK_DIR/design.insts.bin" "$CACHE_DIR/$CACHE_NAME/"
cp "$WORK_DIR/main.pdi" "$CACHE_DIR/$CACHE_NAME/" 2>/dev/null || true
echo "  Installed to: $CACHE_DIR/$CACHE_NAME"
echo ""

echo "=== Phase 40M NPU test complete ==="
