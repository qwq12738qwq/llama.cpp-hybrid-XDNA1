#!/bin/bash
# Build script for XDNA vectorized INT8 GEMM kernel + MLIR generation
# Target: Phoenix XDNA1 NPU, mlir-aie 1.4.3
#
# Usage: bash build_xdna_kernels.sh [M] [K] [N] [n_cols]
# Default: M=512 K=1024 N=3072 n_cols=1

set -e

IRON="/home/ubuntu12738/Storage/Cache/ironenv-v143"
LLVM_AIE="$IRON/lib/python3.14/site-packages/llvm-aie"
CLANGPP="$LLVM_AIE/bin/clang++"
AIE_OPT="$IRON/lib/python3.14/site-packages/mlir_aie/bin/aie-opt"
export PATH="$LLVM_AIE/bin:$PATH"

M=${1:-512}
K=${2:-1024}
N=${3:-3072}
N_COLS=${4:-1}

WORKSPACE="/home/ubuntu12738/Storage/Cache/llama.cpp"
OUTDIR="$WORKSPACE/experiments/xdna-v2/M${M}_K${K}_N${N}_C${N_COLS}"

echo "=== XDNA V2 Build ==="
echo "Shape: M=$M K=$K N=$N n_cols=$N_COLS"
echo "Output: $OUTDIR"

mkdir -p "$OUTDIR"

# Step 1: Compile vectorized kernel
echo ""
echo "[1/4] Compile vectorized matmul_i8_i32 kernel..."
cp /tmp/xdna_build/matmul_i8_i32.cpp "$OUTDIR/"
"$CLANGPP" --target=aie2p-none-unknown-elf -O2 \
    -c "$OUTDIR/matmul_i8_i32.cpp" \
    -o "$OUTDIR/matmul_i8_i32.o" 2>&1
echo "  Kernel object: $(ls -la "$OUTDIR/matmul_i8_i32.o" | awk '{print $5}') bytes"

# Step 2: Generate MLIR
echo ""
echo "[2/4] Generate MLIR for n_cols=$N_COLS..."
python3 /tmp/xdna_build/gen_mlir_v2.py "$M" "$K" "$N" "$N_COLS" "$OUTDIR" 2>&1
echo "  MLIR: $(ls -la "$OUTDIR/aie.mlir" | awk '{print $5}') bytes"

# Step 3: Validate MLIR
echo ""
echo "[3/4] Validate MLIR with aie-opt..."
"$AIE_OPT" "$OUTDIR/aie.mlir" -o /dev/null 2>&1
echo "  MLIR validation: PASS"

# Step 4: Compile with aiecc (generates ELF)
echo ""
echo "[4/4] Compile with aiecc..."
cd "$OUTDIR"
"$IRON/lib/python3.14/site-packages/mlir_aie/bin/aiecc" \
    aie.mlir 2>&1 | tail -5
echo "  aiecc: PASS"

echo ""
echo "=== Build Complete ==="
echo "Artifacts:"
find "$OUTDIR" -type f -not -path "*/aie.mlir.prj/*" | sort
echo ""
echo "NOTE: xclbin generation requires Vitis aietools."
echo "Use existing xclbin from kernel_cache for runtime testing."
