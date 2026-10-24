#!/bin/bash
# Phase 20: Run all shape benchmarks (each in separate process to avoid hw_context conflict)
set -e
DIR="/Storage/Cache/llama.cpp/experiments/phase20"
OUT="$DIR/phase20_all_results.json"
echo "[" > "$OUT"
FIRST=1

run_bench() {
    M=$1 K=$2 N=$3 NC=$4
    LABEL="M${M}_K${K}_N${N}_C${NC}"
    echo "--- $LABEL ---"
    RESULT=$(podman exec npu-dev bash -c "cd /Storage/Cache/llama.cpp && python3 experiments/phase20/phase20_bench_one.py $M $K $N $NC" 2>/dev/null)
    if [ -z "$RESULT" ] || echo "$RESULT" | grep -q '"error"'; then
        echo "  SKIP: kernel not found"
        return
    fi
    echo "  $RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  GOPS_kernel={d['GOPS_kernel']:.2f}  kernel_ms={d['kernel_ms']:.2f}  total_ms={d['total_ms']:.2f}  correct={d['correct']}\")"
    if [ $FIRST -eq 1 ]; then FIRST=0; else echo "," >> "$OUT"; fi
    echo "$RESULT" >> "$OUT"
}

echo "=========================================="
echo "STEP IX: M Scaling (K=1024, N=3072)"
echo "=========================================="
for M in 64 128 256 512; do
    run_bench $M 1024 3072 1
done

echo ""
echo "=========================================="
echo "STEP XI: K Sweep (M=512, N=3072)"
echo "=========================================="
for K in 1024 2048 4096; do
    run_bench 512 $K 3072 1
done

echo ""
echo "=========================================="
echo "STEP X: N_COLS Test (M=512, K=1024, N=3072)"
echo "=========================================="
run_bench 512 1024 3072 1
run_bench 512 1024 3072 4

echo "" >> "$OUT"
echo "]" >> "$OUT"
echo ""
echo "Results saved to $OUT"
