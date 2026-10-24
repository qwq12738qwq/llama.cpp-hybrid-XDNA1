#!/bin/bash
# Phase 20D: Heterogeneous PP benchmark with partition sweep
# Measures PP throughput with Vulkan+XDNA at various partition ratios
# Output: CSV with tokens/s, latency per partition config
#
# Usage: ./bench_hetero.sh [model_path] [output_dir]

set -euo pipefail

MODEL="${1:-/workspace/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf}"
OUTDIR="${2:-/workspace/llama.cpp/experiments/phase20d/results}"
BENCH="/workspace/llama.cpp/build-vk/bin/llama-bench"
REPETITIONS=3

mkdir -p "$OUTDIR"

# Context lengths
CONTEXTS=(4096 8192 16384 32768 65536)

# XDNA offload levels: fraction of MUL_MAT ops sent to XDNA
# Controlled via GGML_XDNA_M_THRESHOLD environment variable
# (threshold in M dimension: ops with M >= threshold go to XDNA)
# Sweep: 0% (Vulkan-only), 10%, 25%, 40%, 50%, 60%, 75%, 90%, 100%
XDNA_THRESHOLDS=(9999 512 256 128 96 64 48 32 1)

echo "=== Phase 20D: Heterogeneous PP Benchmark ==="
echo "Model: $MODEL"
echo "Output: $OUTDIR"
echo "Repetitions: $REPETITIONS"
echo ""

for ctx in "${CONTEXTS[@]}"; do
    echo "--- Context: $ctx tokens ---"

    for thresh in "${XDNA_THRESHOLDS[@]}"; do
        pct_label="thresh${thresh}"
        outfile="$OUTDIR/hetero_ctx${ctx}_${pct_label}.json"

        echo "  XDNA threshold=$thresh ..."

        # Set XDNA threshold via env var
        # The backend checks M >= threshold to decide XDNA offload
        GGML_XDNA_M_THRESHOLD="$thresh" "$BENCH" \
            -m "$MODEL" \
            -p "$ctx" \
            -n 0 \
            -ngl 99 \
            -fa auto \
            -r "$REPETITIONS" \
            -o json \
            -o json \
            2>"$OUTDIR/hetero_ctx${ctx}_${pct_label}.stderr" \
            > "$outfile"

        # Extract PP metrics
        python3 -c "
import json
with open('$outfile') as f:
    data = json.load(f)
for row in data:
    if row.get('test') == 'pp':
        print(f\"    PP: {row.get('avg_ts', 'N/A')} tokens/s\")
        break
" 2>/dev/null || echo "    (parse failed)"
    done
done

echo ""
echo "=== Heterogeneous benchmark complete ==="
echo "Results in: $OUTDIR/hetero_ctx*_thresh*.json"
