#!/bin/bash
# Phase 20D: Vulkan-only PP baseline benchmark
# Measures PP throughput at 4K/8K/16K/32K/64K context lengths
# Output: CSV with tokens/s, latency, and scaling ratios
#
# Usage: ./bench_vulkan_baseline.sh [model_path] [output_dir]

set -euo pipefail

MODEL="${1:-/workspace/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf}"
OUTDIR="${2:-/workspace/llama.cpp/experiments/phase20d/results}"
BENCH="/workspace/llama.cpp/build-vk/bin/llama-bench"
REPETITIONS=3

mkdir -p "$OUTDIR"

# Context lengths to benchmark
CONTEXTS=(4096 8192 16384 32768 65536)

echo "=== Phase 20D: Vulkan-only PP Baseline ==="
echo "Model: $MODEL"
echo "Output: $OUTDIR"
echo "Repetitions: $REPETITIONS"
echo ""

# Run baseline for each context length
for ctx in "${CONTEXTS[@]}"; do
    echo "--- Context: $ctx tokens ---"
    outfile="$OUTDIR/vulkan_baseline_ctx${ctx}.json"

    # Use llama-bench with Vulkan-only (disable XDNA)
    GGML_XDNA_DISABLE=1 "$BENCH" \
        -m "$MODEL" \
        -p "$ctx" \
        -n 0 \
        -ngl 99 \
        -fa auto \
        -r "$REPETITIONS" \
        -o json \
        -o json \
        2>"$OUTDIR/vulkan_baseline_ctx${ctx}.stderr" \
        > "$outfile"

    echo "  Results saved to $outfile"

    # Extract PP metrics
    pp_tokens_per_sec=$(python3 -c "
import json, sys
with open('$outfile') as f:
    data = json.load(f)
for row in data:
    if row.get('test') == 'pp':
        print(f\"  PP: {row.get('avg_ts', 'N/A')} tokens/s, {row.get('avg_ns', 'N/A')} ns avg\")
        break
" 2>/dev/null || echo "  (parse failed)")
    echo "$pp_tokens_per_sec"
done

echo ""
echo "=== Baseline complete ==="
echo "Results in: $OUTDIR/vulkan_baseline_ctx*.json"
