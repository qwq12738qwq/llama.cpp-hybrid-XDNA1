#!/usr/bin/env bash
# collect_graph_dumps.sh - Collect graph dumps via llama-bench
# Each context runs in its own isolated bash invocation
set -euo pipefail

MODEL="/workspace/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf"
LLAMA_BENCH="/workspace/llama.cpp/build/bin/llama-bench"
RESULTS_DIR="/workspace/llama.cpp/experiments/npu-heterogeneous/results/phase7b/graph_dumps"
mkdir -p "$RESULTS_DIR"

CONTEXTS=(256 1024 4096 8192 16384 32768 65536)

for ctx in "${CONTEXTS[@]}"; do
    echo "=== Collecting graph dump for ctx=${ctx} ==="
    
    LLAMA_DUMP_GRAPH=1 bash -c '
        set +e
        "$1" -m "$2" -p "$3" -n 1 -r 1 -ngl 999 -b 2048 -ub 512 2>&1
        exit_code=$?
        echo "EXIT_CODE=$exit_code"
        exit 0
    ' _ "$LLAMA_BENCH" "$MODEL" "$ctx" | tee "$RESULTS_DIR/graph_dump_ctx${ctx}.log"
    
    if ls /tmp/llama_graph_prefill_n*.json 1>/dev/null 2>&1; then
        for f in /tmp/llama_graph_prefill_n*.json; do
            cp "$f" "$RESULTS_DIR/llama_graph_prefill_n${ctx}.json"
            echo "Graph dump saved for ctx=${ctx}"
            rm -f "$f"
        done
    else
        echo "WARNING: No graph dump file found for ctx=${ctx}"
    fi
    
    sleep 2
done

echo "=== Graph dump collection complete ==="
ls -la "$RESULTS_DIR/"
