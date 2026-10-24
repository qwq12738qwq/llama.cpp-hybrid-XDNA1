#!/usr/bin/env python3
"""qwen_graph_profiler.py -- Phase 7B Qwen3.5 graph profiler.

Analyzes graph dumps produced by instrumented llama.cpp.
Generates operator census and XDNA eligibility analysis.

Usage:
    python3 qwen_graph_profiler.py /tmp/llama_graph_prefill_n*.json
    python3 qwen_graph_profiler.py --dir /tmp
"""
import argparse, json, os, sys, statistics
from pathlib import Path
from collections import defaultdict

def load_graph(path):
    with open(path) as f:
        return json.load(f)

def op_category(op):
    if op == "MUL_MAT":
        return "MUL_MAT"
    elif op in ("RMS_NORM", "L2_NORM"):
        return "NORM"
    elif op in ("SILU", "GELU", "RELU", "SWIGLU"):
        return "ACTIVATION"
    elif op in ("ADD", "MUL", "SUB", "DIV"):
        return "ELEMENTWISE"
    elif op in ("ROPE", "ROPE_MULTI"):
        return "ROPE"
    elif op in ("SOFTMAX", "SOFT_MAX"):
        return "SOFTMAX"
    elif op in ("VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "GET_ROWS", "CONT"):
        return "VIEW/RESHAPE"
    elif op in ("SSM_CONV", "RECURRENT_ATTN"):
        return "SSM"
    elif op in ("SUM", "ACC"):
        return "REDUCE"
    elif "embd" in op.lower() or "embed" in op.lower():
        return "EMBEDDING"
    elif "logits" in op.lower() or "lm_head" in op.lower():
        return "LM_HEAD"
    elif "attn" in op.lower():
        return "ATTENTION"
    elif "ffn" in op.lower() or "up" in op.lower() or "down" in op.lower() or "gate" in op.lower():
        return "FFN"
    elif "ssm" in op.lower():
        return "SSM"
    else:
        return "OTHER"

def analyze_graph(data):
    nodes = data.get("nodes", [])
    n_nodes = data.get("n_nodes", len(nodes))
    phase = data.get("phase", "unknown")
    n_tokens = data.get("n_tokens", 0)

    op_counts = defaultdict(int)
    op_bytes = defaultdict(int)
    mul_mat_shapes = defaultdict(lambda: {"count": 0, "bytes": 0, "flops": 0, "names": set()})
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    xdna_eligible = []
    xdna_blocked = []

    for node in nodes:
        op = node.get("op", "UNKNOWN")
        cat = op_category(op)
        n_bytes = node.get("n_bytes", 0)

        op_counts[op] += 1
        op_bytes[op] += n_bytes
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += n_bytes

        if op == "MUL_MAT" and "mul_mat" in node:
            mm = node["mul_mat"]
            w_shape = tuple(mm.get("w_shape", [0,0,0,0]))
            i_shape = tuple(mm.get("i_shape", [0,0,0,0]))
            o_shape = tuple(node.get("ne", [0,0,0,0]))
            w_name = mm.get("w_name", "")
            w_type = mm.get("w_type", 0)

            K = w_shape[0]
            N = w_shape[1]
            M = o_shape[1] if len(o_shape) > 1 else 1

            flops = 2.0 * M * K * N
            shape_key = (M, K, N)
            mul_mat_shapes[shape_key]["count"] += 1
            mul_mat_shapes[shape_key]["bytes"] += n_bytes
            mul_mat_shapes[shape_key]["flops"] += flops
            mul_mat_shapes[shape_key]["names"].add(w_name)

            # XDNA eligibility check
            if K <= 512 and M % 64 == 0 and N % 64 == 0:
                xdna_eligible.append({
                    "shape": shape_key,
                    "w_name": w_name,
                    "w_type": w_type,
                    "flops": flops,
                    "bytes": n_bytes,
                })
            else:
                reasons = []
                if K > 512:
                    reasons.append(f"K={K}>512")
                if M % 64 != 0:
                    reasons.append(f"M={M}%64!=0")
                if N % 64 != 0:
                    reasons.append(f"N={N}%64!=0")
                xdna_blocked.append({
                    "shape": shape_key,
                    "w_name": w_name,
                    "reason": "; ".join(reasons),
                })

    return {
        "phase": phase,
        "n_tokens": n_tokens,
        "n_nodes": n_nodes,
        "op_counts": dict(op_counts),
        "op_bytes": dict(op_bytes),
        "categories": {k: dict(v) for k, v in categories.items()},
        "mul_mat_shapes": {f"M{k[0]}_K{k[1]}_N{k[2]}": {
            "count": v["count"],
            "total_bytes": v["bytes"],
            "total_flops": v["flops"],
            "weight_names": list(v["names"]),
        } for k, v in mul_mat_shapes.items()},
        "xdna_eligible": xdna_eligible,
        "xdna_blocked": xdna_blocked,
    }

def print_summary(result):
    print(f"\n{'='*60}")
    print(f"Phase: {result['phase']}, Tokens: {result['n_tokens']}, Nodes: {result['n_nodes']}")
    print(f"{'='*60}")

    print("\n--- Operator Census ---")
    for op, count in sorted(result["op_counts"].items(), key=lambda x: -x[1]):
        print(f"  {op:20s}: {count:5d}")

    print("\n--- Category Summary ---")
    for cat, info in sorted(result["categories"].items(), key=lambda x: -x[1]["count"]):
        print(f"  {cat:20s}: count={info['count']:5d}, bytes={info['bytes']:12d}")

    print("\n--- MUL_MAT Shapes ---")
    for shape_key, info in sorted(result["mul_mat_shapes"].items()):
        print(f"  {shape_key:20s}: count={info['count']:3d}, "
              f"flops={info['total_flops']:18.0f}, bytes={info['total_bytes']:10d}")
        for name in info["weight_names"]:
            print(f"    -> {name}")

    print(f"\n--- XDNA Eligible ({len(result['xdna_eligible'])}) ---")
    for e in result["xdna_eligible"]:
        print(f"  {e['shape']}: {e['w_name']} flops={e['flops']:.0f}")

    print(f"\n--- XDNA Blocked ({len(result['xdna_blocked'])}) ---")
    for b in result["xdna_blocked"]:
        print(f"  {b['shape']}: {b['w_name']} - {b['reason']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Graph JSON files")
    parser.add_argument("--dir", help="Directory containing graph JSON files")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.dir:
        paths.extend(Path(args.dir).glob("llama_graph_*.json"))

    if not paths:
        print("No graph files found", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for path in sorted(paths):
        print(f"\nAnalyzing: {path}")
        data = load_graph(path)
        result = analyze_graph(data)
        all_results.append(result)
        print_summary(result)

    # Save combined analysis
    out_path = Path(paths[0]).parent / "phase7b_graph_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined analysis saved to {out_path}")

if __name__ == "__main__":
    main()
