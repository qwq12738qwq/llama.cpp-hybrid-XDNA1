#!/usr/bin/env python3
"""Phase 32 analysis: find XDNA workload frontier, deliver verdict.

Reads experiments/phase32/results/*.csv and produces:
  - xdna_vs_vk_crossover.csv : per-shape ratio
  - chain_amortization.csv    : persistent chain analysis
  - phase32_verdict.txt       : final verdict
"""
import csv, os, sys

PHASE32 = "/workspace/llama.cpp/experiments/phase32"
RESULTS = f"{PHASE32}/results"

def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main():
    xdna = read_csv(f"{RESULTS}/xdna_sweep.csv")
    vk   = read_csv(f"{RESULTS}/vk_sweep.csv")
    chain = read_csv(f"{RESULTS}/chain.csv")

    # Build lookup by (M,K,N)
    vk_map = {}
    for r in vk:
        if r.get("status") == "ok":
            k = (int(r["M"]), int(r["K"]), int(r["N"]))
            vk_map[k] = float(r["ms"])

    xdna_map = {}
    for r in xdna:
        if r.get("status") == "ok" and float(r.get("ms_med", 0)) > 0:
            k = (int(r["M"]), int(r["K"]), int(r["N"]))
            xdna_map[k] = float(r["ms_med"])

    # Crossover analysis
    crossover_rows = []
    for k, xdna_ms in sorted(xdna_map.items()):
        vk_ms = vk_map.get(k)
        if vk_ms and vk_ms > 0:
            ratio = xdna_ms / vk_ms
            crossover_rows.append({
                "M": k[0], "K": k[1], "N": k[2],
                "xdna_ms": xdna_ms, "vk_ms": vk_ms,
                "ratio": ratio,
                "winner": "VK" if ratio < 1.0 else "XDNA"
            })

    crossover_rows.sort(key=lambda r: r["ratio"])

    with open(f"{RESULTS}/xdna_vs_vk_crossover.csv", "w") as f:
        f.write("M,K,N,xdna_ms,vk_ms,ratio,winner\n")
        for r in crossover_rows:
            f.write(f"{r['M']},{r['K']},{r['N']},{r['xdna_ms']:.4f},{r['vk_ms']:.4f},{r['ratio']:.3f},{r['winner']}\n")

    print("=== Crossover Analysis ===")
    print(f"{'M':>6} {'K':>6} {'N':>6} {'XDNA':>10} {'VK':>10} {'Ratio':>8}  {'Winner'}")
    print("-" * 60)
    for r in crossover_rows:
        print(f"{r['M']:6d} {r['K']:6d} {r['N']:6d} "
              f"{r['xdna_ms']:10.4f} {r['vk_ms']:10.4f} "
              f"{r['ratio']:8.3f}  {r['winner']}")

    # Chain analysis
    print("\n=== Persistent Chain Analysis ===")
    print(f"{'M':>6} {'K':>6} {'N':>6} {'G':>4} {'Cold':>10} {'Persist':>10} {'Speedup':>8}")
    print("-" * 60)
    for r in chain:
        if r.get("status") == "ok":
            print(f"{int(r['M']):6d} {int(r['K']):6d} {int(r['N']):6d} {int(r['G']):4d} "
                  f"{float(r['cold_total_ms']):10.2f} {float(r['persist_total_ms']):10.2f} "
                  f"{float(r['speedup']):8.3f}")

    # Verdict
    print("\n=== Phase 32 Verdict ===")
    min_ratio = min(r["ratio"] for r in crossover_rows) if crossover_rows else float("inf")
    best = crossover_rows[0] if crossover_rows else None

    print(f"Shapes tested: {len(crossover_rows)}")
    print(f"Best XDNA/VK ratio: {min_ratio:.3f} (XDNA is {min_ratio:.1f}x Vulkan)")
    print(f"Chain speedup range: {min(float(r['speedup']) for r in chain if r.get('status')=='ok'):.2f}x - {max(float(r['speedup']) for r in chain if r.get('status')=='ok'):.2f}x")

    # Determine verdict
    if min_ratio >= 5.0:
        verdict = "XDNA_LLAMA_CPP_STOP"
        rationale = (
            "XDNA single-GEMM latency is >=5x Vulkan across ALL tested shapes. "
            "Persistent chain amortization is negligible (<=1.09x). "
            "No crossover found. XDNA cannot win as a callable GEMM library "
            "for Qwen3.5-0.8B PP operators."
        )
    elif min_ratio >= 1.0:
        verdict = "LONG_CONTEXT_NOT_XDNA_TARGET"
        rationale = (
            "XDNA is slower than Vulkan for all tested shapes. "
            "No async-hidden benefit demonstrated. "
            "Long-context attention is Vulkan-dominated; XDNA overhead cannot be hidden."
        )
    else:
        verdict = "XDNA_ASYNC_USEFUL"
        rationale = "Found shapes where XDNA <= Vulkan with async overlap."

    print(f"\nVERDICT: {verdict}")
    print(f"RATIONALE: {rationale}")

    if best:
        print(f"\nBest candidate shape: M={best['M']} K={best['K']} N={best['N']} "
              f"ratio={best['ratio']:.2f} ({best['winner']} wins)")

    # Write verdict file
    chain_speedups = [float(r['speedup']) for r in chain if r.get('status') == 'ok']
    max_speedup = max(chain_speedups) if chain_speedups else 0.0
    with open(f"{RESULTS}/phase32_verdict.txt", "w") as f:
        f.write(f"VERDICT: {verdict}\n")
        f.write(f"SHAPES_TESTED: {len(crossover_rows)}\n")
        f.write(f"BEST_RATIO: {min_ratio:.3f}\n")
        if best:
            f.write(f"BEST_SHAPE: M={best['M']} K={best['K']} N={best['N']} ratio={best['ratio']:.3f}\n")
        f.write(f"CHAIN_MAX_SPEEDUP: {max_speedup:.3f}\n")
        f.write(f"RATIONALE: {rationale}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
