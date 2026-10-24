#!/usr/bin/env python3
"""
Phase 20D: Scaling Analysis for Long-Context PP
Computes empirical scaling exponents (alpha) via log-log regression.
Compares Vulkan-only vs heterogeneous PP scaling behavior.

Usage:
    python3 analyze_scaling.py results_dir/

Output:
    - Scaling ratios T(N)/T(4K)
    - Empirical alpha from T(N) = a * N^alpha
    - Comparison table
"""

import json
import os
import sys
import math
import glob
from pathlib import Path


def load_pp_results(results_dir):
    """Load all PP benchmark results from JSON files."""
    results = {}

    # Vulkan baseline
    for f in sorted(glob.glob(os.path.join(results_dir, "vulkan_baseline_ctx*.json"))):
        ctx_match = os.path.basename(f).replace("vulkan_baseline_ctx", "").replace(".json", "")
        try:
            ctx = int(ctx_match)
        except ValueError:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            for row in data:
                if row.get("test") == "pp":
                    results[("vulkan", ctx)] = {
                        "tokens_per_sec": float(row.get("avg_ts", 0)),
                        "latency_ms": float(row.get("avg_ns", 0)) / 1e6,
                    }
                    break
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Heterogeneous results
    for f in sorted(glob.glob(os.path.join(results_dir, "hetero_ctx*_thresh*.json"))):
        basename = os.path.basename(f).replace(".json", "")
        parts = basename.split("_")
        ctx = None
        thresh = None
        for p in parts:
            if p.startswith("ctx"):
                try:
                    ctx = int(p[3:])
                except ValueError:
                    pass
            elif p.startswith("thresh"):
                try:
                    thresh = int(p[5:])
                except ValueError:
                    pass
        if ctx is None or thresh is None:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            for row in data:
                if row.get("test") == "pp":
                    results[("hetero", ctx, thresh)] = {
                        "tokens_per_sec": float(row.get("avg_ts", 0)),
                        "latency_ms": float(row.get("avg_ns", 0)) / 1e6,
                    }
                    break
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    return results


def compute_scaling_exponent(contexts, latencies):
    """
    Compute empirical alpha from log-log regression: T(N) = a * N^alpha
    Returns (alpha, a, r_squared).
    """
    if len(contexts) < 2:
        return None, None, None

    log_n = [math.log(c) for c in contexts]
    log_t = [math.log(t) for t in latencies]

    n = len(log_n)
    sum_x = sum(log_n)
    sum_y = sum(log_t)
    sum_xy = sum(x * y for x, y in zip(log_n, log_t))
    sum_x2 = sum(x * x for x in log_n)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None, None, None

    alpha = (n * sum_xy - sum_x * sum_y) / denom
    a = math.exp((sum_y - alpha * sum_x) / n)

    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in log_t)
    ss_res = sum((y - (alpha * x + (sum_y - alpha * sum_x) / n)) ** 2
                 for x, y in zip(log_n, log_t))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0

    return alpha, a, r_squared


def analyze_results(results):
    """Analyze scaling behavior for all configurations."""
    contexts = sorted(set(ctx for (_, ctx, *_) in results.keys() if isinstance(ctx, int)))

    print("=" * 80)
    print("Phase 20D: Long-Context PP Scaling Analysis")
    print("=" * 80)
    print()

    # Vulkan-only analysis
    vulkan_ctxs = []
    vulkan_latencies = []
    print("--- Vulkan-only Baseline ---")
    print(f"{'Context':>10} {'tokens/s':>12} {'latency_ms':>12} {'ratio':>10}")
    print("-" * 50)

    base_latency = None
    for ctx in contexts:
        key = ("vulkan", ctx)
        if key in results:
            r = results[key]
            if base_latency is None:
                base_latency = r["latency_ms"]
            ratio = r["latency_ms"] / base_latency if base_latency else 0
            print(f"{ctx:>10} {r['tokens_per_sec']:>12.1f} {r['latency_ms']:>12.1f} {ratio:>10.2f}x")
            vulkan_ctxs.append(ctx)
            vulkan_latencies.append(r["latency_ms"])

    if len(vulkan_ctxs) >= 2:
        alpha_v, a_v, r2_v = compute_scaling_exponent(vulkan_ctxs, vulkan_latencies)
        print(f"\n  Scaling exponent alpha = {alpha_v:.3f} (R^2 = {r2_v:.4f})")
        print(f"  T(N) = {a_v:.2e} * N^{alpha_v:.3f}")
    print()

    # Heterogeneous analysis (for each threshold)
    thresh_keys = sorted(set(t for (_, _, t) in results.keys() if t is not None))
    for thresh in thresh_keys:
        het_ctxs = []
        het_latencies = []
        print(f"--- Heterogeneous (XDNA threshold={thresh}) ---")
        print(f"{'Context':>10} {'tokens/s':>12} {'latency_ms':>12} {'ratio':>10} {'vs_vulkan':>10}")
        print("-" * 60)

        base_latency = None
        for ctx in contexts:
            key = ("hetero", ctx, thresh)
            vkey = ("vulkan", ctx)
            if key in results:
                r = results[key]
                if base_latency is None:
                    base_latency = r["latency_ms"]
                ratio = r["latency_ms"] / base_latency if base_latency else 0
                speedup = ""
                if vkey in results:
                    speedup = f"{results[vkey]['latency_ms'] / r['latency_ms']:.2f}x"
                print(f"{ctx:>10} {r['tokens_per_sec']:>12.1f} {r['latency_ms']:>12.1f} {ratio:>10.2f}x {speedup:>10}")
                het_ctxs.append(ctx)
                het_latencies.append(r["latency_ms"])

        if len(het_ctxs) >= 2:
            alpha_h, a_h, r2_h = compute_scaling_exponent(het_ctxs, het_latencies)
            print(f"\n  Scaling exponent alpha = {alpha_h:.3f} (R^2 = {r2_h:.4f})")
            print(f"  T(N) = {a_h:.2e} * N^{alpha_h:.3f}")

            if len(vulkan_ctxs) >= 2:
                alpha_v, _, _ = compute_scaling_exponent(vulkan_ctxs, vulkan_latencies)
                delta = alpha_v - alpha_h
                print(f"  Alpha reduction: {delta:+.3f} ({'IMPROVED' if delta > 0 else 'NO IMPROVEMENT'})")
        print()

    # Summary table
    print("=" * 80)
    print("Summary: Scaling Exponents")
    print("=" * 80)
    print(f"{'Configuration':>30} {'alpha':>8} {'R^2':>8} {'verdict':>15}")
    print("-" * 65)

    if len(vulkan_ctxs) >= 2:
        alpha_v, _, r2_v = compute_scaling_exponent(vulkan_ctxs, vulkan_latencies)
        verdict = "near-linear" if alpha_v < 1.2 else ("quadratic" if alpha_v > 1.8 else "super-linear")
        print(f"{'Vulkan-only':>30} {alpha_v:>8.3f} {r2_v:>8.4f} {verdict:>15}")

    for thresh in thresh_keys:
        het_ctxs = []
        het_latencies = []
        for ctx in contexts:
            key = ("hetero", ctx, thresh)
            if key in results:
                het_ctxs.append(ctx)
                het_latencies.append(results[key]["latency_ms"])
        if len(het_ctxs) >= 2:
            alpha_h, _, r2_h = compute_scaling_exponent(het_ctxs, het_latencies)
            verdict = "near-linear" if alpha_h < 1.2 else ("quadratic" if alpha_h > 1.8 else "super-linear")
            print(f"{'Hetero thresh=' + str(thresh):>30} {alpha_h:>8.3f} {r2_h:>8.4f} {verdict:>15}")

    print()

    # STOP conditions check
    print("=" * 80)
    print("STOP Conditions Check")
    print("=" * 80)
    if len(vulkan_ctxs) >= 2 and len(vulkan_latencies) >= 2:
        alpha_v, _, _ = compute_scaling_exponent(vulkan_ctxs, vulkan_latencies)
        print(f"Vulkan alpha: {alpha_v:.3f}")

        best_alpha = alpha_v
        best_thresh = None
        for thresh in thresh_keys:
            het_ctxs = []
            het_latencies = []
            for ctx in contexts:
                key = ("hetero", ctx, thresh)
                if key in results:
                    het_ctxs.append(ctx)
                    het_latencies.append(results[key]["latency_ms"])
            if len(het_ctxs) >= 2:
                alpha_h, _, _ = compute_scaling_exponent(het_ctxs, het_latencies)
                if alpha_h < best_alpha:
                    best_alpha = alpha_h
                    best_thresh = thresh

        if best_thresh is not None:
            print(f"Best heterogeneous alpha: {best_alpha:.3f} (threshold={best_thresh})")
            delta = alpha_v - best_alpha
            if delta > 0.05:
                print(f"  -> LONG_CONTEXT_HETERO_PP_PARTIAL (alpha reduced by {delta:.3f})")
            else:
                print(f"  -> LONG_CONTEXT_HETERO_PP_NOT_VALIDATED (alpha reduction {delta:.3f} < 0.05)")
        else:
            print("  -> No heterogeneous results available")
    else:
        print("  -> Insufficient data for scaling analysis")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} results_dir/")
        sys.exit(1)

    results_dir = sys.argv[1]
    results = load_pp_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        print("Run bench_vulkan_baseline.sh and bench_hetero.sh first.")
        sys.exit(1)

    analyze_results(results)
