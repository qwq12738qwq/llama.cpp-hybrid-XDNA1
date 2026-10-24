#!/usr/bin/env python3
"""Phase 34: scaling efficiency analysis.

Reads all raw/bench_*.json files for a given shape and prints scaling table.

Usage
-----
  python analyze.py --M 512 --K 2048 --N 2048 --work-dir /tmp/p34
"""

import argparse
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    raw_dir = os.path.join(args.work_dir, "raw")
    pattern = os.path.join(raw_dir, "bench_*.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)
        if r["shape"]["M"] == args.M and r["shape"]["K"] == args.K and r["shape"]["N"] == args.N:
            rows.append(r)

    if not rows:
        print(f"No results found in {pattern}")
        sys.exit(1)

    rows.sort(key=lambda r: r["tiles"])
    baseline = next((r for r in rows if r["tiles"] == 1), None)
    if not baseline:
        print("No T=1 baseline found")
        sys.exit(1)

    base_gops = baseline["e2e_gops"]
    print(f"\n=== Scaling Analysis: {args.M}x{args.K}x{args.N} ===")
    print(f"{'Tiles':>6} {'Rows':>5} {'Cols':>5} {'GOPS':>10} {'Scaling':>10} {'Efficiency':>10} {'Correct':>8}")
    print("-" * 62)
    for r in rows:
        t = r["tiles"]
        g = r["e2e_gops"]
        scaling = g / base_gops if base_gops > 0 else 0
        efficiency = scaling / t if t > 0 else 0
        print(f"{t:>6} {r['rows']:>5} {r['cols']:>5} {g:>10.2f} {scaling:>10.2f} {efficiency:>10.2f} {str(r['correct']):>8}")

    # Verdict
    best = max(rows, key=lambda r: r["e2e_gops"])
    best_tiles = best["tiles"]
    best_scaling = best["e2e_gops"] / base_gops if base_gops > 0 else 0
    best_efficiency = best_scaling / best_tiles if best_tiles > 0 else 0

    print(f"\nBest: {best_tiles} tiles ({best['rows']}x{best['cols']}), {best['e2e_gops']:.2f} GOPS, {best_scaling:.2f}x scaling, {best_efficiency:.2f} efficiency")

    if best_tiles >= 8 and best_scaling >= 5:
        verdict = "WHOLE_ARRAY_SCALING_PASS"
    elif best_tiles >= 4 and best_scaling >= 2.5:
        verdict = "WHOLE_ARRAY_SCALING_PARTIAL"
    elif best_scaling < 1.5:
        verdict = "XDNA_ARRAY_SCALING_FAIL"
    else:
        verdict = "XDNA_ARRAY_SCALING_PARTIAL"

    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
