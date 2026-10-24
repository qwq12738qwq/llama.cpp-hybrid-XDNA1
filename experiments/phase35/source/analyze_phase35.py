#!/usr/bin/env python3
"""Phase 35: result analysis and scaling comparison.

Reads all raw/bench_*.json files for a given variant+shape and prints:
  - Per-tile-count scaling table
  - Scaling efficiency
  - Breakdown comparison (if available)
  - Verdict

Usage:
  python analyze_phase35.py --M 512 --K 2048 --N 2048 --work-dir /tmp/p35/baseline
  python analyze_phase35.py --compare-dirs /tmp/p35/baseline /tmp/p35/depth4 /tmp/p35/2d_t16
"""
import argparse
import glob
import json
import os
import sys


def load_results(search_dirs, M, K, N):
    rows = []
    for d in search_dirs:
        pattern = os.path.join(d, "raw", "bench_*.json")
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                r = json.load(f)
            if r["shape"]["M"] == M and r["shape"]["K"] == K and r["shape"]["N"] == N:
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--work-dir", type=str, default=None)
    ap.add_argument("--compare-dirs", type=str, nargs="*", default=None)
    args = ap.parse_args()

    dirs = []
    if args.compare_dirs:
        dirs = args.compare_dirs
    elif args.work_dir:
        dirs = [args.work_dir]
    else:
        print("ERROR: --work-dir or --compare-dirs required")
        sys.exit(1)

    rows = load_results(dirs, args.M, args.K, args.N)
    if not rows:
        print(f"No results found for {args.M}x{args.K}x{args.N}")
        sys.exit(1)

    rows.sort(key=lambda r: (r.get("variant", "baseline"), r["tiles"]))

    # Group by variant
    variants = {}
    for r in rows:
        v = r.get("variant", "baseline")
        variants.setdefault(v, []).append(r)

    for v, vrows in sorted(variants.items()):
        vrows.sort(key=lambda r: r["tiles"])
        base = next((r for r in vrows if r["tiles"] == 1), None)
        if not base:
            print(f"\n[{v}] No T=1 baseline, skipping")
            continue

        base_gops = base.get("npu_gops") or base.get("e2e_gops", 0)
        print(f"\n=== {v}: {args.M}x{args.K}x{args.N} ===")
        print(f"{'Tiles':>6} {'Rows':>5} {'Cols':>5} {'GOPS':>10} {'Scaling':>10} {'Eff':>8} {'Correct':>8} {'Compute%':>10}")
        print("-" * 70)
        for r in vrows:
            t = r["tiles"]
            g = r.get("npu_gops") or r.get("e2e_gops", 0)
            scaling = g / base_gops if base_gops > 0 else 0
            eff = scaling / t if t > 0 else 0
            correct = "PASS" if r.get("correct", False) else "FAIL"
            bd = r.get("breakdown", {})
            cf = bd.get("compute_fraction")
            cf_str = f"{cf:.1%}" if cf is not None else "N/A"
            print(f"{t:>6} {r['rows']:>5} {r['cols']:>5} {g:>10.2f} {scaling:>10.2f} {eff:>8.2f} {correct:>8} {cf_str:>10}")

        best = max(vrows, key=lambda r: r.get("npu_gops") or r.get("e2e_gops", 0))
        best_t = best["tiles"]
        best_g = best.get("npu_gops") or best.get("e2e_gops", 0)
        best_scaling = best_g / base_gops if base_gops > 0 else 0
        best_eff = best_scaling / best_t

        # Verdict
        if best_t >= 8 and best_scaling >= 8 and best_eff >= 0.5:
            verdict = "SCALING_RECOVERY_PASS"
        elif best_t >= 8 and best_scaling >= 6:
            verdict = "SCALING_RECOVERY_PARTIAL"
        elif best_scaling < 5:
            verdict = "SCALING_RECOVERY_FAIL"
        else:
            verdict = "SCALING_RECOVERY_PARTIAL"

        print(f"\n  Best: T={best_t}  {best_g:.2f} GOPS  {best_scaling:.2f}x  eff={best_eff:.2f}")
        print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
