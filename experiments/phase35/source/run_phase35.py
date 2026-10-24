#!/usr/bin/env python3
"""Phase 35 master runner: executes all dataflow experiments and prints PHASE35_RESULT.

Experiments:
  E1: ObjectFifo depth sweep (baseline topology, depth=2/3/4/6/8)
  E2: 2D distribution vs baseline (variant=2d)
  E3: Persistent execution (variant=persistent)
  E4: Depth + 2D combined (variant=2d, fifo_depth=4/8)

For each experiment, runs T=1/2/4/8/16 and measures:
  - GOPS
  - Scaling vs T=1
  - Efficiency
  - Correctness

Breakdown timing is run for the best T=16 variant.

Usage:
  source /Storage/Cache/ironenv-v143/bin/activate
  python run_phase35.py --work-root /tmp/p35_all --iters 20 --warmup 5
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SRC = _BASE
_BENCH = os.path.join(_SRC, "bench_phase35.py")
_GEN = os.path.join(_SRC, "gen_phase35.py")

M, K, N = 512, 2048, 2048
_TILES = [1, 2, 4, 8, 16]
_TILE_RC = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (2, 4), 16: (4, 4)}


def run_cmd(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(cmd)}")
    env = os.environ.copy()
    env["PATH"] = "/Storage/Cache/ironenv-v143/bin:" + env.get("PATH", "")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if r.stdout:
        print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    if check and r.returncode != 0:
        print(f"ERROR: command failed with code {r.returncode}", file=sys.stderr)
        sys.exit(1)
    return r


def compile_design(variant, tiles, fifo_depth, work_dir, n_persist_iters=4):
    """Compile a single design variant."""
    rows, cols = _TILE_RC[tiles]
    cmd = [
        sys.executable, _GEN,
        "--M", str(M), "--K", str(K), "--N", str(N),
        "--tiles", str(tiles),
        "--variant", variant,
        "--fifo-depth", str(fifo_depth),
        "--n-persist-iters", str(n_persist_iters),
        "--work-dir", work_dir,
    ]
    run_cmd(cmd, cwd=_BASE)


def benchmark(variant, tiles, fifo_depth, work_dir, iters=20, warmup=5, breakdown=False,
              n_persist_iters=4):
    """Benchmark a single design variant."""
    rows, cols = _TILE_RC[tiles]
    cmd = [
        sys.executable, _BENCH,
        "--M", str(M), "--K", str(K), "--N", str(N),
        "--tiles", str(tiles),
        "--variant", variant,
        "--fifo-depth", str(fifo_depth),
        "--n-persist-iters", str(n_persist_iters),
        "--work-dir", work_dir,
        "--iters", str(iters),
        "--warmup", str(warmup),
    ]
    if breakdown:
        cmd.append("--breakdown")
    r = run_cmd(cmd, cwd=_BASE, check=False)
    return r.returncode == 0


def load_result(work_dir, tiles):
    path = os.path.join(work_dir, "raw", f"bench_{tiles}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def experiment(name, variants_tiles_depth, iters=20, warmup=5, breakdown_t16=False):
    """Run one experiment across all tile counts.
    
    variants_tiles_depth: list of (variant, tiles, fifo_depth) tuples
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")
    
    results = []
    for variant, tiles, fifo_depth in variants_tiles_depth:
        work_dir = os.path.join(args.work_root, name, f"t{tiles}_d{fifo_depth}")
        os.makedirs(work_dir, exist_ok=True)
        
        print(f"\n--- {variant} T={tiles} depth={fifo_depth} ---")
        
        # Compile
        t0 = time.time()
        try:
            compile_design(variant, tiles, fifo_depth, work_dir)
        except SystemExit:
            print(f"  COMPILE FAILED for {variant} T={tiles}")
            continue
        compile_time = time.time() - t0
        print(f"  Compile time: {compile_time:.1f}s")
        
        # Benchmark
        do_breakdown = breakdown_t16 and tiles == 16
        ok = benchmark(variant, tiles, fifo_depth, work_dir, iters=iters, warmup=warmup,
                        breakdown=do_breakdown)
        
        r = load_result(work_dir, tiles)
        if r:
            results.append(r)
            gops = r.get("npu_gops") or r.get("e2e_gops", 0)
            correct = "PASS" if r.get("correct") else "FAIL"
            bd = r.get("breakdown", {})
            cf = bd.get("compute_fraction")
            cf_str = f"cf={cf:.1%}" if cf is not None else ""
            print(f"  RESULT: T={tiles}  {gops:.1f} GOPS  {correct}  {cf_str}")
        else:
            print(f"  RESULT: T={tiles}  NO DATA")
    
    return results


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=str, default="/tmp/p35_final")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="Quick mode: fewer iters, T=4/8/16 only")
    args = ap.parse_args()

    tiles_to_run = [4, 8, 16] if args.quick else _TILES
    variants_tiles = [(t,) for t in tiles_to_run]

    all_results = {}

    # E0: Baseline sweep (T=1/2/4/8/16, fifo_depth=2)
    print("\n" + "="*60)
    print("E0: BASELINE SWEEP")
    print("="*60)
    e0_results = []
    for tiles in tiles_to_run:
        work_dir = os.path.join(args.work_root, "e0_baseline", f"t{tiles}")
        os.makedirs(work_dir, exist_ok=True)
        print(f"\n--- baseline T={tiles} ---")
        try:
            compile_design("baseline", tiles, 2, work_dir)
        except SystemExit:
            print(f"  COMPILE FAILED")
            continue
        ok = benchmark("baseline", tiles, 2, work_dir, iters=args.iters, warmup=args.warmup,
                        breakdown=(tiles == 16))
        r = load_result(work_dir, tiles)
        if r:
            e0_results.append((tiles, r))
    all_results["e0_baseline"] = e0_results

    # E1a: Depth sweep at T=16
    e1a_results = []
    for depth in [2, 4, 6, 8]:
        work_dir = os.path.join(args.work_root, "e1a_depth_sweep", f"d{depth}")
        os.makedirs(work_dir, exist_ok=True)
        print(f"\n--- E1a: baseline depth={depth} T=16 ---")
        try:
            compile_design("baseline", 16, depth, work_dir)
        except SystemExit:
            print(f"  COMPILE FAILED")
            continue
        ok = benchmark("baseline", 16, depth, work_dir, iters=args.iters, warmup=args.warmup)
        r = load_result(work_dir, 16)
        if r:
            e1a_results.append((depth, r))
            gops = r.get("npu_gops") or r.get("e2e_gops", 0)
            print(f"  RESULT: depth={depth}  {gops:.1f} GOPS  correct={r.get('correct')}")
    all_results["e1a_depth_sweep_t16"] = e1a_results

    # E1b: Depth sweep at T=8
    e1b_results = []
    for depth in [2, 4, 8]:
        work_dir = os.path.join(args.work_root, "e1b_depth_sweep", f"d{depth}")
        os.makedirs(work_dir, exist_ok=True)
        print(f"\n--- E1b: baseline depth={depth} T=8 ---")
        try:
            compile_design("baseline", 8, depth, work_dir)
        except SystemExit:
            continue
        benchmark("baseline", 8, depth, work_dir, iters=args.iters, warmup=args.warmup)
        r = load_result(work_dir, 8)
        if r:
            e1b_results.append((depth, r))
    all_results["e1b_depth_sweep_t8"] = e1b_results

    # E2: 2D distribution
    e2_results = []
    for tiles in tiles_to_run:
        work_dir = os.path.join(args.work_root, "e2_2d", f"t{tiles}")
        os.makedirs(work_dir, exist_ok=True)
        print(f"\n--- E2: 2d-split T={tiles} ---")
        try:
            compile_design("2d", tiles, 2, work_dir)
        except SystemExit:
            continue
        ok = benchmark("2d", tiles, 2, work_dir, iters=args.iters, warmup=args.warmup,
                        breakdown=(tiles == 16))
        r = load_result(work_dir, tiles)
        if r:
            e2_results.append((tiles, r))
    all_results["e2_2d"] = e2_results

    # E3: Persistent
    e3_results = []
    for tiles in tiles_to_run:
        work_dir = os.path.join(args.work_root, "e3_persistent", f"t{tiles}")
        os.makedirs(work_dir, exist_ok=True)
        print(f"\n--- E3: persistent T={tiles} ---")
        try:
            compile_design("persistent", tiles, 2, work_dir)
        except SystemExit:
            continue
        benchmark("persistent", tiles, 2, work_dir, iters=args.iters, warmup=args.warmup)
        r = load_result(work_dir, tiles)
        if r:
            e3_results.append((tiles, r))
    all_results["e3_persistent"] = e3_results

    # E4: Depth + 2D combined at T=16
    e4_results = []
    for depth in [4, 8]:
        work_dir = os.path.join(args.work_root, "e4_depth_2d", f"d{depth}")
        os.makedirs(work_dir, exist_ok=True)
        print(f"\n--- E4: 2d depth={depth} T=16 ---")
        try:
            compile_design("2d", 16, depth, work_dir)
        except SystemExit:
            continue
        benchmark("2d", 16, depth, work_dir, iters=args.iters, warmup=args.warmup)
        r = load_result(work_dir, 16)
        if r:
            e4_results.append((depth, r))
    all_results["e4_depth_2d_t16"] = e4_results

    # Find best result across all experiments
    best = None
    best_gops = 0
    for exp_name, exp_results in all_results.items():
        for item in exp_results:
            if isinstance(item, tuple):
                r = item[-1]
            else:
                r = item
            gops = r.get("npu_gops") or r.get("e2e_gops", 0)
            if gops > best_gops:
                best_gops = gops
                best = (exp_name, r)

    # Find baseline T=1
    baseline_t1 = None
    for exp_name, exp_results in all_results.items():
        for item in exp_results:
            if isinstance(item, tuple):
                r = item[-1]
            else:
                r = item
            if r.get("tiles") == 1:
                baseline_t1 = r
                break
        if baseline_t1:
            break

    # Print final result
    print("\n" + "="*70)
    print("PHASE35_RESULT")
    print("="*70)

    base_gops = (baseline_t1.get("npu_gops") or baseline_t1.get("e2e_gops", 0)) if baseline_t1 else 0

    def fmt_t(tiles, r):
        gops = r.get("npu_gops") or r.get("e2e_gops", 0)
        scaling = gops / base_gops if base_gops > 0 else 0
        eff = scaling / tiles if tiles > 0 else 0
        return f"  T={tiles}: {gops:.1f} GOPS  {scaling:.2f}x  eff={eff:.2f}"

    print(f"\nshape: {M}x{K}x{N} INT8")
    print(f"baseline_T1: {base_gops:.1f} GOPS")

    # Print best variant per tile count
    best_per_t = {}
    for exp_name, exp_results in all_results.items():
        for item in exp_results:
            if isinstance(item, tuple):
                key, r = item
            else:
                r = item
            t = r.get("tiles", 0)
            gops = r.get("npu_gops") or r.get("e2e_gops", 0)
            if t not in best_per_t or gops > best_per_t[t][1].get("npu_gops", 0):
                best_per_t[t] = (exp_name, r)

    for t in sorted(best_per_t):
        exp_name, r = best_per_t[t]
        print(f"best_T{t}: {fmt_t(t, r).strip()} [{exp_name}]")

    if best:
        exp_name, r = best
        tiles = r.get("tiles", 0)
        gops = r.get("npu_gops") or r.get("e2e_gops", 0)
        scaling = gops / base_gops if base_gops > 0 else 0
        eff = scaling / tiles if tiles > 0 else 0
        print(f"\nbest_variant: {exp_name}")
        print(f"best_T16: {gops:.1f} GOPS  {scaling:.2f}x  eff={eff:.2f}")
        print(f"best_scaling: {scaling:.2f}x")
        print(f"best_efficiency: {eff:.2f}")

    # Breakdown for best T=16
    breakdown_t16 = None
    for exp_name, exp_results in all_results.items():
        for item in exp_results:
            if isinstance(item, tuple):
                r = item[-1]
            else:
                r = item
            if r.get("tiles") == 16 and r.get("breakdown"):
                breakdown_t16 = r.get("breakdown")
                break
        if breakdown_t16:
            break

    if breakdown_t16:
        print(f"\nbreakdown_T16:")
        print(f"  compute_us: {breakdown_t16.get('compute_us', 'N/A')}")
        print(f"  dma_us: {breakdown_t16.get('dma_us', 'N/A')}")
        print(f"  compute_fraction: {breakdown_t16.get('compute_fraction', 'N/A')}")

    # Primary bottleneck
    bottleneck = "UNKNOWN"
    if breakdown_t16:
        cf = breakdown_t16.get("compute_fraction")
        if cf is not None:
            if cf < 0.3:
                bottleneck = "DMA"
            elif cf < 0.5:
                bottleneck = "MIXED"
            else:
                bottleneck = "COMPUTE"
    print(f"\nprimary_bottleneck: {bottleneck}")

    # Correctness
    all_correct = all(
        r.get("correct", False)
        for exp_results in all_results.values()
        for item in exp_results
        for r in [item[-1] if isinstance(item, tuple) else item]
    )
    print(f"\ncorrectness: {'ALL PASS' if all_correct else 'SOME FAIL'}")

    # Next action
    if best:
        scaling = best_gops / base_gops if base_gops > 0 else 0
        if scaling >= 8:
            next_action = "extend_best_variant_to_qwen_shapes"
        elif scaling >= 6:
            next_action = "further_optimize_dataflow_depth_and_broadcast"
        else:
            next_action = "investigate_memtile_bandwidth_and_interconnect"
    else:
        next_action = "retry_with_environment_fix"

    print(f"\nnext_action: {next_action}")
    print("="*70)

    # Save summary
    summary = {
        "baseline_T1": base_gops,
        "best_variant": exp_name if best else None,
        "best_T16": best_gops if best else None,
        "best_scaling": scaling if best else None,
        "best_efficiency": eff if best else None,
        "breakdown_T16": breakdown_t16,
        "all_results": all_results,
        "correctness": "ALL PASS" if all_correct else "SOME FAIL",
        "primary_bottleneck": bottleneck,
        "next_action": next_action,
    }
    summary_path = os.path.join(args.work_root, "PHASE35_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
