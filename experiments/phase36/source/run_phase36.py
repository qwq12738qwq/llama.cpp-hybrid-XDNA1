#!/usr/bin/env python3
"""Phase 36: Qwen Real-Shape XDNA Array GEMM orchestrator.

Runs:
  1. Phase 35 baseline reproduction (M=512 K=2048 N=2048, T=1/2/4/8/16)
  2. Real Qwen shapes at M=512, T=1/2/4/8/16
  3. M-scaling sweep for representative shapes at T=16
  4. Vulkan comparison for all shapes

Outputs PHASE36_RESULT summary.

Usage (inside npu-dev container):
  source /Storage/Cache/ironenv-v143/bin/activate
  python run_phase36.py --work-root /tmp/p36_final --iters 20 --warmup 5
"""
import argparse
import json
import os
import subprocess
import sys
import time

_BASE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_BASE, "bench_phase36.py")

M_BASE, K_BASE, N_BASE = 512, 2048, 2048
_TILES = [1, 2, 4, 8, 16]
_TILE_RC = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (2, 4), 16: (4, 4)}

_VULKAN_BENCH = "/workspace/llama.cpp/experiments/phase21/vulkan_gemm_bench"

_QWEN_SHAPES_M512 = [
    {"op": "ffn_gate_up", "M": 512, "K": 1024, "N": 3584},
    {"op": "ffn_down",    "M": 512, "K": 3584, "N": 1024},
    {"op": "attn_qkv",    "M": 512, "K": 1024, "N": 6144},
    {"op": "attn_q",      "M": 512, "K": 1024, "N": 4096},
    {"op": "attn_output", "M": 512, "K": 2048, "N": 1024},
]

_MSCALE_SHAPES = [
    {"op": "K1024_N3584", "K": 1024, "N": 3584},
    {"op": "K3584_N1024", "K": 3584, "N": 1024},
    {"op": "K1024_N4096", "K": 1024, "N": 4096},
    {"op": "K1024_N6144", "K": 1024, "N": 6144},
    {"op": "K2048_N1024", "K": 2048, "N": 1024},
]

_M_VALUES_T16 = [512, 1024, 2048, 4096]


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


def compile_design(M, K, N, tiles, variant, fifo_depth, work_dir, n_persist_iters=4):
    rows, cols = _TILE_RC[tiles]
    cmd = [
        sys.executable, _BENCH,
        "--M", str(M), "--K", str(K), "--N", str(N),
        "--tiles", str(tiles),
        "--variant", variant,
        "--fifo-depth", str(fifo_depth),
        "--n-persist-iters", str(n_persist_iters),
        "--work-dir", work_dir,
    ]
    run_cmd(cmd, cwd=_BASE)


def benchmark_shape(M, K, N, tiles, variant, fifo_depth, work_dir,
                    iters=20, warmup=5, breakdown=False, n_persist_iters=4,
                    vulkan_bench=None):
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
    if vulkan_bench:
        cmd.extend(["--vulkan-bench", vulkan_bench])
    r = run_cmd(cmd, cwd=_BASE, check=False)
    return r.returncode == 0


def load_result(work_dir, tiles):
    path = os.path.join(work_dir, "raw", f"bench_{tiles}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def experiment(name, shapes_tiles, iters=20, warmup=5, breakdown_t16=False,
               vulkan_bench=None, n_persist_iters=4):
    """Run one experiment across shapes x tile counts."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    results = []
    for shape_info in shapes_tiles:
        M = shape_info["M"]
        K = shape_info["K"]
        N = shape_info["N"]
        op = shape_info.get("op", f"M{M}_K{K}_N{N}")
        tiles = shape_info.get("tiles", 16)
        variant = shape_info.get("variant", "persistent")
        fifo_depth = shape_info.get("fifo_depth", 2)

        work_dir = os.path.join(
            args.work_root, name,
            f"{op}_t{tiles}_d{fifo_depth}"
        )
        os.makedirs(work_dir, exist_ok=True)

        print(f"\n--- {op} M={M} K={K} N={N} T={tiles} variant={variant} ---")

        do_breakdown = breakdown_t16 and tiles == 16
        ok = benchmark_shape(
            M, K, N, tiles, variant, fifo_depth, work_dir,
            iters=iters, warmup=warmup,
            breakdown=do_breakdown,
            n_persist_iters=n_persist_iters,
            vulkan_bench=vulkan_bench,
        )

        r = load_result(work_dir, tiles)
        if r:
            results.append(r)
            gops = r.get("npu_gops") or r.get("e2e_gops", 0)
            correct = "PASS" if r.get("correct") else "FAIL"
            vk = r.get("vulkan", {})
            vk_str = ""
            if vk and "vulkan_gops" in vk:
                speedup = vk.get("speedup_vs_vulkan")
                speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
                vk_str = f"  VK={vk['vulkan_gops']:.1f}GOPS spd={speedup_str}"
            bd = r.get("breakdown", {})
            cf = bd.get("compute_fraction")
            cf_str = f"cf={cf:.1%}" if cf is not None else ""
            print(f"  RESULT: T={tiles}  {gops:.1f} GOPS  {correct}  {cf_str}{vk_str}")
        else:
            print(f"  RESULT: T={tiles}  NO DATA")

    return results


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=str, default="/tmp/p36_final")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vulkan-bench", type=str, default=_VULKAN_BENCH)
    ap.add_argument("--skip-vulkan", action="store_true")
    ap.add_argument("--n-persist-iters", type=int, default=4)
    args = ap.parse_args()

    vulkan_bench = None if args.skip_vulkan else args.vulkan_bench
    if vulkan_bench and not os.path.exists(vulkan_bench):
        print(f"WARNING: vulkan_bench not found at {vulkan_bench}, skipping Vulkan comparison")
        vulkan_bench = None

    all_results = {}

    # Step 1: Phase 35 baseline reproduction
    print("\n" + "="*60)
    print("STEP 1: PHASE 35 BASELINE REPRODUCTION")
    print("="*60)
    baseline_results = []
    for tiles in _TILES:
        work_dir = os.path.join(args.work_root, "step1_baseline", f"t{tiles}")
        os.makedirs(work_dir, exist_ok=True)
        print(f"\n--- baseline T={tiles} ---")
        try:
            compile_design(M_BASE, K_BASE, N_BASE, tiles, "persistent", 2, work_dir,
                           n_persist_iters=args.n_persist_iters)
        except SystemExit:
            print(f"  COMPILE FAILED")
            continue
        ok = benchmark_shape(
            M_BASE, K_BASE, N_BASE, tiles, "persistent", 2, work_dir,
            iters=args.iters, warmup=args.warmup,
            breakdown=(tiles == 16),
            n_persist_iters=args.n_persist_iters,
            vulkan_bench=vulkan_bench,
        )
        r = load_result(work_dir, tiles)
        if r:
            baseline_results.append((tiles, r))
    all_results["step1_baseline"] = baseline_results

    # Step 2: Real Qwen shapes at M=512
    print("\n" + "="*60)
    print("STEP 2: REAL QWEN SHAPES (M=512)")
    print("="*60)
    qwen_results = []
    tiles_to_run = [4, 8, 16] if args.quick else _TILES
    for shape in _QWEN_SHAPES_M512:
        for tiles in tiles_to_run:
            shape_entry = dict(shape)
            shape_entry["tiles"] = tiles
            shape_entry["variant"] = "persistent"
            shape_entry["fifo_depth"] = 2
            work_dir = os.path.join(
                args.work_root, "step2_qwen_m512",
                f"{shape['op']}_t{tiles}"
            )
            os.makedirs(work_dir, exist_ok=True)
            print(f"\n--- {shape['op']} M={shape['M']} K={shape['K']} N={shape['N']} T={tiles} ---")
            try:
                compile_design(
                    shape["M"], shape["K"], shape["N"], tiles, "persistent", 2, work_dir,
                    n_persist_iters=args.n_persist_iters,
                )
            except SystemExit:
                print(f"  COMPILE FAILED")
                continue
            ok = benchmark_shape(
                shape["M"], shape["K"], shape["N"], tiles, "persistent", 2, work_dir,
                iters=args.iters, warmup=args.warmup,
                breakdown=(tiles == 16),
                n_persist_iters=args.n_persist_iters,
                vulkan_bench=vulkan_bench,
            )
            r = load_result(work_dir, tiles)
            if r:
                r["op"] = shape["op"]
                qwen_results.append(r)
    all_results["step2_qwen_m512"] = qwen_results

    # Step 3: M-scaling at T=16
    print("\n" + "="*60)
    print("STEP 3: M-SCALING (T=16)")
    print("="*60)
    mscale_results = []
    for shape in _MSCALE_SHAPES:
        for M in _M_VALUES_T16:
            work_dir = os.path.join(
                args.work_root, "step3_mscale_t16",
                f"{shape['op']}_M{M}"
            )
            os.makedirs(work_dir, exist_ok=True)
            print(f"\n--- {shape['op']} M={M} K={shape['K']} N={shape['N']} T=16 ---")
            try:
                compile_design(
                    M, shape["K"], shape["N"], 16, "persistent", 2, work_dir,
                    n_persist_iters=args.n_persist_iters,
                )
            except SystemExit:
                print(f"  COMPILE FAILED")
                continue
            ok = benchmark_shape(
                M, shape["K"], shape["N"], 16, "persistent", 2, work_dir,
                iters=args.iters, warmup=args.warmup,
                breakdown=False,
                n_persist_iters=args.n_persist_iters,
                vulkan_bench=vulkan_bench,
            )
            r = load_result(work_dir, 16)
            if r:
                r["op"] = shape["op"]
                r["M_scaled"] = M
                mscale_results.append(r)
    all_results["step3_mscale_t16"] = mscale_results

    # Save full results
    summary_path = os.path.join(args.work_root, "PHASE36_ALL_RESULTS.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved: {summary_path}")

    # Print PHASE36_RESULT summary
    print("\n" + "="*70)
    print("PHASE36_RESULT")
    print("="*70)

    # Baseline summary
    baseline_t1 = None
    baseline_t16 = None
    for tiles, r in baseline_results:
        if tiles == 1:
            baseline_t1 = r
        if tiles == 16:
            baseline_t16 = r

    def fmt_t(r):
        gops = r.get("npu_gops") or r.get("e2e_gops", 0)
        return f"{gops:.1f} GOPS"

    print(f"\nbaseline_phase35:")
    if baseline_t1:
        g1 = baseline_t1.get("npu_gops") or baseline_t1.get("e2e_gops", 0)
        print(f"  T1: {g1:.1f} GOPS")
    if baseline_t16:
        g16 = baseline_t16.get("npu_gops") or baseline_t16.get("e2e_gops", 0)
        g1 = baseline_t1.get("npu_gops") or baseline_t1.get("e2e_gops", 0) if baseline_t1 else g16 / 13.69
        scaling = g16 / g1 if g1 > 0 else 0
        eff = scaling / 16
        print(f"  T16: {g16:.1f} GOPS")
        print(f"  scaling_efficiency: {eff:.2f}")
    else:
        print("  T16: N/A")
        print("  scaling_efficiency: N/A")

    # Qwen shapes summary
    print(f"\nqwen_shapes:")
    qwen_summary = {}
    for r in qwen_results:
        op = r.get("op", "unknown")
        tiles = r.get("tiles", 0)
        gops = r.get("npu_gops") or r.get("e2e_gops", 0)
        vk = r.get("vulkan", {})
        vk_gops = vk.get("vulkan_gops") if vk else None
        speedup = vk.get("speedup_vs_vulkan") if vk else None
        correct = r.get("correct", False)
        key = f"{op}_T{tiles}"
        qwen_summary[key] = {
            "gops": round(gops, 2),
            "correct": correct,
            "vulkan_gops": round(vk_gops, 2) if vk_gops else None,
            "speedup_vs_vulkan": round(speedup, 4) if speedup else None,
        }
        spd_str = f"spd={speedup:.2f}x" if speedup else "no_vk"
        print(f"  {op} T={tiles}: {gops:.1f} GOPS  correct={correct}  {spd_str}")

    # M-scaling summary
    print(f"\nm_scaling:")
    mscale_summary = {}
    for r in mscale_results:
        op = r.get("op", "unknown")
        M_val = r.get("M_scaled", r.get("shape", {}).get("M", 0))
        tiles = r.get("tiles", 16)
        gops = r.get("npu_gops") or r.get("e2e_gops", 0)
        vk = r.get("vulkan", {})
        vk_gops = vk.get("vulkan_gops") if vk else None
        key = f"{op}_M{M_val}"
        mscale_summary[key] = {
            "gops": round(gops, 2),
            "vulkan_gops": round(vk_gops, 2) if vk_gops else None,
        }
        vk_str = f"VK={vk_gops:.1f}" if vk_gops else ""
        print(f"  {op} M={M_val}: {gops:.1f} GOPS  {vk_str}")

    # Identify crossover
    print(f"\ncrossover:")
    crossover_points = {}
    for r in qwen_results + mscale_results:
        op = r.get("op", "unknown")
        vk = r.get("vulkan", {})
        if not vk or "vulkan_us" not in vk:
            continue
        xdna_us = r.get("npu_exec_us", {}).get("mean") or r.get("host_e2e_us", {}).get("mean")
        vk_us = vk.get("vulkan_us")
        if xdna_us and vk_us:
            M_val = r.get("M_scaled", r.get("shape", {}).get("M", 0))
            tiles = r.get("tiles", 0)
            key = f"{op}_M{M_val}_T{tiles}"
            if xdna_us <= vk_us:
                crossover_points[key] = {
                    "M": M_val,
                    "tiles": tiles,
                    "xdna_us": round(xdna_us, 2),
                    "vulkan_us": round(vk_us, 2),
                    "xdna_gops": round(r.get("npu_gops") or r.get("e2e_gops", 0), 2),
                    "vulkan_gops": round(vk.get("vulkan_gops", 0), 2),
                }
                print(f"  CROSSOVER: {key} XDNA={xdna_us:.0f}us <= VK={vk_us:.0f}us")
            else:
                print(f"  NO_CROSSOVER: {key} XDNA={xdna_us:.0f}us > VK={vk_us:.0f}us")

    # Bottleneck analysis
    print(f"\nbottleneck:")
    breakdown_found = False
    for r in baseline_results + qwen_results:
        bd = r.get("breakdown", {})
        cf = bd.get("compute_fraction")
        if cf is not None:
            breakdown_found = True
            op = r.get("op", "baseline")
            tiles = r.get("tiles", 0)
            if cf < 0.3:
                bottleneck = "DMA"
            elif cf < 0.5:
                bottleneck = "MIXED"
            else:
                bottleneck = "COMPUTE"
            print(f"  {op} T={tiles}: compute_frac={cf:.1%} -> {bottleneck}")
    if not breakdown_found:
        print("  No breakdown data available")

    # Verdict
    print(f"\nverdict:")
    any_crossover = len(crossover_points) > 0
    any_partial = False
    for r in qwen_results:
        vk = r.get("vulkan", {})
        if not vk or "speedup_vs_vulkan" not in vk:
            continue
        speedup = vk.get("speedup_vs_vulkan")
        if speedup is not None and speedup < 1.0:
            any_crossover = True
            break
        if speedup is not None and speedup < 2.0:
            any_partial = True

    if any_crossover:
        verdict = "QWEN_SHAPE_COMPETITIVE"
        next_action = "PHASE37_LARGE_PP_VALIDATION"
    elif any_partial:
        verdict = "QWEN_SHAPE_PARTIAL"
        next_action = "PHASE37_LARGE_PP_VALIDATION"
    else:
        verdict = "QWEN_SHAPE_NO_CROSSOVER"
        next_action = "identify_specific_bottleneck"

    print(f"  {verdict}")
    print(f"  next_action: {next_action}")
    print("="*70)

    # Save summary
    summary = {
        "baseline_phase35": {
            "T1": baseline_t1.get("npu_gops") or baseline_t1.get("e2e_gops", 0) if baseline_t1 else None,
            "T16": baseline_t16.get("npu_gops") or baseline_t16.get("e2e_gops", 0) if baseline_t16 else None,
            "scaling_efficiency": round(
                (baseline_t16.get("npu_gops") or baseline_t16.get("e2e_gops", 0)) /
                (baseline_t1.get("npu_gops") or baseline_t1.get("e2e_gops", 1)) / 16, 4
            ) if baseline_t1 and baseline_t16 else None,
        },
        "qwen_shapes": qwen_summary,
        "m_scaling": mscale_summary,
        "crossover": crossover_points,
        "verdict": verdict,
        "next_action": next_action,
    }
    summary_path = os.path.join(args.work_root, "PHASE36_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
