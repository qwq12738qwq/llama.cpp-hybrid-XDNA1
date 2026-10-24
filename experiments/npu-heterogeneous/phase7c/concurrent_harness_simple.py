#!/usr/bin/env python3
"""
Phase 7C: GPU+NPU Concurrent Execution Harness (Simple Version)

Uses threading with subprocess calls and host timestamps.
No FIFO protocol needed - just proper async submission.
"""

import argparse
import json
import os
import sys
import time
import subprocess
import threading
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


SCRIPT_DIR = Path(__file__).parent
GPU_BENCH = SCRIPT_DIR.parent / "gpu_bench"
NPU_GEMM = Path("/workspace/llama.cpp/experiments/npu-int8-gemm/int8_gemm_nparallel.py")


@dataclass
class TimingResult:
    """Single iteration timing result"""
    mode: str = ""
    
    # GPU timings (ms)
    gpu_h2d_ms: float = 0.0
    gpu_compute_ms: float = 0.0
    gpu_d2h_ms: float = 0.0
    gpu_total_ms: float = 0.0
    gpu_gops: float = 0.0
    
    # NPU timings (ms)
    npu_host_ms: float = 0.0
    npu_exec_ms: float = 0.0
    npu_gops: float = 0.0
    
    # Concurrent metadata (ms)
    wall_start_ns: int = 0
    wall_end_ns: int = 0
    sequential_time_ms: float = 0.0
    concurrent_time_ms: float = 0.0
    overlap_ms: float = 0.0
    speedup: float = 1.0
    gpu_slowdown_pct: float = 0.0


def run_gpu_bench(M: int, K: int, N: int, iters: int = 1, warmup: int = 0) -> Optional[TimingResult]:
    """Run GPU GEMM benchmark once, return timing result"""
    cmd = [
        str(GPU_BENCH), "--M", str(M), "--K", str(K), "--N", str(N),
        "--iters", str(iters), "--warmup", str(warmup)
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/vulkan:" + env.get("LD_LIBRARY_PATH", "")
    
    start_ns = time.perf_counter_ns()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GPU_BENCH.parent), env=env, timeout=120)
    end_ns = time.perf_counter_ns()
    
    if proc.returncode != 0:
        print(f"GPU bench failed: {proc.stderr[:200]}", file=sys.stderr)
        return None
    
    json_start = proc.stdout.rfind("{")
    if json_start < 0:
        return None
    
    try:
        data = json.loads(proc.stdout[json_start:])
        r = TimingResult(
            mode="gpu_only",
            gpu_h2d_ms=data.get("h2d_ms", 0),
            gpu_compute_ms=data.get("compute_ms", 0),
            gpu_d2h_ms=data.get("d2h_ms", 0),
            gpu_total_ms=data.get("total_ms", 0),
            gpu_gops=data.get("gops", 0),
            wall_start_ns=start_ns,
            wall_end_ns=end_ns,
        )
        return r
    except json.JSONDecodeError:
        return None


def run_npu_bench(M: int, K: int, N: int, cols: int, iters: int = 1) -> Optional[TimingResult]:
    """Run NPU N-parallel GEMM once, return timing result"""
    cmd = [
        sys.executable, str(NPU_GEMM),
        "--M", str(M), "--K", str(K), "--N", str(N),
        "--cols", str(cols), "--iters", str(iters)
    ]
    
    start_ns = time.perf_counter_ns()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NPU_GEMM.parent), timeout=180)
    end_ns = time.perf_counter_ns()
    
    if proc.returncode != 0:
        print(f"NPU bench failed: {proc.stderr[:200]}", file=sys.stderr)
        return None
    
    # Parse output (NPU outputs in microseconds, convert to ms)
    host_ms = 0.0
    npu_ms = 0.0
    npu_gops = 0.0
    for line in proc.stdout.split("\n"):
        if "Host submission latency" in line and "mean=" in line:
            parts = line.split("mean=")[1].split(" ")[0]
            host_ms = float(parts) / 1000.0  # us -> ms
        if "NPU execution latency" in line and "mean=" in line:
            parts = line.split("mean=")[1].split(" ")[0]
            npu_ms = float(parts) / 1000.0  # us -> ms
        if "Effective GOPS (NPU exec):" in line:
            npu_gops = float(line.split(":")[1].strip())
    
    if host_ms > 0 and npu_ms > 0:
        r = TimingResult(
            mode="npu_only",
            npu_host_ms=host_ms,
            npu_exec_ms=npu_ms,
            npu_gops=npu_gops,
            wall_start_ns=start_ns,
            wall_end_ns=end_ns,
        )
        return r
    return None


def run_concurrent(gpu_M: int, gpu_K: int, gpu_N: int, 
                   npu_M: int, npu_K: int, npu_N: int, npu_cols: int,
                   warmup: int = 1) -> Optional[TimingResult]:
    """Run GPU and NPU concurrently using threads"""
    
    # Warmup
    for _ in range(warmup):
        run_gpu_bench(gpu_M, gpu_K, gpu_N, 1, 0)
        run_npu_bench(npu_M, npu_K, npu_N, npu_cols, 1)
    
    results = {"gpu": None, "npu": None}
    wall_start = time.perf_counter_ns()
    
    def gpu_worker():
        results["gpu"] = run_gpu_bench(gpu_M, gpu_K, gpu_N, 1, 0)
    
    def npu_worker():
        results["npu"] = run_npu_bench(npu_M, npu_K, npu_N, npu_cols, 1)
    
    gpu_thread = threading.Thread(target=gpu_worker)
    npu_thread = threading.Thread(target=npu_worker)
    
    gpu_thread.start()
    npu_thread.start()
    
    gpu_thread.join()
    npu_thread.join()
    
    wall_end = time.perf_counter_ns()
    
    if not results["gpu"] or not results["npu"]:
        return None
    
    gpu = results["gpu"]
    npu = results["npu"]
    
    gpu_time_ms = gpu.gpu_total_ms
    npu_time_ms = npu.npu_host_ms
    sequential_time_ms = gpu_time_ms + npu_time_ms
    concurrent_time_ms = (wall_end - wall_start) / 1e6
    overlap_ms = sequential_time_ms - concurrent_time_ms
    speedup = sequential_time_ms / concurrent_time_ms if concurrent_time_ms > 0 else 1.0
    
    # GPU slowdown: compare concurrent GPU time vs solo GPU time
    # We need a baseline GPU-only run for this
    gpu_slowdown_pct = 0.0  # Will be computed after baseline
    
    r = TimingResult(
        mode="concurrent",
        gpu_h2d_ms=gpu.gpu_h2d_ms,
        gpu_compute_ms=gpu.gpu_compute_ms,
        gpu_d2h_ms=gpu.gpu_d2h_ms,
        gpu_total_ms=gpu.gpu_total_ms,
        gpu_gops=gpu.gpu_gops,
        npu_host_ms=npu.npu_host_ms,
        npu_exec_ms=npu.npu_exec_ms,
        npu_gops=npu.npu_gops,
        wall_start_ns=wall_start,
        wall_end_ns=wall_end,
        sequential_time_ms=sequential_time_ms,
        concurrent_time_ms=concurrent_time_ms,
        overlap_ms=max(0, overlap_ms),
        speedup=speedup,
        gpu_slowdown_pct=gpu_slowdown_pct,
    )
    return r


def run_sequential_gpu_then_npu(gpu_M: int, gpu_K: int, gpu_N: int,
                                 npu_M: int, npu_K: int, npu_N: int, npu_cols: int) -> Optional[TimingResult]:
    """Run GPU then NPU sequentially"""
    gpu = run_gpu_bench(gpu_M, gpu_K, gpu_N, 1, 0)
    npu = run_npu_bench(npu_M, npu_K, npu_N, npu_cols, 1)
    if not gpu or not npu:
        return None
    return TimingResult(
        mode="gpu_then_npu",
        gpu_h2d_ms=gpu.gpu_h2d_ms,
        gpu_compute_ms=gpu.gpu_compute_ms,
        gpu_d2h_ms=gpu.gpu_d2h_ms,
        gpu_total_ms=gpu.gpu_total_ms,
        gpu_gops=gpu.gpu_gops,
        npu_host_ms=npu.npu_host_ms,
        npu_exec_ms=npu.npu_exec_ms,
        npu_gops=npu.npu_gops,
        sequential_time_ms=gpu.gpu_total_ms + npu.npu_host_ms,
        concurrent_time_ms=gpu.gpu_total_ms + npu.npu_host_ms,
        speedup=1.0,
    )


def run_sequential_npu_then_gpu(gpu_M: int, gpu_K: int, gpu_N: int,
                                 npu_M: int, npu_K: int, npu_N: int, npu_cols: int) -> Optional[TimingResult]:
    """Run NPU then GPU sequentially"""
    npu = run_npu_bench(npu_M, npu_K, npu_N, npu_cols, 1)
    gpu = run_gpu_bench(gpu_M, gpu_K, gpu_N, 1, 0)
    if not gpu or not npu:
        return None
    return TimingResult(
        mode="npu_then_gpu",
        gpu_h2d_ms=gpu.gpu_h2d_ms,
        gpu_compute_ms=gpu.gpu_compute_ms,
        gpu_d2h_ms=gpu.gpu_d2h_ms,
        gpu_total_ms=gpu.gpu_total_ms,
        gpu_gops=gpu.gpu_gops,
        npu_host_ms=npu.npu_host_ms,
        npu_exec_ms=npu.npu_exec_ms,
        npu_gops=npu.npu_gops,
        sequential_time_ms=npu.npu_host_ms + gpu.gpu_total_ms,
        concurrent_time_ms=npu.npu_host_ms + gpu.gpu_total_ms,
        speedup=1.0,
    )


def aggregate(values: List[float]) -> Dict[str, float]:
    """Compute statistics from list of values"""
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p10": 0, "p90": 0, "std": 0, "n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min": s[0],
        "max": s[-1],
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p10": s[max(0, int(n * 0.1))],
        "p90": s[min(n-1, int(n * 0.9))],
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "n": n
    }


def print_result(r: TimingResult, label: str):
    print(f"\n=== {label} ===")
    print(f"Mode: {r.mode}")
    if r.gpu_total_ms > 0:
        print(f"GPU: total={r.gpu_total_ms:.2f}ms compute={r.gpu_compute_ms:.2f}ms h2d={r.gpu_h2d_ms:.2f}ms d2h={r.gpu_d2h_ms:.2f}ms GOPS={r.gpu_gops:.2f}")
    if r.npu_host_ms > 0:
        print(f"NPU: host={r.npu_host_ms:.2f}ms exec={r.npu_exec_ms:.2f}ms GOPS={r.npu_gops:.2f}")
    if r.mode == "concurrent":
        print(f"Concurrent: wall={r.concurrent_time_ms:.2f}ms seq={r.sequential_time_ms:.2f}ms overlap={r.overlap_ms:.2f}ms speedup={r.speedup:.3f}x")


def save_results(results: Dict[str, List[TimingResult]], output_path: Path):
    output = {}
    for mode, runs in results.items():
        output[mode] = [asdict(r) for r in runs]
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


def run_mode(mode: str, args) -> List[TimingResult]:
    """Run a specific mode multiple times"""
    runs = []
    for i in range(args.iters):
        print(f"\n  Iteration {i+1}/{args.iters}...")
        if mode == "gpu_only":
            r = run_gpu_bench(args.gpu_M, args.gpu_K, args.gpu_N, 1, args.warmup)
        elif mode == "npu_only":
            r = run_npu_bench(args.M, args.K, args.N, args.npu_cols, 1)
        elif mode == "gpu_then_npu":
            r = run_sequential_gpu_then_npu(args.gpu_M, args.gpu_K, args.gpu_N, args.M, args.K, args.N, args.npu_cols)
        elif mode == "npu_then_gpu":
            r = run_sequential_npu_then_gpu(args.gpu_M, args.gpu_K, args.gpu_N, args.M, args.K, args.N, args.npu_cols)
        elif mode == "concurrent":
            r = run_concurrent(args.gpu_M, args.gpu_K, args.gpu_N, args.M, args.K, args.N, args.npu_cols, args.warmup)
        else:
            r = None
        
        if r:
            runs.append(r)
            print_result(r, f"{mode} run {i+1}")
        else:
            print(f"  FAILED")
    return runs


def main():
    parser = argparse.ArgumentParser(description="Phase 7C GPU+NPU Concurrent Execution Harness")
    parser.add_argument("--mode", choices=["gpu_only", "npu_only", "gpu_then_npu", "npu_then_gpu", "concurrent", "all"],
                        default="all", help="Execution mode")
    parser.add_argument("--M", type=int, default=512, help="NPU GEMM M dimension")
    parser.add_argument("--K", type=int, default=1024, help="NPU GEMM K dimension")
    parser.add_argument("--N", type=int, default=3584, help="NPU GEMM N dimension")
    parser.add_argument("--gpu-M", type=int, default=2048, help="GPU GEMM M dimension")
    parser.add_argument("--gpu-K", type=int, default=4096, help="GPU GEMM K dimension")
    parser.add_argument("--gpu-N", type=int, default=4096, help="GPU GEMM N dimension")
    parser.add_argument("--npu-cols", type=int, default=4, help="NPU columns (1,2,4)")
    parser.add_argument("--iters", type=int, default=5, help="Iterations per mode")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--output", type=str, default="phase7c_results.json", help="Output JSON file")
    args = parser.parse_args()

    results: Dict[str, List[TimingResult]] = {}
    modes = ["gpu_only", "npu_only", "gpu_then_npu", "npu_then_gpu", "concurrent"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*60}")
        results[mode] = run_mode(mode, args)

    # Compute GPU slowdown for concurrent runs
    if "gpu_only" in results and "concurrent" in results:
        gpu_only_times = [r.gpu_total_ms for r in results["gpu_only"] if r.gpu_total_ms > 0]
        concurrent_gpu_times = [r.gpu_total_ms for r in results["concurrent"] if r.gpu_total_ms > 0]
        if gpu_only_times and concurrent_gpu_times:
            gpu_baseline = statistics.median(gpu_only_times)
            for r in results["concurrent"]:
                if r.gpu_total_ms > 0:
                    r.gpu_slowdown_pct = (r.gpu_total_ms - gpu_baseline) / gpu_baseline * 100

    # Print aggregated stats
    print(f"\n{'='*60}")
    print("AGGREGATED STATISTICS")
    print(f"{'='*60}")
    for mode, runs in results.items():
        if not runs:
            continue
        print(f"\n--- {mode} ({len(runs)} runs) ---")
        if mode in ("gpu_only", "concurrent", "gpu_then_npu", "npu_then_gpu"):
            gpu_totals = [r.gpu_total_ms for r in runs if r.gpu_total_ms > 0]
            if gpu_totals:
                stats = aggregate(gpu_totals)
                print(f"  GPU total: median={stats['median']:.2f}ms mean={stats['mean']:.2f}ms p10={stats['p10']:.2f} p90={stats['p90']:.2f}")
        if mode in ("npu_only", "concurrent", "gpu_then_npu", "npu_then_gpu"):
            npu_host = [r.npu_host_ms for r in runs if r.npu_host_ms > 0]
            if npu_host:
                stats = aggregate(npu_host)
                print(f"  NPU host: median={stats['median']:.2f}ms mean={stats['mean']:.2f}ms p10={stats['p10']:.2f} p90={stats['p90']:.2f}")
        if mode == "concurrent":
            speedups = [r.speedup for r in runs if r.speedup > 0]
            overlaps = [r.overlap_ms for r in runs if r.overlap_ms > 0]
            slowdowns = [r.gpu_slowdown_pct for r in runs if r.gpu_slowdown_pct != 0]
            if speedups:
                stats = aggregate(speedups)
                print(f"  Speedup: median={stats['median']:.3f}x mean={stats['mean']:.3f}x p10={stats['p10']:.3f} p90={stats['p90']:.3f}")
            if overlaps:
                stats = aggregate(overlaps)
                print(f"  Overlap: median={stats['median']:.2f}ms mean={stats['mean']:.2f}ms")
            if slowdowns:
                stats = aggregate(slowdowns)
                print(f"  GPU slowdown: median={stats['median']:.1f}% mean={stats['mean']:.1f}%")

    save_results(results, Path(args.output))

    # Decision summary
    if "gpu_only" in results and "concurrent" in results and results["gpu_only"] and results["concurrent"]:
        gpu_median = aggregate([r.gpu_total_ms for r in results["gpu_only"] if r.gpu_total_ms > 0])["median"]
        concurrent_median = aggregate([r.concurrent_time_ms for r in results["concurrent"] if r.concurrent_time_ms > 0])["median"]
        if gpu_median > 0 and concurrent_median > 0:
            e2e_speedup = gpu_median / concurrent_median
            print(f"\n{'='*60}")
            print(f"E2E SPEEDUP (GPU-only vs GPU+NPU concurrent): {e2e_speedup:.3f}x")
            if e2e_speedup >= 1.20:
                print("DECISION: STRONG SUCCESS (>=20% faster)")
            elif e2e_speedup >= 1.05:
                print("DECISION: MODERATE SUCCESS (5-20% faster)")
            elif e2e_speedup >= 1.0:
                print("DECISION: MARGINAL SUCCESS (0-5% faster)")
            else:
                print("DECISION: REGRESSION (GPU+NPU slower than GPU-only)")


if __name__ == "__main__":
    main()