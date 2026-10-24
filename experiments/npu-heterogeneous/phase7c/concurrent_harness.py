#!/usr/bin/env python3
"""
Phase 7C: GPU+NPU Concurrent Execution Harness

Supports:
- GPU-only (Vulkan GEMM)
- NPU-only (N-parallel INT8 GEMM)
- GPU→NPU sequential
- NPU→GPU sequential
- GPU || NPU concurrent (true async with FIFO sync)

Timing instrumentation:
- GPU: Vulkan timestamps + host wall time
- NPU: XRT execution time + host wall time
- Concurrent: synchronized start via FIFOs, measure overlap
"""

import argparse
import json
import os
import sys
import time
import subprocess
import threading
import statistics
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
import tempfile


SCRIPT_DIR = Path(__file__).parent
GPU_BENCH = SCRIPT_DIR.parent / "gpu_bench"
GPU_BURN = SCRIPT_DIR.parent / "gpu_burn"
NPU_GEMM = Path("/workspace/llama.cpp/experiments/npu-int8-gemm/int8_gemm_nparallel.py")


@dataclass
class TimingResult:
    """Single iteration timing result"""
    # GPU timings
    gpu_wall_start_ns: int = 0
    gpu_wall_end_ns: int = 0
    gpu_submit_ns: int = 0
    gpu_start_ns: int = 0
    gpu_end_ns: int = 0
    gpu_h2d_ms: float = 0.0
    gpu_compute_ms: float = 0.0
    gpu_d2h_ms: float = 0.0
    gpu_total_ms: float = 0.0
    gpu_gops: float = 0.0

    # NPU timings
    npu_wall_start_ns: int = 0
    npu_wall_end_ns: int = 0
    npu_submit_ns: int = 0
    npu_start_ns: int = 0
    npu_end_ns: int = 0
    npu_host_ms: float = 0.0
    npu_exec_ms: float = 0.0
    npu_gops: float = 0.0

    # Concurrent metadata
    mode: str = "gpu_only"
    overlap_ns: int = 0
    sequential_time_ns: int = 0
    concurrent_time_ns: int = 0
    speedup: float = 1.0


@dataclass
class AggregatedStats:
    """Aggregated statistics over multiple iterations"""
    min: float
    max: float
    mean: float
    median: float
    p10: float
    p90: float
    std: float
    n: int


def aggregate(values: List[float]) -> AggregatedStats:
    """Compute statistics from list of values"""
    if not values:
        return AggregatedStats(0, 0, 0, 0, 0, 0, 0, 0)
    s = sorted(values)
    n = len(s)
    return AggregatedStats(
        min=s[0],
        max=s[-1],
        mean=statistics.mean(s),
        median=statistics.median(s),
        p10=s[max(0, int(n * 0.1))],
        p90=s[min(n-1, int(n * 0.9))],
        std=statistics.stdev(s) if n > 1 else 0.0,
        n=n
    )


def run_gpu_only(M: int, K: int, N: int, iters: int, warmup: int) -> List[TimingResult]:
    """Run GPU-only GEMM benchmark"""
    results = []
    for i in range(iters):
        cmd = [
            str(GPU_BENCH), "--M", str(M), "--K", str(K), "--N", str(N),
            "--iters", "1", "--warmup", str(warmup)
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/vulkan:" + env.get("LD_LIBRARY_PATH", "")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GPU_BENCH.parent), env=env, timeout=120)
        if proc.returncode != 0:
            print(f"GPU run {i} failed: {proc.stderr[:200]}")
            continue
        json_start = proc.stdout.rfind("{")
        if json_start < 0:
            continue
        try:
            data = json.loads(proc.stdout[json_start:])
            r = TimingResult(
                mode="gpu_only",
                gpu_h2d_ms=data.get("h2d_ms", 0),
                gpu_compute_ms=data.get("compute_ms", 0),
                gpu_d2h_ms=data.get("d2h_ms", 0),
                gpu_total_ms=data.get("total_ms", 0),
                gpu_gops=data.get("gops", 0),
                gpu_wall_start_ns=0,
                gpu_wall_end_ns=int(data.get("total_ms", 0) * 1e6),
            )
            results.append(r)
        except json.JSONDecodeError:
            continue
    return results


def run_npu_only(M: int, K: int, N: int, cols: int, iters: int) -> List[TimingResult]:
    """Run NPU-only N-parallel GEMM benchmark"""
    results = []
    for i in range(iters):
        cmd = [
            sys.executable, str(NPU_GEMM),
            "--M", str(M), "--K", str(K), "--N", str(N),
            "--cols", str(cols), "--iters", "1"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NPU_GEMM.parent), timeout=180)
        if proc.returncode != 0:
            print(f"NPU run {i} failed: {proc.stderr[:200]}")
            continue

        # Parse output for timing
        host_ms = 0.0
        npu_ms = 0.0
        npu_gops = 0.0
        for line in proc.stdout.split("\n"):
            if "Host submission latency" in line and "mean=" in line:
                parts = line.split("mean=")[1].split(" ")[0]
                host_ms = float(parts)
            if "NPU execution latency" in line and "mean=" in line:
                parts = line.split("mean=")[1].split(" ")[0]
                npu_ms = float(parts)
            if "Effective GOPS (NPU exec):" in line:
                npu_gops = float(line.split(":")[1].strip())

        if host_ms > 0 and npu_ms > 0:
            r = TimingResult(
                mode="npu_only",
                npu_host_ms=host_ms,
                npu_exec_ms=npu_ms,
                npu_gops=npu_gops,
                npu_wall_start_ns=0,
                npu_wall_end_ns=int(host_ms * 1000),
            )
            results.append(r)
    return results


class ConcurrentRunner:
    """Runs GPU and NPU concurrently using FIFO synchronization"""

    def __init__(self, M: int, K: int, N: int, gpu_M: int, gpu_K: int, gpu_N: int, npu_cols: int):
        self.M = M
        self.K = K
        self.N = N
        self.gpu_M = gpu_M
        self.gpu_K = gpu_K
        self.gpu_N = gpu_N
        self.npu_cols = npu_cols

        # Create temp directory for FIFOs
        self.tmpdir = tempfile.mkdtemp(prefix="phase7c_")
        self.gpu_ready_fifo = os.path.join(self.tmpdir, "gpu_ready")
        self.gpu_go_fifo = os.path.join(self.tmpdir, "gpu_go")
        self.gpu_result_fifo = os.path.join(self.tmpdir, "gpu_result")
        self.npu_ready_fifo = os.path.join(self.tmpdir, "npu_ready")
        self.npu_go_fifo = os.path.join(self.tmpdir, "npu_go")
        self.npu_result_fifo = os.path.join(self.tmpdir, "npu_result")

        for f in [self.gpu_ready_fifo, self.gpu_go_fifo, self.gpu_result_fifo,
                  self.npu_ready_fifo, self.npu_go_fifo, self.npu_result_fifo]:
            os.mkfifo(f, 0o666)

        self.gpu_process: Optional[subprocess.Popen] = None
        self.npu_thread: Optional[threading.Thread] = None
        self.npu_result: Optional[TimingResult] = None
        self.gpu_result: Optional[TimingResult] = None

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def run_npu_worker(self):
        """NPU worker thread - waits for GO, runs NPU GEMM, reports timing"""
        # Wait for GO signal
        with open(self.npu_go_fifo, 'r') as f:
            f.read()

        # Run NPU GEMM with timing
        npu_wall_start = time.perf_counter_ns()
        cmd = [
            sys.executable, str(NPU_GEMM),
            "--M", str(self.M), "--K", str(self.K), "--N", str(self.N),
            "--cols", str(self.npu_cols), "--iters", "1"
        ]
        npu_submit = time.perf_counter_ns()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NPU_GEMM.parent), timeout=180)
        npu_wall_end = time.perf_counter_ns()

        # Parse output
        host_ms = 0.0
        npu_ms = 0.0
        npu_gops = 0.0
        for line in proc.stdout.split("\n"):
            if "Host submission latency" in line and "mean=" in line:
                parts = line.split("mean=")[1].split(" ")[0]
                host_ms = float(parts)
            if "NPU execution latency" in line and "mean=" in line:
                parts = line.split("mean=")[1].split(" ")[0]
                npu_ms = float(parts)
            if "Effective GOPS (NPU exec):" in line:
                npu_gops = float(line.split(":")[1].strip())

        self.npu_result = TimingResult(
            mode="concurrent",
            npu_wall_start_ns=npu_wall_start,
            npu_wall_end_ns=npu_wall_end,
            npu_submit_ns=npu_submit,
            npu_host_ms=host_ms,
            npu_exec_ms=npu_ms,
            npu_gops=npu_gops,
        )

        # Write result
        with open(self.npu_result_fifo, 'w') as f:
            f.write(json.dumps(asdict(self.npu_result)))

    def run_concurrent(self, warmup: int = 2) -> TimingResult:
        """Run GPU and NPU concurrently"""
        # Warmup runs (sequential)
        for _ in range(warmup):
            run_gpu_only(self.gpu_M, self.gpu_K, self.gpu_N, 1, 0)
            run_npu_only(self.M, self.K, self.N, self.npu_cols, 1)

        # Start GPU process (waits for GO at FIFO)
        gpu_cmd = [
            str(GPU_BURN),
            "--width", str(self.gpu_N),
            "--height", str(self.gpu_M),
            "--kdim", str(self.gpu_K),
            "--warmup", str(warmup),
            "--iters", "1",
            "--ready-fifo", self.gpu_ready_fifo,
            "--go-fifo", self.gpu_go_fifo,
            "--result-fifo", self.gpu_result_fifo,
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/vulkan:" + env.get("LD_LIBRARY_PATH", "")
        self.gpu_process = subprocess.Popen(gpu_cmd, cwd=str(GPU_BURN.parent), env=env,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Start NPU thread
        self.npu_thread = threading.Thread(target=self.run_npu_worker)
        self.npu_thread.start()

        # Wait for both READY signals
        with open(self.gpu_ready_fifo, 'r') as f:
            gpu_ready = f.read().strip()
        with open(self.npu_ready_fifo, 'r') as f:
            npu_ready = f.read().strip()

        print(f"  GPU ready: {gpu_ready}")
        print(f"  NPU ready: {npu_ready}")

        # Synchronized GO
        go_time = time.perf_counter_ns()
        with open(self.gpu_go_fifo, 'w') as f:
            f.write("GO")
        with open(self.npu_go_fifo, 'w') as f:
            f.write("GO")

        # Collect results
        with open(self.gpu_result_fifo, 'r') as f:
            gpu_result_str = f.read().strip()
        with open(self.npu_result_fifo, 'r') as f:
            npu_result_str = f.read().strip()

        self.gpu_process.wait(timeout=120)
        self.npu_thread.join(timeout=120)

        # Parse GPU result (CSV: wall_start,wall_end,gpu_exec_ns,ts_period,ts_start,ts_end)
        gpu_parts = gpu_result_str.split(",")
        if len(gpu_parts) >= 6:
            gpu_wall_start = int(gpu_parts[0])
            gpu_wall_end = int(gpu_parts[1])
            gpu_exec_ns = int(gpu_parts[2])
        else:
            gpu_wall_start = go_time
            gpu_wall_end = time.perf_counter_ns()
            gpu_exec_ns = 0

        # Parse NPU result
        npu_data = json.loads(npu_result_str) if npu_result_str else {}

        # Compute overlap
        wall_start = min(gpu_wall_start, npu_data.get("npu_wall_start_ns", go_time))
        wall_end = max(gpu_wall_end, npu_data.get("npu_wall_end_ns", go_time))
        concurrent_time = wall_end - wall_start

        gpu_time = gpu_wall_end - gpu_wall_start
        npu_time = npu_data.get("npu_wall_end_ns", 0) - npu_data.get("npu_wall_start_ns", 0)
        sequential_time = gpu_time + npu_time
        overlap = sequential_time - concurrent_time if sequential_time > concurrent_time else 0
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1.0

        result = TimingResult(
            mode="concurrent",
            gpu_wall_start_ns=gpu_wall_start,
            gpu_wall_end_ns=gpu_wall_end,
            gpu_compute_ms=gpu_exec_ns / 1e6,
            gpu_total_ms=gpu_time / 1e6,
            npu_wall_start_ns=npu_data.get("npu_wall_start_ns", 0),
            npu_wall_end_ns=npu_data.get("npu_wall_end_ns", 0),
            npu_host_ms=npu_data.get("npu_host_ms", 0),
            npu_exec_ms=npu_data.get("npu_exec_ms", 0),
            npu_gops=npu_data.get("npu_gops", 0),
            sequential_time_ns=sequential_time,
            concurrent_time_ns=concurrent_time,
            overlap_ns=overlap,
            speedup=speedup,
        )

        return result

    def run_sequential_gpu_then_npu(self) -> TimingResult:
        """Run GPU then NPU sequentially"""
        gpu_results = run_gpu_only(self.gpu_M, self.gpu_K, self.gpu_N, 1, 0)
        npu_results = run_npu_only(self.M, self.K, self.N, self.npu_cols, 1)
        if gpu_results and npu_results:
            gpu = gpu_results[0]
            npu = npu_results[0]
            return TimingResult(
                mode="gpu_then_npu",
                gpu_wall_start_ns=0,
                gpu_wall_end_ns=int(gpu.gpu_total_ms * 1e6),
                gpu_total_ms=gpu.gpu_total_ms,
                gpu_gops=gpu.gpu_gops,
                npu_wall_start_ns=int(gpu.gpu_total_ms * 1e6),
                npu_wall_end_ns=int((gpu.gpu_total_ms + npu.npu_host_ms) * 1e6),
                npu_host_ms=npu.npu_host_ms,
                npu_exec_ms=npu.npu_exec_ms,
                npu_gops=npu.npu_gops,
                sequential_time_ns=int((gpu.gpu_total_ms + npu.npu_host_ms) * 1e6),
                concurrent_time_ns=int((gpu.gpu_total_ms + npu.npu_host_ms) * 1e6),
                speedup=1.0,
            )
        return TimingResult(mode="gpu_then_npu")

    def run_sequential_npu_then_gpu(self) -> TimingResult:
        """Run NPU then GPU sequentially"""
        npu_results = run_npu_only(self.M, self.K, self.N, self.npu_cols, 1)
        gpu_results = run_gpu_only(self.gpu_M, self.gpu_K, self.gpu_N, 1, 0)
        if npu_results and gpu_results:
            npu = npu_results[0]
            gpu = gpu_results[0]
            return TimingResult(
                mode="npu_then_gpu",
                npu_wall_start_ns=0,
                npu_wall_end_ns=int(npu.npu_host_ms * 1e6),
                npu_host_ms=npu.npu_host_ms,
                npu_exec_ms=npu.npu_exec_ms,
                npu_gops=npu.npu_gops,
                gpu_wall_start_ns=int(npu.npu_host_ms * 1e6),
                gpu_wall_end_ns=int((npu.npu_host_ms + gpu.gpu_total_ms) * 1e6),
                gpu_total_ms=gpu.gpu_total_ms,
                gpu_gops=gpu.gpu_gops,
                sequential_time_ns=int((npu.npu_host_ms + gpu.gpu_total_ms) * 1e6),
                concurrent_time_ns=int((npu.npu_host_ms + gpu.gpu_total_ms) * 1e6),
                speedup=1.0,
            )
        return TimingResult(mode="npu_then_gpu")


def print_timing_result(r: TimingResult, label: str):
    print(f"\n=== {label} ===")
    print(f"Mode: {r.mode}")
    if r.mode in ("gpu_only", "concurrent", "gpu_then_npu", "npu_then_gpu"):
        print(f"GPU: total={r.gpu_total_ms:.2f}ms compute={r.gpu_compute_ms:.2f}ms GOPS={r.gpu_gops:.2f}")
    if r.mode in ("npu_only", "concurrent", "gpu_then_npu", "npu_then_gpu"):
        print(f"NPU: host={r.npu_host_ms:.2f}ms exec={r.npu_exec_ms:.2f}ms GOPS={r.npu_gops:.2f}")
    if r.mode == "concurrent":
        print(f"Concurrent: wall={r.concurrent_time_ns/1e6:.2f}ms seq={r.sequential_time_ns/1e6:.2f}ms overlap={r.overlap_ns/1e6:.2f}ms speedup={r.speedup:.3f}x")


def save_results(results: Dict[str, List[TimingResult]], output_path: Path):
    """Save all results to JSON"""
    output = {}
    for mode, runs in results.items():
        output[mode] = [asdict(r) for r in runs]
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7C GPU+NPU Concurrent Execution Harness")
    parser.add_argument("--mode", choices=["gpu_only", "npu_only", "gpu_then_npu", "npu_then_gpu", "concurrent", "all"],
                        default="all", help="Execution mode")
    parser.add_argument("--M", type=int, default=512, help="NPU GEMM M dimension")
    parser.add_argument("--K", type=int, default=1024, help="NPU GEMM K dimension")
    parser.add_argument("--N", type=int, default=3584, help="NPU GEMM N dimension")
    parser.add_argument("--gpu-M", type=int, default=1024, help="GPU GEMM M dimension")
    parser.add_argument("--gpu-K", type=int, default=4096, help="GPU GEMM K dimension")
    parser.add_argument("--gpu-N", type=int, default=4096, help="GPU GEMM N dimension")
    parser.add_argument("--npu-cols", type=int, default=4, help="NPU columns (1,2,4)")
    parser.add_argument("--iters", type=int, default=10, help="Iterations per mode")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--output", type=str, default="phase7c_results.json", help="Output JSON file")
    args = parser.parse_args()

    results: Dict[str, List[TimingResult]] = {}

    if args.mode in ("gpu_only", "all"):
        print(f"\n{'='*60}")
        print("GPU-ONLY BASELINE")
        print(f"{'='*60}")
        runs = run_gpu_only(args.gpu_M, args.gpu_K, args.gpu_N, args.iters, args.warmup)
        results["gpu_only"] = runs
        for i, r in enumerate(runs):
            print_timing_result(r, f"GPU-only run {i+1}")

    if args.mode in ("npu_only", "all"):
        print(f"\n{'='*60}")
        print("NPU-ONLY BASELINE")
        print(f"{'='*60}")
        runs = run_npu_only(args.M, args.K, args.N, args.npu_cols, args.iters)
        results["npu_only"] = runs
        for i, r in enumerate(runs):
            print_timing_result(r, f"NPU-only run {i+1}")

    if args.mode in ("gpu_then_npu", "all"):
        print(f"\n{'='*60}")
        print("SEQUENTIAL: GPU -> NPU")
        print(f"{'='*60}")
        runs = []
        for i in range(args.iters):
            runner = ConcurrentRunner(args.M, args.K, args.N, args.gpu_M, args.gpu_K, args.gpu_N, args.npu_cols)
            r = runner.run_sequential_gpu_then_npu()
            runs.append(r)
            print_timing_result(r, f"GPU->NPU run {i+1}")
        results["gpu_then_npu"] = runs

    if args.mode in ("npu_then_gpu", "all"):
        print(f"\n{'='*60}")
        print("SEQUENTIAL: NPU -> GPU")
        print(f"{'='*60}")
        runs = []
        for i in range(args.iters):
            runner = ConcurrentRunner(args.M, args.K, args.N, args.gpu_M, args.gpu_K, args.gpu_N, args.npu_cols)
            r = runner.run_sequential_npu_then_gpu()
            runs.append(r)
            print_timing_result(r, f"NPU->GPU run {i+1}")
        results["npu_then_gpu"] = runs

    if args.mode in ("concurrent", "all"):
        print(f"\n{'='*60}")
        print("CONCURRENT: GPU || NPU")
        print(f"{'='*60}")
        runs = []
        for i in range(args.iters):
            runner = ConcurrentRunner(args.M, args.K, args.N, args.gpu_M, args.gpu_K, args.gpu_N, args.npu_cols)
            r = runner.run_concurrent(args.warmup)
            runs.append(r)
            print_timing_result(r, f"Concurrent run {i+1}")
        results["concurrent"] = runs

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
                print(f"  GPU total: median={stats.median:.2f}ms mean={stats.mean:.2f}ms p10={stats.p10:.2f} p90={stats.p90:.2f}")
        if mode in ("npu_only", "concurrent", "gpu_then_npu", "npu_then_gpu"):
            npu_host = [r.npu_host_ms for r in runs if r.npu_host_ms > 0]
            if npu_host:
                stats = aggregate(npu_host)
                print(f"  NPU host: median={stats.median:.2f}ms mean={stats.mean:.2f}ms p10={stats.p10:.2f} p90={stats.p90:.2f}")
        if mode == "concurrent":
            speedups = [r.speedup for r in runs if r.speedup > 0]
            overlaps = [r.overlap_ns / 1e6 for r in runs if r.overlap_ns > 0]
            if speedups:
                stats = aggregate(speedups)
                print(f"  Speedup: median={stats.median:.3f}x mean={stats.mean:.3f}x p10={stats.p10:.3f} p90={stats.p90:.3f}")
            if overlaps:
                stats = aggregate(overlaps)
                print(f"  Overlap: median={stats.median:.2f}ms mean={stats.mean:.2f}ms")

    save_results(results, Path(args.output))

    # Decision summary
    if "gpu_only" in results and "concurrent" in results and results["gpu_only"] and results["concurrent"]:
        gpu_median = aggregate([r.gpu_total_ms for r in results["gpu_only"] if r.gpu_total_ms > 0]).median
        concurrent_median = aggregate([r.concurrent_time_ns / 1e6 for r in results["concurrent"] if r.concurrent_time_ns > 0]).median
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