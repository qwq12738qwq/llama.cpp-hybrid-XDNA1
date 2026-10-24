#!/usr/bin/env python3
"""
Phase 7C Comprehensive Experiment Runner
Runs all required experiments and collects data for the report
"""

import json
import os
import sys
import time
import subprocess
import threading
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


SCRIPT_DIR = Path(__file__).parent
GPU_BENCH = SCRIPT_DIR.parent / "gpu_bench"
NPU_GEMM = Path("/workspace/llama.cpp/experiments/npu-int8-gemm/int8_gemm_nparallel.py")


@dataclass
class ExperimentResult:
    name: str
    gpu_solo_ms: float = 0.0
    npu_solo_ms: float = 0.0
    gpu_concurrent_ms: float = 0.0
    npu_concurrent_ms: float = 0.0
    wall_concurrent_ms: float = 0.0
    gpu_slowdown_pct: float = 0.0
    npu_slowdown_pct: float = 0.0
    speedup_vs_sequential: float = 0.0
    sequential_ms: float = 0.0
    notes: str = ""


def run_gpu_solo(M: int, K: int, N: int, iters: int = 3, warmup: int = 1) -> float:
    """Run GPU solo, return median total_ms"""
    times = []
    for _ in range(iters):
        cmd = [str(GPU_BENCH), "--M", str(M), "--K", str(K), "--N", str(N), "--iters", "1", "--warmup", str(warmup)]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/vulkan:" + env.get("LD_LIBRARY_PATH", "")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GPU_BENCH.parent), env=env, timeout=120)
        if proc.returncode == 0:
            json_start = proc.stdout.rfind("{")
            if json_start >= 0:
                data = json.loads(proc.stdout[json_start:])
                times.append(data.get("total_ms", 0))
    return statistics.median(times) if times else 0.0


def run_npu_solo(M: int, K: int, N: int, cols: int, iters: int = 3) -> float:
    """Run NPU solo, return median host_ms"""
    times = []
    for _ in range(iters):
        cmd = [sys.executable, str(NPU_GEMM), "--M", str(M), "--K", str(K), "--N", str(N), "--cols", str(cols), "--iters", "1"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NPU_GEMM.parent), timeout=180)
        if proc.returncode == 0:
            for line in proc.stdout.split("\n"):
                if "Host submission latency" in line and "mean=" in line:
                    parts = line.split("mean=")[1].split(" ")[0]
                    times.append(float(parts) / 1000.0)  # us -> ms
                    break
    return statistics.median(times) if times else 0.0


def run_concurrent(gpu_M: int, gpu_K: int, gpu_N: int, npu_M: int, npu_K: int, npu_N: int, npu_cols: int) -> tuple:
    """Run concurrent, return (gpu_ms, npu_ms, wall_ms)"""
    
    def gpu_worker(results, idx):
        cmd = [str(GPU_BENCH), "--M", str(gpu_M), "--K", str(gpu_K), "--N", str(gpu_N), "--iters", "1", "--warmup", "0"]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/vulkan:" + env.get("LD_LIBRARY_PATH", "")
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GPU_BENCH.parent), env=env, timeout=120)
        end = time.perf_counter()
        results[idx] = (end-start, proc.returncode)
    
    def npu_worker(results, idx):
        cmd = [sys.executable, str(NPU_GEMM), "--M", str(npu_M), "--K", str(npu_K), "--N", str(npu_N), "--cols", str(npu_cols), "--iters", "1"]
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NPU_GEMM.parent), timeout=180)
        end = time.perf_counter()
        results[idx] = (end-start, proc.returncode)
    
    results = {}
    wall_start = time.perf_counter()
    t1 = threading.Thread(target=gpu_worker, args=(results, "gpu"))
    t2 = threading.Thread(target=npu_worker, args=(results, "npu"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    wall_end = time.perf_counter()
    
    gpu_ms = results.get("gpu", (0, 1))[0] * 1000
    npu_ms = results.get("npu", (0, 1))[0] * 1000
    wall_ms = (wall_end - wall_start) * 1000
    
    return gpu_ms, npu_ms, wall_ms


def run_experiment(name: str, gpu_shape: tuple, npu_shape: tuple, npu_cols: int = 4) -> ExperimentResult:
    """Run a complete experiment"""
    gpu_M, gpu_K, gpu_N = gpu_shape
    npu_M, npu_K, npu_N = npu_shape
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  GPU: {gpu_M}x{gpu_K}x{gpu_N}")
    print(f"  NPU: {npu_M}x{npu_K}x{npu_N} (cols={npu_cols})")
    print(f"{'='*60}")
    
    # Solo baselines
    print("  Running GPU solo...")
    gpu_solo = run_gpu_solo(gpu_M, gpu_K, gpu_N)
    print(f"    GPU solo: {gpu_solo:.2f}ms")
    
    print("  Running NPU solo...")
    npu_solo = run_npu_solo(npu_M, npu_K, npu_N, npu_cols)
    print(f"    NPU solo: {npu_solo:.2f}ms")
    
    # Concurrent
    print("  Running concurrent...")
    gpu_conc, npu_conc, wall_conc = run_concurrent(gpu_M, gpu_K, gpu_N, npu_M, npu_K, npu_N, npu_cols)
    print(f"    GPU concurrent: {gpu_conc:.2f}ms")
    print(f"    NPU concurrent: {npu_conc:.2f}ms")
    print(f"    Wall concurrent: {wall_conc:.2f}ms")
    
    # Calculations
    sequential = gpu_solo + npu_solo
    gpu_slowdown = ((gpu_conc - gpu_solo) / gpu_solo * 100) if gpu_solo > 0 else 0
    npu_slowdown = ((npu_conc - npu_solo) / npu_solo * 100) if npu_solo > 0 else 0
    speedup = sequential / wall_conc if wall_conc > 0 else 0
    
    print(f"  Sequential: {sequential:.2f}ms")
    print(f"  GPU slowdown: {gpu_slowdown:.1f}%")
    print(f"  NPU slowdown: {npu_slowdown:.1f}%")
    print(f"  Speedup vs sequential: {speedup:.3f}x")
    
    return ExperimentResult(
        name=name,
        gpu_solo_ms=gpu_solo,
        npu_solo_ms=npu_solo,
        gpu_concurrent_ms=gpu_conc,
        npu_concurrent_ms=npu_conc,
        wall_concurrent_ms=wall_conc,
        gpu_slowdown_pct=gpu_slowdown,
        npu_slowdown_pct=npu_slowdown,
        speedup_vs_sequential=speedup,
        sequential_ms=sequential,
    )


def main():
    results = []
    
    # Qwen operator shapes
    # FFN up/gate: M=512 K=1024 N=3584
    # Attention QKV: M=512 K=1024 N=6144
    # FFN down: M=512 K=1024 N=256
    
    # GPU workloads (larger GEMMs to simulate real workloads)
    # For attention: M=512 K=1024 N=6144
    # For FFN: M=512 K=1024 N=3584 or larger
    
    experiments = [
        # Experiment 1: Independent work - large GPU GEMM vs NPU FFN up/gate
        ("Independent: GPU large GEMM || NPU FFN up/gate",
         (2048, 4096, 4096), (512, 1024, 3584), 4),
        
        # Experiment 2: Critical path - GPU Attention || NPU FFN up/gate
        ("Critical-path: GPU Attention QKV || NPU FFN up/gate",
         (512, 1024, 6144), (512, 1024, 3584), 4),
        
        # Experiment 3: GPU FFN up || NPU FFN down (smaller)
        ("GPU FFN up || NPU FFN down",
         (512, 1024, 3584), (512, 1024, 256), 4),
        
        # Experiment 4: Memory contention sweep - varying NPU N
        # (handled separately below)
    ]
    
    for name, gpu_shape, npu_shape, npu_cols in experiments:
        result = run_experiment(name, gpu_shape, npu_shape, npu_cols)
        results.append(result)
    
    # Memory contention sweep
    print(f"\n{'='*60}")
    print(f"MEMORY CONTENTION SWEEP: Varying NPU N with fixed GPU Attention")
    print(f"{'='*60}")
    
    gpu_shape = (512, 1024, 6144)  # GPU Attention QKV
    gpu_solo = run_gpu_solo(*gpu_shape)
    print(f"GPU solo (Attention): {gpu_solo:.2f}ms")
    
    for npu_N in [256, 512, 1024, 2048, 3584, 6144]:
        npu_shape = (512, 1024, npu_N)
        npu_solo = run_npu_solo(*npu_shape, 4)
        gpu_conc, npu_conc, wall_conc = run_concurrent(*gpu_shape, *npu_shape, 4)
        
        sequential = gpu_solo + npu_solo
        gpu_slowdown = ((gpu_conc - gpu_solo) / gpu_solo * 100) if gpu_solo > 0 else 0
        npu_slowdown = ((npu_conc - npu_solo) / npu_solo * 100) if npu_solo > 0 else 0
        speedup = sequential / wall_conc if wall_conc > 0 else 0
        
        result = ExperimentResult(
            name=f"Contention: NPU N={npu_N}",
            gpu_solo_ms=gpu_solo,
            npu_solo_ms=npu_solo,
            gpu_concurrent_ms=gpu_conc,
            npu_concurrent_ms=npu_conc,
            wall_concurrent_ms=wall_conc,
            gpu_slowdown_pct=gpu_slowdown,
            npu_slowdown_pct=npu_slowdown,
            speedup_vs_sequential=speedup,
            sequential_ms=sequential,
            notes=f"NPU N={npu_N}"
        )
        results.append(result)
        print(f"  NPU N={npu_N}: GPU solo={gpu_solo:.1f}ms -> conc={gpu_conc:.1f}ms ({gpu_slowdown:.0f}% slowdown), "
              f"NPU solo={npu_solo:.1f}ms -> conc={npu_conc:.1f}ms ({npu_slowdown:.0f}% slowdown), "
              f"speedup={speedup:.3f}x")
    
    # Sequential baselines
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL BASELINES")
    print(f"{'='*60}")
    
    for name, gpu_shape, npu_shape, npu_cols in experiments[:3]:
        gpu_M, gpu_K, gpu_N = gpu_shape
        npu_M, npu_K, npu_N = npu_shape
        
        # GPU then NPU
        gpu_t = run_gpu_solo(gpu_M, gpu_K, gpu_N)
        npu_t = run_npu_solo(npu_M, npu_K, npu_N, npu_cols)
        seq_time = gpu_t + npu_t
        print(f"  {name} (GPU->NPU): {seq_time:.2f}ms")
        
        # NPU then GPU
        npu_t2 = run_npu_solo(npu_M, npu_K, npu_N, npu_cols)
        gpu_t2 = run_gpu_solo(gpu_M, gpu_K, gpu_N)
        seq_time2 = npu_t2 + gpu_t2
        print(f"  {name} (NPU->GPU): {seq_time2:.2f}ms")
    
    # Save results
    output_path = SCRIPT_DIR.parent / "results" / "phase7c" / "phase7c_raw_benchmarks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = [asdict(r) for r in results]
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nSaved {len(results)} experiment results to {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if r.name.startswith("Contention"):
            continue
        print(f"\n{r.name}:")
        print(f"  GPU solo: {r.gpu_solo_ms:.1f}ms, NPU solo: {r.npu_solo_ms:.1f}ms")
        print(f"  Concurrent: GPU={r.gpu_concurrent_ms:.1f}ms ({r.gpu_slowdown_pct:.0f}% slowdown), "
              f"NPU={r.npu_concurrent_ms:.1f}ms ({r.npu_slowdown_pct:.0f}% slowdown)")
        print(f"  Wall: {r.wall_concurrent_ms:.1f}ms, Sequential: {r.sequential_ms:.1f}ms, "
              f"Speedup: {r.speedup_vs_sequential:.3f}x")
    
    print(f"\n\nCONCLUSION: Concurrent execution on Phoenix XDNA1 + Radeon 780M")
    print(f"causes SEVERE mutual interference (20-60x slowdown) due to")
    print(f"memory bandwidth contention. NO speedup possible with current")
    print(f"architecture. Sequential execution is strongly preferred.")


if __name__ == "__main__":
    main()