#!/usr/bin/env python3
"""
Phase 7C Scheduling Variants Test (Case A/B/C/D)
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


SCRIPT_DIR = Path(__file__).parent
GPU_BENCH = SCRIPT_DIR.parent / "gpu_bench"
NPU_GEMM = Path("/workspace/llama.cpp/experiments/npu-int8-gemm/int8_gemm_nparallel.py")


def run_gpu(M: int, K: int, N: int) -> float:
    cmd = [str(GPU_BENCH), "--M", str(M), "--K", str(K), "--N", str(N), "--iters", "1", "--warmup", "0"]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/vulkan:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GPU_BENCH.parent), env=env, timeout=120)
    if proc.returncode == 0:
        json_start = proc.stdout.rfind("{")
        if json_start >= 0:
            data = json.loads(proc.stdout[json_start:])
            return data.get("total_ms", 0)
    return 0


def run_npu(M: int, K: int, N: int, cols: int) -> float:
    cmd = [sys.executable, str(NPU_GEMM), "--M", str(M), "--K", str(K), "--N", str(N), "--cols", str(cols), "--iters", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NPU_GEMM.parent), timeout=180)
    if proc.returncode == 0:
        for line in proc.stdout.split("\n"):
            if "Host submission latency" in line and "mean=" in line:
                return float(line.split("mean=")[1].split(" ")[0]) / 1000.0
    return 0


def run_concurrent(gpu_M, gpu_K, gpu_N, npu_M, npu_K, npu_N, npu_cols) -> tuple:
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


def test_case_a():
    """Case A: Naive offload - same operation on both GPU and NPU"""
    print("\n=== Case A: Naive offload (same op on both) ===")
    # Both do M=512 K=1024 N=3584
    gpu_solo = run_gpu(512, 1024, 3584)
    npu_solo = run_npu(512, 1024, 3584, 4)
    gpu_conc, npu_conc, wall = run_concurrent(512, 1024, 3584, 512, 1024, 3584, 4)
    seq = gpu_solo + npu_solo
    print(f"  GPU solo: {gpu_solo:.1f}ms, NPU solo: {npu_solo:.1f}ms")
    print(f"  Concurrent: GPU={gpu_conc:.1f}ms, NPU={npu_conc:.1f}ms, Wall={wall:.1f}ms")
    print(f"  Sequential: {seq:.1f}ms, Speedup: {seq/wall:.3f}x")
    return {"gpu_solo": gpu_solo, "npu_solo": npu_solo, "gpu_conc": gpu_conc, "npu_conc": npu_conc, "wall": wall, "seq": seq}


def test_case_b():
    """Case B: Asynchronous offload - GPU attention, NPU FFN"""
    print("\n=== Case B: Asynchronous offload (GPU Attention || NPU FFN) ===")
    gpu_solo = run_gpu(512, 1024, 6144)
    npu_solo = run_npu(512, 1024, 3584, 4)
    gpu_conc, npu_conc, wall = run_concurrent(512, 1024, 6144, 512, 1024, 3584, 4)
    seq = gpu_solo + npu_solo
    print(f"  GPU solo: {gpu_solo:.1f}ms, NPU solo: {npu_solo:.1f}ms")
    print(f"  Concurrent: GPU={gpu_conc:.1f}ms, NPU={npu_conc:.1f}ms, Wall={wall:.1f}ms")
    print(f"  Sequential: {seq:.1f}ms, Speedup: {seq/wall:.3f}x")
    return {"gpu_solo": gpu_solo, "npu_solo": npu_solo, "gpu_conc": gpu_conc, "npu_conc": npu_conc, "wall": wall, "seq": seq}


def test_case_c():
    """Case C: Serialized offload - GPU -> NPU -> GPU"""
    print("\n=== Case C: Serialized offload (GPU -> NPU -> GPU) ===")
    # Simulate: GPU does attention, then NPU does FFN, then GPU does output projection
    gpu1 = run_gpu(512, 1024, 6144)  # Attention QKV
    npu_t = run_npu(512, 1024, 3584, 4)  # FFN up/gate
    gpu2 = run_gpu(512, 1024, 256)  # FFN down / output
    seq = gpu1 + npu_t + gpu2
    print(f"  GPU stage 1: {gpu1:.1f}ms, NPU: {npu_t:.1f}ms, GPU stage 2: {gpu2:.1f}ms")
    print(f"  Total sequential: {seq:.1f}ms")
    return {"gpu1": gpu1, "npu": npu_t, "gpu2": gpu2, "seq": seq}


def test_case_d():
    """Case D: Overlapped pipeline - GPU stage i || NPU stage i -> GPU stage i+1"""
    print("\n=== Case D: Overlapped pipeline ===")
    # This would require multi-step pipeline, simplified:
    # Iteration 1: GPU attention || NPU FFN up
    # Iteration 2: GPU FFN down || NPU (next layer FFN up)
    # For now just measure the pipeline overhead
    print("  Pipeline simulation requires multi-iteration setup")
    print("  Skipping detailed measurement - would need persistent buffers")
    return {}


def main():
    results = {}
    
    results["case_a"] = test_case_a()
    results["case_b"] = test_case_b()
    results["case_c"] = test_case_c()
    results["case_d"] = test_case_d()
    
    output_path = SCRIPT_DIR.parent / "results" / "phase7c" / "phase7c_scheduling_variants.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nSaved scheduling variant results to {output_path}")
    
    print("\n=== SUMMARY ===")
    for case, data in results.items():
        if data and "wall" in data:
            speedup = data["seq"] / data["wall"] if data["wall"] > 0 else 0
            print(f"  {case}: sequential={data['seq']:.1f}ms, concurrent={data['wall']:.1f}ms, speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()