#!/usr/bin/env python3
"""cpu_bench.py -- CPU baseline for MUL_MAT / GEMM.

Uses numpy BLAS (OpenBLAS/MKL) as CPU reference.
Outputs JSON with detailed timing breakdown.

Usage:
    python3 cpu_bench.py --M 2 --K 1024 --N 16 --iters 20 --warmup 5
    python3 cpu_bench.py --M 128 --K 4096 --N 4096 --iters 10
"""
import argparse, json, sys, time, os
import numpy as np

def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0], "mean": sum(samples) / len(samples),
        "median": s[n // 2],
        "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)], "max": s[-1],
    }

def benchmark(M, K, N, iters=20, warmup=5, dtype=np.float32, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.random((M, K)).astype(dtype)
    B = rng.random((K, N)).astype(dtype)

    # Warmup
    for _ in range(warmup):
        C = A @ B

    # Timed runs
    compute_times = []
    total_times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        C = A @ B
        t1 = time.perf_counter_ns()
        compute_times.append((t1 - t0) / 1000.0)  # us
        total_times.append((t1 - t0) / 1000.0)

    # Ensure result is computed (numpy is lazy sometimes)
    _ = C.copy()

    compute_s = _stats(compute_times)
    total_s = _stats(total_times)

    ops = 2.0 * M * K * N
    compute_ms_median = compute_s["median"] / 1000.0
    gops = ops / (compute_ms_median * 1e-3) / 1e9 if compute_ms_median > 0 else 0

    return {
        "status": "OK",
        "backend": "CPU",
        "op": "GEMM",
        "M": M, "K": K, "N": N,
        "dtype": str(dtype),
        "iters": iters,
        "warmup": warmup,
        "compute_us": compute_s,
        "total_us": total_s,
        "compute_ms_median": compute_ms_median,
        "total_ms_median": total_s["median"] / 1000.0,
        "gops": gops,
        "total_ops": ops,
        "gmac": ops / 1e9,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
    }
    dtype = dtype_map.get(args.dtype, np.float32)

    result = benchmark(args.M, args.K, args.N, args.iters, args.warmup, dtype, args.seed)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "OK" else 1)

if __name__ == "__main__":
    main()
