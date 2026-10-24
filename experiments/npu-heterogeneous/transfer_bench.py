#!/usr/bin/env python3
"""transfer_bench.py -- Measure pure host<->NPU transfer costs.

Measures:
  - H2D: host -> NPU input transfer time
  - D2H: NPU -> host output transfer time
  - E2E: host -> NPU -> host round-trip (without compute)

Usage:
    python3 transfer_bench.py --M 128 --K 512 --N 4096 --iters 20
"""
import argparse, json, sys, time, os, numpy as np

try:
    import aie.iron as iron
    IRON_AVAILABLE = True
except ImportError:
    IRON_AVAILABLE = False

def benchmark_transfer(M, K, N, iters=20, seed=42):
    if not IRON_AVAILABLE:
        return {"status": "NO_IRON", "error": "aie.iron not available"}

    rng = np.random.default_rng(seed)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")

    # Warmup transfers
    for _ in range(3):
        A._sync_to_device()
        _ = A.numpy()
        B._sync_to_device()
        _ = B.numpy()

    # Measure H2D for A
    h2d_times = []
    for i in range(iters):
        A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        A.data[:] = A_np
        t0 = time.perf_counter_ns()
        A._sync_to_device()
        t1 = time.perf_counter_ns()
        h2d_times.append((t1 - t0) / 1000.0)

    # Measure D2H for A
    d2h_times = []
    for i in range(iters):
        t0 = time.perf_counter_ns()
        _ = A.numpy()
        t1 = time.perf_counter_ns()
        d2h_times.append((t1 - t0) / 1000.0)

    # Measure round-trip H2D + D2H
    e2e_times = []
    for i in range(iters):
        A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        A.data[:] = A_np
        t0 = time.perf_counter_ns()
        A._sync_to_device()
        _ = A.numpy()
        t1 = time.perf_counter_ns()
        e2e_times.append((t1 - t0) / 1000.0)

    def _stats(samples):
        s = sorted(samples)
        n = len(s)
        return {
            "min": s[0], "mean": sum(samples) / len(samples),
            "median": s[n // 2],
            "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)], "max": s[-1],
        }

    bytes_A = M * K * 1  # int8
    bytes_B = K * N * 1
    bytes_total = bytes_A + bytes_B

    return {
        "status": "OK",
        "backend": "NPU_TRANSFER",
        "M": M, "K": K, "N": N,
        "bytes_A": bytes_A, "bytes_B": bytes_B, "bytes_total": bytes_total,
        "iters": iters,
        "h2d_A_us": _stats(h2d_times),
        "d2h_A_us": _stats(d2h_times),
        "e2e_A_us": _stats(e2e_times),
        "h2d_A_bw_gbps": (bytes_A / 1e9) / (_stats(h2d_times)["median"] / 1e6) if _stats(h2d_times)["median"] > 0 else 0,
        "d2h_A_bw_gbps": (bytes_A / 1e9) / (_stats(d2h_times)["median"] / 1e6) if _stats(d2h_times)["median"] > 0 else 0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = benchmark_transfer(args.M, args.K, args.N, args.iters, args.seed)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("status") == "OK" else 1)

if __name__ == "__main__":
    main()
