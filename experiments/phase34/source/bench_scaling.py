#!/usr/bin/env python3
"""Phase 34: multi-tile GEMM scaling benchmark using Phase21-based design.

Runs the compiled multi-tile GEMM for a given shape + tile count,
measures host e2e latency and NPU execution latency, and writes
CSV + JSON results.

Usage
-----
  source /Storage/Cache/ironenv-v143/bin/activate
  python bench_scaling.py --M 512 --K 2048 --N 2048 --tiles 4 --work-dir /tmp/p34
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.device import from_name
from gen_phase21 import build_design


_TILE_COUNT_TO_ROWS_COLS = {
    1:  (1, 1),
    2:  (1, 2),
    4:  (1, 4),
    8:  (2, 4),
    16: (4, 4),
}


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0],
        "mean": sum(samples) / n,
        "median": s[n // 2],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--tiles", type=int, required=True, choices=[1, 2, 4, 8, 16])
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cores = args.tiles
    n_rows, n_cols = _TILE_COUNT_TO_ROWS_COLS[n_cores]

    iron.set_current_device(from_name("npu", n_cols=4))

    @iron.jit
    def design(A: In, B: In, C: Out, *,
               M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
               mt: CompileTime[int] = 64, kt: CompileTime[int] = 64, nt: CompileTime[int] = 64,
               rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
               din: CompileTime[str] = "i8", dout: CompileTime[str] = "i32", act: CompileTime[str] = "none"):
        return build_design(M, K, N, 64, 64, 64, rows, cols, np.int8, np.int32, "none")

    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    if not os.path.exists(xp):
        print(f"ERROR: xclbin not found at {xp}. Run gen_phase21.py first.")
        sys.exit(1)

    # Load compiled design (reuses cached xclbin/insts)
    compiled = design.specialize(
        M=M, K=K, N=N, mt=64, kt=64, nt=64,
        rows=n_rows, cols=n_cols, din="i8", dout="i32", act="none",
    ).compile(xclbin_path=xp, inst_path=ip)
    print(f"Loaded: {compiled[0]}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    # Warmup
    for _ in range(args.warmup):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        design(A, B, C, M=M, K=K, N=N)

    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)

    host_times = []
    npu_times = []
    mismatches_list = []

    for i in range(args.iters):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        t0 = time.perf_counter()
        ret = design(A, B, C, M=M, K=K, N=N)
        e2e_us = (time.perf_counter() - t0) * 1e6
        host_times.append(e2e_us)

        npu_ns = None
        if isinstance(ret, tuple):
            for item in ret:
                if hasattr(item, "npu_time"):
                    npu_ns = item.npu_time
                    break
        elif hasattr(ret, "npu_time"):
            npu_ns = ret.npu_time

        if npu_ns is not None:
            npu_times.append(npu_ns / 1000.0)

        got = C.numpy()
        mismatches = int(np.sum(got != ref))
        mismatches_list.append(mismatches)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.iters}  host={e2e_us:.0f}us  mismatches={mismatches}")

    host_stats = _stats(host_times)
    npu_stats = _stats(npu_times) if npu_times else None
    max_mismatches = max(mismatches_list)

    total_ops = 2.0 * M * K * N
    e2e_gops = total_ops / (host_stats["mean"] * 1e-6) / 1e9
    npu_gops = total_ops / (npu_stats["mean"] * 1e-6) / 1e9 if npu_stats else None

    result = {
        "shape": {"M": M, "K": K, "N": N},
        "tiles": n_cores,
        "rows": n_rows,
        "cols": n_cols,
        "total_ops": total_ops,
        "host_e2e_us": host_stats,
        "npu_exec_us": npu_stats,
        "e2e_gops": e2e_gops,
        "npu_gops": npu_gops,
        "max_mismatches": max_mismatches,
        "correct": max_mismatches == 0,
        "iters": args.iters,
    }

    print(f"\n=== Phase34 {M}x{K}x{N} tiles={n_cores} ({n_rows}x{n_cols}) ===")
    print(f"Total ops: {total_ops:.0f}")
    print(f"Host e2e: mean={host_stats['mean']:.0f}us  median={host_stats['median']:.0f}us")
    if npu_stats:
        print(f"NPU exec: mean={npu_stats['mean']:.0f}us  median={npu_stats['median']:.0f}us")
    print(f"E2E GOPS: {e2e_gops:.2f}")
    if npu_gops:
        print(f"NPU GOPS: {npu_gops:.2f}")
    print(f"Correctness: {max_mismatches} mismatches ({'PASS' if max_mismatches == 0 else 'FAIL'})")

    raw_path = os.path.join(args.work_dir, "raw", f"bench_{n_cores}.json")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results: {raw_path}")

    sys.exit(0 if max_mismatches == 0 else 1)


if __name__ == "__main__":
    main()
