#!/usr/bin/env python3
# int8_gemm_v3.py — Phoenix XDNA1 INT8 GEMM workload (v3)
#
# C = A x B,  where A,B are int8 and C is int32.
# Supports arbitrary M, K, N shapes with proper tiling.
# Uses mlir-aie kernels.mm() with the aie2 int8->int32 vectorized path.
#
# Run inside the npu-dev:mlir container with:
#   source /root/src/mlir-aie-venv/bin/activate
#   python int8_gemm_v3.py --M 1024 --K 1024 --N 1024

import argparse
import sys
import time
import statistics
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, TaskGroup
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.iron.controlflow import range_

# Default tile size for the GEMM.
# Must be compatible with aie2 int8 MAC dims (4, 8, 8).
_TILE_M = 64
_TILE_K = 64
_TILE_N = 64


def _build_design(M, K, N, m, k, n, vectorized=True):
    """Build an @iron.jit matmul design for the given shape and tile."""
    matmul_kernel = mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=np.int8,
        output_dtype=np.int32,
        vectorized=vectorized,
    )
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    a_ty = np.ndarray[(m * k,), np.dtype[np.int8]]
    b_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
    c_ty = np.ndarray[(m * n,), np.dtype[np.int32]]

    # Use group_tiler without pattern_repeat for correct tile access patterns
    a_tiler = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k)
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n)
    )[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))

    fifo_A = iron.ObjectFifo(a_ty, name="A")
    fifo_B = iron.ObjectFifo(b_ty, name="B")
    fifo_C = iron.ObjectFifo(c_ty, name="C")

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            for i in range_(m * n):
                elem_out[i] = 0
            for _ in range_(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = iron.Worker(
        core_fn,
        [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), matmul_kernel],
    )

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // m):
            tg = TaskGroup()
            a_prod.fill(A, tap=a_tiler[tile_row], group=tg)
            b_prod.fill(B, tap=b_tap, group=tg)
            c_cons.drain(C, tap=c_taps[tile_row], group=tg, wait=True)
            tg.finish()

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A.prod(), fifo_B.prod(), fifo_C.cons()],
    )
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit
def _build_design_jit(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 512,
    K: CompileTime[int] = 512,
    N: CompileTime[int] = 512,
    m: CompileTime[int] = _TILE_M,
    k: CompileTime[int] = _TILE_K,
    n: CompileTime[int] = _TILE_N,
    vectorized: CompileTime[bool] = True,
):
    return _build_design(M, K, N, m, k, n, vectorized=vectorized)


def cpu_int8_gemm(A, B):
    """CPU reference: int8 multiply, int32 accumulation."""
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    return A_i32 @ B_i32


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0],
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
    }


def _validate_shape(M, K, N, m, k, n, vectorized):
    """Validate shape compatibility with aie2 int8 MAC dims (4, 8, 8)."""
    if vectorized:
        r, s, t = 4, 8, 8
        if M % r != 0 or K % s != 0 or N % t != 0:
            sys.exit(
                f"Shape {M}x{K}x{N} not compatible with aie2 int8 MAC dims ({r},{s},{t})."
            )
    if M % m != 0 or K % k != 0 or N % n != 0:
        sys.exit(
            f"Shape {M}x{K}x{N} not divisible by tile size ({m},{k},{n})."
        )
    # Limit tile size to avoid DMA BD issues with very large tiles
    if m * n > 16383:
        sys.exit(f"Tile output size {m}x{n}={m*n} exceeds 16383 element limit.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=_TILE_M)
    parser.add_argument("--tile-k", type=int, default=_TILE_K)
    parser.add_argument("--tile-n", type=int, default=_TILE_N)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--scalar", action="store_true", help="Use scalar kernel (for M=1)")
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    m, k, n = args.tile_m, args.tile_k, args.tile_n
    vectorized = not args.scalar

    _validate_shape(M, K, N, m, k, n, vectorized)

    dev = iron.get_current_device()
    print(f"device: {type(dev).__name__}  columns: {dev.cols}")
    print(f"Building INT8 GEMM: A={M}x{K}, B={K}x{N}, C={M}x{N}  tile=({m},{k},{n})  vectorized={vectorized}")

    # Random int8 inputs in range [-128, 127]
    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    # Warmup
    _build_design_jit(A, B, C, M=M, K=K, N=N, m=m, k=k, n=n, vectorized=vectorized)

    # CPU reference
    ref = cpu_int8_gemm(A_np, B_np)

    # Benchmark
    host_times = []
    npu_times = []
    for i in range(args.iters):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        start = time.perf_counter()
        ret = _build_design_jit(A, B, C, M=M, K=K, N=N, m=m, k=k, n=n, vectorized=vectorized)
        e2e_us = (time.perf_counter() - start) * 1e6
        host_times.append(e2e_us)
        npu_ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", None)
        if npu_ns is not None:
            npu_times.append(npu_ns / 1000.0)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.iters} done")

    host_stats = _stats(host_times)
    npu_stats = _stats(npu_times) if npu_times else None

    total_ops = 2.0 * M * K * N
    e2e_gops = total_ops / (host_stats["mean"] * 1e-6) / 1e9
    npu_gops = total_ops / (npu_stats["mean"] * 1e-6) / 1e9 if npu_stats else None

    print(f"\n=== INT8 GEMM {M}x{K}x{N} ===")
    print(f"Total INT8 MAC operations: {total_ops:.0f}")
    print(f"\nHost submission latency (us):")
    for k, v in host_stats.items():
        print(f"  {k}={v:.1f}")
    if npu_stats:
        print(f"\nNPU execution latency (us):")
        for k, v in npu_stats.items():
            print(f"  {k}={v:.1f}")
    print(f"\nEffective GOPS (host e2e): {e2e_gops:.2f}")
    if npu_gops:
        print(f"Effective GOPS (NPU exec): {npu_gops:.2f}")

    # Correctness check
    got = C.numpy()
    mismatches = int(np.sum(got != ref))
    print(f"\nCorrectness check: {mismatches} mismatches out of {M*N}")
    if mismatches == 0:
        print("PASS - output matches CPU int32 reference")
    else:
        print("FAIL - output does not match CPU int32 reference")
        print("got[:8] =", got[:8].flatten())
        print("ref[:8] =", ref[:8].flatten())
        sys.exit(1)


if __name__ == "__main__":
    main()
