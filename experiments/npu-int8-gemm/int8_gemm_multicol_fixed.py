#!/usr/bin/env python3
# int8_gemm_multicol_fixed.py — Phoenix XDNA1 INT8 GEMM with multi-N-column fix
#
# This is the canonical fixed implementation after Phase 7B-R.
# Key fix: pattern_repeat=1 in A-tile TensorTiler2D to avoid A BD aliasing
# when M//m > 1 AND N//n > 1 AND K > 512.
#
# Run inside the npu-dev container with:
#   source /root/src/mlir-aie-venv/bin/activate
#   python int8_gemm_multicol_fixed.py --M 128 --K 1024 --N 128

import argparse
import sys
import time
import statistics
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.utils.verify import assert_pass

# Tile size for the single-core GEMM.
# Must be a multiple of the MAC micro-kernel dims (r, s, t).
# For aie2 + int8/int32: (4, 8, 8).
_TILE_M = 64
_TILE_K = 64
_TILE_N = 64

MAX_BD_BYTES = 65536  # aiecc iteration-wrap limit


def _build_design(M, K, N):
    """Build an @iron.jit matmul design for the given shape."""
    m, k, n = _TILE_M, _TILE_K, _TILE_N

    matmul_kernel = mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=np.int8,
        output_dtype=np.int32,
        vectorized=True,
    )
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    a_ty = np.ndarray[(m * k,), np.dtype[np.int8]]
    b_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
    c_ty = np.ndarray[(m * n,), np.dtype[np.int32]]

    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    # B L2L1: always col-major TAS to match core acquire order (N-outer, K-inner)
    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3"
    )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in iron.controlflow.range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            for i in iron.controlflow.range_(m * n):
                elem_out[i] = 0
            for _ in iron.controlflow.range_(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = iron.Worker(
        core_fn,
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel],
    )

    # A taps: one tap per M-tile row, each covering all K tiles for that row.
    # pattern_repeat MUST be 1. Using N//n here causes A BD aliasing when M//m > 1,
    # because the resulting TD has (N//n, 0) as the first dim with stride 0,
    # making multiple M-row groups map to the same physical offset.
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=1
    )

    # B L2L1 tap: col-major for correct core acquire order
    b_tap_l2l1 = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
    )[0]

    # B L3L2 taps: split by N tiles only when K*N > MAX_BD_BYTES
    # For K<=512 with n=64: K*N <= 32768 < 65536, single tap works
    # For K>512: split into per-N-column taps (each K*n bytes)
    b_bytes = K * N
    if b_bytes <= MAX_BD_BYTES:
        # Single tap for full B matrix (original behavior)
        b_taps_l3l2 = [TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
        )[0]]
    else:
        # Split by N tiles: each tap covers K x n (one N column)
        b_taps_l3l2 = []
        for nt in range(N // n):
            offset = nt * n
            b_tap_chunk = TensorAccessPattern(
                tensor_dims=(K, N),
                offset=offset,
                sizes=[K, n],
                strides=[N, 1],
            )
            b_taps_l3l2.append(b_tap_chunk)

    c_taps = []
    for tile_row in range(M // m):
        for tile_col in range(N // n):
            offset = tile_row * m * N + tile_col * n
            c_tap = TensorAccessPattern(
                tensor_dims=(M, N),
                offset=offset,
                sizes=[m, n],
                strides=[N, 1],
            )
            c_taps.append(c_tap)

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // m):
            for tile_col in range(N // n):
                tg = iron.TaskGroup()
                a_prod.fill(A, tap=a_taps[tile_row], group=tg)
                # B L3L2: use the appropriate tap for this output tile's N column
                b_tap_idx = tile_col if len(b_taps_l3l2) > 1 else 0
                b_prod.fill(B, tap=b_taps_l3l2[b_tap_idx], group=tg)
                c_cons.drain(C, tap=c_taps[tile_row * (N // n) + tile_col], group=tg, wait=True)
                tg.finish()

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


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


def main():
    global _TILE_M, _TILE_K, _TILE_N
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tile-m", type=int, default=_TILE_M)
    parser.add_argument("--tile-k", type=int, default=_TILE_K)
    parser.add_argument("--tile-n", type=int, default=_TILE_N)
    parser.add_argument("--work-dir", type=str, default=None)
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    _TILE_M, _TILE_K, _TILE_N = args.tile_m, args.tile_k, args.tile_n

    # Validate tile divisibility against MAC dims (4, 8, 8) on aie2 int8.
    r, s, t = 4, 8, 8
    if M % r != 0 or K % s != 0 or N % t != 0:
        sys.exit(
            f"Shape {M}x{K}x{N} not compatible with aie2 int8 MAC dims ({r},{s},{t})."
        )
    if M % _TILE_M != 0 or K % _TILE_K != 0 or N % _TILE_N != 0:
        sys.exit(
            f"Shape {M}x{K}x{N} not divisible by tile size ({_TILE_M},{_TILE_K},{_TILE_N})."
        )

    dev = iron.get_current_device()
    cols = dev.cols
    print(f"device: {type(dev).__name__}  columns: {cols}")

    print(f"Building INT8 GEMM: A={M}x{K}, B={K}x{N}, C={M}x{N}  tile=({_TILE_M},{_TILE_K},{_TILE_N})")

    # Build the JIT design
    @iron.jit
    def int8_gemm_design(
        A: In,
        B: In,
        C: Out,
        *,
        M: CompileTime[int] = M,
        K: CompileTime[int] = K,
        N: CompileTime[int] = N,
    ):
        return _build_design(M, K, N)

    # Compile the design
    import os
    compile_kwargs = {}
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        compile_kwargs["xclbin_path"] = os.path.join(args.work_dir, "design.xclbin")
        compile_kwargs["inst_path"] = os.path.join(args.work_dir, "design.insts.bin")
    compiled = int8_gemm_design.specialize(M=M, K=K, N=N).compile(**compile_kwargs)
    print(f"  xclbin: {compiled[0]}")
    print(f"  insts:  {compiled[1]}")

    # Random int8 inputs in range [-128, 127]
    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    # Warmup
    int8_gemm_design(A, B, C, M=M, K=K, N=N)

    # CPU reference
    ref = cpu_int8_gemm(A_np, B_np)

    # Benchmark
    host_times = []
    npu_times = []
    for i in range(args.iters):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        start = time.perf_counter()
        ret = int8_gemm_design(A, B, C, M=M, K=K, N=N)
        e2e_us = (time.perf_counter() - start) * 1e6
        host_times.append(e2e_us)
        npu_ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", None)
        if npu_ns is not None:
            npu_times.append(npu_ns / 1000.0)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.iters} iterations done")

    host_stats = _stats(host_times)
    npu_stats = _stats(npu_times) if npu_times else None

    total_ops = 2.0 * M * K * N  # multiply-adds
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