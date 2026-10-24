#!/usr/bin/env python3
# int8_gemm_fixed.py — Fixed version with chunked B L3L2 for K>512

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

_TILE_M = 64
_TILE_K = 64
_TILE_N = 64

MAX_BD_BYTES = 65536  # aiecc iteration-wrap limit


def _build_design(M, K, N):
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

    # B: always use col-major TAS for L2L1 to match core acquire order
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

    # A taps: unchanged
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )

    # B taps for L2L1: always col-major to match core acquire order (N-outer, K-inner)
    b_tap_l2l1 = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
    )[0]

    # B taps for L3L2: split into chunks to keep each BD under MAX_BD_BYTES
    # Each chunk covers a subset of K tiles, all N tiles
    b_taps_l3l2 = []
    bytes_per_k_tile = k * N  # int8 bytes for one K tile (all N columns)
    k_tiles_per_chunk = max(1, MAX_BD_BYTES // bytes_per_k_tile)
    # But we also need k_tiles_per_chunk to divide K//k evenly
    K_tiles = K // k
    # Find largest divisor of K_tiles that keeps chunk under limit
    for divisor in range(K_tiles, 0, -1):
        if K_tiles % divisor == 0:
            chunk_k_tiles = divisor
            chunk_bytes = chunk_k_tiles * bytes_per_k_tile
            if chunk_bytes <= MAX_BD_BYTES:
                k_tiles_per_chunk = chunk_k_tiles
                break
    
    num_chunks = K_tiles // k_tiles_per_chunk
    print(f"  B L3L2 chunking: K_tiles={K_tiles}, k_tiles_per_chunk={k_tiles_per_chunk}, num_chunks={num_chunks}")
    
    for chunk_idx in range(num_chunks):
        k_start = chunk_idx * k_tiles_per_chunk * k
        k_chunk_size = k_tiles_per_chunk * k
        offset = k_start * N
        b_tap_chunk = TensorAccessPattern(
            tensor_dims=(K, N),
            offset=offset,
            sizes=[k_chunk_size, N],
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
                # B: issue multiple fills, one per L3L2 chunk
                for b_tap_chunk in b_taps_l3l2:
                    b_prod.fill(B, tap=b_tap_chunk, group=tg)
                c_cons.drain(C, tap=c_taps[tile_row * (N // n) + tile_col], group=tg, wait=True)
                tg.finish()

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def cpu_int8_gemm(A, B):
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

    r, s, t = 4, 8, 8
    if M % r != 0 or K % s != 0 or N % t != 0:
        sys.exit(f"Shape {M}x{K}x{N} not compatible with aie2 int8 MAC dims ({r},{s},{t}).")
    if M % _TILE_M != 0 or K % _TILE_K != 0 or N % _TILE_N != 0:
        sys.exit(f"Shape {M}x{K}x{N} not divisible by tile size ({_TILE_M},{_TILE_K},{_TILE_N}).")

    dev = iron.get_current_device()
    cols = dev.cols
    print(f"device: {type(dev).__name__}  columns: {cols}")

    print(f"Building INT8 GEMM: A={M}x{K}, B={K}x{N}, C={M}x{N}  tile=({_TILE_M},{_TILE_K},{_TILE_N})")

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

    import os
    compile_kwargs = {}
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        compile_kwargs["xclbin_path"] = os.path.join(args.work_dir, "design.xclbin")
        compile_kwargs["inst_path"] = os.path.join(args.work_dir, "design.insts.bin")
    compiled = int8_gemm_design.specialize(M=M, K=K, N=N).compile(**compile_kwargs)
    print(f"  xclbin: {compiled[0]}")
    print(f"  insts:  {compiled[1]}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    int8_gemm_design(A, B, C, M=M, K=K, N=N)

    ref = cpu_int8_gemm(A_np, B_np)

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