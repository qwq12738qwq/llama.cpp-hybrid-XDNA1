#!/usr/bin/env python3
# int8_gemm_multicore.py — Phoenix XDNA1 INT8 GEMM workload (multi-core)
#
# Uses all 4 AIE columns on XDNA1 for parallel GEMM computation.
# C = A x B, where A,B are int8 and C is int32.
#
# Run inside the npu-dev:mlir container with:
#   source /root/src/mlir-aie-venv/bin/activate
#   python int8_gemm_multicore.py --M 1024 --K 1024 --N 1024

import sys
import time
import statistics
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, TaskGroup, Worker
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.iron.controlflow import range_

_TILE_M = 64
_TILE_K = 64
_TILE_N = 64


def _build_design(M, K, N, n_aie_cols=4):
    m, k, n = _TILE_M, _TILE_K, _TILE_N
    n_aie_rows = 4

    matmul_kernel = mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=np.int8, output_dtype=np.int32, vectorized=True,
    )
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    A_l2_ty = np.ndarray[(m * k * n_aie_rows,), np.dtype[np.int8]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[np.int32]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[np.int8]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[np.int8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[np.int32]]

    A_l3l2 = [None] * n_aie_cols
    A_l2l1_fifos = [None] * n_aie_cols
    B_l3l2 = [None] * n_aie_cols
    B_l2l1 = [None] * n_aie_cols
    C_l2l3 = [None] * n_aie_cols
    C_l1l2_fifos = [None] * n_aie_cols

    for col in range(n_aie_cols):
        A_l3l2[col] = iron.ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=2)
        A_l2l1_fifos[col] = (
            A_l3l2[col]
            .cons()
            .split(
                [m * k * j for j in range(n_aie_rows)],
                obj_types=[A_l1_ty] * n_aie_rows,
                names=[f"A_L2L1_{col}_{row}" for row in range(n_aie_rows)],
                dims_to_stream=[
                    [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
                    for _ in range(n_aie_rows)
                ],
            )
        )

        B_l3l2[col] = iron.ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
        B_l2l1[col] = (
            B_l3l2[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=[(k // s, s * n), (n // t, t), (s, n), (t, 1)],
            )
        )

        C_l2l3[col] = iron.ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=2,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        C_l1l2_fifos[col] = (
            C_l2l3[col]
            .prod()
            .join(
                [iron.ObjectFifo(C_l1_ty, name=f"C_L1L2_{col}_{row}") for row in range(n_aie_rows)],
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            )
        )

    def core_fn(of_a_rows, of_b, of_c_rows, matmul):
        for row in range(n_aie_rows):
            def row_core(row_idx, of_a_row, of_c_row):
                for _ in range_(K // k):
                    elem_out = of_c_row.acquire(1)
                    for i in range_(m * n):
                        elem_out[i] = 0
                    elem_in_a = of_a_row.acquire(1)
                    elem_in_b = of_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out)
                    of_a_row.release(1)
                    of_b.release(1)
                    of_c_row.release(1)
            row_core(row, of_a_rows[row], of_c_rows[row])

    workers = []
    for col in range(n_aie_cols):
        for row in range(n_aie_rows):
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[col][row].cons(),
                        B_l2l1[col].cons(),
                        C_l1l2_fifos[col][row].prod(),
                        matmul_kernel,
                    ],
                    stack_size=0xD00,
                )
            )

    def sequence(A, B, C, a_prods, b_prods, c_cons):
        for col in range(n_aie_cols):
            tg = TaskGroup()
            a_prods[col].fill(A, group=tg)
            b_prods[col].fill(B, group=tg)
            c_cons[col].drain(C, group=tg, wait=True)
            tg.finish()

    A_prods = [f.prod() for f in A_l3l2]
    B_prods = [f.prod() for f in B_l3l2]
    C_conses = [f.cons() for f in C_l2l3]

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses],
    )
    return iron.Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def cpu_int8_gemm(A, B):
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    return A_i32 @ B_i32


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0], "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)], "max": s[-1],
    }


def main():
    M, K, N = 1024, 1024, 1024
    iters = 100
    n_aie_cols = 4

    r, s, t = 4, 8, 8
    if M % r != 0 or K % s != 0 or N % t != 0:
        sys.exit(f"Shape {M}x{K}x{N} not compatible with aie2 int8 MAC dims ({r},{s},{t}).")
    if M % _TILE_M != 0 or K % _TILE_K != 0 or N % _TILE_N != 0:
        sys.exit(f"Shape {M}x{K}x{N} not divisible by tile size ({_TILE_M},{_TILE_K},{_TILE_N}).")
    if N % (_TILE_N * n_aie_cols) != 0:
        sys.exit(f"N ({N}) must be divisible by tile_n ({_TILE_N}) * n_aie_cols ({n_aie_cols}).")

    dev = iron.get_current_device()
    print(f"device: {type(dev).__name__}  columns: {dev.cols}")

    print(f"Building INT8 GEMM (multi-core): A={M}x{K}, B={K}x{N}, C={M}x{N}  cols={n_aie_cols} tile=({_TILE_M},{_TILE_K},{_TILE_N})")
    compiled = _build_design(M, K, N, n_aie_cols).specialize(
        M=M, K=K, N=N, n_aie_cols=n_aie_cols
    ).compile()
    print(f"  xclbin: {compiled[0]}")
    print(f"  insts:  {compiled[1]}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    int8_gemm_design = _build_design(M, K, N, n_aie_cols)
    int8_gemm_design(A, B, C)  # warmup

    ref = cpu_int8_gemm(A_np, B_np)

    host_times, npu_times = [], []
    for i in range(iters):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        start = time.perf_counter()
        ret = int8_gemm_design(A, B, C)
        host_times.append((time.perf_counter() - start) * 1e6)
        npu_ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", None)
        if npu_ns is not None:
            npu_times.append(npu_ns / 1000.0)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{iters} done")

    host_stats = _stats(host_times)
    npu_stats = _stats(npu_times) if npu_times else None

    total_ops = 2.0 * M * K * N
    e2e_gops = total_ops / (host_stats["mean"] * 1e-6) / 1e9
    npu_gops = total_ops / (npu_stats["mean"] * 1e-6) / 1e9 if npu_stats else None

    print(f"\n=== INT8 GEMM {M}x{K}x{N} (multi-core {n_aie_cols} cols) ===")
    print(f"Host submission latency (us): min={host_stats['min']:.1f} mean={host_stats['mean']:.1f} median={host_stats['median']:.1f} p95={host_stats['p95']:.1f} p99={host_stats['p99']:.1f} max={host_stats['max']:.1f}")
    if npu_stats:
        print(f"NPU execution latency (us): min={npu_stats['min']:.1f} mean={npu_stats['mean']:.1f} median={npu_stats['median']:.1f} p95={npu_stats['p95']:.1f} p99={npu_stats['p99']:.1f} max={npu_stats['max']:.1f}")
    print(f"Effective GOPS (host e2e): {e2e_gops:.2f}")
    if npu_gops:
        print(f"Effective GOPS (NPU exec): {npu_gops:.2f}")

    got = C.numpy()
    mismatches = int(np.sum(got != ref))
    print(f"Correctness: {mismatches} mismatches / {M*N}")
    if mismatches == 0:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
