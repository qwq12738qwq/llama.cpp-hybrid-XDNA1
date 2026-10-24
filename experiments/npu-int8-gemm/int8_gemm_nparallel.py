#!/usr/bin/env python3
# int8_gemm_nparallel.py — Phoenix XDNA1 INT8 GEMM with N-parallel multi-column
#
# N-parallel design: Each column computes independent output columns
#   Column 0: C[:, 0:N/2] = A × B[:, 0:N/2]
#   Column 1: C[:, N/2:N] = A × B[:, N/2:N]
#   ...
#   Then concatenate: C = [C0, C1, ...]
#
# No numerical reduction required - pure spatial parallelism.
#
# Run inside the npu-dev container with:
#   python int8_gemm_nparallel.py --M 128 --K 1024 --N 256 --cols 2

import argparse
import sys
import time
import statistics
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, TaskGroup, Worker
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron.controlflow import range_


_TILE_M = 64
_TILE_K = 64
_TILE_N = 64

MAX_BD_BYTES = 65536


def _build_nparallel_design(M, K, N, n_cols):
    """Build N-parallel GEMM design using n_cols columns."""
    m, k, n = _TILE_M, _TILE_K, _TILE_N
    n_rows = 4  # 4 compute rows per column

    matmul_kernel = mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=np.int8, output_dtype=np.int32, vectorized=True,
    )
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    a_ty = np.ndarray[(m * k,), np.dtype[np.int8]]
    b_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
    c_ty = np.ndarray[(m * n,), np.dtype[np.int32]]

    # N tiles per column
    N_per_col = N // n_cols
    assert N % n_cols == 0, f"N={N} must be divisible by n_cols={n_cols}"
    n_tiles_per_col = N_per_col // n

    # Create ObjectFifos for each column
    A_l3l2 = []
    A_l2l1 = []
    B_l3l2 = []
    B_l2l1 = []
    C_l1l2 = []
    C_l2l3 = []

    for col in range(n_cols):
        # A: each column gets full A matrix (broadcast)
        A_l3l2.append(iron.ObjectFifo(a_ty, name=f"A_L3L2_{col}", depth=2))
        tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
        A_l2l1.append(
            A_l3l2[col].cons().forward(
                dims_to_stream=tap_A_L2L1.transformation_dims, name=f"A_L2L1_{col}"
            )
        )

        # B: each column gets its N-column slice
        B_l3l2.append(iron.ObjectFifo(b_ty, name=f"B_L3L2_{col}", depth=2))
        tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
        B_l2l1.append(
            B_l3l2[col].cons().forward(
                dims_to_stream=tap_B_L2L1.transformation_dims, name=f"B_L2L1_{col}"
            )
        )

        # C: each column produces its output slice
        C_l1l2.append(iron.ObjectFifo(c_ty, name=f"C_L1L2_{col}", depth=2))
        tap_C_L1L2 = TensorAccessPattern(
            tensor_dims=(m, n),
            offset=0,
            sizes=[m // r, r, n // t, t],
            strides=[r * n, t, r * t, 1],
        )
        C_l2l3.append(
            C_l1l2[col].cons().forward(
                dims_to_stream=list(tap_C_L1L2.transformation_dims), name=f"C_L2L3_{col}"
            )
        )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in range_(M // m * n_tiles_per_col):
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

    # Create workers for each column (1 row per column for simplicity)
    workers = []
    for col in range(n_cols):
        workers.append(Worker(
            core_fn,
            [A_l2l1[col].cons(), B_l2l1[col].cons(), C_l1l2[col].prod(), matmul_kernel],
            stack_size=0x800,
        ))

    # A taps: one per M-row (same for all columns since A is broadcast)
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=1
    )

    # B taps: per column, each covering its N-column slice with correct BD splitting
    b_taps_l3l2 = []
    for col in range(n_cols):
        slice_bytes = K * N_per_col
        col_offset = col * N_per_col
        if slice_bytes <= MAX_BD_BYTES:
            # Single tap for this column's slice
            b_tap = TensorAccessPattern(
                tensor_dims=(K, N),
                offset=col_offset,
                sizes=[K, N_per_col],
                strides=[N, 1],
            )
            b_taps_l3l2.append([b_tap])
        else:
            # Split by N-tiles within this column's slice
            col_taps = []
            n_tiles_in_col = N_per_col // n
            for nt in range(n_tiles_in_col):
                offset = col_offset + nt * n
                b_tap = TensorAccessPattern(
                    tensor_dims=(K, N),
                    offset=offset,
                    sizes=[K, n],
                    strides=[N, 1],
                )
                col_taps.append(b_tap)
            b_taps_l3l2.append(col_taps)

    # C taps: per column, each covering its output region
    c_taps = []
    for col in range(n_cols):
        col_taps = []
        for tile_row in range(M // m):
            for tile_col in range(n_tiles_per_col):
                offset = tile_row * m * N + col * N_per_col + tile_col * n
                c_tap = TensorAccessPattern(
                    tensor_dims=(M, N),
                    offset=offset,
                    sizes=[m, n],
                    strides=[N, 1],
                )
                col_taps.append(c_tap)
        c_taps.append(col_taps)

    def sequence(A, B, C, a_prods, b_prods, c_conses):
        for tile_row in range(M // m):
            for col in range(n_cols):
                for tile_col in range(n_tiles_per_col):
                    tg = TaskGroup()
                    a_prods[col].fill(A, tap=a_taps[tile_row], group=tg)
                    b_tap_idx = tile_col if len(b_taps_l3l2[col]) > 1 else 0
                    b_prods[col].fill(B, tap=b_taps_l3l2[col][b_tap_idx], group=tg)
                    c_conses[col].drain(C, tap=c_taps[col][tile_row * n_tiles_per_col + tile_col], group=tg, wait=True)
                    tg.finish()

    # Build producers/consumers lists
    A_prods = [f.prod() for f in A_l3l2]
    B_prods = [f.prod() for f in B_l3l2]
    C_conses = [f.cons() for f in C_l2l3]

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses],
    )
    return iron.Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def cpu_int8_gemm(A, B):
    return A.astype(np.int32) @ B.astype(np.int32)


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0], "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)], "max": s[-1],
    }


def main():
    global _TILE_M, _TILE_K, _TILE_N
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--tile-m", type=int, default=_TILE_M)
    parser.add_argument("--tile-k", type=int, default=_TILE_K)
    parser.add_argument("--tile-n", type=int, default=_TILE_N)
    args = parser.parse_args()

    M, K, N, n_cols = args.M, args.K, args.N, args.cols
    _TILE_M, _TILE_K, _TILE_N = args.tile_m, args.tile_k, args.tile_n

    m, k, n = _TILE_M, _TILE_K, _TILE_N
    r, s, t = 4, 8, 8

    if M % r != 0 or K % s != 0 or N % t != 0:
        sys.exit(f"Shape {M}x{K}x{N} not compatible with MAC dims ({r},{s},{t}).")
    if M % m != 0 or K % k != 0 or N % n != 0:
        sys.exit(f"Shape {M}x{K}x{N} not divisible by tile size ({m},{k},{n}).")
    if N % (n * n_cols) != 0:
        sys.exit(f"N={N} must be divisible by tile_n*n_cols={n*n_cols}.")

    dev = iron.get_current_device()
    print(f"device: {type(dev).__name__}  columns: {dev.cols}")
    print(f"Building N-parallel INT8 GEMM: A={M}x{K}, B={K}x{N}, C={M}x{N}  cols={n_cols} tile=({m},{k},{n})")

    @iron.jit
    def int8_gemm_design(
        A: In, B: In, C: Out,
        *,
        M: CompileTime[int] = M,
        K: CompileTime[int] = K,
        N: CompileTime[int] = N,
        n_cols: CompileTime[int] = n_cols,
    ):
        return _build_nparallel_design(M, K, N, n_cols)

    compiled = int8_gemm_design.specialize(M=M, K=K, N=N, n_cols=n_cols).compile()
    print(f"  xclbin: {compiled[0]}")
    print(f"  insts:  {compiled[1]}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    # Warmup
    int8_gemm_design(A, B, C, M=M, K=K, N=N, n_cols=n_cols)

    ref = cpu_int8_gemm(A_np, B_np)

    host_times, npu_times = [], []
    for i in range(args.iters):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        start = time.perf_counter()
        ret = int8_gemm_design(A, B, C, M=M, K=K, N=N, n_cols=n_cols)
        host_times.append((time.perf_counter() - start) * 1e6)
        npu_ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", None)
        if npu_ns is not None:
            npu_times.append(npu_ns / 1000.0)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{args.iters} done")

    host_stats = _stats(host_times)
    npu_stats = _stats(npu_times) if npu_times else None

    total_ops = 2.0 * M * K * N
    e2e_gops = total_ops / (host_stats["mean"] * 1e-6) / 1e9
    npu_gops = total_ops / (npu_stats["mean"] * 1e-6) / 1e9 if npu_stats else None

    print(f"\n=== N-parallel INT8 GEMM {M}x{K}x{N} (cols={n_cols}) ===")
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
        for col in range(n_cols):
            col_start = col * (N // n_cols)
            col_mismatches = int(np.sum(got[:, col_start:col_start+N//n_cols] != ref[:, col_start:col_start+N//n_cols]))
            if col_mismatches > 0:
                print(f"  Column {col} wrong: {col_mismatches} mismatches")
        sys.exit(1)


if __name__ == "__main__":
    main()