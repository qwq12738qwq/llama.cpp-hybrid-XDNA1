#!/usr/bin/env python3
"""
Parameterized XDNA GEMM Kernel Generator

This script generates compile-time specialized XDNA GEMM kernels via MLIR-AIE/IRON.
It can be invoked from the ggml-xdna kernel registry to produce xclbin/insts/pdi
for arbitrary M/K/N shapes.

Usage:
    python gen_xdna_gemm.py --M 64 --K 1024 --N 128 --work-dir /Storage/Cache/xdna_kernels/M64_K1024_N128
    python gen_xdna_gemm.py --M 128 --K 1024 --N 512 --work-dir /Storage/Cache/xdna_kernels/M128_K1024_N512 --cols 2

Output:
    {work_dir}/design.xclbin
    {work_dir}/design.insts.bin
    {work_dir}/main.pdi
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron.controlflow import range_ as iron_range

# Default tile sizes (match aie2 int8 MAC dims 4x8x8)
TILE_M = 64
TILE_K = 64
TILE_N = 64

MAX_BD_BYTES = 65536  # aiecc iteration-wrap limit


def build_gemm_design(M, K, N, n_cols=1):
    """Build an N-parallel GEMM design for the given shape."""
    m, k, n = TILE_M, TILE_K, TILE_N
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

    # Create ObjectFifos for each column
    A_l3l2 = []
    A_l2l1 = []
    B_l3l2 = []
    B_l2l1 = []
    C_l1l2 = []
    C_l2l3 = []

    N_per_col = N // n_cols
    assert N % n_cols == 0, f"N={N} must be divisible by n_cols={n_cols}"
    n_tiles_per_col = N_per_col // n

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
        for _ in iron_range(M // m * n_tiles_per_col):
            elem_out = of_c.acquire(1)
            for i in iron_range(m * n):
                elem_out[i] = 0
            for _ in iron_range(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    # Create workers for each column
    workers = []
    for col in range(n_cols):
        workers.append(iron.Worker(
            core_fn,
            [A_l2l1[col].cons(), B_l2l1[col].cons(), C_l1l2[col].prod(), matmul_kernel],
            stack_size=0x800,
        ))

    # A taps: one per M-row (same for all columns since A is broadcast)
    # pattern_repeat=1 is CRITICAL to avoid BD aliasing when M//m > 1 AND N//n > 1
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=1
    )

    # B taps: per column, each covering its N-column slice with correct BD splitting
    b_taps_l3l2 = []
    for col in range(n_cols):
        slice_bytes = K * N_per_col
        col_offset = col * N_per_col
        # Always split B per N-tile. The single-BD path fed a whole column
        # slice per (M-row, N-tile) step and produced wrong results for every
        # shape it handled; see experiments/phase20d/verify_kernels.py.
        col_taps = []
        for nt in range(n_tiles_per_col):
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
                    tg = iron.TaskGroup()
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


def main():
    global TILE_M, TILE_K, TILE_N
    parser = argparse.ArgumentParser(description="Parameterized XDNA GEMM Kernel Generator")
    parser.add_argument("--M", type=int, required=True, help="M dimension")
    parser.add_argument("--K", type=int, required=True, help="K dimension")
    parser.add_argument("--N", type=int, required=True, help="N dimension")
    parser.add_argument("--cols", type=int, default=1, help="Number of parallel columns (N-parallel)")
    parser.add_argument("--tile-m", type=int, default=TILE_M)
    parser.add_argument("--tile-k", type=int, default=TILE_K)
    parser.add_argument("--tile-n", type=int, default=TILE_N)
    parser.add_argument("--work-dir", type=str, required=True, help="Output directory for xclbin/insts/pdi")
    parser.add_argument("--dtype", type=str, default="int8", choices=["int8"], help="Input data type")
    parser.add_argument("--accum-dtype", type=str, default="int32", choices=["int32"], help="Accumulation data type")
    args = parser.parse_args()

    M, K, N, n_cols = args.M, args.K, args.N, args.cols
    TILE_M, TILE_K, TILE_N = args.tile_m, args.tile_k, args.tile_n

    m, k, n = TILE_M, TILE_K, TILE_N
    r, s, t = 4, 8, 8  # aie2 int8 MAC dims

    # Validate shape compatibility
    if M % r != 0 or K % s != 0 or N % t != 0:
        sys.exit(f"Shape {M}x{K}x{N} not compatible with aie2 int8 MAC dims ({r},{s},{t}).")
    if M % m != 0 or K % k != 0 or N % n != 0:
        sys.exit(f"Shape {M}x{K}x{N} not divisible by tile size ({m},{k},{n}).")
    if N % (n * n_cols) != 0:
        sys.exit(f"N={N} must be divisible by tile_n*n_cols={n*n_cols}.")

    dev = iron.get_current_device()
    print(f"device: {type(dev).__name__}  columns: {dev.cols}")
    print(f"Building XDNA GEMM: A={M}x{K}, B={K}x{N}, C={M}x{N}  cols={n_cols} tile=({m},{k},{n})")

    os.makedirs(args.work_dir, exist_ok=True)

    @iron.jit
    def int8_gemm_design(
        A: In, B: In, C: Out,
        *,
        M: CompileTime[int] = M,
        K: CompileTime[int] = K,
        N: CompileTime[int] = N,
        n_cols: CompileTime[int] = n_cols,
    ):
        return build_gemm_design(M, K, N, n_cols)

    xclbin_path = os.path.join(args.work_dir, "design.xclbin")
    inst_path = os.path.join(args.work_dir, "design.insts.bin")
    pdi_path = os.path.join(args.work_dir, "main.pdi")

    print(f"Compiling...")
    t0 = time.time()
    try:
        compiled = int8_gemm_design.specialize(M=M, K=K, N=N, n_cols=n_cols).compile(
            xclbin_path=xclbin_path,
            inst_path=inst_path,
            pdi_path=pdi_path,
        )
    except Exception as exc:
        print(f"COMPILE FAILED: {exc}")
        sys.exit(2)
    compile_time = time.time() - t0

    actual_xclbin = compiled[0]
    actual_insts = compiled[1]
    actual_pdi = pdi_path

    print(f"  compile_time: {compile_time:.1f}s")
    print(f"  xclbin: {actual_xclbin} ({os.path.getsize(actual_xclbin)} bytes)")
    print(f"  insts:  {actual_insts} ({os.path.getsize(actual_insts)} bytes)")
    print(f"  pdi:    {actual_pdi} ({os.path.getsize(actual_pdi)} bytes)")

    # Write metadata
    meta = {
        "M": M, "K": K, "N": N,
        "n_cols": n_cols,
        "tile_m": TILE_M, "tile_k": TILE_K, "tile_n": TILE_N,
        "dtype": args.dtype,
        "accum_dtype": args.accum_dtype,
        "compile_time_s": round(compile_time, 3),
        "xclbin_path": str(actual_xclbin),
        "insts_path": str(actual_insts),
        "pdi_path": str(actual_pdi),
        "xclbin_size_bytes": os.path.getsize(actual_xclbin),
        "insts_size_bytes": os.path.getsize(actual_insts),
        "pdi_size_bytes": os.path.getsize(actual_pdi),
    }
    with open(os.path.join(args.work_dir, "kernel_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Kernel metadata written to {args.work_dir}/kernel_meta.json")
    print("SUCCESS")


if __name__ == "__main__":
    main()