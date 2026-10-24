#!/usr/bin/env python3
# verify_fix.py — Verify the multi-N-column fix in int8_gemm.py

import sys
import os
import time
import numpy as np

os.environ['MLIR_AIE_INSTALL_DIR'] = '/root/src/mlir-aie'

# Use the FIXED int8_gemm.py
sys.path.insert(0, "/home/ubuntu12738/Storage/Cache/llama.cpp/experiments/npu-int8-gemm")

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

_TILE_M = 64
_TILE_K = 64
_TILE_N = 64
MAX_BD_BYTES = 65536


def cpu_int8_gemm(A, B):
    return A.astype(np.int32) @ B.astype(np.int32)


def build_design(M, K, N):
    """Build GEMM design using the FIXED A-tile pattern_repeat=1."""
    m, k, n = _TILE_M, _TILE_K, _TILE_N

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

    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n), offset=0,
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

    # FIXED: pattern_repeat=1 (was N//n, which caused A BD aliasing)
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=1
    )

    b_tap_l2l1 = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
    )[0]

    b_bytes = K * N
    if b_bytes <= MAX_BD_BYTES:
        b_taps_l3l2 = [TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
        )[0]]
    else:
        b_taps_l3l2 = []
        for nt in range(N // n):
            b_tap_chunk = TensorAccessPattern(
                tensor_dims=(K, N),
                offset=nt * n,
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
                b_tap_idx = tile_col if len(b_taps_l3l2) > 1 else 0
                b_prod.fill(B, tap=b_taps_l3l2[b_tap_idx], group=tg)
                c_cons.drain(C, tap=c_taps[tile_row * (N // n) + tile_col], group=tg, wait=True)
                tg.finish()

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def run_test(M, K, N, work_dir):
    m, k, n = _TILE_M, _TILE_K, _TILE_N
    r, s, t = 4, 8, 8

    if M % r != 0 or K % s != 0 or N % t != 0:
        print(f"  SKIP: shape not compatible with MAC dims")
        return None
    if M % m != 0 or K % k != 0 or N % n != 0:
        print(f"  SKIP: shape not divisible by tile size")
        return None

    print(f"\n[M={M} K={K} N={N}] tiles=({M//m},{K//k},{N//n})")

    @iron.jit
    def int8_gemm_design(
        A: In, B: In, C: Out,
        *,
        M: CompileTime[int] = M,
        K: CompileTime[int] = K,
        N: CompileTime[int] = N,
    ):
        return build_design(M, K, N)

    os.makedirs(work_dir, exist_ok=True)
    compiled = int8_gemm_design.specialize(M=M, K=K, N=N).compile(
        xclbin_path=os.path.join(work_dir, "design.xclbin"),
        inst_path=os.path.join(work_dir, "design.insts.bin"),
    )
    print(f"  compiled: {os.path.basename(compiled[0])}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    int8_gemm_design(A, B, C, M=M, K=K, N=N)

    ref = cpu_int8_gemm(A_np, B_np)
    got = C.numpy()
    mismatches = int(np.sum(got != ref))
    total = M * N
    status = "PASS" if mismatches == 0 else "FAIL"
    print(f"  result: {status}  mismatches={mismatches}/{total}")

    if mismatches > 0:
        for tile_col in range(N // n):
            col_start = tile_col * n
            tile_mismatches = int(np.sum(got[:, col_start:col_start+n] != ref[:, col_start:col_start+n]))
            if tile_mismatches > 0:
                print(f"    wrong tile column: col={tile_col}  mismatches={tile_mismatches}/{M*n}")

    return mismatches == 0


def main():
    test_cases = [
        (64, 1024, 64,  "verify_baseline_M64_N64"),
        (64, 1024, 128, "verify_M64_N128"),
        (128, 1024, 64, "verify_M128_N64"),
        (128, 1024, 128, "verify_M128_N128"),
        (128, 1024, 256, "verify_M128_N256"),
        (128, 1024, 512, "verify_M128_N512"),
        (256, 1024, 128, "verify_M256_N128"),
        (512, 1024, 16,  "verify_qwen_ssm"),
    ]

    print("=" * 60)
    print("VERIFYING FIX: pattern_repeat=1 for A-tile TAS")
    print("=" * 60)

    results = {}
    for M, K, N, desc in test_cases:
        print(f"\n--- {desc} ---")
        work_dir = f"/workspace/llama.cpp/experiments/npu-int8-gemm/verify_{desc}"
        ok = run_test(M, K, N, work_dir)
        results[(M, K, N)] = ok

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Shape':<20} {'Status':<10}")
    print("-" * 30)
    for (M, K, N), ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"{M}x{K}x{N:<10} {status}")


if __name__ == "__main__":
    main()
