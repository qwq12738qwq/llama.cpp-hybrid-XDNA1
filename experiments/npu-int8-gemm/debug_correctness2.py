#!/usr/bin/env python3
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.iron.controlflow import range_

M, K, N = 128, 512, 4096
m, k, n = 64, 64, 64

def _build_design(M, K, N):
    matmul_kernel = mm(dim_m=m, dim_k=k, dim_n=n,
                       input_dtype=np.int8, output_dtype=np.int32, vectorized=True)
    r, s, t = matmul_kernel.mac_dims
    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]
    a_ty = np.ndarray[(m * k), np.dtype[np.int8]]
    b_ty = np.ndarray[(k * n), np.dtype[np.int8]]
    c_ty = np.ndarray[(m * n), np.dtype[np.int32]]
    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1")
    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1")
    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n), offset=0, sizes=[m // r, r, n // t, t], strides=[r * n, t, r * t, 1])
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3")
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
    worker = iron.Worker(core_fn, [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel])
    a_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k), pattern_repeat=(N // n))
    b_tap = TensorTiler2D.group_tiler((K, N), (k, n), (K // k, N // n), tile_group_col_major=True)[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))
    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // m):
            tg = iron.TaskGroup()
            a_prod.fill(A, tap=a_taps[tile_row], group=tg)
            b_prod.fill(B, tap=b_tap, group=tg)
            c_cons.drain(C, tap=c_taps[tile_row], group=tg, wait=True)
            tg.finish()
    rt = iron.Runtime(sequence, [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

rng = np.random.default_rng(42)
A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
A = iron.tensor(A_np, dtype=np.int8, device="npu")
B = iron.tensor(B_np, dtype=np.int8, device="npu")
C = iron.tensor((M, N), dtype=np.int32, device="npu")

def main():
    global _build_design
    @iron.jit
    def int8_gemm_design(
        A: In, B: In, C: Out,
        *, M: CompileTime[int]=M, K: CompileTime[int]=K, N: CompileTime[int]=N,
    ):
        return _build_design(M, K, N)
    compiled = int8_gemm_design.specialize(M=M, K=K, N=N).compile()
    print(f"Compiled: {compiled[0]}")
    ret = int8_gemm_design(A, B, C, M=M, K=K, N=N)
    got = C.numpy()
    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)
    mm = int(np.sum(got != ref))
    print(f"mismatches: {mm} / {M*N}")
    # WHERE are mismatches?
    diff = got != ref
    print(f"Rows with mismatches: {np.any(diff, axis=1).sum()}")
    print(f"Cols with mismatches: {np.any(diff, axis=0).sum()}")
    # Check if it's transposed
    print(f"got.T matches ref: {np.array_equal(got.T, ref)}")
    # Check row 0 vs row 1
    print(f"Row 0 matches: {np.array_equal(got[0], ref[0])}")
    print(f"Row 1 matches: {np.array_equal(got[1], ref[1])}")
    print(f"Row 63 matches: {np.array_equal(got[63], ref[63])}")
    print(f"Row 64 matches: {np.array_equal(got[64], ref[64])}")
    # How many rows match fully
    rows_match = sum(1 for i in range(M) if np.array_equal(got[i], ref[i]))
    print(f"Rows fully matching: {rows_match}/{M}")
    # For the first mismatched row, find the mismatch columns
    for row in range(M):
        if not np.array_equal(got[row], ref[row]):
            diffs = np.where(got[row] != ref[row])[0]
            print(f"First mismatched row {row}: cols {diffs[:10]}... ({len(diffs)} mismatches in row)")
            print(f"  got[{row}, {diffs[0]}] = {got[row, diffs[0]]}, ref[{row}, {diffs[0]}] = {ref[row, diffs[0]]}")
            print(f"  ref[{row}, :4] = {ref[row, :4]}")
            print(f"  got[{row}, :4] = {got[row, :4]}")
            break

if __name__ == "__main__":
    main()
