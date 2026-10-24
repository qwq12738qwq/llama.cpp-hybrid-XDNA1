#!/usr/bin/env python3
# Test GEMV (M=1) with non-vectorized kernel

import argparse
import sys
import time
import statistics
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.verify import assert_pass

def cpu_int8_gemv(A, B):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--work-dir", type=str, default=None)
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    
    # MAC dims for non-vectorized: (1, 8, 8) or (4, 8, 8)?
    # Try with vectorized=False
    matmul_kernel = mm(
        dim_m=16,  # tile_m
        dim_k=64,  # tile_k
        dim_n=16,  # tile_n
        input_dtype=np.int8,
        output_dtype=np.int32,
        vectorized=False,  # Try non-vectorized
    )
    r, s, t = matmul_kernel.mac_dims
    print(f"MAC dims: {r}x{s}x{t}")

    # Check divisibility
    if M % r != 0 or K % s != 0 or N % t != 0:
        sys.exit(f"Shape {M}x{K}x{N} not compatible with MAC dims ({r},{s},{t}).")

    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    a_ty = np.ndarray[(16 * 64,), np.dtype[np.int8]]
    b_ty = np.ndarray[(64 * 16,), np.dtype[np.int8]]
    c_ty = np.ndarray[(16 * 16,), np.dtype[np.int32]]

    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((16, 64), (r, s), (16 // r, 64 // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((64, 16), (s, t), (64 // s, 16 // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(16, 16),
        offset=0,
        sizes=[16 // r, r, 16 // t, t],
        strides=[r * 16, t, r * t, 1],
    )
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3"
    )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in iron.controlflow.range_(M // 16 * N // 16):
            elem_out = of_c.acquire(1)
            for i in iron.controlflow.range_(16 * 16):
                elem_out[i] = 0
            for _ in iron.controlflow.range_(K // 64):
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

    # Taps for full matrices
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (16, 64), (1, K // 64), pattern_repeat=(N // 16)
    )

    b_taps_l3l2 = [TensorTiler2D.group_tiler(
        (K, N), (64, 16), (K // 64, N // 16), tile_group_col_major=True
    )[0]]

    c_taps = []
    for tile_row in range(M // 16):
        for tile_col in range(N // 16):
            offset = tile_row * 16 * N + tile_col * 16
            c_tap = TensorAccessPattern(
                tensor_dims=(M, N),
                offset=offset,
                sizes=[16, 16],
                strides=[N, 1],
            )
            c_taps.append(c_tap)

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // 16):
            for tile_col in range(N // 16):
                tg = iron.TaskGroup()
                a_prod.fill(A, tap=a_taps[tile_row], group=tg)
                b_prod.fill(B, tap=b_taps_l3l2[0], group=tg)
                c_cons.drain(C, tap=c_taps[tile_row * (N // 16) + tile_col], group=tg, wait=True)
                tg.finish()

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    prog = iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    import os
    compile_kwargs = {}
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        compile_kwargs["xclbin_path"] = os.path.join(args.work_dir, "design.xclbin")
        compile_kwargs["inst_path"] = os.path.join(args.work_dir, "design.insts.bin")

    @iron.jit
    def gemv_design(
        A: In,
        B: In,
        C: Out,
        *,
        M: CompileTime[int] = M,
        K: CompileTime[int] = K,
        N: CompileTime[int] = N,
    ):
        return prog

    compiled = gemv_design.specialize(M=M, K=K, N=N).compile(**compile_kwargs)
    print(f"  xclbin: {compiled[0]}")
    print(f"  insts:  {compiled[1]}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    gemv_design(A, B, C, M=M, K=K, N=N)

    ref = cpu_int8_gemv(A_np, B_np)

    host_times = []
    npu_times = []
    for i in range(args.iters):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        start = time.perf_counter()
        ret = gemv_design(A, B, C, M=M, K=K, N=N)
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

    print(f"\n=== INT8 GEMV {M}x{K}x{N} ===")
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