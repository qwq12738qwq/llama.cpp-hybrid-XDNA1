#!/usr/bin/env python3
"""Phase 20 V2: Vectorized INT8 GEMM, compiled via IRON JIT + manual aiecc.
Follows the exact pattern from int8_gemm.py which is proven to work."""
import os, sys, time, json
import numpy as np

sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

def _build_design(M, K, N, tile_m=64, tile_k=64, tile_n=64):
    """Build vectorized GEMM design -- same as int8_gemm.py's _build_design."""
    matmul_kernel = mm(dim_m=tile_m, dim_k=tile_k, dim_n=tile_n,
                       input_dtype=np.int8, output_dtype=np.int32, vectorized=True)
    r, s, t = matmul_kernel.mac_dims
    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]
    a_ty = np.ndarray[(tile_m * tile_k), np.dtype[np.int8]]
    b_ty = np.ndarray[(tile_k * tile_n), np.dtype[np.int8]]
    c_ty = np.ndarray[(tile_m * tile_n), np.dtype[np.int32]]

    MAX_BD_BYTES = 65536
    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((tile_m, tile_k), (r, s), (tile_m // r, tile_k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1")

    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    # Split B L3L2 BDs when K*N > 65536 (aiecc iteration-wrap limit)
    if K * tile_n > MAX_BD_BYTES:
        b_l3l2_taps = TensorTiler2D.group_tiler(
            (K, N), (tile_k, tile_n), (K // tile_k, 1),
            pattern_repeat=1)[0]
    else:
        b_l3l2_taps = None
    tap_B_L2L1 = TensorTiler2D.group_tiler((tile_k, tile_n), (s, t), (tile_k // s, tile_n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1")

    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(tile_m, tile_n), offset=0,
        sizes=[tile_m // r, r, tile_n // t, t], strides=[r * tile_n, t, r * t, 1])
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3")

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in range_(M // tile_m * N // tile_n):
            elem_out = of_c.acquire(1)
            for i in range_(tile_m * tile_n):
                elem_out[i] = 0
            for _ in range_(K // tile_k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = iron.Worker(core_fn,
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel])

    a_taps = TensorTiler2D.group_tiler(
        (M, K), (tile_m, tile_k), (1, K // tile_k), pattern_repeat=(N // tile_n))
    if b_l3l2_taps is not None:
        b_tap = b_l3l2_taps
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (tile_k, tile_n), (K // tile_k, N // tile_n),
            tile_group_col_major=True)[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (tile_m, tile_n), (1, N // tile_n))

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // tile_m):
            tg = iron.TaskGroup()
            a_prod.fill(A, tap=a_taps[tile_row], group=tg)
            if b_l3l2_taps is not None:
                b_prod.fill(B, tap=b_tap, group=tg)
            else:
                b_prod.fill(B, tap=b_tap, group=tg)
            c_cons.drain(C, tap=c_taps[tile_row], group=tg, wait=True)
            tg.finish()

    rt = iron.Runtime(sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    M, K, N = 512, 1024, 3072
    work_dir = "/Storage/Cache/llama.cpp/experiments/phase20/v2_design.prj"
    os.makedirs(work_dir, exist_ok=True)

    dev = iron.get_current_device()
    print(f"device: {type(dev).__name__}  columns: {dev.cols}")

    # Define @iron.jit INSIDE main -- same pattern as int8_gemm.py
    @iron.jit
    def v2_design(A: In, B: In, C: Out,
                  *, M: CompileTime[int] = M,
                     K: CompileTime[int] = K,
                     N: CompileTime[int] = N):
        return _build_design(M, K, N)

    # Compile with explicit output paths
    compiled = v2_design.specialize(M=M, K=K, N=N).compile(
        xclbin_path=os.path.join(work_dir, "design.xclbin"),
        inst_path=os.path.join(work_dir, "design.insts.bin"),
    )
    print(f"Compiled: xclbin={compiled[0]}, insts={compiled[1]}")

    # Generate test data
    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    # First run + correctness check
    C.data.fill(0)
    v2_design(A, B, C, M=M, K=K, N=N)
    got = C.numpy()
    max_err = int(np.max(np.abs(got - ref)))
    n_mismatch = int(np.sum(got != ref))
    print(f"Correctness: max_err={max_err}, mismatches={n_mismatch}/{M*N}")

    if max_err > 0:
        print("FAIL - correctness mismatch")
        # Save for debugging
        np.save(os.path.join(work_dir, "A.npy"), A_np)
        np.save(os.path.join(work_dir, "B.npy"), B_np)
        np.save(os.path.join(work_dir, "got.npy"), got)
        np.save(os.path.join(work_dir, "ref.npy"), ref)
        return

    # Benchmark
    n_warmup, n_runs = 5, 20
    for _ in range(n_warmup):
        C.data.fill(0)
        v2_design(A, B, C, M=M, K=K, N=N)

    kernel_times = []
    for _ in range(n_runs):
        C.data.fill(0)
        t0 = time.perf_counter()
        v2_design(A, B, C, M=M, K=K, N=N)
        t1 = time.perf_counter()
        kernel_times.append((t1 - t0) * 1000)

    kt = np.array(kernel_times)
    flops = 2.0 * M * K * N
    gops = flops / (np.mean(kt) * 1e-3) / 1e9

    print(f"\nV2 Vectorized M={M} K={K} N={N}")
    print(f"  Kernel: {np.mean(kt):.2f} ms (p50={np.percentile(kt,50):.2f}, p95={np.percentile(kt,95):.2f})")
    print(f"  GOPS:   {gops:.2f}")
    print(f"  V1 was: 53.96 GOPS -> speedup: {gops/53.96:.2f}x")

    result = {"M": M, "K": K, "N": N, "kernel_ms": float(np.mean(kt)),
              "GOPS_kernel": gops, "max_err": max_err, "correct": True}
    with open("/Storage/Cache/llama.cpp/experiments/phase20/phase20_v2_vectorized.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
