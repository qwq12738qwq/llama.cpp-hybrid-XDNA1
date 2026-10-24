#!/usr/bin/env python3
"""Phase 20 V2: Use int8_gemm.py directly to confirm V1 = depth=2 double buffered.
Then benchmark to establish that the 53 GOPS ceiling includes double buffering."""
import os, sys, time, json
import numpy as np

sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern

_TILE_M = 64
_TILE_K = 64
_TILE_N = 64
MAX_BD_BYTES = 65536

def _build_design(M, K, N):
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
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1")

    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    if K * n > MAX_BD_BYTES:
        b_taps = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, 1),
            pattern_repeat=1)[0]
    else:
        b_taps = None
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1")

    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n), offset=0,
        sizes=[m // r, r, n // t, t], strides=[r * n, t, r * t, 1])
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3")

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

    worker = iron.Worker(core_fn,
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel])

    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n))
    if b_taps is not None:
        b_tap = b_taps
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K // k, N // n), tile_group_col_major=True)[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // m):
            tg = iron.TaskGroup()
            a_prod.fill(A, tap=a_taps[tile_row], group=tg)
            b_prod.fill(B, tap=b_tap, group=tg)
            c_cons.drain(C, tap=c_taps[tile_row], group=tg, wait=True)
            tg.finish()

    rt = iron.Runtime(sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    M, K, N = 512, 1024, 3072
    work_dir = "/Storage/Cache/llama.cpp/experiments/phase20/v2_confirm.prj"
    os.makedirs(work_dir, exist_ok=True)

    dev = iron.get_current_device()
    print(f"device: {type(dev).__name__}  columns: {dev.cols}")

    @iron.jit
    def v2_design(A: In, B: In, C: Out,
                  *, M: CompileTime[int] = M,
                     K: CompileTime[int] = K,
                     N: CompileTime[int] = N):
        return _build_design(M, K, N)

    compiled = v2_design.specialize(M=M, K=K, N=N).compile(
        xclbin_path=os.path.join(work_dir, "design.xclbin"),
        inst_path=os.path.join(work_dir, "design.insts.bin"),
    )
    print(f"Compiled: xclbin={compiled[0]}, insts={compiled[1]}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")

    C.data.fill(0)
    v2_design(A, B, C, M=M, K=K, N=N)
    got = C.numpy()
    max_err = int(np.max(np.abs(got - ref)))
    n_mismatch = int(np.sum(got != ref))
    print(f"Correctness: max_err={max_err}, mismatches={n_mismatch}/{M*N}")

    if max_err > 0:
        print("FAIL")
        return

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

    print(f"\nV2 Confirm M={M} K={K} N={N}")
    print(f"  Kernel: {np.mean(kt):.2f} ms (p50={np.percentile(kt,50):.2f}, p95={np.percentile(kt,95):.2f})")
    print(f"  GOPS:   {gops:.2f}")
    print(f"  V1 was: 53.96 GOPS -> ratio: {gops/53.96:.2f}x")

    result = {"M": M, "K": K, "N": N, "kernel_ms": float(np.mean(kt)),
              "GOPS_kernel": gops, "max_err": max_err, "correct": True,
              "note": "Same design as V1 (depth=2 already default)"}
    out = "/Storage/Cache/llama.cpp/experiments/phase20/phase20_v2_confirm.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
