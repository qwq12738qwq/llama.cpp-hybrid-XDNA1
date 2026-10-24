#!/usr/bin/env python3
"""
Phase 9H-A: Minimal M=1 K=32 N=16 scalar XDNA design via CompilableDesign.

Recovery path from Phase 9G (scalar kernel compilation failure):
- Uses CompilableDesign.compile() so compile_external_kernel() is invoked.
- Patched mm() emits -DSCALAR_ONLY for vectorized=False.

Run inside npu-dev:mlir container with:
  source /root/src/mlir-aie-venv/bin/activate
  python gen_m1_k32_n16_v2.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.benchmark import run_iters

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def build_design(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 1,
    K: CompileTime[int] = 32,
    N: CompileTime[int] = 16,
):
    matmul_kernel = mm(
        dim_m=1,
        dim_k=32,
        dim_n=16,
        input_dtype=np.int8,
        output_dtype=np.int32,
        vectorized=False,
    )

    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(K, N), np.dtype[np.int8]]
    C_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    a_ty = np.ndarray[(1 * 32,), np.dtype[np.int8]]
    b_ty = np.ndarray[(32 * 16,), np.dtype[np.int8]]
    c_ty = np.ndarray[(1 * 16,), np.dtype[np.int32]]

    fifo_A_L3L2 = iron.ObjectFifo(a_ty, name="A_L3L2")
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=[(1 * 32, 1)], name="A_L2L1"
    )

    fifo_B_L3L2 = iron.ObjectFifo(b_ty, name="B_L3L2")
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=[(32 * 16, 1)], name="B_L2L1"
    )

    fifo_C_L1L2 = iron.ObjectFifo(c_ty, name="C_L1L2")
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=[(1, 16), (16, 1)], name="C_L2L3"
    )

    # DMA access patterns - explicit, no TensorTiler2D ambiguity
    # A: 1x32, single tile
    tap_A = TensorAccessPattern(
        (M, K), 0, [1, 32], [K, 1]
    )

    # B: 32x16, single tile  
    tap_B = TensorAccessPattern(
        (K, N), 0, [32, 16], [N, 1]
    )

    # C: 1x16, single tile
    tap_C = TensorAccessPattern(
        (M, N), 0, [1, 16], [N, 1]
    )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in iron.controlflow.range_(1):
            elem_out = of_c.acquire(1)
            for i in iron.controlflow.range_(1 * 16):
                elem_out[i] = 0
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matmul(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    worker = iron.Worker(
        core_fn,
        [
            fifo_A_L2L1.cons(),
            fifo_B_L2L1.cons(),
            fifo_C_L1L2.prod(),
            matmul_kernel,
        ],
    )

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        tg = iron.TaskGroup()
        a_prod.fill(A, tap=tap_A, group=tg)
        b_prod.fill(B, tap=tap_B, group=tg)
        c_cons.drain(C, tap=tap_C, group=tg, wait=True)
        tg.finish()

    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    return iron.Program(
        iron.get_current_device(), rt, workers=[worker]
    ).resolve_program()


def cpu_int8_gemm(A, B):
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    return A_i32 @ B_i32


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9H: M=1 scalar XDNA kernel compilation and correctness"
    )
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--tile-m", type=int, default=1)
    parser.add_argument("--tile-k", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=16)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--work-dir", type=str, default=None)
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    TILE_M, TILE_K, TILE_N = args.tile_m, args.tile_k, args.tile_n

    if M != 1:
        sys.exit(f"This script validates M=1 only; got M={M}.")

    dev = iron.get_current_device()
    print(f"device: {type(dev).__name__}  columns: {dev.cols}")
    print(
        f"Building scalar M=1 GEMM: A={M}x{K}, B={K}x{N}, C={M}x{N}  "
        f"tile=({TILE_M},{TILE_K},{TILE_N})"
    )

    @iron.jit
    def m1_design(
        A: In,
        B: In,
        C: Out,
        *,
        M: CompileTime[int] = 1,
        K: CompileTime[int] = 32,
        N: CompileTime[int] = 16,
    ):
        return build_design(A, B, C, M=M, K=K, N=N)

    xclbin_path = None
    insts_path = None
    pdi_path = None
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        xclbin_path = os.path.join(args.work_dir, "design.xclbin")
        insts_path = os.path.join(args.work_dir, "design.insts.bin")
        pdi_path = os.path.join(args.work_dir, "main.pdi")

    t0 = time.time()
    try:
        compiled = m1_design.specialize(M=M, K=K, N=N).compile(
            xclbin_path=xclbin_path,
            inst_path=insts_path,
            pdi_path=pdi_path,
        )
    except Exception as exc:
        print(f"COMPILE FAILED: {exc}")
        sys.exit(2)
    compile_time = time.time() - t0

    actual_xclbin = compiled[0]
    actual_insts = compiled[1]
    actual_pdi = pdi_path or str(
        m1_design.compilable.get_pdi_path() or "N/A"
    )
    print(f"  compile_time: {compile_time:.1f}s")
    print(f"  xclbin: {actual_xclbin}")
    print(f"  insts:  {actual_insts}")
    print(f"  pdi:    {actual_pdi}")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    ref = cpu_int8_gemm(A_np, B_np)

    def make_tensors():
        return (
            iron.tensor(A_np, dtype=np.int8, device="npu"),
            iron.tensor(B_np, dtype=np.int8, device="npu"),
            iron.tensor((M, N), dtype=np.int32, device="npu"),
        )

    # Warmup
    a_in, b_in, c_out = make_tensors()
    m1_design(a_in, b_in, c_out, M=M, K=K, N=N)

    # The output tensor is mutated in-place; capture the last one.
    c_last = [None]

    def run_once():
        a_in, b_in, c_out = make_tensors()
        m1_design(a_in, b_in, c_out, M=M, K=K, N=N)
        c_last[0] = c_out
        return c_out

    result = run_iters(run_once, warmup=args.warmup, iters=args.iters)

    e2e = result.e2e
    npu = result.npu
    print(f"\n=== M={M} K={K} N={N} scalar XDNA ===")
    print("Host submission latency (us):")
    for k in ("min_us", "avg_us", "max_us"):
        print(f"  {k}={getattr(e2e, k):.1f}")
    if npu:
        print("NPU execution latency (us):")
        for k in ("min_us", "avg_us", "max_us"):
            print(f"  {k}={getattr(npu, k):.1f}")

    total_ops = 2.0 * M * K * N
    e2e_gops = total_ops / (e2e.avg_us * 1e-6) / 1e9
    npu_gops = total_ops / (npu.avg_us * 1e-6) / 1e9 if npu else None
    print(f"\nEffective GOPS (host e2e): {e2e_gops:.2f}")
    if npu_gops:
        print(f"Effective GOPS (NPU exec): {npu_gops:.2f}")

    # Final correctness check
    a_in, b_in, c_out = make_tensors()
    m1_design(a_in, b_in, c_out, M=M, K=K, N=N)
    got = c_out.numpy()
    mismatches = int(np.sum(got != ref))
    max_abs_error = int(np.max(np.abs(got.astype(np.int64) - ref.astype(np.int64))))
    print(f"\nCorrectness: {mismatches} mismatches out of {M*N}")
    print(f"Max abs error: {max_abs_error}")
    if mismatches == 0 and max_abs_error == 0:
        print("PASS - output matches CPU int32 reference exactly")
    else:
        print("FAIL - output does not match CPU int32 reference")
        print("got[:8] =", got[:8].flatten())
        print("ref[:8] =", ref[:8].flatten())
        sys.exit(1)

    compile_info: dict[str, Any] = {
        "phase": "9h_m1_k32",
        "kernel": "m1_k32_n16_matmul_scalar_i8_i32",
        "M": M,
        "K": K,
        "N": N,
        "tile_m": TILE_M,
        "tile_k": TILE_K,
        "tile_n": TILE_N,
        "vectorized": False,
        "scalar_path": True,
        "k_chunked": False,
        "a_tile_bytes": TILE_M * TILE_K,
        "b_tile_bytes": TILE_K * TILE_N,
        "c_tile_bytes": TILE_M * TILE_N * 4,
        "compile_time_s": round(compile_time, 3),
        "xclbin_path": str(actual_xclbin),
        "insts_path": str(actual_insts),
        "pdi_path": str(actual_pdi),
        "xclbin_exists": os.path.exists(actual_xclbin),
        "insts_exists": os.path.exists(actual_insts),
        "xclbin_size_bytes": os.path.getsize(actual_xclbin) if os.path.exists(actual_xclbin) else 0,
        "insts_size_bytes": os.path.getsize(actual_insts) if os.path.exists(actual_insts) else 0,
        "pdi_size_bytes": os.path.getsize(actual_pdi) if os.path.exists(actual_pdi) else 0,
        "host_e2e": {
            "min_us": round(e2e.min_us, 3),
            "avg_us": round(e2e.avg_us, 3),
            "max_us": round(e2e.max_us, 3),
        },
        "npu_exec": {
            "min_us": round(npu.min_us, 3),
            "avg_us": round(npu.avg_us, 3),
            "max_us": round(npu.max_us, 3),
        } if npu else None,
        "mismatches": mismatches,
        "max_abs_error": max_abs_error,
        "correct": mismatches == 0 and max_abs_error == 0,
    }

    out_json = os.path.join(WORK_DIR, "phase9h_m1_k32_compile.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(compile_info, f, indent=2)
    print(f"\nCompile info written: {out_json}")


if __name__ == "__main__":
    main()
