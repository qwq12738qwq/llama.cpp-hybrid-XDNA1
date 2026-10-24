#!/usr/bin/env python3
"""npu_workload.py -- Minimal persistent NPU INT8 GEMM for overlap experiment.

Extracted from experiments/npu-int8-gemm/persistent_benchmark.py.
Reuses the proven 128x512x4096 INT8 GEMM design that passes correctness.

All compile/BO setup happens at construction; .warmup() runs a warm iteration.
The .run() method executes one GEMM and returns timing + correctness.
"""
import time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.iron.controlflow import range_

_TILE_M = 64
_TILE_K = 64
_TILE_N = 64


def _build_design(M, K, N):
    m, k, n = _TILE_M, _TILE_K, _TILE_N
    matmul_kernel = mm(dim_m=m, dim_k=k, dim_n=n,
                       input_dtype=np.int8, output_dtype=np.int32, vectorized=True)
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
    worker = iron.Worker(
        core_fn,
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel],
    )
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
    )[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))
    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // m):
            tg = iron.TaskGroup()
            a_prod.fill(A, tap=a_taps[tile_row], group=tg)
            b_prod.fill(B, tap=b_tap, group=tg)
            c_cons.drain(C, tap=c_taps[tile_row], group=tg, wait=True)
            tg.finish()
    rt = iron.Runtime(
        sequence,
        [A_ty, B_ty, C_ty, fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()],
    )
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


class NPUWorkload:
    """Persistent NPU INT8 GEMM workload. Pre-compiles at construction."""

    def __init__(self, M=128, K=512, N=4096, seed=42):
        self.M, self.K, self.N = M, K, N
        assert M % 64 == 0 and K % 64 == 0 and N % 64 == 0
        assert M % 4 == 0 and K % 8 == 0 and N % 8 == 0

        @iron.jit
        def design(A: In, B: In, C: Out,
                   *, M: CompileTime[int]=M, K: CompileTime[int]=K, N: CompileTime[int]=N):
            return _build_design(M, K, N)

        self.design = design
        self.compiled = design.specialize(M=M, K=K, N=N).compile()

        rng = np.random.default_rng(seed)
        self.A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        self.B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

        self.A = iron.tensor(self.A_np, dtype=np.int8, device="npu")
        self.B = iron.tensor(self.B_np, dtype=np.int8, device="npu")

        self.ref = self.A_np.astype(np.int32) @ self.B_np.astype(np.int32)

    def warmup(self):
        C = iron.tensor((self.M, self.N), dtype=np.int32, device="npu")
        ret = self.design(self.A, self.B, C, M=self.M, K=self.K, N=self.N)
        got = C.numpy()
        mm = int(np.sum(got != self.ref))
        assert mm == 0, f"Warmup correctness failed: {mm} mismatches"
        npu_ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", None)
        return npu_ns / 1e6 if npu_ns else None

    def run(self):
        """Execute one NPU GEMM. Returns dict with timing and result."""
        C = iron.tensor((self.M, self.N), dtype=np.int32, device="npu")
        wall_start = time.perf_counter_ns()
        ret = self.design(self.A, self.B, C, M=self.M, K=self.K, N=self.N)
        wall_end = time.perf_counter_ns()

        npu_ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", None)
        got = C.numpy()
        mismatches = int(np.sum(got != self.ref))

        return {
            "wall_start_ns": wall_start,
            "wall_end_ns": wall_end,
            "npu_exec_ns": npu_ns,
            "wall_exec_ns": wall_end - wall_start,
            "mismatches": mismatches,
        }
