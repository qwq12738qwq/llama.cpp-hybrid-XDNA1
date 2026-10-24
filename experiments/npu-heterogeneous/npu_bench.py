#!/usr/bin/env python3
"""npu_bench.py -- NPU MUL_MAT / GEMM benchmark using iron.

Requires iron (aie.iron) available in the environment.
M, K, N must be divisible by tile size (64, 64, 64).
K <= 512 for direct execution (DMA BD generation limit).

Usage:
    python3 npu_bench.py --M 128 --K 512 --N 4096 --iters 20 --warmup 3
    python3 npu_bench.py --M 2 --K 1024 --N 16 --iters 20
"""
import argparse, json, sys, time, os, statistics
import numpy as np

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out
    from aie.iron.kernels import mm
    from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
    from aie.iron.controlflow import range_
    IRON_AVAILABLE = True
except ImportError:
    IRON_AVAILABLE = False

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

def _get_npu_time_us(ret):
    if isinstance(ret, tuple):
        result_obj = ret[1] if len(ret) > 1 else ret
    else:
        result_obj = ret
    npu_ns = getattr(result_obj, "npu_time", None)
    return npu_ns / 1000.0 if npu_ns is not None else None

def _stats(samples):
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0], "mean": sum(samples) / len(samples),
        "median": s[n // 2],
        "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)], "max": s[-1],
    }

class NPUBenchmark:
    def __init__(self, M, K, N, seed=42):
        self.M, self.K, self.N = M, K, N
        if not IRON_AVAILABLE:
            raise RuntimeError("aie.iron not available")
        if M % _TILE_M != 0 or K % _TILE_K != 0 or N % _TILE_N != 0:
            raise ValueError(f"Shape {M}x{K}x{N} not divisible by tile ({_TILE_M},{_TILE_K},{_TILE_N})")
        if K > 512:
            raise ValueError(f"K={K} > 512 exceeds DMA BD generation limit")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.A_np = self.rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        self.B_np = self.rng.integers(-128, 127, size=(K, N), dtype=np.int8)
        self.ref = self.A_np.astype(np.int32) @ self.B_np.astype(np.int32)

        @iron.jit
        def design(A: In, B: In, C: Out,
                   *, M: CompileTime[int]=M, K: CompileTime[int]=K, N: CompileTime[int]=N):
            return _build_design(M, K, N)

        self.design = design
        t0 = time.perf_counter()
        self.compiled = design.specialize(M=M, K=K, N=N).compile()
        self.init_time_ms = (time.perf_counter() - t0) * 1000.0

        self.A = iron.tensor(self.A_np, dtype=np.int8, device="npu")
        self.B = iron.tensor(self.B_np, dtype=np.int8, device="npu")

    def warmup(self):
        C = iron.tensor((self.M, self.N), dtype=np.int32, device="npu")
        ret = self.design(self.A, self.B, C, M=self.M, K=self.K, N=self.N)
        got = C.numpy()
        mm = int(np.sum(got != self.ref))
        if mm != 0:
            raise AssertionError(f"Warmup correctness failed: {mm} mismatches")
        return _get_npu_time_us(ret)

    def run_iteration(self, new_input=True):
        if new_input:
            A_np = self.rng.integers(-128, 127, size=(self.M, self.K), dtype=np.int8)
        else:
            A_np = self.A_np

        B_np = self.B_np  # B is constant

        A = iron.tensor(A_np, dtype=np.int8, device="npu")
        B = iron.tensor(B_np, dtype=np.int8, device="npu")
        C = iron.tensor((self.M, self.N), dtype=np.int32, device="npu")

        t_wall_start = time.perf_counter_ns()
        ret = self.design(A, B, C, M=self.M, K=self.K, N=self.N)
        t_wall_end = time.perf_counter_ns()

        npu_us = _get_npu_time_us(ret)
        got = C.numpy()
        ref = A_np.astype(np.int32) @ B_np.astype(np.int32)
        mismatches = int(np.sum(got != ref))

        return {
            "wall_start_ns": t_wall_start,
            "wall_end_ns": t_wall_end,
            "wall_us": (t_wall_end - t_wall_start) / 1000.0,
            "npu_us": npu_us,
            "mismatches": mismatches,
        }

    def benchmark(self, iters=20, warmup_iters=3):
        # Warmup (creates fresh tensors each time)
        for _ in range(warmup_iters):
            r = self.run_iteration(new_input=False)
            if r["mismatches"] != 0:
                raise AssertionError(f"Warmup correctness failed: {r['mismatches']} mismatches")

        # Timed runs
        wall_times = []
        npu_times = []
        host_overheads = []
        all_mismatches = []

        for i in range(iters):
            r = self.run_iteration(new_input=(i % 5 == 0))
            wall_times.append(r["wall_us"])
            if r["npu_us"] is not None:
                npu_times.append(r["npu_us"])
                host_overheads.append(r["wall_us"] - r["npu_us"])
            all_mismatches.append(r["mismatches"])

        max_mismatches = max(all_mismatches) if all_mismatches else 0
        cold_npu_us = None
        if npu_times:
            cold_npu_us = npu_times[0] if warmup_iters == 0 else None

        wall_median = _stats(wall_times).get("median", 0)
        npu_median = _stats(npu_times).get("median", 0) if npu_times else 0
        overhead_median = _stats(host_overheads).get("median", 0) if host_overheads else 0

        ops = 2.0 * self.M * self.K * self.N
        result = {
            "status": "OK",
            "backend": "NPU",
            "op": "GEMM",
            "M": self.M, "K": self.K, "N": self.N,
            "tile_m": _TILE_M, "tile_k": _TILE_K, "tile_n": _TILE_N,
            "dtype": "int8/int32",
            "iters": iters,
            "warmup": warmup_iters,
            "init_ms": self.init_time_ms,
            "cold_npu_us": cold_npu_us,
            "mismatches": max_mismatches,
            "wall_us": _stats(wall_times),
            "npu_us": _stats(npu_times) if npu_times else {},
            "host_overhead_us": _stats(host_overheads) if host_overheads else {},
            "wall_ms_median": wall_median / 1000.0,
            "npu_ms_median": npu_median / 1000.0,
            "overhead_ms_median": overhead_median / 1000.0,
        }
        if npu_median > 0:
            result["gops_npu"] = ops / (npu_median * 1e-6) / 1e9
            result["gops_wall"] = ops / (wall_median * 1e-6) / 1e9
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not IRON_AVAILABLE:
        print(json.dumps({"status": "NO_IRON", "error": "aie.iron not available"}))
        sys.exit(2)

    try:
        bench = NPUBenchmark(args.M, args.K, args.N, args.seed)
        warmup_npu_us = bench.warmup()
        result = bench.benchmark(args.iters, args.warmup)
        result["warmup_npu_us"] = warmup_npu_us

        # Compute derived metrics
        wall_median = result["wall_us"].get("median", 0)
        npu_median = result["npu_us"].get("median", 0)
        overhead_median = result["host_overhead_us"].get("median", 0)

        result["wall_ms_median"] = wall_median / 1000.0
        result["npu_ms_median"] = npu_median / 1000.0
        result["overhead_ms_median"] = overhead_median / 1000.0

        ops = 2.0 * args.M * args.K * args.N
        if npu_median > 0:
            result["gops_npu"] = ops / (npu_median * 1e-6) / 1e9
            result["gops_wall"] = ops / (wall_median * 1e-6) / 1e9

        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "OK" and result["mismatches"] == 0 else 1)
    except (ValueError, AssertionError, RuntimeError) as e:
        print(json.dumps({"status": "BLOCKED", "M": args.M, "K": args.K, "N": args.N,
                          "error": str(e)}))
        sys.exit(2)

if __name__ == "__main__":
    main()
