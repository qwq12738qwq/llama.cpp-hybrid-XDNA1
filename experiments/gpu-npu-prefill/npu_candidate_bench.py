#!/usr/bin/env python3
"""npu_candidate_bench.py -- NPU candidate harness for prefill characterization.

Measures per-operation NPU latency and overhead for candidate offload ops.
Outputs CSV to stdout for parsing by classifier.

Usage:
    python3 npu_candidate_bench.py --op GEMM --M 1024 --K 4096 --N 4096 --iters 10
    python3 npu_candidate_bench.py --op GEMM --M 128 --K 512 --N 4096 --iters 10

Ops:
    GEMM       - INT8 matmul A[M,K] @ B[K,N] -> C[M,N]
    ELEMENTWISE - elementwise ops (estimated from bandwidth)
    RMSNORM    - RMS norm (estimated)

For GEMM, M,K,N must be divisible by tile size (64,64,64) and K <= 512
for direct NPU execution. For K > 512, reports BLOCKED.
"""
import argparse, sys, time, statistics, os, json
import numpy as np

# Try importing iron - fail gracefully if not available
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


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0], "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)], "max": s[-1],
    }


def _get_npu_time(ret):
    """Extract npu_time from iron runtime result."""
    if isinstance(ret, tuple):
        result_obj = ret[1] if len(ret) > 1 else ret
    else:
        result_obj = ret
    npu_ns = getattr(result_obj, "npu_time", None)
    return npu_ns / 1000.0 if npu_ns is not None else None


def measure_gemm(M, K, N, iters=10):
    """Measure INT8 GEMM [M,K] @ [K,N] on NPU.
    Returns dict with timing breakdown or BLOCKED status.
    """
    if not IRON_AVAILABLE:
        return {"status": "NO_IRON", "error": "aie.iron not available"}

    m, k, n = _TILE_M, _TILE_K, _TILE_N
    if M % m != 0 or K % k != 0 or N % n != 0:
        return {"status": "BLOCKED", "error": f"Shape {M}x{K}x{N} not divisible by tile ({m},{k},{n})"}

    # Check if K > 512 (DMA BD limitation)
    if K > 512:
        return {
            "status": "BLOCKED",
            "error": f"K={K} > 512 exceeds DMA BD generation limit (mlir-aie 1.4.1)",
            "note": "Would require tiling K into 512-wide chunks"
        }

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    print(f"[NPU] Building design M={M} K={K} N={N}...", flush=True)

    @iron.jit
    def design(A: In, B: In, C: Out,
               *, M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N):
        return _build_design(M, K, N)

    # Phase 1: One-time init
    t_init = time.perf_counter()
    compiled = design.specialize(M=M, K=K, N=N).compile()
    t_compile = (time.perf_counter() - t_init) * 1e3

    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")
    t_bo = (time.perf_counter() - t_init) * 1e3 - t_compile

    # Phase 2: Cold start
    t_cold = time.perf_counter()
    ret = design(A, B, C, M=M, K=K, N=N)
    t_cold_total = (time.perf_counter() - t_cold) * 1e6
    npu_cold = _get_npu_time(ret)

    # Phase 3: Warm loop
    e2e_times = []
    npu_times = []
    h2d_times = []
    d2h_times = []
    host_overheads = []

    for i in range(iters):
        # H2D: sync input A
        A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        A.data[:] = A_np
        t0 = time.perf_counter_ns()
        A._sync_to_device()
        h2d_us = (time.perf_counter_ns() - t0) / 1000.0
        h2d_times.append(h2d_us)

        # Reset output C
        C.data.fill(0)
        t0 = time.perf_counter_ns()
        C._sync_to_device()
        d2h_reset_us = (time.perf_counter_ns() - t0) / 1000.0

        # Submit + wait
        t0 = time.perf_counter_ns()
        ret = design(A, B, C, M=M, K=K, N=N)
        t_e2e_us = (time.perf_counter_ns() - t0) / 1000.0
        e2e_times.append(t_e2e_us)

        npu_us = _get_npu_time(ret)
        if npu_us is not None:
            npu_times.append(npu_us)
            host_overheads.append(t_e2e_us - npu_us)

        # D2H: read back C
        t0 = time.perf_counter_ns()
        got = C.numpy()
        d2h_us = (time.perf_counter_ns() - t0) / 1000.0
        d2h_times.append(d2h_us + d2h_reset_us)

        # Correctness check every N iters
        if (i + 1) % max(1, iters // 3) == 0:
            ref = A_np.astype(np.int32) @ B_np.astype(np.int32)
            mm = int(np.sum(got != ref))
            if mm != 0:
                print(f"  WARNING: {mm} mismatches at iter {i+1}", flush=True)

    e2e_s = _stats(e2e_times)
    npu_s = _stats(npu_times) if npu_times else None
    h2d_s = _stats(h2d_times)
    d2h_s = _stats(d2h_times)
    host_s = _stats(host_overheads) if host_overheads else None

    total_ops = 2.0 * M * K * N
    result = {
        "status": "OK",
        "op": "GEMM",
        "M": M, "K": K, "N": N,
        "tile_m": m, "tile_k": k, "tile_n": n,
        "total_ops": total_ops,
        "gmac": total_ops / 1e9,
        "compile_ms": t_compile,
        "bo_create_ms": t_bo,
        "init_total_ms": t_compile + t_bo,
        "cold_total_us": t_cold_total,
        "cold_npu_us": npu_cold,
        "warm": {
            "e2e": e2e_s,
            "npu": npu_s,
            "h2d": h2d_s,
            "d2h": d2h_s,
            "host_overhead": host_s,
        },
    }

    if npu_s:
        e2e_median = e2e_s["median"]
        npu_median = npu_s["median"]
        overhead = e2e_median - npu_median
        result["warm"]["overhead_median_us"] = overhead
        result["warm"]["overhead_ratio"] = e2e_median / npu_median if npu_median > 0 else 0
        result["warm"]["e2e_median_us"] = e2e_median
        result["warm"]["npu_median_us"] = npu_median
        result["warm"]["h2d_median_us"] = h2d_s["median"]
        result["warm"]["d2h_median_us"] = d2h_s["median"]
        result["gops_e2e"] = total_ops / (e2e_median * 1e-6) / 1e9
        result["gops_npu"] = total_ops / (npu_median * 1e-6) / 1e9

    return result


def measure_elementwise(M, N, iters=10):
    """Estimate elementwise op (RMSNorm, activation) on NPU.
    These are memory-bound; estimate based on memory bandwidth.
    For now, mark as ESTIMATED since iron doesn't have direct elementwise kernels.
    """
    total_elements = M * N
    bytes_per_element = 4  # FP32
    total_bytes = total_elements * bytes_per_element

    # NPU memory bandwidth estimate: ~10 GB/s for XDNA1
    npu_bw_gbps = 10.0
    npu_compute_ms = (total_bytes / 1e9) / npu_bw_gbps * 1000.0

    # Host overhead estimate from GEMM measurements
    host_overhead_us = 500.0  # conservative estimate

    return {
        "status": "ESTIMATED",
        "op": "ELEMENTWISE",
        "M": M, "N": N,
        "total_elements": total_elements,
        "npu_compute_ms": npu_compute_ms,
        "host_overhead_us": host_overhead,
        "note": "Estimated from memory bandwidth; iron does not expose standalone elementwise kernel"
    }


def measure_rmsnorm(M, N, iters=10):
    """Estimate RMSNorm on NPU.
    RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    This is a reduction + elementwise. Estimate as 2x elementwise.
    """
    return measure_elementwise(M, N, iters)


def _build_design(M, K, N):
    """Build iron Program for INT8 GEMM."""
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
        tensor_dims=(m, n),
        offset=0,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=["GEMM", "ELEMENTWISE", "RMSNORM"])
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if args.op == "GEMM":
        if args.K == 0:
            print("ERROR: --K required for GEMM op", flush=True)
            sys.exit(1)
        result = measure_gemm(args.M, args.K, args.N, args.iters)
    elif args.op == "ELEMENTWISE":
        result = measure_elementwise(args.M, args.N, args.iters)
    elif args.op == "RMSNORM":
        result = measure_rmsnorm(args.M, args.N, args.iters)
    else:
        result = {"status": "UNKNOWN_OP", "error": f"Unknown op: {args.op}"}

    # Output as JSON for easy parsing
    print(json.dumps(result, indent=2), flush=True)

    # Return exit code based on status
    if result.get("status") in ("OK", "ESTIMATED"):
        sys.exit(0)
    else:
        sys.exit(2)  # BLOCKED or error


if __name__ == "__main__":
    main()
