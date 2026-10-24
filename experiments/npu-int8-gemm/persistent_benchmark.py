#!/usr/bin/env python3
# persistent_benchmark.py -- Phoenix XDNA1 persistent INT8 GEMM benchmark
#
# Measures cold start + warm loop latency for a persistent XRT/AIE context.
# All one-time setup (compile, xclbin load, BO creation, DMA init) done ONCE.
# Then N iterations run on the SAME BOs with only input data refresh per iter.
#
# Reuse the proven design pattern from int8_gemm.py.
import argparse, sys, time, statistics, numpy as np
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

def _get_npu_time(ret):
    if isinstance(ret, tuple):
        result_obj = ret[1] if len(ret) > 1 else ret
    else:
        result_obj = ret
    npu_ns = getattr(result_obj, "npu_time", None)
    return npu_ns / 1000.0 if npu_ns is not None else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    m, k, n = _TILE_M, _TILE_K, _TILE_N
    r, s, t = 4, 8, 8
    if M % m != 0 or K % k != 0 or N % n != 0:
        sys.exit(f"Shape {M}x{K}x{N} not divisible by tile size ({m},{k},{n}).")

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    print(f"=== Persistent Benchmark: M={M} K={K} N={N} tile=({m},{k},{n}) ===", flush=True)

    @iron.jit
    def design(A: In, B: In, C: Out,
               *, M: CompileTime[int]=M, K: CompileTime[int]=K, N: CompileTime[int]=N):
        return _build_design(M, K, N)

    # --- Phase 1: One-time initialization ---
    print("[INIT] Building design...", flush=True)
    t_init = time.perf_counter()
    t0 = time.perf_counter()
    compiled = design.specialize(M=M, K=K, N=N).compile()
    t_compile = (time.perf_counter() - t0) * 1e3
    print(f"  compile: {t_compile:.1f} ms  xclbin={compiled[0]}", flush=True)

    t0 = time.perf_counter()
    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    C = iron.tensor((M, N), dtype=np.int32, device="npu")
    t_bo = (time.perf_counter() - t0) * 1e3
    print(f"  BO creation: {t_bo:.1f} ms", flush=True)
    t_init_total = (time.perf_counter() - t_init) * 1e3
    print(f"  init total: {t_init_total:.1f} ms", flush=True)

    # --- Phase 2: Cold start ---
    print("[COLD] First execution...", flush=True)
    t_cold = time.perf_counter()
    ret = design(A, B, C, M=M, K=K, N=N)
    t_cold_us = (time.perf_counter() - t_cold) * 1e6
    npu_cold = _get_npu_time(ret)

    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)
    got = C.numpy()
    mm_cold = int(np.sum(got != ref))
    npu_str = f"{npu_cold:.1f}" if npu_cold else "None"
    print(f"  cold_total_latency: {t_cold_us:.1f} us  npu_time={npu_str}", flush=True)
    print(f"  cold correctness: {mm_cold} mismatches / {M*N}", flush=True)
    assert mm_cold == 0, "Cold start correctness failed"

    # --- Phase 3: Warm loop ---
    print(f"[WARM] {args.iters} iterations...", flush=True)
    e2e_times = []
    npu_times = []
    a_sync_times = []
    c_reset_times = []
    host_ohead_times = []

    for i in range(args.iters):
        # 1. Write new input activation to A
        A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        A.data[:] = A_np
        t0 = time.perf_counter_ns()
        A._sync_to_device()
        a_sync_times.append((time.perf_counter_ns() - t0))

        # Reset output C (zero + sync to device)
        C.data.fill(0)
        t0 = time.perf_counter_ns()
        C._sync_to_device()
        c_reset_times.append((time.perf_counter_ns() - t0))

        # 2. Submit + wait (timed as e2e)
        t0 = time.perf_counter_ns()
        ret = design(A, B, C, M=M, K=K, N=N)
        t_e2e_ns = time.perf_counter_ns() - t0
        e2e_times.append(t_e2e_ns / 1000.0)

        npu_us = _get_npu_time(ret)
        if npu_us is not None:
            npu_times.append(npu_us)
            host_ohead_times.append(t_e2e_ns / 1000.0 - npu_us)

        # 4. Verify correctness every N iterations
        if (i + 1) % 100 == 0:
            got = C.numpy()
            ref_i = A_np.astype(np.int32) @ B_np.astype(np.int32)
            mm = int(np.sum(got != ref_i))
            if mm != 0:
                print(f"  WARNING: mismatch at iter {i+1}: {mm}/{M*N}", flush=True)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.iters} done", flush=True)

    # --- Phase 4: Final correctness ---
    got = C.numpy()
    ref_final = A_np.astype(np.int32) @ B_np.astype(np.int32)
    mm_final = int(np.sum(got != ref_final))
    print(f"\n[VERIFY] Final correctness: {mm_final} mismatches / {M*N}", flush=True)
    assert mm_final == 0, "Final warm correctness failed"
    print("PASS - all 0 mismatches", flush=True)

    # --- Phase 5: Statistics ---
    e2e_s = _stats(e2e_times)
    npu_s = _stats(npu_times) if npu_times else None
    a_sync_s = _stats([t/1000.0 for t in a_sync_times])
    c_reset_s = _stats([t/1000.0 for t in c_reset_times])

    total_ops = 2.0 * M * K * N
    print(f"\n=== Results: M={M} K={K} N={N} ===", flush=True)
    print(f"MAC ops: {total_ops:.0f}  ({total_ops/1e9:.2f} GMAC)", flush=True)
    print(f"\n--- Initialization (one-time) ---")
    print(f"  compile: {t_compile:.1f} ms", flush=True)
    print(f"  BO creation: {t_bo:.1f} ms", flush=True)
    print(f"  init total: {t_init_total:.1f} ms", flush=True)
    print(f"\n--- Cold start ---")
    print(f"  cold_total_latency: {t_cold_us:.1f} us", flush=True)
    if npu_cold is not None:
        print(f"  cold NPU execution (npu_time): {npu_cold:.1f} us", flush=True)

    print(f"\n--- Warm end-to-end latency (us, {args.iters} iters) ---")
    for key in ("min", "mean", "median", "p95", "p99", "max"):
        print(f"  {key}={e2e_s[key]:.1f}", flush=True)

    if npu_s:
        print(f"\n--- Warm NPU execution latency (npu_time, us) ---")
        for key in ("min", "mean", "median", "p95", "p99", "max"):
            print(f"  {key}={npu_s[key]:.1f}", flush=True)

        e2e_mean = e2e_s["mean"]
        npu_mean = npu_s["mean"]
        overhead = e2e_mean - npu_mean
        ratio = e2e_mean / npu_mean
        e2e_gops = total_ops / (e2e_mean * 1e-6) / 1e9
        npu_gops = total_ops / (npu_mean * 1e-6) / 1e9
        print(f"\n--- Host overhead ---")
        print(f"  warm end-to-end (mean):      {e2e_mean:.1f} us", flush=True)
        print(f"  warm NPU execution (mean):   {npu_mean:.1f} us", flush=True)
        print(f"  warm host/runtime overhead:  {overhead:.1f} us ({(overhead/e2e_mean*100):.1f}% of e2e)", flush=True)
        print(f"  overhead_ratio (e2e/NPU):    {ratio:.2f}x", flush=True)
        print(f"  GOPS (host e2e): {e2e_gops:.2f}", flush=True)
        print(f"  GOPS (NPU exec): {npu_gops:.2f}", flush=True)

        print(f"\n--- Host breakdown (us, mean) ---")
        print(f"  A sync_to_device:          {a_sync_s['mean']:.1f}  (min={a_sync_s['min']:.1f} max={a_sync_s['max']:.1f})", flush=True)
        print(f"  C zero+sync_to_device:     {c_reset_s['mean']:.1f}  (min={c_reset_s['min']:.1f} max={c_reset_s['max']:.1f})", flush=True)
        print(f"  XRT submit+wait overhead:  {overhead:.1f} us  (e2e - npu_time)", flush=True)

        print(f"\n=== VERDICT ===", flush=True)
        if ratio > 10:
            print(f"warm host overhead ({overhead:.0f}us) >> NPU execution ({npu_mean:.0f}us), ratio={ratio:.1f}x", flush=True)
            print("CONCLUSION: NPU invocation/runtime overhead dominated", flush=True)
            print("ACTION: Hold off on llama.cpp / ggml-xdna prototype", flush=True)
        else:
            print(f"warm host overhead ({overhead:.0f}us) ~ NPU execution ({npu_mean:.0f}us), ratio={ratio:.2f}x", flush=True)
            print("CONCLUSION: persistent invocation path is potentially viable", flush=True)
            print("ACTION: proceed to llama.cpp / ggml-xdna prototype", flush=True)
    else:
        print(f"\n  (npu_time not available from runtime)", flush=True)

if __name__ == "__main__":
    main()