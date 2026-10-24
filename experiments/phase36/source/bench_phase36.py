#!/usr/bin/env python3
"""Phase 36: Qwen Real-Shape XDNA Array GEMM benchmark.

Extends Phase 35 persistent worker infrastructure to:
  - Benchmark real Qwen3.5-0.8B GEMM shapes
  - Compare XDNA vs Vulkan on same shapes
  - M-scaling sweep
  - Breakdown timing (compute vs DMA)

Reuses gen_phase35.py for XDNA compilation and execution.
Uses vulkan_gemm_bench binary for Vulkan comparison.

Usage:
  source /Storage/Cache/ironenv-v143/bin/activate
  python bench_phase36.py \\
    --M 512 --K 1024 --N 3584 --tiles 16 --variant persistent \\
    --work-dir /tmp/p36/ffn_gate_up_t16 \\
    --vulkan-bench /workspace/llama.cpp/experiments/phase21/vulkan_gemm_bench \\
    --iters 20 --warmup 5
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_GEN = os.path.join(_BASE, "gen_phase35.py")
_P35_SRC = "/workspace/llama.cpp/experiments/phase35/source"
sys.path.insert(0, _BASE)
sys.path.insert(0, _P35_SRC)

import aie.iron as iron
from aie.iron import In, Out, CompileTime, kernels, ObjectFifo, Worker, TaskGroup, Runtime
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.helpers.taplib import TensorAccessPattern
from gen_phase35 import build, cpu_int8_gemm, _TM, _TK, _TN

_TILE_COUNT_TO_ROWS_COLS = {
    1:  (1, 1),
    2:  (1, 2),
    4:  (1, 4),
    8:  (2, 4),
    16: (4, 4),
}


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0],
        "mean": sum(samples) / n,
        "median": s[n // 2],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
    }


def _run_xdna_design(design, A_np, B_np, M, K, N, n_iters=20, warmup=5):
    """Run XDNA design and return (host_stats, npu_stats, mismatches, C_out)."""
    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int8, device="npu")
    ref = A_np.astype(np.int32) @ B_np.astype(np.int32)

    host_times = []
    npu_times = []
    mismatches_list = []

    for i in range(warmup):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        design(A, B, C, M=M, K=K, N=N)

    for i in range(n_iters):
        C = iron.tensor((M, N), dtype=np.int32, device="npu")
        t0 = time.perf_counter()
        ret = design(A, B, C, M=M, K=K, N=N)
        e2e_us = (time.perf_counter() - t0) * 1e6
        host_times.append(e2e_us)

        npu_ns = None
        if isinstance(ret, tuple):
            for item in ret:
                if hasattr(item, "npu_time"):
                    npu_ns = item.npu_time
                    break
        elif hasattr(ret, "npu_time"):
            npu_ns = ret.npu_time

        if npu_ns is not None:
            npu_times.append(npu_ns / 1000.0)

        got = C.numpy()
        mm = int(np.sum(got != ref))
        mismatches_list.append(mm)

        if (i + 1) % 10 == 0:
            npu_str = f"npu={npu_times[-1]:.0f}us" if npu_times else "npu=N/A"
            print(f"  {i + 1}/{n_iters}  host={e2e_us:.0f}us  {npu_str}  mm={mm}")

    host_stats = _stats(host_times)
    npu_stats = _stats(npu_times) if npu_times else None
    max_mm = max(mismatches_list) if mismatches_list else -1
    return host_stats, npu_stats, max_mm, ref


def _run_vulkan_bench(vulkan_bench_path, M, K, N, iters=10):
    """Run vulkan_gemm_bench and return (ms, gops)."""
    cmd = [vulkan_bench_path, str(M), str(K), str(N), str(iters)]
    env = os.environ.copy()
    env["PATH"] = "/workspace/llama.cpp/build-vk/bin:" + env.get("PATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = r.stdout + r.stderr
    ms = None
    gops = None
    for line in output.splitlines():
        if line.startswith("CSV "):
            parts = line.split(",")
            if len(parts) >= 6:
                ms = float(parts[4])
                gops = float(parts[5])
                break
    if ms is None:
        for line in output.splitlines():
            if "ms=" in line and "gops=" in line:
                import re
                m_ms = re.search(r"ms=([\d.]+)", line)
                m_gops = re.search(r"gops=([\d.]+)", line)
                if m_ms and m_gops:
                    ms = float(m_ms.group(1))
                    gops = float(m_gops.group(1))
                    break
    return ms, gops, output


def _build_zero_design(M, K, N, n_rows, n_cols, fifo_depth=2):
    """Same topology as baseline but with zero kernel only (no matmul)."""
    m, k, n = _TM, _TK, _TN
    n_cores = n_rows * n_cols
    dtype_in = np.int8
    dtype_out = np.int32

    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=dtype_in, output_dtype=dtype_out,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    nb_m_per_col = M // m // n_rows
    nb_n_per_col = N // n // n_cols
    n_tiles_per_core = nb_m_per_col * nb_n_per_col

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    b_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    c_l2_ty = np.ndarray[(m * n * n_rows,), np.dtype[dtype_out]]
    a_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos = []
    A_l2l1_fifos = []
    B_l3l2_fifos = []
    B_l2l1_fifos = []
    C_l1l2_fifos = [[] for _ in range(n_rows)]
    C_l2l3_fifos = []

    n_shim_A = n_rows if n_cols > n_rows else n_cols
    n_A_per_shim = n_rows // n_cols if n_cols < n_rows else 1

    for i in range(n_shim_A):
        a_l3l2 = ObjectFifo(
            np.ndarray[(m * k * n_A_per_shim,), np.dtype[dtype_in]],
            name=f"A_L3L2_{i}", depth=fifo_depth,
        )
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_per_shim
        stop_row = start_row + n_A_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        a_dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * (stop_row - start_row)
        a_tmp = a_l3l2.cons().split(
            of_offsets,
            obj_types=[a_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp)

    for col in range(n_cols):
        b_l3l2 = ObjectFifo(b_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(obj_type=b_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims)
        )

        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(c_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims)
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_rows)]
        c_tmp = c_l2l3.prod().join(
            of_offsets,
            obj_types=[c_l1_ty] * n_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_rows)],
            depths=[fifo_depth] * n_rows,
        )
        for j in range(n_rows):
            C_l1l2_fifos[j].append(c_tmp[j])

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        n_rows, n_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )
    flat_workers = [w for row in workers for w in row]

    A_tiles = []
    for mt in range(nb_m_per_col):
        offset = mt * m * n_rows * K
        a_tap = TensorAccessPattern(
            tensor_dims=(M, K),
            offset=offset,
            sizes=[m * n_rows, K],
            strides=[K, 1],
        )
        A_tiles.append(a_tap)

    B_tiles = []
    for col in range(n_cols):
        col_n_offset = col * (N // n_cols)
        b_tap = TensorAccessPattern(
            tensor_dims=(K, N),
            offset=col_n_offset,
            sizes=[K, N // n_cols],
            strides=[N, 1],
        )
        B_tiles.append(b_tap)

    C_tiles = []
    for col in range(n_cols):
        col_n_offset = col * (N // n_cols)
        for mt in range(nb_m_per_col):
            for nt in range(nb_n_per_col):
                offset = (mt * m * n_rows) * N + col_n_offset + nt * n
                c_tap = TensorAccessPattern(
                    tensor_dims=(M, N),
                    offset=offset,
                    sizes=[m * n_rows, n],
                    strides=[N, 1],
                )
                C_tiles.append(c_tap)

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]
    nA, nB = len(A_prods), len(B_prods)

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        c_index = 0
        for mt in range(nb_m_per_col):
            for col in range(n_cols):
                tg = TaskGroup()
                for row in range(n_rows):
                    tile_offset = (mt * n_shim_A + row) % len(A_tiles)
                    A_hs[row].fill(A, tap=A_tiles[tile_offset], group=tg)
                B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                c_index += 1
                tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def _run_breakdown(M, K, N, n_cores, work_dir, variant="baseline", fifo_depth=2,
                   n_iters=10, warmup=3):
    """Run breakdown measurement: full vs DMA-only to estimate compute fraction."""
    n_rows, n_cols = _TILE_COUNT_TO_ROWS_COLS[n_cores]
    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    iron.set_current_device(from_name("npu", n_cols=4))

    # Full GEMM
    @iron.jit
    def design_full(A: In, B: In, C: Out, *,
                    M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
                    mt: CompileTime[int] = _TM, kt: CompileTime[int] = _TK, nt: CompileTime[int] = _TN,
                    rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
                    din: CompileTime[str] = "i8", dout: CompileTime[str] = "i32", act: CompileTime[str] = "none"):
        return build(M, K, N, n_cores, variant, fifo_depth=fifo_depth,
                      dtype_in=np.int8, dtype_out=np.int32)

    xp = os.path.join(work_dir, "design.xclbin")
    ip = os.path.join(work_dir, "design.insts.bin")
    design_full.specialize(M=M, K=K, N=N, mt=_TM, kt=_TK, nt=_TN,
                           rows=n_rows, cols=n_cols, din="i8", dout="i32", act="none"
                           ).compile(xclbin_path=xp, inst_path=ip)

    print(f"  [breakdown] Running full GEMM ({n_iters} iters)...")
    host_full, npu_full, mm_full, _ = _run_xdna_design(
        design_full, A_np, B_np, M, K, N,
        n_iters=n_iters, warmup=warmup,
    )

    # DMA-only (zero kernel, skip matmul)
    zero_dir = work_dir + "_zero"
    os.makedirs(zero_dir, exist_ok=True)

    @iron.jit
    def design_zero(A: In, B: In, C: Out, *,
                    M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
                    mt: CompileTime[int] = _TM, kt: CompileTime[int] = _TK, nt: CompileTime[int] = _TN,
                    rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
                    din: CompileTime[str] = "i8", dout: CompileTime[str] = "i32", act: CompileTime[str] = "none"):
        return _build_zero_design(M, K, N, n_rows, n_cols, fifo_depth=fifo_depth)

    xpz = os.path.join(zero_dir, "design.xclbin")
    ipz = os.path.join(zero_dir, "design.insts.bin")
    design_zero.specialize(M=M, K=K, N=N, mt=_TM, kt=_TK, nt=_TN,
                           rows=n_rows, cols=n_cols, din="i8", dout="i32", act="none"
                           ).compile(xclbin_path=xpz, inst_path=ipz)

    print(f"  [breakdown] Running DMA-only ({n_iters} iters)...")
    host_zero, npu_zero, mm_zero, _ = _run_xdna_design(
        design_zero, A_np, B_np, M, K, N,
        n_iters=n_iters, warmup=warmup,
    )

    breakdown = {
        "full_npu_mean_us": npu_full["mean"] if npu_full else None,
        "zero_npu_mean_us": npu_zero["mean"] if npu_zero else None,
        "full_host_mean_us": host_full["mean"],
        "zero_host_mean_us": host_zero["mean"],
    }
    if npu_full and npu_zero:
        compute_us = max(0, npu_full["mean"] - npu_zero["mean"])
        breakdown["compute_us"] = compute_us
        breakdown["dma_us"] = npu_zero["mean"]
        breakdown["compute_fraction"] = compute_us / npu_full["mean"] if npu_full["mean"] > 0 else 0
    else:
        breakdown["compute_us"] = None
        breakdown["dma_us"] = None
        breakdown["compute_fraction"] = None

    print(f"  [breakdown] full={npu_full['mean']:.0f}us  zero={npu_zero['mean']:.0f}us  "
          f"compute_frac={breakdown['compute_fraction']:.2%}" if breakdown["compute_fraction"] else "")

    return breakdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--tiles", type=int, required=True, choices=[1, 2, 4, 8, 16])
    ap.add_argument("--variant", type=str, default="persistent",
                    choices=["baseline", "depth", "2d", "persistent"])
    ap.add_argument("--fifo-depth", type=int, default=2)
    ap.add_argument("--n-persist-iters", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--vulkan-bench", type=str, default=None,
                    help="Path to vulkan_gemm_bench binary")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--dtype-in", type=str, choices=["i8", "i16", "bf16"], default="i8")
    ap.add_argument("--dtype-out", type=str, choices=["i8", "i16", "i32", "bf16", "f32"], default="i32")
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cores = args.tiles
    variant = args.variant
    n_rows, n_cols = _TILE_COUNT_TO_ROWS_COLS[n_cores]

    iron.set_current_device(from_name("npu", n_cols=4))

    @iron.jit
    def design(A: In, B: In, C: Out, *,
               M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
               mt: CompileTime[int] = _TM, kt: CompileTime[int] = _TK, nt: CompileTime[int] = _TN,
               rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
               din: CompileTime[str] = args.dtype_in, dout: CompileTime[str] = args.dtype_out,
               act: CompileTime[str] = "none"):
        return build(M, K, N, n_cores, variant, fifo_depth=args.fifo_depth,
                      n_persist_iters=args.n_persist_iters,
                      dtype_in=np.int8, dtype_out=np.int32)

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    if not os.path.exists(xp):
        print(f"[phase36] Compiling M={M} K={K} N={N} tiles={n_cores} ({n_rows}x{n_cols}) "
              f"variant={variant} fifo_depth={args.fifo_depth}...")
        t0 = time.time()
        compiled = design.specialize(
            M=M, K=K, N=N, mt=_TM, kt=_TK, nt=_TN,
            rows=n_rows, cols=n_cols, din=args.dtype_in, dout=args.dtype_out, act="none",
        ).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"[phase36] compiled in {time.time() - t0:.1f}s -> {compiled[0]}")
    else:
        print(f"[phase36] Using cached xclbin: {xp}")
        compiled = design.specialize(
            M=M, K=K, N=N, mt=_TM, kt=_TK, nt=_TN,
            rows=n_rows, cols=n_cols, din=args.dtype_in, dout=args.dtype_out, act="none",
        ).compile(xclbin_path=xp, inst_path=ip)

    rng = np.random.default_rng(42)
    A_np = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
    B_np = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

    print(f"[phase36] Running XDNA benchmark: variant={variant} tiles={n_cores} ({n_rows}x{n_cols})")
    host_stats, npu_stats, max_mm, ref = _run_xdna_design(
        design, A_np, B_np, M, K, N,
        n_iters=args.iters, warmup=args.warmup,
    )

    total_ops = 2.0 * M * K * N
    e2e_gops = total_ops / (host_stats["mean"] * 1e-6) / 1e9
    npu_gops = (total_ops / (npu_stats["mean"] * 1e-6) / 1e9) if npu_stats else None

    breakdown = {}
    if args.breakdown:
        breakdown = _run_breakdown(M, K, N, n_cores, args.work_dir, variant, args.fifo_depth)

    # Vulkan comparison
    vulkan_result = {}
    if args.vulkan_bench:
        print(f"[phase36] Running Vulkan benchmark M={M} K={K} N={N}...")
        vk_ms, vk_gops, vk_output = _run_vulkan_bench(args.vulkan_bench, M, K, N, iters=10)
        if vk_ms is not None:
            vulkan_result = {
                "vulkan_ms": vk_ms,
                "vulkan_gops": vk_gops,
                "vulkan_us": vk_ms * 1000,
            }
            if npu_gops is not None:
                vulkan_result["speedup_vs_vulkan"] = vk_gops / npu_gops if npu_gops > 0 else None
            print(f"[phase36] Vulkan: {vk_ms:.4f}ms  {vk_gops:.2f} GOPS")
        else:
            print(f"[phase36] Vulkan benchmark failed")
            vulkan_result = {"error": "benchmark_failed", "output": vk_output}

    result = {
        "shape": {"M": M, "K": K, "N": N},
        "tiles": n_cores,
        "rows": n_rows,
        "cols": n_cols,
        "variant": variant,
        "fifo_depth": args.fifo_depth,
        "total_ops": total_ops,
        "host_e2e_us": host_stats,
        "npu_exec_us": npu_stats,
        "e2e_gops": e2e_gops,
        "npu_gops": npu_gops,
        "max_mismatches": max_mm,
        "correct": max_mm == 0,
        "iters": args.iters,
        "warmup": args.warmup,
        "breakdown": breakdown,
        "vulkan": vulkan_result,
    }

    raw_dir = os.path.join(args.work_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"bench_{n_cores}.json")
    with open(raw_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Phase36 {variant} {M}x{K}x{N} tiles={n_cores} ({n_rows}x{n_cols}) ===")
    print(f"Total ops: {total_ops:.0f}")
    print(f"Host e2e: mean={host_stats['mean']:.0f}us  median={host_stats['median']:.0f}us")
    if npu_stats:
        print(f"NPU exec: mean={npu_stats['mean']:.0f}us  median={npu_stats['median']:.0f}us")
    print(f"E2E GOPS: {e2e_gops:.2f}")
    if npu_gops:
        print(f"NPU GOPS: {npu_gops:.2f}")
    if breakdown:
        print(f"Breakdown: compute={breakdown.get('compute_us', 'N/A')}us  "
              f"DMA={breakdown.get('dma_us', 'N/A')}us  "
              f"compute_frac={breakdown.get('compute_fraction', 'N/A')}")
    if vulkan_result:
        print(f"Vulkan: {vulkan_result.get('vulkan_us', 'N/A'):.0f}us  "
              f"{vulkan_result.get('vulkan_gops', 'N/A'):.2f} GOPS  "
              f"speedup={vulkan_result.get('speedup_vs_vulkan', 'N/A')}")
    print(f"Correctness: {max_mm} mismatches ({'PASS' if max_mm == 0 else 'FAIL'})")
    print(f"Results: {raw_path}")

    sys.exit(0 if max_mm == 0 else 1)


if __name__ == "__main__":
    main()
