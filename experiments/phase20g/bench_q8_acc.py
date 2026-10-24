#!/usr/bin/env python3
"""Phase 20H bench: compile / run / verify / trace a Q8_0 GEMM candidate.

Uses the IRON runtime path (NPUKernel + TraceConfig) so hardware trace is
available, and splits host packing time from device run time.
"""
import argparse
import os
import time

import numpy as np

import aie.iron as iron
from aie.utils import DefaultNPURuntime, HostRuntime, NPUKernel, TraceConfig, tensor
import gen_q8_scale_acc as G


def gen_data(M, K, N, seed=0, uniform=False):
    rng = np.random.default_rng(seed)
    A = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    nb = K // 32
    if uniform:
        As = np.ones((M, nb), np.float32)
        Bs = np.ones((nb, N), np.float32)
    else:
        As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
        Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)
    return A, As, B, Bs


def ref(A, As, B, Bs):
    M, K = A.shape
    N = B.shape[1]
    nb = K // 32
    C = np.zeros((M, N), np.float64)
    for b in range(nb):
        p = A[:, b * 32:(b + 1) * 32].astype(np.int64) @ B[b * 32:(b + 1) * 32, :].astype(np.int64)
        C += p * As[:, b:b + 1].astype(np.float64) * Bs[b:b + 1, :].astype(np.float64)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, default=32)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--trace-size", type=int, default=0)
    ap.add_argument("--uniform", type=int, default=0)
    ap.add_argument("--force", type=int, default=1)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    mt, nt, kc, cols, depth = args.m_tile, args.n_tile, args.k_chunk, args.cols, args.depth

    A, As, B, Bs = gen_data(M, K, N, uniform=bool(args.uniform))
    tp0 = time.perf_counter()
    Ap = G.pack_A(A, As, M, K, mt, kc)
    tp1 = time.perf_counter()
    Bp = G.pack_B(B, Bs, K, N, nt, kc)
    tp2 = time.perf_counter()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")
    ts = args.trace_size

    if args.force or not (os.path.exists(xp) and os.path.exists(ip)):
        d = G.design.specialize(M=M, K=K, N=N, m_tile=mt, n_tile=nt, k_chunk=kc,
                                n_cols=cols, buffer_depth=depth, trace_size=ts)
        tc0 = time.perf_counter()
        d.compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        tcomp = time.perf_counter() - tc0
    else:
        tcomp = 0.0

    Abuf = np.ascontiguousarray(Ap)
    Bbuf = np.ascontiguousarray(Bp)
    Cout = np.zeros(M * N, np.float32)
    buffers = [tensor(Abuf, dtype=Abuf.dtype), tensor(Bbuf, dtype=Bbuf.dtype),
               tensor(Cout, dtype=Cout.dtype)]

    tcfg = None
    if ts > 0:
        tf = os.path.join(args.work_dir, "trace.txt")
        tcfg = TraceConfig(trace_size=ts, trace_file=tf, reuse_output_buffer=True,
                           last_tensor_shape=Cout.shape, last_tensor_dtype=Cout.dtype)
        HostRuntime.prepare_args_for_trace(buffers, tcfg)

    kern = NPUKernel(xp, ip, trace_config=tcfg)
    h = DefaultNPURuntime.load(kern)

    # warmup
    DefaultNPURuntime.run(h, buffers)
    it = max(1, args.iters)
    t0 = time.perf_counter()
    for _ in range(it):
        DefaultNPURuntime.run(h, buffers)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / it * 1e3

    if tcfg is not None:
        tb, _ = HostRuntime.extract_trace_from_args(buffers, tcfg)
        tcfg.write_trace(tb.view(np.uint32))
        print(f"trace -> {tcfg.trace_file}")

    last = np.asarray(buffers[-1])
    if ts > 0:
        C = last.view(np.float32)[:M * N].reshape(M, N)
    else:
        C = last.astype(np.float32).reshape(M, N)

    R = ref(A, As, B, Bs)
    diff = np.abs(C.astype(np.float64) - R)
    denom = np.abs(R)
    bad = int(np.sum(diff > 1e-3 * np.maximum(denom, 1.0)))
    max_rel = float(np.max(diff[np.abs(R) > 1e-6] / np.abs(R[np.abs(R) > 1e-6]))) if np.any(np.abs(R) > 1e-6) else 0.0

    print(f"BENCH M={M} K={K} N={N} mt={mt} nt={nt} kc={kc} cols={cols} depth={depth} "
          f"trace={ts}")
    print(f"  compile={tcomp:.1f}s packA={1e3*(tp1-tp0):.2f}ms packB={1e3*(tp2-tp1):.2f}ms "
          f"run_avg={avg_ms:.2f}ms iters={it}")
    print(f"  gmac_s={M*N*K/avg_ms/1e6:.2f} max_rel={max_rel:.3e} bad={bad}/{M*N}")


if __name__ == "__main__":
    main()
