#!/usr/bin/env python3
"""Phase 20G bench: Q8_0 scale-aware matrix-MAC GEMM.

Acceptance: matches the Q8_0 reference; report gmac_s vs the old vector kernel.
"""
import argparse
import os
import time

import numpy as np

from aie.utils import DefaultNPURuntime, NPUKernel, tensor
import gen_q8_mmul2 as G


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
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--uniform", type=int, default=0)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()
    M, K, N = args.M, args.K, args.N

    A, As, B, Bs = gen_data(M, K, N, uniform=bool(args.uniform))
    Ap = G.pack_A(A, As, M, K, args.m_tile, args.k_chunk)
    Bp = G.pack_B(B, Bs, K, N, args.n_tile, args.k_chunk)

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")
    if not (os.path.exists(xp) and os.path.exists(ip)):
        d = G.design.specialize(M=M, K=K, N=N, m_tile=args.m_tile, n_tile=args.n_tile,
                                k_chunk=args.k_chunk, depth=args.depth)
        t0 = time.time()
        d.compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"compile={time.time() - t0:.1f}s")

    Abuf = np.ascontiguousarray(Ap)
    Bbuf = np.ascontiguousarray(Bp)
    Cout = np.zeros(M * N, np.float32)
    buffers = [tensor(Abuf, dtype=Abuf.dtype), tensor(Bbuf, dtype=Bbuf.dtype),
               tensor(Cout, dtype=Cout.dtype)]
    kern = NPUKernel(xp, ip)
    h = DefaultNPURuntime.load(kern)
    DefaultNPURuntime.run(h, buffers)
    it = max(1, args.iters)
    t0 = time.perf_counter()
    for _ in range(it):
        DefaultNPURuntime.run(h, buffers)
    avg_ms = (time.perf_counter() - t0) / it * 1e3

    C = np.asarray(buffers[-1]).astype(np.float64).reshape(M, N)
    R = ref(A, As, B, Bs)
    diff = np.abs(C - R)
    bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(R), 1.0)))
    max_rel = float(np.max(diff[np.abs(R) > 1e-6] / np.abs(R[np.abs(R) > 1e-6]))) if np.any(np.abs(R) > 1e-6) else 0.0
    print(f"Q8MMUL M={M} K={K} N={N} mt={args.m_tile} nt={args.n_tile} kc={args.k_chunk} "
          f"depth={args.depth} uniform={args.uniform}")
    print(f"  run_avg={avg_ms:.2f}ms gmac_s={M * N * K / avg_ms / 1e6:.2f} "
          f"max_rel={max_rel:.3e} bad={bad}/{M * N}")


if __name__ == "__main__":
    main()
