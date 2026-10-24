#!/usr/bin/env python3
"""Phase 20G bench: pure int8 matrix-MAC GEMM (int32 out).

Acceptance: exact int32 correctness; report gmac_s vs the vector kernel.
"""
import argparse
import os
import time

import numpy as np

from aie.utils import DefaultNPURuntime, NPUKernel, tensor
import gen_i8_mmul as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, default=16)
    ap.add_argument("--n-tile", type=int, default=32)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()
    M, K, N = args.M, args.K, args.N

    rng = np.random.default_rng(0)
    A = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    Ap = G.pack_A(A, M, K, args.m_tile, args.k_chunk)
    Bp = G.pack_B(B, K, N, args.n_tile, args.k_chunk)

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
    Cout = np.zeros(M * N, np.int32)
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

    C = G.unpack_C(np.asarray(buffers[-1]).astype(np.int64), M, N, args.m_tile, args.n_tile)
    R = A.astype(np.int64) @ B.astype(np.int64)
    bad = int(np.sum(C != R))
    print(f"I8MMUL M={M} K={K} N={N} mt={args.m_tile} nt={args.n_tile} kc={args.k_chunk} depth={args.depth}")
    print(f"  run_avg={avg_ms:.2f}ms gmac_s={M * N * K / avg_ms / 1e6:.2f} bad={bad}/{M * N}")


if __name__ == "__main__":
    main()
