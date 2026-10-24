#!/usr/bin/env python3
"""Phase 20E E4: multi-tile full-K Q8_0 GEMM, ONE XRT submit.

M is partitioned across `n_tiles` compute tiles (row dimension).  B is shared
by all tiles through a broadcast ObjectFifo (one fill, N consumers).  Each tile
produces its own [m_tile, n_tile] C block.

  A[i] chunk: [m_tile, k_chunk] int8 + [m_tile, k_chunk/32] float32  (per tile)
  B chunk   : [k_chunk, n_tile] int8 + [k_chunk/32, n_tile] float32  (broadcast)
  C[i] block: [m_tile, n_tile] float32
"""
import argparse
import os
import time

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
from aie.iron.controlflow import range_ as iron_range
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config

KERNEL_SRC = r"""
#include <stdint.h>

extern "C" void mm_chunk(
    const int8_t * __restrict a_packed,
    const int8_t * __restrict b_packed,
    float * __restrict c,
    int32_t m, int32_t kc, int32_t n)
{
    const int32_t nblk = kc / 32;
    const int8_t * a_data = a_packed;
    const float  * a_scales = (const float *)(a_packed + m * kc);
    const int8_t * b_data = b_packed;
    const float  * b_scales = (const float *)(b_packed + kc * n);

    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int32_t kb = 0; kb < nblk; kb++) {
                int32_t psum = 0;
                const int8_t * ar = a_data + i * kc + kb * 32;
                const int8_t * bc = b_data + (kb * 32) * n + j;
                for (int32_t l = 0; l < 32; l++)
                    psum += (int32_t)ar[l] * (int32_t)bc[l * n];
                acc += a_scales[i * nblk + kb] * b_scales[kb * n + j] * (float)psum;
            }
            c[i * n + j] += acc;
        }
    }
}
"""


def build(M, K, N, m_tile, n_tile, k_chunk, n_tiles):
    assert K % k_chunk == 0 and k_chunk % 32 == 0
    assert M % (m_tile * n_tiles) == 0 and N % n_tile == 0
    n_mt = M // m_tile            # one M-tile per compute tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    assert n_mt == n_tiles, f"M/m_tile={n_mt} must equal n_tiles={n_tiles}"
    nblk = k_chunk // 32

    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    c_elems = m_tile * n_tile
    c_bytes = c_elems * 4
    BD_MAX = 16383 * 4
    for nm, v in (("a_bytes", a_bytes), ("b_bytes", b_bytes)):
        if v > BD_MAX:
            raise ValueError(f"{nm}={v} exceeds BD limit {BD_MAX}")

    depth_a, depth_b, depth_c = 1, 1, 1
    stack_bytes = 0x2000
    l1_per_tile = (stack_bytes + depth_a * a_bytes + depth_b * b_bytes
                   + depth_c * c_bytes)
    L1_MAX = 64 * 1024
    print(f"  per-tile L1: stack={stack_bytes} A={depth_a * a_bytes} "
          f"B={depth_b * b_bytes} C={depth_c * c_bytes} total={l1_per_tile}/{L1_MAX}")
    if l1_per_tile > L1_MAX:
        raise ValueError(f"per-tile L1 {l1_per_tile} exceeds {L1_MAX}")

    # B is broadcast (1 BD covers all consumers); A/C are per-tile.
    n_bd = n_tiles * n_kc + n_kc + n_tiles * n_nt
    print(f"  BD: A={n_tiles * n_kc} B={n_kc}(broadcast) C={n_tiles * n_nt} "
          f"total={n_bd}/16")
    if n_bd > 16:
        raise ValueError(f"{n_bd} simultaneous BDs exceeds 16")

    fused = ExternalFunction(
        "mm_chunk",
        source_string=KERNEL_SRC,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.float32]],
            np.int32, np.int32, np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]

    fifo_a = [ObjectFifo(a_ty, name=f"a{i}", depth=depth_a) for i in range(n_tiles)]
    fifo_b = ObjectFifo(b_ty, name="b", depth=depth_b)
    fifo_c = [ObjectFifo(c_ty, name=f"c{i}", depth=depth_c) for i in range(n_tiles)]

    def core_fn(of_a, of_b, of_c, kern, m, kc, n, nkc):
        for _nt in iron_range(n_nt):
            elem_c = of_c.acquire(1)
            for x in iron_range(m * n):
                elem_c[x] = 0.0
            for _kc in iron_range(nkc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c, m, kc, n)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    workers = [
        Worker(
            core_fn,
            [fifo_a[i].cons(), fifo_b.cons(), fifo_c[i].prod(), fused,
             m_tile, k_chunk, n_tile, n_kc],
            stack_size=stack_bytes,
        )
        for i in range(n_tiles)
    ]

    A_ty = np.ndarray[(n_tiles * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [[TensorAccessPattern(
        tensor_dims=(n_tiles * n_kc * a_bytes,),
        offset=(i * n_kc + kc) * a_bytes, sizes=[a_bytes], strides=[1])
        for kc in range(n_kc)] for i in range(n_tiles)]

    b_taps = [TensorAccessPattern(
        tensor_dims=(n_kc * b_bytes,),
        offset=kc * b_bytes, sizes=[b_bytes], strides=[1])
        for kc in range(n_kc)]

    c_taps = [[TensorAccessPattern(
        tensor_dims=(M, N),
        offset=(i * m_tile) * N + nt * n_tile,
        sizes=[m_tile, n_tile], strides=[N, 1])
        for nt in range(n_nt)] for i in range(n_tiles)]

    def sequence(A, B, C, a_p, b_p, c_c):
        for kc in range(n_kc):
            b_p.fill(B, tap=b_taps[kc])
            for i in range(n_tiles):
                a_p[i].fill(A, tap=a_taps[i][kc])
        for i in range(n_tiles):
            for nt in range(n_nt):
                c_c[i].drain(C, tap=c_taps[i][nt], wait=True)

    rt = Runtime(sequence, [A_ty, B_ty, C_ty,
                            [f.prod() for f in fifo_a], fifo_b.prod(),
                            [f.cons() for f in fifo_c]])
    return iron.Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, required=True)
    ap.add_argument("--n-tile", type=int, required=True)
    ap.add_argument("--k-chunk", type=int, required=True)
    ap.add_argument("--n-tiles", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.m_tile, args.n_tile,
                     args.k_chunk, args.n_tiles)

    print(f"E4 multi: M={args.M} K={args.K} N={args.N} m_tile={args.m_tile} "
          f"n_tile={args.n_tile} k_chunk={args.k_chunk} tiles={args.n_tiles}")
    t0 = time.time()
    design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time()-t0:.1f}s")

    import glob
    cores = sorted(glob.glob(os.path.join(os.path.dirname(ip) or ".",
                                          "*.prj", "main_core_*.elf")))
    if not cores:
        cores = sorted(glob.glob(os.path.join(args.work_dir, "**",
                                              "main_core_*.elf"), recursive=True))
    print(f"compute tile artifacts ({len(cores)}): "
          f"{[os.path.basename(c) for c in cores]}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
