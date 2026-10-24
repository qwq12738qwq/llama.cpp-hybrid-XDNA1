#!/usr/bin/env python3
"""Phase 20G step 5: IRON generator for a tiled, scale-aware Q8_0 full-K GEMM.

One XRT submit covers the whole K. The AIE core loops over K-blocks (32 elements)
on the device and applies per-block scales, so there is no host K chunking and no
per-block submit.

Host-visible operands are tile-packed (this is the packing adapter of step 4/5):
  A_packed: (M//m)*nb tiles, each = [m*32 int8 data][m f32 scale]
  B_packed: (N//n)*nb tiles, each = [32*n int8 data][n f32 scale]
  C       : f32 [M, N]

C[m,n] = sum_blk ( sum_{k in blk} A[m,k]*B[k,n] ) * As[m,blk] * Bs[blk,n]

Usage:
  python gen_q8_scale_gemm.py --M 64 --K 1024 --N 64 --work-dir /tmp/q8s
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

// One K-block (32) of a Q8_0 GEMM with per-block scales applied on the device.
// a: [m*32 int8][m f32], b: [32*n int8][n f32], c: [m*n f32] accumulator.
extern "C" void q8_block_mm(
    const int8_t * __restrict a,
    const int8_t * __restrict b,
    float        * __restrict c,
    int32_t m, int32_t n)
{
    const int8_t * ad = a;
    const float  * as = (const float *)(a + m * 32);
    const int8_t * bd = b;
    const float  * bs = (const float *)(b + 32 * n);

    for (int32_t i = 0; i < m; i++) {
        const float as_i = as[i];
        const int8_t * arow = ad + i * 32;
        for (int32_t j = 0; j < n; j++) {
            int32_t psum = 0;
            for (int32_t l = 0; l < 32; l++) {
                psum += (int32_t)arow[l] * (int32_t)bd[l * n + j];
            }
            c[i * n + j] += as_i * bs[j] * (float)psum;
        }
    }
}
"""


def build(M, K, N, m_tile, n_tile, k_tile=32):
    nb = K // k_tile
    m, n = m_tile, n_tile
    assert M % m == 0 and N % n == 0 and K % k_tile == 0

    a_tile_bytes = m * k_tile + m * 4
    b_tile_bytes = k_tile * n + n * 4
    a_tiles = (M // m) * nb
    b_tiles = (N // n) * nb

    a_ty = np.ndarray[(a_tile_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_tile_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(m * n,), np.dtype[np.float32]]

    # 2D host tensors so a single fill can push all nb K-block tiles at once.
    A_ty = np.ndarray[(a_tiles, a_tile_bytes), np.dtype[np.int8]]
    B_ty = np.ndarray[(b_tiles, b_tile_bytes), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    kern = ExternalFunction(
        "q8_block_mm",
        source_string=KERNEL_SRC,
        arg_types=[a_ty, b_ty, c_ty, np.int32, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    fifo_a = ObjectFifo(a_ty, name="a", depth=2)
    fifo_b = ObjectFifo(b_ty, name="b", depth=2)
    fifo_c = ObjectFifo(c_ty, name="c", depth=2)

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range((M // m) * (N // n)):
            elem_c = of_c.acquire(1)
            for i in iron_range(m * n):
                elem_c[i] = 0.0
            for _blk in iron_range(nb):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c, m, n)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(core_fn, [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), kern], stack_size=0x2000)

    # one tap per M-row / N-col covering all nb K-block tiles
    a_taps = [TensorAccessPattern(tensor_dims=(a_tiles, a_tile_bytes), offset=mt * nb * a_tile_bytes,
                                  sizes=[nb, a_tile_bytes], strides=[a_tile_bytes, 1])
              for mt in range(M // m)]
    b_taps = [TensorAccessPattern(tensor_dims=(b_tiles, b_tile_bytes), offset=nt * nb * b_tile_bytes,
                                  sizes=[nb, b_tile_bytes], strides=[b_tile_bytes, 1])
              for nt in range(N // n)]

    c_taps = []
    for mt in range(M // m):
        row = []
        for nt in range(N // n):
            row.append(TensorAccessPattern(tensor_dims=(M, N), offset=mt * m * N + nt * n,
                                           sizes=[m, n], strides=[N, 1]))
        c_taps.append(row)

    def sequence(A, B, C, a_p, b_p, c_c):
        for mt in range(M // m):
            for nt in range(N // n):
                tg = iron.TaskGroup()
                a_p.fill(A, tap=a_taps[mt], group=tg)
                b_p.fill(B, tap=b_taps[nt], group=tg)
                c_c.drain(C, tap=c_taps[mt][nt], group=tg, wait=True)
                tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, fifo_a.prod(), fifo_b.prod(), fifo_c.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, default=64)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.m_tile, args.n_tile)

    print(f"Building scale-aware Q8_0 full-K: M={args.M} K={args.K} N={args.N} "
          f"m_tile={args.m_tile} n_tile={args.n_tile}")
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print(f"  xclbin: {compiled[0]} ({os.path.getsize(compiled[0])} bytes)")
    print(f"  insts:  {compiled[1]} ({os.path.getsize(compiled[1])} bytes)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
