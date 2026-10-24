#!/usr/bin/env python3
"""Phase 20E E3: full-K tiled Q8_0 GEMM, ONE XRT submit per logical GEMM.

Grid: (M/m_tile) x (N/n_tile) output tiles, each accumulating over K/k_chunk
K-chunks.  The whole grid is encoded into one runtime sequence (one instruction
stream -> one XRT submit).

  A chunk: [m_tile, k_chunk] int8 + [m_tile, k_chunk/32] float32
  B chunk: [k_chunk, n_tile] int8 + [k_chunk/32, n_tile] float32
  C tile : [m_tile, n_tile] float32, stays in L1 across all K-chunks
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
#include <aie_api/aie.hpp>

// mmul layout A: [m/4][kc/32][4 s-blocks][4x8] int8, then [m][kc/32] float32
// mmul layout B: [kc/32][n/8][4 s-blocks][8x8] int8, then [kc/32][n] float32
// C: [m][n] float32
extern "C" void mm_chunk_mmul(
    const int8_t * __restrict a_packed,
    const int8_t * __restrict b_packed,
    float * __restrict c,
    int32_t m, int32_t kc, int32_t n)
{
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    constexpr int32_t SA = MMUL::size_A;   // 32
    constexpr int32_t SB = MMUL::size_B;   // 64
    constexpr int32_t SC = MMUL::size_C;   // 32
    constexpr int32_t RS = 4;
    constexpr int32_t TS = 8;

    const int32_t nblk = kc / 32;
    const int32_t n_r = m / RS;
    const int32_t n_t = n / TS;
    const int8_t * a_data = a_packed;
    const float  * a_scales = (const float *)(a_packed + m * kc);
    const int8_t * b_data = b_packed;
    const float  * b_scales = (const float *)(b_packed + kc * n);

    for (int32_t mi = 0; mi < n_r; mi++) {
        for (int32_t ni = 0; ni < n_t; ni++) {
            for (int32_t kb = 0; kb < nblk; kb++) {
                MMUL pacc(aie::zeros<int32_t, SC>());
                for (int32_t l = 0; l < 4; l++) {
                    const int8_t * ap = a_data + ((((mi * nblk) + kb) * 4) + l) * SA;
                    const int8_t * bp = b_data + ((((kb * n_t) + ni) * 4) + l) * SB;
                    aie::vector<int8_t, SA> av = aie::load_v<SA>(ap);
                    aie::vector<int8_t, SB> bv = aie::load_v<SB>(bp);
                    pacc.mac(av, bv);
                }
                int32_t pbuf[SC];
                aie::store_v(pbuf, pacc.template to_vector<int32_t>());
                for (int32_t i = 0; i < RS; i++) {
                    float as_i = a_scales[(mi * RS + i) * nblk + kb];
                    float * crow = c + (mi * RS + i) * n + ni * TS;
                    const float * brow = b_scales + kb * n + ni * TS;
                    for (int32_t j = 0; j < TS; j++)
                        crow[j] += as_i * brow[j] * (float)pbuf[i * TS + j];
                }
            }
        }
    }
}
"""


def build(M, K, N, m_tile, n_tile, k_chunk):
    assert K % k_chunk == 0 and k_chunk % 32 == 0
    assert M % m_tile == 0 and N % n_tile == 0
    n_mt, n_nt, n_kc = M // m_tile, N // n_tile, K // k_chunk
    nblk = k_chunk // 32

    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    c_elems = m_tile * n_tile
    c_bytes = c_elems * 4
    BD_MAX = 16383 * 4
    for nm, v in (("a_bytes", a_bytes), ("b_bytes", b_bytes)):
        if v > BD_MAX:
            raise ValueError(f"{nm}={v} exceeds BD limit {BD_MAX}")

    # L1 budget: ObjectFifo allocates `depth` buffers per FIFO on the compute tile.
    depth_a, depth_b, depth_c = 1, 1, 1
    stack_bytes = 0x2000
    l1_total = stack_bytes + depth_a * a_bytes + depth_b * b_bytes + depth_c * c_bytes
    L1_MAX = 64 * 1024
    print(f"  L1: stack={stack_bytes} A={depth_a * a_bytes} B={depth_b * b_bytes} "
          f"C={depth_c * c_bytes} total={l1_total}/{L1_MAX}")
    if l1_total > L1_MAX:
        raise ValueError(f"L1 budget {l1_total} exceeds {L1_MAX}. "
                         f"Reduce m_tile/n_tile/k_chunk.")

    # BD budget: every ObjectFifo fill becomes its own BD (no reuse).
    n_bd = n_mt * n_nt * (2 * n_kc + 1)
    print(f"  BD: {n_bd} (shim tile limit 16)")
    if n_bd > 16:
        raise ValueError(f"{n_bd} simultaneous BDs exceeds 16")

    fused = ExternalFunction(
        "mm_chunk_mmul",
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

    fifo_a = ObjectFifo(a_ty, name="a", depth=depth_a)
    fifo_b = ObjectFifo(b_ty, name="b", depth=depth_b)
    fifo_c = ObjectFifo(c_ty, name="c", depth=depth_c)

    def core_fn(of_a, of_b, of_c, kern, m, kc, n, nmt, nnt, nkc):
        for _mt in iron_range(nmt):
            for _nt in iron_range(nnt):
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

    worker = Worker(
        core_fn,
        [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), fused,
         m_tile, k_chunk, n_tile, n_mt, n_nt, n_kc],
        stack_size=stack_bytes,
    )

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_kc * n_nt * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [[TensorAccessPattern(
        tensor_dims=(n_mt * n_kc * a_bytes,),
        offset=(mt * n_kc + kc) * a_bytes, sizes=[a_bytes], strides=[1])
        for kc in range(n_kc)] for mt in range(n_mt)]

    b_taps = [[TensorAccessPattern(
        tensor_dims=(n_kc * n_nt * b_bytes,),
        offset=(kc * n_nt + nt) * b_bytes, sizes=[b_bytes], strides=[1])
        for nt in range(n_nt)] for kc in range(n_kc)]

    c_taps = [[TensorAccessPattern(
        tensor_dims=(M, N),
        offset=(mt * m_tile) * N + nt * n_tile,
        sizes=[m_tile, n_tile], strides=[N, 1])
        for nt in range(n_nt)] for mt in range(n_mt)]

    def sequence(A, B, C, a_p, b_p, c_c):
        for mt in range(n_mt):
            for nt in range(n_nt):
                for kc in range(n_kc):
                    a_p.fill(A, tap=a_taps[mt][kc])
                    b_p.fill(B, tap=b_taps[kc][nt])
                c_c.drain(C, tap=c_taps[mt][nt], wait=True)

    rt = Runtime(sequence, [A_ty, B_ty, C_ty,
                            fifo_a.prod(), fifo_b.prod(), fifo_c.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, default=32)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.m_tile, args.n_tile, args.k_chunk)

    print(f"E3 mmul: M={args.M} K={args.K} N={args.N} "
          f"m_tile={args.m_tile} n_tile={args.n_tile} k_chunk={args.k_chunk}")
    t0 = time.time()
    design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time()-t0:.1f}s")
    print("SUCCESS")


if __name__ == "__main__":
    main()
