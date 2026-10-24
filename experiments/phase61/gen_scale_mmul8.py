#!/usr/bin/env python3
"""Phase 61: scale-preserving Q8_0 GEMM kernel with a 16-lane float accumulator.

The phase 60 production kernel (phase 40M `q8_c4_mmul_v2`, engine
`c4_mmul_v2`) keeps Q8_0's per-32-block scales:

    C[m,n] = sum_b sA[m,b] * sW[b,n] * ( sum_{k in b} A_q[m,k] * W_q[k,n] )

but it flushes the C tile once per 32-block, so the scale multiply and the C
read-modify-write are repeated NBLK times per output element.

This kernel keeps the whole K chunk in a float accumulator in registers:

  * native `aie::mmul<8,8,8>` (the v64acc is the hardware's own accumulator
    width, so the 4-mul chain of a 32-block is exact - the 4x8 formulation
    breaks on long chains),
  * the per-32-block int32 partial is converted to float, multiplied by the
    rank-1 (row x column) block scale and added to a 16-lane (512-bit) float
    accumulator,
  * the C tile is touched once per (i,j) per K chunk instead of once per
    (i,j,block),
  * two independent (i,j) chains so the matrix unit has two dependency chains.

Numerics are identical to the production kernel: the same int32 partial sums
and the same per-block scale products, only the summation order of the C tile
changes (register accumulation first, one C read-modify-write per chunk).

Usage:
  gen_scale_mmul8.py --M 512 --K 1024 --N 3584 --cols 4 \
      --m-tile 64 --n-tile 64 --k-tile 256 --cache-dir <dir>
"""
import argparse
import os
import time

import numpy as np

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False


KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// Q8_0 scale-preserving GEMM, native 8x8 int8 matrix unit, 16-lane float
// accumulator (phase 61).
//
// A chunk: (MA * KC/8) tiles of 8x8 int8, then (MT * KC/32) f32 row scales
// B chunk: (NA * KC/8) tiles of 8x8 int8, then (KC/32 * NT) f32 col scales
// C:       MT x NT f32, accumulated over the K chunks
//
// The 8x8 tile is four 16-lane groups; group g holds rows 2g and 2g+1 of the
// eight, so the block scale of a group is
//     [ sA[row 2g] * sW[col 0..7] , sA[row 2g+1] * sW[col 0..7] ]
// which is the rank-1 product of the per-row and the per-column block scale.
extern "C" void q8_c4_mmul_v3(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT   = %(MT)d;
    constexpr int NT   = %(NT)d;
    constexpr int KCC  = %(KCC)d;
    constexpr int NBLK = KCC / 32;
    constexpr int MA   = MT / 8;
    constexpr int NA   = NT / 8;
    constexpr int KA   = KCC / 8;
    constexpr int KB   = KCC / 8;
    const int8_t * a_tiles = at;
    const float  * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float  * bsc = (const float *)(bt + KCC * NT);

    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        const int8_t * ai = a_tiles + i * KA * 64;
        const float  * asci = asc + i * 8 * NBLK;
        for (int j = 0; j + 1 < NA; j += 2) {
            const int8_t * bj0 = b_tiles + j * KB * 64;
            const int8_t * bj1 = b_tiles + (j + 1) * KB * 64;
            aie::vector<float, 16> f0[4], f1[4];
            for (int g = 0; g < 4; g++) {
                f0[g] = aie::broadcast<float, 16>(0.0f);
                f1[g] = aie::broadcast<float, 16>(0.0f);
            }
            for (int bb = 0; bb < NBLK; bb++) {
                const int8_t * ap = ai + bb * 4 * 64;
                aie::vector<int8_t, 64> a0 = aie::load_v<64>(ap);
                aie::vector<int8_t, 64> a1 = aie::load_v<64>(ap + 64);
                aie::vector<int8_t, 64> a2 = aie::load_v<64>(ap + 128);
                aie::vector<int8_t, 64> a3 = aie::load_v<64>(ap + 192);
                const int8_t * bp0 = bj0 + bb * 4 * 64;
                aie::vector<int8_t, 64> b00 = aie::load_v<64>(bp0);
                aie::vector<int8_t, 64> b01 = aie::load_v<64>(bp0 + 64);
                aie::vector<int8_t, 64> b02 = aie::load_v<64>(bp0 + 128);
                aie::vector<int8_t, 64> b03 = aie::load_v<64>(bp0 + 192);
                const int8_t * bp1 = bj1 + bb * 4 * 64;
                aie::vector<int8_t, 64> b10 = aie::load_v<64>(bp1);
                aie::vector<int8_t, 64> b11 = aie::load_v<64>(bp1 + 64);
                aie::vector<int8_t, 64> b12 = aie::load_v<64>(bp1 + 128);
                aie::vector<int8_t, 64> b13 = aie::load_v<64>(bp1 + 192);

                aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
                MMUL acc0(zero);
                MMUL acc1(zero);
                acc0.mac(a0, b00);
                acc0.mac(a1, b01);
                acc0.mac(a2, b02);
                acc0.mac(a3, b03);
                acc1.mac(a0, b10);
                acc1.mac(a1, b11);
                acc1.mac(a2, b12);
                acc1.mac(a3, b13);

                aie::vector<float, 64> p0 = aie::to_float(acc0.to_vector<int32_t>());
                aie::vector<float, 64> p1 = aie::to_float(acc1.to_vector<int32_t>());

                aie::vector<float, 8> bs0 = aie::load_v<8>(bsc + bb * NT + j * 8);
                aie::vector<float, 8> bs1 = aie::load_v<8>(bsc + bb * NT + (j + 1) * 8);
                aie::vector<float, 16> w0 = aie::broadcast<float, 16>(0.0f);
                w0.insert<8>(0, bs0);
                w0.insert<8>(1, bs0);
                aie::vector<float, 16> w1 = aie::broadcast<float, 16>(0.0f);
                w1.insert<8>(0, bs1);
                w1.insert<8>(1, bs1);

                const float * ap2 = asci + bb;
                for (int g = 0; g < 4; g++) {
                    aie::vector<float, 16> a16 = aie::broadcast<float, 16>(0.0f);
                    a16.insert<8>(0, aie::broadcast<float, 8>(ap2[(2 * g) * NBLK]));
                    a16.insert<8>(1, aie::broadcast<float, 8>(ap2[(2 * g + 1) * NBLK]));
                    aie::vector<float, 16> sc0 = aie::mul(a16, w0).to_vector<float>();
                    aie::vector<float, 16> sc1 = aie::mul(a16, w1).to_vector<float>();
                    f0[g] = aie::add(f0[g], aie::mul(p0.template extract<16>(g), sc0).to_vector<float>());
                    f1[g] = aie::add(f1[g], aie::mul(p1.template extract<16>(g), sc1).to_vector<float>());
                }
            }
            for (int g = 0; g < 4; g++) {
                float * c0 = c + (i * 8 + 2 * g + 0) * NT + j * 8;
                float * c1 = c + (i * 8 + 2 * g + 1) * NT + j * 8;
                float * c2 = c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8;
                float * c3 = c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8;
                aie::store_v(c0, aie::add(aie::load_v<8>(c0), f0[g].template extract<8>(0)));
                aie::store_v(c1, aie::add(aie::load_v<8>(c1), f0[g].template extract<8>(1)));
                aie::store_v(c2, aie::add(aie::load_v<8>(c2), f1[g].template extract<8>(0)));
                aie::store_v(c3, aie::add(aie::load_v<8>(c3), f1[g].template extract<8>(1)));
            }
        }
    }
}
"""


def build(M, K, N, n_cols, m_tile, n_tile, k_tile, fifo_depth=1, b_fifo_depth=1,
          c_fifo_depth=1, stack_size=0x2000):
    assert M % m_tile == 0 and N % n_tile == 0 and K % k_tile == 0
    assert N % n_cols == 0
    n_mt = M // m_tile
    n_nt_c = (N // n_cols) // n_tile
    n_kc = K // k_tile
    nblk = k_tile // 32

    a_bytes = m_tile * k_tile + m_tile * nblk * 4
    b_bytes = k_tile * n_tile + nblk * n_tile * 4
    c_elems = m_tile * n_tile

    lmem = fifo_depth * a_bytes + b_fifo_depth * b_bytes + c_fifo_depth * c_elems * 4
    if lmem > 64 * 1024:
        raise ValueError(f"LMEM budget {lmem} exceeds 64KB")

    src = KERNEL_SRC % {"MT": m_tile, "NT": n_tile, "KCC": k_tile}
    kern = ExternalFunction(
        "q8_c4_mmul_v3",
        source_string=src,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.float32]],
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]

    fifo_a, fifo_b, fifo_c = [], [], []
    workers = []

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt_c):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0.0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    for col in range(n_cols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=fifo_depth)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=b_fifo_depth)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=c_fifo_depth)
        fifo_a.append(fa)
        fifo_b.append(fb)
        fifo_c.append(fc)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern],
                              stack_size=stack_size))

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_cols * n_nt_c * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * a_bytes,),
                                  offset=mt * n_kc * a_bytes,
                                  sizes=[n_kc * a_bytes], strides=[1])
              for mt in range(n_mt)]
    b_taps = [[TensorAccessPattern(tensor_dims=(n_cols * n_nt_c * n_kc * b_bytes,),
                                   offset=(col * n_nt_c + nt) * n_kc * b_bytes,
                                   sizes=[n_kc * b_bytes], strides=[1])
               for nt in range(n_nt_c)] for col in range(n_cols)]
    c_taps = [[[TensorAccessPattern(tensor_dims=(M, N),
                                    offset=mt * m_tile * N + (col * n_nt_c + nt) * n_tile,
                                    sizes=[m_tile, n_tile], strides=[N, 1])
                for nt in range(n_nt_c)] for mt in range(n_mt)] for col in range(n_cols)]

    def sequence(A, B, C, *fifos):
        a_p = fifos[0:n_cols]
        b_p = fifos[n_cols:2 * n_cols]
        c_c = fifos[2 * n_cols:3 * n_cols]
        for mt in range(n_mt):
            for col in range(n_cols):
                for nt in range(n_nt_c):
                    tg = iron.TaskGroup()
                    a_p[col].fill(A, tap=a_taps[mt], group=tg)
                    b_p[col].fill(B, tap=b_taps[col][nt], group=tg)
                    c_c[col].drain(C, tap=c_taps[col][mt][nt], group=tg, wait=True)
                    tg.finish()

    prod_cons = []
    for col in range(n_cols):
        prod_cons.append(fifo_a[col].prod())
    for col in range(n_cols):
        prod_cons.append(fifo_b[col].prod())
    for col in range(n_cols):
        prod_cons.append(fifo_c[col].cons())
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--m-tile", type=int, default=64)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-tile", type=int, default=256)
    ap.add_argument("--suffix", type=str, default="Q8SCALEV_MMUL8")
    ap.add_argument("--cache-dir", type=str, required=True)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cols = args.cols

    out = os.path.join(args.cache_dir,
                       f"M{M}_K{K}_N{N}_C{n_cols}_{args.suffix}")
    os.makedirs(out, exist_ok=True)
    xp = os.path.join(out, "design.xclbin")
    ip = os.path.join(out, "design.insts.bin")
    pp = os.path.join(out, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(M, K, N, n_cols, args.m_tile, args.n_tile, args.k_tile)

    print(f"Phase61 scale mmul8 kernel: M={M} K={K} N={N} cols={n_cols} "
          f"tile={args.m_tile}x{args.n_tile}x{args.k_tile} -> {out}")
    t0 = time.time()
    design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print("SUCCESS")


if __name__ == "__main__":
    main()
