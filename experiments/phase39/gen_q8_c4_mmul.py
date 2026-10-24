#!/usr/bin/env python3
"""Phase 39C: C4 Q8_0 GEMM using AIE mmul + per-block scales.

Combines gen_q8_scale_mmul.py's mmul<4,8,8> + scale kernel with
gen_q8_scale_multicol.py's N-split across 4 AIE columns.

One XRT submit covers the whole K. The device kernel loops over K-chunks (256),
and within each chunk loops over 32-element blocks applying per-block Q8_0 scales.

Tile-major operand layout (aie::mmul tile order):
  A chunk: (16 * KC/8) tiles of 4x8 int8, then 64*(KC/32) f32 scales
  B chunk: (8  * KC/8) tiles of 8x8 int8, then (KC/32)*64 f32 scales
  C: float [M, N], accumulated across K-chunks

Usage:
  python gen_q8_c4_mmul.py --M 512 --K 1024 --N 3584 --work-dir /tmp/q8c4
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

M_TILE = 64
N_TILE = 64
KC = 256
NBLK = KC // 32
KA = KC // 8
KB = KC // 8

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// at: (16 * KC/8) tiles of 4x8 int8, then 64*(KC/32) f32 scales
// bt: (8  * KC/8) tiles of 8x8 int8, then (KC/32)*64 f32 scales
// c:  64x64 f32 accumulator
extern "C" void q8_mmul_scale(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int KC    = 256;
    constexpr int NBLK  = KC / 32;
    constexpr int KA    = KC / 8;
    constexpr int KB    = KC / 8;
    const int8_t * a_tiles = at;
    const float  * as = (const float *)(at + 64 * KC);
    const int8_t * b_tiles = bt;
    const float  * bs = (const float *)(bt + KC * 64);

    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int bb = 0; bb < NBLK; bb++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 8; j++) {
                MMUL acc;
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(a_tiles + (i * KA + bb * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
                alignas(32) int32_t tmp[32];
                aie::store_v(tmp, acc.to_vector<int32_t>());
                aie::vector<float, 8> bsv = aie::load_v<8>(bs + bb * 64 + j * 8);
                for (int rr = 0; rr < 4; rr++) {
                    int row = i * 4 + rr;
                    float asi = as[row * NBLK + bb];
                    aie::vector<float, 8> pvf = aie::to_float(aie::load_v<8>(tmp + rr * 8));
                    aie::vector<float, 8> f = aie::mul(asi, aie::mul(bsv, pvf).to_vector<float>()).to_vector<float>();
                    float * cp = c + row * 64 + j * 8;
                    aie::store_v(cp, aie::add(aie::load_v<8>(cp), f));
                }
            }
        }
    }
}
"""


def build(M, K, N, n_cols):
    assert M % M_TILE == 0 and N % N_TILE == 0 and K % KC == 0
    assert N % n_cols == 0
    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    n_kc = K // KC

    a_bytes = M_TILE * KC + M_TILE * NBLK * 4
    b_bytes = KC * N_TILE + NBLK * N_TILE * 4
    c_elems = M_TILE * N_TILE

    kern = ExternalFunction(
        "q8_mmul_scale",
        source_string=KERNEL_SRC,
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
            for _ch in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    for col in range(n_cols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=1)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=1)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=1)
        fifo_a.append(fa)
        fifo_b.append(fb)
        fifo_c.append(fc)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern], stack_size=0x1000))

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_cols * n_nt_c * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [TensorAccessPattern(
        tensor_dims=(n_mt * n_kc * a_bytes,),
        offset=mt * n_kc * a_bytes, sizes=[n_kc * a_bytes], strides=[1])
        for mt in range(n_mt)]

    b_taps = [[TensorAccessPattern(
        tensor_dims=(n_cols * n_nt_c * n_kc * b_bytes,),
        offset=(col * n_nt_c + nt) * n_kc * b_bytes, sizes=[n_kc * b_bytes], strides=[1])
        for nt in range(n_nt_c)] for col in range(n_cols)]

    c_taps = [[[TensorAccessPattern(
        tensor_dims=(M, N),
        offset=mt * M_TILE * N + (col * n_nt_c + nt) * N_TILE,
        sizes=[M_TILE, N_TILE], strides=[N, 1])
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
    return iron.Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.cols)

    print(f"[gen_q8_c4_mmul] M={args.M} K={args.K} N={args.N} cols={args.cols}")
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print(f"  xclbin: {compiled[0]} ({os.path.getsize(compiled[0])} bytes)")
    print(f"  insts:  {compiled[1]} ({os.path.getsize(compiled[1])} bytes)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
