#!/usr/bin/env python3
"""Phase 20G step 5c: scale-aware Q8_0 full-K GEMM using the AIE matrix MAC unit.

One kernel call consumes a whole K-chunk (KC = 256 = 8 Q8_0 blocks). Inside, per
32-block, the device runs an aie::mmul<4,8,8> int8 matmul (the AIE matrix MAC
unit) into an int32 tile, applies the Q8_0 scales with vector float ops, and
accumulates into the float C tile.

Tile-major operand layout (aie::mmul tile order):
  A chunk: (16 * KC/8) tiles of 4x8 int8, then 64*KC/32 f32 scales
  B chunk: (8  * KC/8) tiles of 8x8 int8, then KC/32*64 f32 scales
  C: float [M, N], accumulated across K-chunks

Usage:
  python gen_q8_scale_mmul.py --M 512 --K 1024 --N 1024 --work-dir /tmp/q8m
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


def build(M, K, N):
    assert M % M_TILE == 0 and N % N_TILE == 0 and K % KC == 0
    n_mt, n_nt, n_chunk = M // M_TILE, N // N_TILE, K // KC

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

    fifo_a = ObjectFifo(a_ty, name="a", depth=1)
    fifo_b = ObjectFifo(b_ty, name="b", depth=1)
    fifo_c = ObjectFifo(c_ty, name="c", depth=1)

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0.0
            for _ch in iron_range(n_chunk):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(core_fn, [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), kern], stack_size=0x1000)

    A_ty = np.ndarray[(n_mt * n_chunk * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_nt * n_chunk * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [TensorAccessPattern(
        tensor_dims=(n_mt * n_chunk * a_bytes,),
        offset=mt * n_chunk * a_bytes, sizes=[n_chunk * a_bytes], strides=[1])
        for mt in range(n_mt)]
    b_taps = [TensorAccessPattern(
        tensor_dims=(n_nt * n_chunk * b_bytes,),
        offset=nt * n_chunk * b_bytes, sizes=[n_chunk * b_bytes], strides=[1])
        for nt in range(n_nt)]
    c_taps = [[TensorAccessPattern(
        tensor_dims=(M, N),
        offset=mt * M_TILE * N + nt * N_TILE,
        sizes=[M_TILE, N_TILE], strides=[N, 1])
        for nt in range(n_nt)] for mt in range(n_mt)]

    def sequence(A, B, C, a_p, b_p, c_c):
        for mt in range(n_mt):
            for nt in range(n_nt):
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
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N)

    print(f"mmul: M={args.M} K={args.K} N={args.N}")
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print(f"  xclbin: {compiled[0]} ({os.path.getsize(compiled[0])} bytes)")
    print(f"  insts:  {compiled[1]} ({os.path.getsize(compiled[1])} bytes)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
