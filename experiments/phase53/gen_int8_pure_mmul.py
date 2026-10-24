#!/usr/bin/env python3
"""Phase 53: pure int8 full-K GEMM (no device-side scales).

Adapted from experiments/phase40/gen_q8_c4_mmul_v2.py: removes the per-32
scale arrays from A/B and the device-side scale multiply, accumulates int32
in the AIE, outputs int32. Scales (per-row/per-col) are applied on the host.

Tile: M_TILE=32, N_TILE=64, KC=256, n_cols=4.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False

M_TILE = 32
N_TILE = 64
KC = 256
MA = M_TILE // 4
NA = N_TILE // 8
KA = KC // 8
KB = KC // 8

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void int8_pure_mmul(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    int32_t      * __restrict c)
{
    constexpr int MT = 32;
    constexpr int NT = 64;
    constexpr int KCC = 256;
    constexpr int MA = MT / 4;
    constexpr int NA = NT / 8;
    constexpr int KA = KCC / 8;
    constexpr int KB = KCC / 8;
    const int8_t * a_tiles = at;
    const int8_t * b_tiles = bt;

    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        for (int j = 0; j < NA; j++) {
            MMUL acc;
            for (int kk = 0; kk < KA; kk++) {
                aie::vector<int8_t, 32> av = aie::load_v<32>(a_tiles + (i * KA + kk) * 32);
                aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (kk * NA + j) * 64);
                acc.mac(av, bv);
            }
            aie::store_v(c + (i * NA + j) * 32, acc.to_vector<int32_t>());
        }
    }
}
"""


def build(M, K, N, n_cols, fifo_depth=2):
    assert M % M_TILE == 0 and N % N_TILE == 0 and K % KC == 0
    assert N % n_cols == 0
    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    assert n_nt_c >= 1
    n_kc = K // KC

    a_bytes = M_TILE * KC
    b_bytes = KC * N_TILE
    c_elems = M_TILE * N_TILE

    kern = ExternalFunction(
        "int8_pure_mmul",
        source_string=KERNEL_SRC,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.int32]],
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.int32]]

    fifo_a, fifo_b, fifo_c = [], [], []
    workers = []

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt_c):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    for col in range(n_cols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=fifo_depth)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=fifo_depth)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=1)
        fifo_a.append(fa)
        fifo_b.append(fb)
        fifo_c.append(fc)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern]))

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_cols * n_nt_c * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.int32]]

    a_taps = [__import__("aie.helpers.taplib", fromlist=["TensorAccessPattern"]).TensorAccessPattern(
        tensor_dims=(n_mt * n_kc * a_bytes,),
        offset=mt * n_kc * a_bytes, sizes=[n_kc * a_bytes], strides=[1])
        for mt in range(n_mt)]

    Tap = __import__("aie.helpers.taplib", fromlist=["TensorAccessPattern"]).TensorAccessPattern
    b_taps = [[Tap(
        tensor_dims=(n_cols * n_nt_c * n_kc * b_bytes,),
        offset=(col * n_nt_c + nt) * n_kc * b_bytes, sizes=[n_kc * b_bytes], strides=[1])
        for nt in range(n_nt_c)] for col in range(n_cols)]

    c_taps = [[[Tap(
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
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cols = args.cols

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(M, K, N, n_cols)

    print("Compiling...")
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print(f"  xclbin: {compiled[0]} ({os.path.getsize(compiled[0])} bytes)")
    print(f"  insts:  {compiled[1]} ({os.path.getsize(compiled[1])} bytes)")
    print("SUCCESS")


if __name__ == "__main__":
    main()