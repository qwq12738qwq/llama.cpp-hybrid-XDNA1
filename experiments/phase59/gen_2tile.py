#!/usr/bin/env python3
"""Phase 59: 2-tile GEMM (2 rows, 1 col) - no sharing, each tile independent.

Each tile handles M/2 x N output. A and B are loaded separately per tile.
This wastes DMA bandwidth but uses 2 compute tiles.
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction, TaskGroup
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False

M_TILE = 32
N_TILE = 64
KC = 256
NBLK = KC // 32
MA = M_TILE // 4
NA = N_TILE // 8
KA = KC // 8
KB = KC // 8

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void int8_pure_mmul_v2(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    int32_t      * __restrict c)
{
    constexpr int MT   = 32;
    constexpr int NT   = 64;
    constexpr int KCC  = 256;
    constexpr int NBLK = KCC / 32;
    constexpr int MA   = MT / 4;
    constexpr int NA   = NT / 8;
    constexpr int KA   = KCC / 8;
    constexpr int KB   = KCC / 8;
    const int8_t * a_tiles = at;
    const int8_t * b_tiles = bt;

    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int bb = 0; bb < NBLK; bb++) {
        for (int i = 0; i < MA; i++) {
            for (int j = 0; j < NA; j++) {
                MMUL acc;
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(a_tiles + (i * KA + bb * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
                alignas(32) int32_t tmp[32];
                aie::store_v(tmp, acc.to_vector<int32_t>());
                for (int rr = 0; rr < 4; rr++) {
                    for (int ll = 0; ll < 8; ll++) {
                        c[(i * 4 + rr) * NT + j * 8 + ll] += tmp[rr * 8 + ll];
                    }
                }
            }
        }
    }
}
"""


def build(M, K, N, n_rows, n_cols, fifo_depth=2):
    assert M % (M_TILE * n_rows) == 0 and N % (N_TILE * n_cols) == 0 and K % KC == 0

    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    assert n_nt_c >= 1
    n_kc = K // KC
    nb_m_per_row = M // M_TILE // n_rows
    nb_n_per_col = N // N_TILE // n_cols
    n_tiles_per_core = nb_m_per_row * nb_n_per_col

    a_bytes = M_TILE * KC
    b_bytes = KC * N_TILE
    c_elems = M_TILE * N_TILE

    matmul_kernel = ExternalFunction(
        "int8_pure_mmul_v2",
        source_string=KERNEL_SRC,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.int32]],
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_chunk = a_bytes
    b_chunk = b_bytes

    # Per-worker fifos (no sharing)
    workers = []
    all_a_fifos = []
    all_b_fifos = []
    all_c_fifos = []

    for row in range(n_rows):
        for col in range(n_cols):
            fa = ObjectFifo(np.ndarray[(a_chunk,), np.dtype[np.int8]],
                            name=f"a_{row}_{col}", depth=fifo_depth)
            fb = ObjectFifo(np.ndarray[(b_chunk,), np.dtype[np.int8]],
                            name=f"b_{row}_{col}", depth=fifo_depth)
            fc = ObjectFifo(np.ndarray[(c_elems,), np.dtype[np.int32]],
                            name=f"c_{row}_{col}", depth=1)
            all_a_fifos.append(fa)
            all_b_fifos.append(fb)
            all_c_fifos.append(fc)

            def core_fn(in_a, in_b, out_c, matmul, _ntpc=n_tiles_per_core, _ce=c_elems, _nkc=n_kc):
                loop = iron_range(_ntpc)
                for _ in loop:
                    elem_out = out_c.acquire(1)
                    for x in iron_range(_ce):
                        elem_out[x] = 0
                    for _ch in iron_range(_nkc):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        in_a.release(1)
                        in_b.release(1)
                    out_c.release(1)

            workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), matmul_kernel]))

    # Taps
    a_taps = []
    for row in range(n_rows):
        for mt in range(nb_m_per_row):
            a_taps.append(TensorAccessPattern(
                tensor_dims=(n_mt * n_kc * a_chunk,),
                offset=(row * nb_m_per_row + mt) * n_kc * a_chunk,
                sizes=[n_kc * a_chunk],
                strides=[1]))

    b_taps = []
    for col in range(n_cols):
        for nt in range(nb_n_per_col):
            b_taps.append(TensorAccessPattern(
                tensor_dims=(n_cols * n_nt_c * n_kc * b_chunk,),
                offset=(col * nb_n_per_col + nt) * n_kc * b_chunk,
                sizes=[n_kc * b_chunk],
                strides=[1]))

    c_taps = []
    for row in range(n_rows):
        for col in range(n_cols):
            for mt in range(nb_m_per_row):
                for nt in range(nb_n_per_col):
                    c_taps.append(TensorAccessPattern(
                        tensor_dims=(M, N),
                        offset=(row * nb_m_per_row + mt) * M_TILE * N + (col * nb_n_per_col + nt) * N_TILE,
                        sizes=[M_TILE, N_TILE],
                        strides=[N, 1]))

    A_ty = np.ndarray[(n_mt * n_kc * a_chunk,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_cols * n_nt_c * n_kc * b_chunk,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.int32]]

    a_idx = 0
    b_idx = 0
    c_idx = 0

    def sequence(A, B, C, *fifos):
        a_hs = fifos[:n_rows * n_cols]
        b_hs = fifos[n_rows * n_cols:n_rows * n_cols * 2]
        c_hs = fifos[n_rows * n_cols * 2:]
        ci = 0
        for mt in range(nb_m_per_row):
            for nt in range(nb_n_per_col):
                for row in range(n_rows):
                    for col in range(n_cols):
                        tg = TaskGroup()
                        ai = row * n_cols + col
                        bi = row * n_cols + col  # same B for all rows in this col
                        # Actually B is per-col, so bi should be col * n_rows + row? No.
                        # B is indexed by col only in 1x4. For multi-row, each row
                        # needs the same B (same col). So bi = col * n_rows? No.
                        # Let's just use col index for B.
                        bi = col  # simplified: one B fifo per col
                        # Wait, we created n_rows*n_cols B fifos. Let me use row*n_cols+col.
                        bi = row * n_cols + col
                        a_hs[ai].fill(A, tap=a_taps[row * nb_m_per_row + mt], group=tg)
                        b_hs[bi].fill(B, tap=b_taps[col * nb_n_per_col + nt], group=tg)
                        c_hs[ci].drain(C, tap=c_taps[ci], wait=True, group=tg)
                        tg.finish()
                        ci += 1

    prod_cons = []
    for f in all_a_fifos:
        prod_cons.append(f.prod())
    for f in all_b_fifos:
        prod_cons.append(f.prod())
    for f in all_c_fifos:
        prod_cons.append(f.cons())

    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_rows, n_cols = args.rows, args.cols

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    iron.set_current_device(iron.device.from_name("npu", n_cols=n_cols))

    @iron.jit
    def design(A: In, B: In, C: Out, *,
               M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
               rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
               dummy: CompileTime[int] = 0):
        return build(M, K, N, rows, cols)

    print("Compiling...")
    t0 = time.time()
    try:
        compiled = design.specialize(
            M=M, K=K, N=N, rows=n_rows, cols=n_cols, dummy=0
        ).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"Compiled in {time.time()-t0:.1f}s -> {compiled[0]}")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print("SUCCESS")


if __name__ == "__main__":
    main()
