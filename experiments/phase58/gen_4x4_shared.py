#!/usr/bin/env python3
"""Phase 58: 4x4 GEMM with shared external DMA channels.

Fixes the ShimNOCTile DMA budget issue by sharing B data across rows
(one L3L2 fifo per column instead of per-worker).

Based on phase40/gen_q8_c4_mmul_v2_v4.py with shared B fifos.
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
    assert M % M_TILE == 0 and N % N_TILE == 0 and K % KC == 0
    assert N % n_cols == 0
    assert M % (M_TILE * n_rows) == 0

    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    assert n_nt_c >= 1
    n_kc = K // KC
    nb_m_per_col = M // M_TILE // n_rows
    nb_n_per_col = N // N_TILE // n_cols
    n_tiles_per_core = nb_m_per_col * nb_n_per_col

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

    # A: shared per-row at L3L2, split per-row at L2L1
    n_shim_mem_A = n_rows if n_cols > n_rows else n_cols
    n_A_tiles_per_shim = n_rows // n_cols if n_cols < n_rows else 1

    A_l3l2_fifos = []
    A_l2l1_fifos = []
    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(np.ndarray[(a_chunk * n_A_tiles_per_shim,), np.dtype[np.int8]],
                            name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [j * a_chunk for j in range(stop_row - start_row)]
        a_dims = [[(M_TILE // 4, 4 * KC), (KC // 8, 8), (4, KC), (8, 1)]] * (stop_row - start_row)
        a_tmp = a_l3l2.cons().split(
            of_offsets,
            obj_types=[np.ndarray[(a_chunk,), np.dtype[np.int8]]] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp)

    # B: shared per-column at L3L2, duplicate data n_rows times
    # This avoids the "Cons endpoint not set" issue with offset=0 splits
    B_l3l2_fifos = []
    B_l2l1_map = {}  # (row, col) -> fifo
    for col in range(n_cols):
        b_l3l2 = ObjectFifo(np.ndarray[(b_chunk * n_rows,), np.dtype[np.int8]],
                            name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims = [(KC // 8, 8 * N_TILE), (N_TILE // 8, 8), (8, N_TILE), (8, 1)]
        b_tmp = b_l3l2.cons().split(
            [i * b_chunk for i in range(n_rows)],
            obj_types=[np.ndarray[(b_chunk,), np.dtype[np.int8]]] * n_rows,
            names=[f"B_L2L1_{row}_{col}" for row in range(n_rows)],
            dims_to_stream=[b_dims] * n_rows,
        )
        for row in range(n_rows):
            B_l2l1_map[(row, col)] = b_tmp[row]

    # C: per-worker fifos (no join, like 1x4 design)
    C_l1l2_fifos = {}
    C_l1l2_cons = {}
    C_l1l2_prod = {}
    for row in range(n_rows):
        for col in range(n_cols):
            c_fifo = ObjectFifo(np.ndarray[(c_elems,), np.dtype[np.int32]],
                                name=f"C_L1L2_{row}_{col}", depth=1)
            C_l1l2_fifos[(row, col)] = c_fifo
            C_l1l2_cons[(row, col)] = c_fifo.cons()
            C_l1l2_prod[(row, col)] = c_fifo.prod()

    def core_fn(in_a, in_b, out_c, matmul):
        loop = iron_range(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            for x in iron_range(c_elems):
                elem_out[x] = 0
            for _ch in iron_range(n_kc):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    def make_worker(row, col):
        return Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_map[(row, col)].cons(),
                C_l1l2_prod[(row, col)],
                matmul_kernel,
            ],
            stack_size=0x800,
        )

    # C runtime: no L2L3 fifos, drain each worker's C fifo directly
    # (handled by c_taps in sequence)
    workers = []
    for row in range(n_rows):
        for col in range(n_cols):
            workers.append(make_worker(row, col))

    # Taps
    a_taps = [[TensorAccessPattern(
        tensor_dims=(n_mt * n_kc * a_chunk,),
        offset=(row * nb_m_per_col + mt) * n_kc * a_chunk,
        sizes=[n_kc * a_chunk],
        strides=[1])
        for mt in range(nb_m_per_col)]
        for row in range(n_rows)]

    b_taps = [TensorAccessPattern(
        tensor_dims=(n_cols * n_nt_c * n_kc * b_chunk,),
        offset=(col * n_nt_c + nt) * n_kc * b_chunk,
        sizes=[n_kc * b_chunk],
        strides=[1])
        for col in range(n_cols) for nt in range(n_nt_c)]

    c_taps = [[[[TensorAccessPattern(
        tensor_dims=(M, N),
        offset=(row * nb_m_per_col + mt) * M_TILE * N + (col * nb_n_per_col + nt) * N_TILE,
        sizes=[M_TILE, N_TILE],
        strides=[N, 1])
        for nt in range(nb_n_per_col)]
        for mt in range(nb_m_per_col)]
        for row in range(n_rows)]
        for col in range(n_cols)]

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    nA, nB = len(A_prods), len(B_prods)

    # C: per-worker cons (already created above)
    pass

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        idx = 0
        for mt in range(nb_m_per_col):
            for nt in range(nb_n_per_col):
                for col in range(n_cols):
                    for row in range(n_rows):
                        tg = TaskGroup()
                        A_hs[row % nA].fill(A, tap=a_taps[row][mt], group=tg)
                        B_hs[col].fill(B, tap=b_taps[col * n_nt_c + nt], group=tg)
                        ci = row * n_cols + col
                        C_hs[ci].drain(C, tap=c_taps[col][row][mt][nt], wait=True, group=tg)
                        tg.finish()

    prod_cons = []
    for f in A_l3l2_fifos:
        prod_cons.append(f.prod())
    for f in B_l3l2_fifos:
        prod_cons.append(f.prod())
    for row in range(n_rows):
        for col in range(n_cols):
            prod_cons.append(C_l1l2_cons[(row, col)])

    A_ty = np.ndarray[(n_mt * n_kc * a_chunk,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_cols * n_nt_c * n_kc * b_chunk,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.int32]]

    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
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
