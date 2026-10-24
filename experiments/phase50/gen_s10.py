#!/usr/bin/env python3
"""Phase 50 candidate s10: kernel/dataflow redesign around the measured
bottlenecks (float path = 93% of cycles; fp32 runs at half rate on the AIE2
bf16 datapath; ~84 of the 206 block-loop bundles are bank shuffles/spills).

Changes vs production c4_mmul_v2 (each audited against the phase40 source):
 1. C tile layout in core memory is [i][j][4][8] (32 contiguous floats per
    output block): the C read-modify-write becomes two 16-lane load/add/store
    pairs; the drain tap scatters (i,j,rr,lane) into row-major DDR C.
 2. Host pre-broadcasts the a-scales to 32 floats per (bb,i) =
    [a(i*4+0)x8, a(i*4+1)x8, a(i*4+2)x8, a(i*4+3)x8] and duplicates the
    b-scales to 16 floats per (bb,j) = [bsv, bsv]. The per-block scale
    products are then single 16-lane muls with no in-kernel broadcast/insert.
 3. Only 16-lane float ops; the 4 a-scale scalars leave the block loop.

Q8_0 per-32 semantics are unchanged: one int32 MMUL group (4 MACs) per
32-element k block, exact int32->fp32 via the production to_float, then
scale = (a-scale x b-scale) per (row, col) and fp32 accumulation.

Usage (ironenv python):
  python gen_s10.py --M 512 --K 1024 --N 3584 --cols 4 --work-dir /tmp/p50_s10
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))

import numpy as np

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void q8_c4_mmul_v2(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float * __restrict c0, float * __restrict c1,
    float * __restrict c2, float * __restrict c3,
    float * __restrict c4, float * __restrict c5,
    float * __restrict c6, float * __restrict c7)
{
    constexpr int MT = 32;
    constexpr int NT = 64;
    constexpr int KC = 256;
    constexpr int NBLK = KC / 32;
    constexpr int MA = MT / 4;
    constexpr int NA = NT / 8;
    constexpr int KA = KC / 8;
    constexpr int KB = KC / 8;
    const int8_t * a_tiles = at;
    const float * as = (const float *)(at + MT * KC);
    const int8_t * b_tiles = bt;
    const float * bs = (const float *)(bt + KC * NT);
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
#pragma clang loop unroll_count(1)
    for (int bb = 0; bb < NBLK; bb++) {
        const float * bsp = bs + bb * NA * 16;
#pragma clang loop unroll_count(1)
        for (int i = 0; i < MA; i++) {
            const float * asp = as + (bb * MA + i) * 32;
            aie::vector<float, 16> ab01 = aie::load_v<16>(asp);
            aie::vector<float, 16> ab23 = aie::load_v<16>(asp + 16);
            const int8_t * ak = a_tiles + (i * KA + bb * 4) * 32;
            float * cb = (i == 0) ? c0 : (i == 1) ? c1 : (i == 2) ? c2 : (i == 3) ? c3 :
                         (i == 4) ? c4 : (i == 5) ? c5 : (i == 6) ? c6 : c7;
#pragma clang loop unroll_count(1)
            for (int j = 0; j < NA; j++) {
                aie::vector<int32_t, 32> zero = aie::broadcast<int32_t, 32>(0);
                MMUL acc(zero);
                const int8_t * bk = b_tiles + (j * KB + bb * 4) * 64;
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(ak + kk * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bk + kk * 64);
                    acc.mac(av, bv);
                }
                aie::vector<float, 32> p = aie::to_float(acc.to_vector<int32_t>());
                aie::vector<float, 16> bv16 = aie::load_v<16>(bsp + j * 16);
                aie::vector<float, 16> t01 = aie::mul(ab01, bv16).to_vector<float>();
                aie::vector<float, 16> t23 = aie::mul(ab23, bv16).to_vector<float>();
                float * q = cb + j * 32;
                aie::store_v(q, aie::add(aie::load_v<16>(q),
                    aie::mul(p.template extract<16>(0), t01).to_vector<float>()));
                aie::store_v(q + 16, aie::add(aie::load_v<16>(q + 16),
                    aie::mul(p.template extract<16>(1), t23).to_vector<float>()));
            }
        }
    }
}
"""

M_TILE, N_TILE, KC, NBLK = 32, 64, 256, 8


def build(M, K, N, n_cols, fifo_depth=2, b_fifo_depth=2, c_fifo_depth=1, stack_size=0x800):
    import aie.iron as iron
    from aie.iron import ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config

    MA, NA = M_TILE // 4, N_TILE // 8
    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    n_kc = K // KC

    a_bytes = M_TILE * KC + NBLK * MA * 32 * 4
    b_bytes = KC * N_TILE + NBLK * NA * 16 * 4
    c_elems = M_TILE * N_TILE
    # C fifo holds MA objects of NA*32 floats (one i-row-group each) so each
    # per-i drain pops exactly one object. The kernel sees the flat 2048 view.
    c_obj = NA * 32

    c_arg = np.ndarray[(c_obj,), np.dtype[np.float32]]
    kern = ExternalFunction(
        "q8_c4_mmul_v2",
        source_string=KERNEL_SRC,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            c_arg, c_arg, c_arg, c_arg, c_arg, c_arg, c_arg, c_arg,
        ],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[f for f in os.environ.get("P50_KFLAGS", "").split(",") if f],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_obj,), np.dtype[np.float32]]

    b_depth = fifo_depth if b_fifo_depth is None else b_fifo_depth
    workers = []

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt_c):
            c0, c1, c2, c3, c4, c5, c6, c7 = of_c.acquire(MA)
            for cc in (c0, c1, c2, c3, c4, c5, c6, c7):
                for x in iron_range(c_obj):
                    cc[x] = 0.0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, c0, c1, c2, c3, c4, c5, c6, c7)
                of_a.release(1)
                of_b.release(1)
            of_c.release(MA)

    fifo_a, fifo_b, fifo_c = [], [], []
    for col in range(n_cols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=fifo_depth)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=b_depth)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=c_fifo_depth)
        fifo_a.append(fa)
        fifo_b.append(fb)
        fifo_c.append(fc)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern], stack_size=stack_size))

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

    # core C tile layout [i][j][rr][lane] contiguous = i*(NA*32) + j*32 + rr*8 + lane.
    # The shim DMA BD supports 3 access dims + 1 repeat dim, so a single tap
    # cannot split (i, j, rr, lane): rr has DDR stride N while lane has stride 1.
    # Drain one i-row-group at a time (8 drains per group, sequential 256-elem
    # pops matching the kernel's (i,j,x) write order): 3D tap (j, rr, lane)
    # with sizes=[NA,4,8] strides=[8,N,1], left-padded to keep sizes[0]=1.
    c_taps = [[[[TensorAccessPattern(
        tensor_dims=(M, N),
        offset=mt * M_TILE * N + (col * n_nt_c + nt) * N_TILE + i * 4 * N,
        sizes=[1, NA, 4, 8], strides=[0, 8, N, 1])
        for i in range(MA)] for nt in range(n_nt_c)] for mt in range(n_mt)] for col in range(n_cols)]

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
                    for i in range(MA):
                        c_c[col].drain(C, tap=c_taps[col][mt][nt][i], group=tg, wait=(i == MA - 1))
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


def pack_A_s10(A, As, M, K):
    """A: int8 tile-major (unchanged) + per (bb,i) 32 pre-broadcast a-scales."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
    from gen_q8_c4_mmul_v2 import pack_A
    nb = KC // 32  # scale blocks per k-chunk (pack_A layout: [row][nb])
    n_mt, n_kc = M // M_TILE, K // KC
    MA = M_TILE // 4
    a_chunk = M_TILE * KC
    Ap = pack_A(A, As, M, K, M_TILE, KC)
    out = np.zeros(n_mt * n_kc * (a_chunk + NBLK * MA * 32 * 4), np.int8)
    new_a_chunk = a_chunk + NBLK * MA * 32 * 4
    for mt in range(n_mt):
        for kc in range(n_kc):
            off = (mt * n_kc + kc) * (a_chunk + M_TILE * nb * 4)
            sc = Ap[off + a_chunk:off + a_chunk + M_TILE * nb * 4].view(np.float32).reshape(M_TILE, nb)
            bc = np.empty((NBLK, MA, 32), np.float32)
            for bb in range(NBLK):
                for i in range(MA):
                    for r in range(4):
                        bc[bb, i, r * 8:(r + 1) * 8] = sc[i * 4 + r, bb]
            o = (mt * n_kc + kc) * new_a_chunk
            out[o:o + a_chunk] = Ap[off:off + a_chunk]
            out[o + a_chunk:o + a_chunk + NBLK * MA * 32 * 4] = bc.reshape(-1).view(np.int8)
    return out


def pack_B_s10(B, Bs, K, N):
    """B: int8 tile-major (unchanged) + per (bb,j) 16 duplicated b-scales."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
    from gen_q8_c4_mmul_v2 import pack_B
    nb = KC // 32  # NBLK
    n_nt, n_kc = N // N_TILE, K // KC
    NA = N_TILE // 8
    b_chunk = KC * N_TILE
    Bp = pack_B(B, Bs, K, N, N_TILE, KC)
    out = np.zeros(n_nt * n_kc * (b_chunk + NBLK * NA * 16 * 4), np.int8)
    new_b_chunk = b_chunk + NBLK * NA * 16 * 4
    for nt in range(n_nt):
        for kc in range(n_kc):
            off = (nt * n_kc + kc) * (b_chunk + NBLK * N_TILE * 4)
            sc = Bp[off + b_chunk:off + b_chunk + NBLK * N_TILE * 4].view(np.float32).reshape(NBLK, N_TILE)
            bc = np.empty((NBLK, NA, 16), np.float32)
            for bb in range(NBLK):
                for j in range(NA):
                    bc[bb, j, 0:8] = sc[bb, j * 8:j * 8 + 8]
                    bc[bb, j, 8:16] = sc[bb, j * 8:j * 8 + 8]
            o = (nt * n_kc + kc) * new_b_chunk
            out[o:o + b_chunk] = Bp[off:off + b_chunk]
            out[o + b_chunk:o + b_chunk + NBLK * NA * 16 * 4] = bc.reshape(-1).view(np.int8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    # the iron JIT cache key does not cover source_string kernel bodies
    nonce = int(os.environ.get("P50_NONCE", "0"))

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.cols,
                     fifo_depth=int(os.environ.get("P50_ADEPTH", "2")),
                     b_fifo_depth=int(os.environ.get("P50_BDEPTH", "2")),
                     c_fifo_depth=int(os.environ.get("P50_CDEPTH", "1")),
                     stack_size=int(os.environ.get("P50_STACK", "0x800"), 16))

    t0 = time.time()
    compiled = design.specialize(dummy=nonce).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"[gen_s10] compiled in {time.time()-t0:.1f}s -> {compiled[0]}")


if __name__ == "__main__":
    main()
