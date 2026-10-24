#!/usr/bin/env python3
"""Phase 40M v4: Q8SCALEV-C4 mmul with per-worker B dataflow.

Key change from v2: B is distributed per-worker at L3L2 level,
eliminating the shared-fifo race condition. A remains shared per-row
(which is safe because all workers in a row need the same A data).

Uses 16 L3L2 B fifos + 4 L3L2 A fifos + 4 L3L2 C fifos = 24 total,
which fits within the NPU's 32 output + 32 input ShimNOCTile DMA budget.
"""
import argparse
import os
import time

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction, TaskGroup
)
from aie.iron.controlflow import range_ as iron_range
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils import config


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

extern "C" void q8_c4_mmul_v2(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT   = 32;
    constexpr int NT   = 64;
    constexpr int KC   = 256;
    constexpr int NBLK = KC / 32;
    constexpr int MA   = MT / 4;
    constexpr int NA   = NT / 8;
    constexpr int KA   = KC / 8;
    constexpr int KB   = KC / 8;
    const int8_t * a_tiles = at;
    const float  * as = (const float *)(at + MT * KC);
    const int8_t * b_tiles = bt;
    const float  * bs = (const float *)(bt + KC * NT);

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
                aie::vector<float, 8> bsv = aie::load_v<8>(bs + bb * NT + j * 8);
                for (int rr = 0; rr < 4; rr++) {
                    int row = i * 4 + rr;
                    float asi = as[row * NBLK + bb];
                    aie::vector<float, 8> pvf = aie::to_float(aie::load_v<8>(tmp + rr * 8));
                    aie::vector<float, 8> f = aie::mul(asi, aie::mul(bsv, pvf).to_vector<float>()).to_vector<float>();
                    float * cp = c + row * NT + j * 8;
                    aie::store_v(cp, aie::add(aie::load_v<8>(cp), f));
                }
            }
        }
    }
}
"""


def build(M, K, N, n_cols, n_rows=4, fifo_depth=1):
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

    a_bytes = M_TILE * KC + M_TILE * NBLK * 4
    b_bytes = KC * N_TILE + NBLK * N_TILE * 4
    c_elems = M_TILE * N_TILE

    matmul_kernel = ExternalFunction(
        "q8_c4_mmul_v2",
        source_string=KERNEL_SRC,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.float32]],
        ],
        include_dirs=[config.cxx_header_path()],
    )
    r, s, t = 4, 8, 8
    assert M_TILE % r == 0 and KC % s == 0 and N_TILE % t == 0

    a_chunk = M_TILE * KC + M_TILE * NBLK * 4
    b_chunk = KC * N_TILE + NBLK * N_TILE * 4
    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    n_kc = K // KC

    n_shim_mem_A = n_rows if n_cols > n_rows else n_cols
    n_A_tiles_per_shim = n_rows // n_cols if n_cols < n_rows else 1

    A_ty = np.ndarray[(n_mt * n_kc * a_chunk,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_cols * n_nt_c * n_kc * b_chunk,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_l2_ty = np.ndarray[(a_chunk * n_A_tiles_per_shim,), np.dtype[np.int8]]
    b_l2_ty = np.ndarray[(b_chunk,), np.dtype[np.int8]]
    c_l2_ty = np.ndarray[(M_TILE * N_TILE * n_rows,), np.dtype[np.float32]]
    a_l1_ty = np.ndarray[(M_TILE * KC + M_TILE * NBLK * 4,), np.dtype[np.int8]]
    b_l1_ty = np.ndarray[(KC * N_TILE + NBLK * N_TILE * 4,), np.dtype[np.int8]]
    c_l1_ty = np.ndarray[(M_TILE * N_TILE,), np.dtype[np.float32]]

    A_l3l2_fifos = []
    A_l2l1_fifos = []
    B_l3l2_fifos = []
    B_l2l1_fifos = []
    C_l1l2_fifos = [[] for _ in range(n_rows)]
    C_l2l3_fifos = []

    # A: shared per-row at L3L2, split per-row at L2L1
    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(a_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [j * a_chunk for j in range(stop_row - start_row)]
        a_dims = [[(M_TILE // r, r * KC), (KC // s, s), (r, KC), (s, 1)]] * (stop_row - start_row)
        a_tmp = a_l3l2.cons().split(
            of_offsets,
            obj_types=[a_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp)

    # B: per-worker at L3L2, forwarded per-worker at L2L1
    for row in range(n_rows):
        for col in range(n_cols):
            b_l3l2 = ObjectFifo(b_l2_ty, name=f"B_L3L2_{row}_{col}", depth=fifo_depth)
            B_l3l2_fifos.append(b_l3l2)
            b_dims = [(KC // s, s * N_TILE), (N_TILE // t, t), (s, N_TILE), (t, 1)]
            B_l2l1_fifos.append(
                b_l3l2.cons().forward(obj_type=b_l1_ty, name=f"B_L2L1_{row}_{col}", dims_to_stream=b_dims)
            )

    # C: per-worker at L1L2, joined per-column at L2L3
    for col in range(n_cols):
        c_l2l3 = ObjectFifo(c_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth)
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [M_TILE * N_TILE * i for i in range(n_rows)]
        c_tmp = c_l2l3.prod().join(
            of_offsets,
            obj_types=[c_l1_ty] * n_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_rows)],
            depths=[fifo_depth] * n_rows,
        )
        for j in range(n_rows):
            C_l1l2_fifos[j].append(c_tmp[j])

    def core_fn(in_a, in_b, out_c, matmul):
        loop = iron_range(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            for x in iron_range(c_elems):
                elem_out[x] = 0.0
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
                B_l2l1_fifos[row * n_cols + col].cons(),
                C_l1l2_fifos[row][col].prod(),
                matmul_kernel,
            ],
            stack_size=0x800,
        )

    workers = Worker.grid(n_rows, n_cols, make_worker)
    flat_workers = [w for row in workers for w in row]

    # Taps for runtime sequence
    a_taps = [[TensorAccessPattern(
        tensor_dims=(n_mt * n_kc * a_chunk,),
        offset=(row * nb_m_per_col + mt) * n_kc * a_chunk,
        sizes=[n_kc * a_chunk],
        strides=[1])
        for mt in range(nb_m_per_col)]
        for row in range(n_rows)]

    b_taps = [[TensorAccessPattern(
        tensor_dims=(n_cols * n_nt_c * n_kc * b_chunk,),
        offset=(col * n_nt_c + nt) * n_kc * b_chunk,
        sizes=[n_kc * b_chunk],
        strides=[1])
        for nt in range(n_nt_c)]
        for col in range(n_cols)]

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
    C_conses = [f.cons() for f in C_l2l3_fifos]
    nA, nB = len(A_prods), len(B_prods)

    def sequence(A, B, C, *fifos):
        A_hs = fifos[:nA]
        B_hs = fifos[nA:nA + nB]
        C_hs = fifos[nA + nB:]
        for mt in range(nb_m_per_col):
            for nt in range(nb_n_per_col):
                for col in range(n_cols):
                    for row in range(n_rows):
                        tg = TaskGroup()
                        A_hs[row].fill(A, tap=a_taps[row][mt], group=tg)
                        b_index = row * n_cols + col
                        B_hs[b_index].fill(B, tap=b_taps[col][nt], group=tg)
                        c_index = row * nb_n_per_col * n_cols + col * nb_n_per_col + nt
                        C_hs[col].drain(C, tap=c_taps[col][row][mt][nt], wait=True, group=tg)
                        tg.finish()

    print(f"[mmul-v2-v4] cols={n_cols} rows={n_rows} fifo_depth={fifo_depth} "
          f"n_shim_mem_A={n_shim_mem_A} n_A_tiles_per_shim={n_A_tiles_per_shim} "
          f"A_tiles={n_mt} B_tiles={n_cols * n_nt_c} C_tiles={n_rows * n_cols * nb_m_per_col * nb_n_per_col} "
          f"n_tiles_per_core={n_tiles_per_core} "
          f"L3L2_fifos={len(A_l3l2_fifos) + len(B_l3l2_fifos) + len(C_l2l3_fifos)}",
          flush=True)

    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + A_prods + B_prods + C_conses)
    return iron.Program(iron.get_current_device(), rt, workers=flat_workers).resolve_program()


def pack_A(A, As, M, K, m_tile, k_chunk):
    n_mt, n_kc = M // m_tile, K // k_chunk
    MA_tile, nb = m_tile // 4, k_chunk // 32
    ai = np.empty((n_mt, n_kc, MA_tile, nb, 4, 32), np.int8)
    asc = np.empty((n_mt, n_kc, m_tile, nb), np.float32)
    for mt in range(n_mt):
        for kc in range(n_kc):
            for bb in range(nb):
                for im in range(MA_tile):
                    for kk in range(4):
                        r0 = mt * m_tile + im * 4
                        c0 = kc * k_chunk + bb * 32 + kk * 8
                        ai[mt, kc, im, bb, kk] = A[r0:r0 + 4, c0:c0 + 8].reshape(-1)
            asc[mt, kc] = As[mt * m_tile:(mt + 1) * m_tile,
                             kc * nb:(kc + 1) * nb]
    int8 = ai.reshape(n_mt, n_kc, -1)
    sc = asc.reshape(n_mt, n_kc, -1).view(np.int8)
    out = np.concatenate([int8, sc], axis=2)
    return out.reshape(-1)


def pack_B(B, Bs, K, N, n_tile, k_chunk):
    n_nt, n_kc = N // n_tile, K // k_chunk
    NA_tile, nb = n_tile // 8, k_chunk // 32
    bi = np.empty((n_nt, n_kc, NA_tile, nb, 4, 64), np.int8)
    bsc = np.empty((n_nt, n_kc, nb, n_tile), np.float32)
    for nt in range(n_nt):
        for kc in range(n_kc):
            for bb in range(nb):
                for kk in range(4):
                    for jn in range(NA_tile):
                        r0 = kc * k_chunk + bb * 32 + kk * 8
                        c0 = nt * n_tile + jn * 8
                        bi[nt, kc, jn, bb, kk] = B[r0:r0 + 8, c0:c0 + 8].reshape(-1)
                bsc[nt, kc, bb] = Bs[kc * nb + bb,
                                     nt * n_tile:(nt + 1) * n_tile]
    int8 = bi.reshape(n_nt, n_kc, -1)
    sc = bsc.reshape(n_nt, n_kc, -1).view(np.int8)
    return np.concatenate([int8, sc], axis=2).reshape(-1)


def ref_q8_gemm(A_i8, As, B_i8, Bs):
    M, K = A_i8.shape
    N = B_i8.shape[1]
    nb = K // 32
    C = np.zeros((M, N), dtype=np.float64)
    for b in range(nb):
        k_start = b * 32
        k_end = k_start + 32
        A_blk = A_i8[:, k_start:k_end].astype(np.int64)
        B_blk = B_i8[k_start:k_end, :].astype(np.int64)
        p = A_blk @ B_blk
        scale = As[:, b:b+1].astype(np.float64) * Bs[b:b+1, :].astype(np.float64)
        C += p * scale
    return C.astype(np.float32)


def unpack_A(Ap, M, K, m_tile, k_chunk):
    n_mt, n_kc = M // m_tile, K // k_chunk
    a_bytes = m_tile * k_chunk + m_tile * (k_chunk // 32) * 4
    packed = Ap.reshape(n_mt, n_kc, -1)
    ai = packed[:, :, :m_tile * k_chunk].reshape(n_mt, n_kc, m_tile // 4, k_chunk // 32, 4, 32)
    sc = packed[:, :, m_tile * k_chunk:].reshape(n_mt, n_kc, m_tile, k_chunk // 32)
    A_i8 = np.concatenate([ai.reshape(n_mt, n_kc, m_tile, k_chunk).transpose(0, 2, 1, 3).reshape(M, K)], axis=0)
    As = np.concatenate([sc.reshape(n_mt, n_kc, m_tile, k_chunk // 32).transpose(0, 2, 1, 3).reshape(M, k_chunk // 32)], axis=0)
    return A_i8, As


def unpack_B(Bp, K, N, n_tile, k_chunk):
    n_nt, n_kc = N // n_tile, K // k_chunk
    b_bytes = k_chunk * n_tile + (k_chunk // 32) * n_tile * 4
    packed = Bp.reshape(n_nt, n_kc, -1)
    bi = packed[:, :, :k_chunk * n_tile].reshape(n_nt, n_kc, n_tile // 8, k_chunk // 32, 4, 64)
    bsc = packed[:, :, k_chunk * n_tile:].reshape(n_nt, n_kc, k_chunk // 32, n_tile)
    B_i8 = np.concatenate([bi.transpose(0, 3, 2, 4, 5, 1).reshape(n_nt, n_tile, k_chunk).transpose(1, 0, 2).reshape(N, K)], axis=0)
    Bs = np.concatenate([bsc.transpose(0, 3, 2, 1).reshape(n_nt, n_tile, k_chunk // 32).transpose(1, 0, 2).reshape(N, k_chunk // 32)], axis=0)
    return B_i8, Bs


def simulate_kernel_cpu(Ap, Bp, M, K, N, m_tile, n_tile, k_chunk):
    A_i8, As = unpack_A(Ap, M, K, m_tile, k_chunk)
    B_i8, Bs = unpack_B(Bp, K, N, n_tile, k_chunk)
    return ref_q8_gemm(A_i8, As, B_i8, Bs)


def test_correctness(M, K, N, n_cols, n_rows=4, pattern='geometric'):
    rng = np.random.default_rng(42)
    nb = K // 32
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)

    if pattern == 'uniform':
        As = np.ones((M, nb), dtype=np.float32)
        Bs = np.ones((nb, N), dtype=np.float32)
    elif pattern == 'two_scale':
        As = np.ones((M, nb), dtype=np.float32)
        Bs = np.ones((nb, N), dtype=np.float32)
        As[:, nb//2:] = 2.0
        Bs[nb//2:, :] = 2.0
    elif pattern == 'geometric':
        factors = np.array([min(2.0 ** i, 65504.0) for i in range(nb)], dtype=np.float32)
        As = np.ones((M, nb), dtype=np.float32) * factors[np.newaxis, :]
        Bs = np.ones((nb, N), dtype=np.float32) * factors[:, np.newaxis]
    elif pattern == 'realistic':
        As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
        Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    C_ref = ref_q8_gemm(A_i8, As, B_i8, Bs)

    Ap = pack_A(A_i8, As, M, K, M_TILE, KC)
    Bp = pack_B(B_i8, Bs, K, N, N_TILE, KC)

    C_dev = simulate_kernel_cpu(Ap, Bp, M, K, N, M_TILE, N_TILE, KC)

    diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
    max_err = float(np.max(diff))
    rel_err = float(np.max(diff[np.abs(C_ref) > 1e-6] / np.abs(C_ref[np.abs(C_ref) > 1e-6])))
    bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))

    print(f"  Pattern={pattern}: M={M} K={K} N={N} cols={n_cols} rows={n_rows}")
    print(f"    C_ref[0,0]={C_ref[0,0]:.6f} C_dev[0,0]={C_dev[0,0]:.6f}")
    print(f"    max_err={max_err:.4e} rel_err={rel_err:.4e} bad={bad}/{M*N}")

    return bad == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=512)
    parser.add_argument('--K', type=int, default=1024)
    parser.add_argument('--N', type=int, default=3584)
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--fifo-depth', type=int, default=1)
    parser.add_argument('--work-dir', type=str, required=True)
    parser.add_argument('--pattern', type=str, default='geometric',
                        choices=['uniform', 'two_scale', 'geometric', 'realistic'])
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cols, n_rows = args.cols, args.rows

    print(f"[mmul-v2-v4] cols={n_cols} rows={n_rows} fifo_depth={args.fifo_depth}")

    iron.set_current_device(iron.device.from_name("npu", n_cols=n_cols))

    @iron.jit
    def design(A: In, B: In, C: Out, *,
               M: CompileTime[int] = M, K: CompileTime[int] = K, N: CompileTime[int] = N,
               rows: CompileTime[int] = n_rows, cols: CompileTime[int] = n_cols,
               dummy: CompileTime[int] = 0):
        return build(M, K, N, cols, n_rows=rows, fifo_depth=args.fifo_depth)

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    print(f"  Compiling...")
    t0 = time.time()
    try:
        compiled = design.specialize(M=M, K=K, N=N, rows=n_rows, cols=n_cols, dummy=0).compile(
            xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"  Compiled in {time.time()-t0:.1f}s -> {compiled[0]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        raise

    passed = test_correctness(M, K, N, n_cols, n_rows=n_rows, pattern=args.pattern)
    print(f"  Correctness: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
