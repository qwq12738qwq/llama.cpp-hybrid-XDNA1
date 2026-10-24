#!/usr/bin/env python3
"""Phase 40M: Q8SCALEV-C4 replacement using aie::mmul<4,8,8>.

Replaces the software-loop mm_chunk_vec GEMM in Q8SCALEV-C4 with an
AIE matrix-mac mmul kernel while preserving exact Q8_0 per-block scale
semantics.

Tile-major operand layout (aie::mmul tile order):
  A chunk: (MA * KC/8) tiles of 4x8 int8, then (M_TILE * KC/32) f32 scales
  B chunk: (NA * KC/8) tiles of 8x8 int8, then (KC/32 * N_TILE) f32 scales
  C:       M_TILE x N_TILE f32 accumulator, accumulated across K-chunks

where MA = M_TILE/4, NA = N_TILE/8.

Usage:
  python gen_q8_c4_mmul_v2.py --M 512 --K 1024 --N 3584 --cols 4 --work-dir /tmp/q8c4mmul
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
    from aie.utils import config, DefaultNPURuntime, NPUKernel, tensor
    from aie.utils.trace.events import CoreEvent
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

// Q8_0 C4 mmul with per-block device-side scaling (Phase 40M).
// Tile-major operand layout (aie::mmul tile order):
//   A chunk: (MA * KC/8) tiles of 4x8 int8, then (M_TILE * KC/32) f32 scales
//   B chunk: (NA * KC/8) tiles of 8x8 int8, then (KC/32 * N_TILE) f32 scales
//   C:       M_TILE x N_TILE f32 accumulator, accumulated across K-chunks
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

KERNEL_SRC_P32 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void q8_c4_mmul_v2(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float * __restrict c)
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
    for (int bb = 0; bb < NBLK; bb++) {
        for (int i = 0; i < MA; i++) {
            for (int j = 0; j < NA; j++) {
                aie::vector<int32_t, 32> zero = aie::broadcast<int32_t, 32>(0);
                MMUL acc(zero);
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(a_tiles + (i * KA + bb * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
                aie::vector<float, 32> p = aie::to_float(acc.to_vector<int32_t>());
                aie::vector<float, 8> bsv = aie::load_v<8>(bs + bb * NT + j * 8);
                aie::vector<float, 8> s0 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 0) * NBLK + bb]), bsv);
                aie::vector<float, 8> s1 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 1) * NBLK + bb]), bsv);
                aie::vector<float, 8> s2 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 2) * NBLK + bb]), bsv);
                aie::vector<float, 8> s3 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 3) * NBLK + bb]), bsv);
                aie::vector<float, 32> scale = aie::broadcast<float, 32>(0.0f);
                scale.insert<8>(0, s0);
                scale.insert<8>(1, s1);
                scale.insert<8>(2, s2);
                scale.insert<8>(3, s3);
                p = aie::mul(p, scale);
                for (int rr = 0; rr < 4; rr++) {
                    float * cp = c + (i * 4 + rr) * NT + j * 8;
                    aie::store_v(cp, aie::add(aie::load_v<8>(cp), p.template extract<8>(rr)));
                }
            }
        }
    }
}
"""

KERNEL_SRC_P32_TRACE = KERNEL_SRC_P32.replace(
    "                for (int kk = 0; kk < 4; kk++) {",
    "                event0();\n                for (int kk = 0; kk < 4; kk++) {",
).replace(
    "                aie::vector<float, 32> p =",
    "                event1();\n                event0();\n                aie::vector<float, 32> p =",
).replace(
    "                }\n            }\n        }\n    }\n}",
    "                }\n                event1();\n            }\n        }\n    }\n}",
)

def build(M, K, N, n_cols, trace_size=0, trace_col=0, fifo_depth=2, stack_size=0x800, b_fifo_depth=2, c_fifo_depth=1, kernel_variant="p32"):
    assert M % M_TILE == 0 and N % N_TILE == 0 and K % KC == 0
    assert N % n_cols == 0
    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    assert n_nt_c >= 1
    n_kc = K // KC

    a_bytes = M_TILE * KC + M_TILE * NBLK * 4
    b_bytes = KC * N_TILE + NBLK * N_TILE * 4
    c_elems = M_TILE * N_TILE

    if kernel_variant == "p32":
        kernel_source = KERNEL_SRC_P32_TRACE if trace_size > 0 else KERNEL_SRC_P32
    else:
        kernel_source = KERNEL_SRC

    kern = ExternalFunction(
        "q8_c4_mmul_v2",
        source_string=kernel_source,
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
    b_depth = fifo_depth if b_fifo_depth is None else b_fifo_depth
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
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=b_depth)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=c_fifo_depth)
        fifo_a.append(fa)
        fifo_b.append(fb)
        fifo_c.append(fc)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern], stack_size=stack_size, trace=(1 if trace_size > 0 and col == trace_col else 0)))

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
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    if trace_size > 0:
        prog.enable_trace(trace_size, workers=[workers[trace_col]], reuse_output_buffer=True, egress_shim_col=3, coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL, CoreEvent.STREAM_STALL])
    return prog.resolve_program()


def pack_A(A, As, M, K, m_tile, k_chunk):
    """Pack A in tile-major format for aie::mmul kernel (Phase 40M)."""
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
    """Pack B in tile-major format for aie::mmul kernel (Phase 40M)."""
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
    """Unpack A from tile-major mmul layout to row-major [M, K] int8 + [M, nb] float32 scales."""
    n_mt, n_kc = M // m_tile, K // k_chunk
    nb = K // 32
    nb_per_chunk = k_chunk // 32
    MA_tile = m_tile // 4
    a_chunk = m_tile * k_chunk + m_tile * nb_per_chunk * 4

    A_out = np.zeros((M, K), dtype=np.int8)
    As_out = np.zeros((M, nb), dtype=np.float32)

    for mt in range(n_mt):
        for kc in range(n_kc):
            offset = (mt * n_kc + kc) * a_chunk
            chunk = Ap[offset:offset + a_chunk]
            int8_part = chunk[:m_tile * k_chunk]
            scale_part = chunk[m_tile * k_chunk:].view(np.float32).reshape(m_tile, nb_per_chunk)

            ai = int8_part.reshape(MA_tile, nb_per_chunk, 4, 32)
            for im in range(MA_tile):
                for bb in range(nb_per_chunk):
                    for kk in range(4):
                        r0 = mt * m_tile + im * 4
                        c0 = kc * k_chunk + bb * 32 + kk * 8
                        A_out[r0:r0 + 4, c0:c0 + 8] = ai[im, bb, kk].reshape(4, 8)

            As_out[mt * m_tile:(mt + 1) * m_tile, kc * nb_per_chunk:(kc + 1) * nb_per_chunk] = scale_part

    return A_out, As_out


def unpack_B(Bp, K, N, n_tile, k_chunk):
    """Unpack B from tile-major mmul layout to row-major [K, N] int8 + [nb, N] float32 scales."""
    n_nt, n_kc = N // n_tile, K // k_chunk
    nb = K // 32
    nb_per_chunk = k_chunk // 32
    NA_tile = n_tile // 8
    b_chunk = k_chunk * n_tile + nb_per_chunk * n_tile * 4

    B_out = np.zeros((K, N), dtype=np.int8)
    Bs_out = np.zeros((nb, N), dtype=np.float32)

    for nt in range(n_nt):
        for kc in range(n_kc):
            offset = (nt * n_kc + kc) * b_chunk
            chunk = Bp[offset:offset + b_chunk]
            int8_part = chunk[:k_chunk * n_tile]
            scale_part = chunk[k_chunk * n_tile:].view(np.float32).reshape(nb_per_chunk, n_tile)

            bi = int8_part.reshape(NA_tile, nb_per_chunk, 4, 64)
            for jn in range(NA_tile):
                for bb in range(nb_per_chunk):
                    for kk in range(4):
                        r0 = kc * k_chunk + bb * 32 + kk * 8
                        c0 = nt * n_tile + jn * 8
                        B_out[r0:r0 + 8, c0:c0 + 8] = bi[jn, bb, kk].reshape(8, 8)

            Bs_out[kc * nb_per_chunk:(kc + 1) * nb_per_chunk, nt * n_tile:(nt + 1) * n_tile] = scale_part

    return B_out, Bs_out


def simulate_kernel_cpu(Ap, Bp, M, K, N, m_tile, n_tile, k_chunk):
    """CPU simulation of the mmul kernel with tile-major layout."""
    A_i8, As = unpack_A(Ap, M, K, m_tile, k_chunk)
    B_i8, Bs = unpack_B(Bp, K, N, n_tile, k_chunk)
    return ref_q8_gemm(A_i8, As, B_i8, Bs)


def test_correctness(M, K, N, n_cols, pattern='geometric'):
    """Test kernel correctness against CPU reference."""
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

    print(f"  Pattern={pattern}: M={M} K={K} N={N} cols={n_cols}")
    print(f"    C_ref[0,0]={C_ref[0,0]:.6f} C_dev[0,0]={C_dev[0,0]:.6f}")
    print(f"    max_err={max_err:.4e} rel_err={rel_err:.4e} bad={bad}/{M*N}")

    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="geometric",
                    choices=["uniform", "two_scale", "geometric", "realistic"])
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cols = args.cols

    print(f"Phase 40M: Q8SCALEV-C4 mmul v2 correctness test")
    print(f"  M={M} K={K} N={N} cols={n_cols} m_tile={M_TILE} n_tile={N_TILE} k_chunk={KC}")
    print(f"  nb={K//32} n_mt={M//M_TILE} n_nt_c={(N//n_cols)//N_TILE} n_kc={K//KC}")
    print()

    # Phase 40M-7: First correctness test
    print("Phase 40M-7: First correctness test")
    passed = test_correctness(M, K, N, n_cols, pattern=args.pattern)
    print()

    # Phase 40M-8: Non-uniform scale stress test
    print("Phase 40M-8: Non-uniform scale stress test")
    for pat in ['uniform', 'two_scale', 'geometric', 'realistic']:
        ok = test_correctness(M, K, N, n_cols, pattern=pat)
        passed = passed and ok
    print()

    if passed:
        print("Phase 40M: ALL TESTS PASSED")
    else:
        print("Phase 40M: SOME TESTS FAILED")


if __name__ == "__main__":
    main()
