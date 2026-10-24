#!/usr/bin/env python3
"""Phase 62 kernel harness: the per-chunk (Q8_0-256) scale-preserving GEMM.

Phase 61's kernel (s4) applies the Q8_0 per-32-block scale inside the block
loop, which costs two float multiplies and one int32->float conversion per
output element per 32-block, and forces eight float accumulators to stay live
across the block loop (the AIE2 has eight vector registers, so the kernel
spills).

This harness implements the candidate that moves the scale application out of
the block loop:

    the weight block scale is per 256 elements (one K chunk) instead of per
    32, so the whole chunk is one int32 matrix-unit chain and the scale is
    applied once per chunk:

        C[m,n] = sum_c sA[m,c] * sW[c,n] * ( sum_b P[m,b,n] )

The inner loop is 32 dependent `aie::mmul<8,8,8>` MACs plus 32 int32 vector
adds, and the float unit is touched 4 times per chunk instead of 4 + 16 times
per block.

Two data layouts are supported:
  * "sep"  : the A and the B chunk travel in separate ObjectFifos (the phase 61
             layout).  Each fill costs two ShimNOCTile DMA channels, so the
             design cannot go past four workers.
  * "mrg"  : the A and the B chunk travel in one ObjectFifo (the kernel
             derives the two pointers from one base).  One fill per worker
             instead of two, which is what makes the 8- and 16-worker designs
             fit into the shim DMA budget.

usage:
  kbench62.py --M 512 --K 1024 --N 3584 --wcols 4 --m-tile 64 --n-tile 64 \
              --k-chunk 256 --layout mrg --work-dir DIR --recompile
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.jsonl")

# --------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------
KERNEL_C256 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// Q8_0 GEMM with a per-256-element (per K chunk) scale.
//
// A chunk: (MA * KCC/8) tiles of 8x8 int8, then MT f32 row scales
// B chunk: (NA * KCC/8) tiles of 8x8 int8, then NT f32 column scales
// C:       MT x NT f32, accumulated over the K chunks
//
// The 8x8 tile is four 16-lane groups; group g holds rows 2g and 2g+1 of the
// eight, so the chunk scale of a group is
//     [ sA[row 2g] * sW[col 0..7] , sA[row 2g+1] * sW[col 0..7] ]
extern "C" void kern(%(ARGS)s)
{
    constexpr int MT   = %(MT)d;
    constexpr int NT   = %(NT)d;
    constexpr int KCC  = %(KCC)d;
    constexpr int NBLK = KCC / 32;
    constexpr int MA   = MT / 8;
    constexpr int NA   = NT / 8;
    %(ADEFS)s
    const int8_t * a_tiles = at;
    const float  * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = %(BTILES)s;
    const float  * bsc = (const float *)(%(BTILES)s + KCC * NT);

    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        const int8_t * ai = a_tiles + i * (KCC / 8) * 64;
        for (int j = 0; j < NA; j++) {
            const int8_t * bj = b_tiles + j * (KCC / 8) * 64;
            aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
            MMUL acc(zero);
            for (int bb = 0; bb < NBLK; bb++) {
                const int8_t * ap = ai + bb * 4 * 64;
                const int8_t * bp = bj + bb * 4 * 64;
                acc.mac(aie::load_v<64>(ap),      aie::load_v<64>(bp));
                acc.mac(aie::load_v<64>(ap + 64),  aie::load_v<64>(bp + 64));
                acc.mac(aie::load_v<64>(ap + 128), aie::load_v<64>(bp + 128));
                acc.mac(aie::load_v<64>(ap + 192), aie::load_v<64>(bp + 192));
            }
            aie::vector<float, 64> p = aie::to_float(acc.to_vector<int32_t>());
            aie::vector<float, 8>  w = aie::load_v<8>(bsc + j * 8);
            aie::vector<float, 16> w16 = aie::broadcast<float, 16>(0.0f);
            w16.insert<8>(0, w);
            w16.insert<8>(1, w);
            for (int g = 0; g < 4; g++) {
                const float * ap2 = asc + i * 8 + 2 * g;
                aie::vector<float, 16> a16 = aie::broadcast<float, 16>(0.0f);
                a16.insert<8>(0, aie::broadcast<float, 8>(ap2[0]));
                a16.insert<8>(1, aie::broadcast<float, 8>(ap2[1]));
                aie::vector<float, 16> sc = aie::mul(a16, w16).to_vector<float>();
                aie::vector<float, 16> v =
                    aie::mul(p.template extract<16>(g), sc).to_vector<float>();
                float * c0 = c + (i * 8 + 2 * g + 0) * NT + j * 8;
                float * c1 = c + (i * 8 + 2 * g + 1) * NT + j * 8;
                aie::store_v(c0, aie::add(aie::load_v<8>(c0), v.template extract<8>(0)));
                aie::store_v(c1, aie::add(aie::load_v<8>(c1), v.template extract<8>(1)));
            }
        }
    }
}
"""

# The same kernel with two independent (i,j) chains, to test whether the extra
# live state pays for itself in this structure.
KERNEL_C256_2CH = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern(%(ARGS)s)
{
    constexpr int MT   = %(MT)d;
    constexpr int NT   = %(NT)d;
    constexpr int KCC  = %(KCC)d;
    constexpr int NBLK = KCC / 32;
    constexpr int MA   = MT / 8;
    constexpr int NA   = NT / 8;
    %(ADEFS)s
    const int8_t * a_tiles = at;
    const float  * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = %(BTILES)s;
    const float  * bsc = (const float *)(%(BTILES)s + KCC * NT);

    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        const int8_t * ai = a_tiles + i * (KCC / 8) * 64;
        const float  * asi = asc + i * 8;
        for (int j = 0; j + 1 < NA; j += 2) {
            const int8_t * bj0 = b_tiles + j * (KCC / 8) * 64;
            const int8_t * bj1 = b_tiles + (j + 1) * (KCC / 8) * 64;
            aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
            MMUL acc0(zero);
            MMUL acc1(zero);
            for (int bb = 0; bb < NBLK; bb++) {
                const int8_t * ap = ai + bb * 4 * 64;
                aie::vector<int8_t, 64> a0 = aie::load_v<64>(ap);
                aie::vector<int8_t, 64> a1 = aie::load_v<64>(ap + 64);
                aie::vector<int8_t, 64> a2 = aie::load_v<64>(ap + 128);
                aie::vector<int8_t, 64> a3 = aie::load_v<64>(ap + 192);
                const int8_t * bp0 = bj0 + bb * 4 * 64;
                const int8_t * bp1 = bj1 + bb * 4 * 64;
                acc0.mac(a0, aie::load_v<64>(bp0));
                acc0.mac(a1, aie::load_v<64>(bp0 + 64));
                acc0.mac(a2, aie::load_v<64>(bp0 + 128));
                acc0.mac(a3, aie::load_v<64>(bp0 + 192));
                acc1.mac(a0, aie::load_v<64>(bp1));
                acc1.mac(a1, aie::load_v<64>(bp1 + 64));
                acc1.mac(a2, aie::load_v<64>(bp1 + 128));
                acc1.mac(a3, aie::load_v<64>(bp1 + 192));
            }
            aie::vector<float, 64> p0 = aie::to_float(acc0.to_vector<int32_t>());
            aie::vector<float, 64> p1 = aie::to_float(acc1.to_vector<int32_t>());
            aie::vector<float, 8>  w0 = aie::load_v<8>(bsc + j * 8);
            aie::vector<float, 8>  w1 = aie::load_v<8>(bsc + (j + 1) * 8);
            aie::vector<float, 16> w16a = aie::broadcast<float, 16>(0.0f);
            w16a.insert<8>(0, w0);
            w16a.insert<8>(1, w0);
            aie::vector<float, 16> w16b = aie::broadcast<float, 16>(0.0f);
            w16b.insert<8>(0, w1);
            w16b.insert<8>(1, w1);
            for (int g = 0; g < 4; g++) {
                const float * ap2 = asi + 2 * g;
                aie::vector<float, 16> a16 = aie::broadcast<float, 16>(0.0f);
                a16.insert<8>(0, aie::broadcast<float, 8>(ap2[0]));
                a16.insert<8>(1, aie::broadcast<float, 8>(ap2[1]));
                aie::vector<float, 16> sca = aie::mul(a16, w16a).to_vector<float>();
                aie::vector<float, 16> scb = aie::mul(a16, w16b).to_vector<float>();
                aie::vector<float, 16> v0 =
                    aie::mul(p0.template extract<16>(g), sca).to_vector<float>();
                aie::vector<float, 16> v1 =
                    aie::mul(p1.template extract<16>(g), scb).to_vector<float>();
                float * c0 = c + (i * 8 + 2 * g + 0) * NT + j * 8;
                float * c1 = c + (i * 8 + 2 * g + 1) * NT + j * 8;
                float * c2 = c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8;
                float * c3 = c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8;
                aie::store_v(c0, aie::add(aie::load_v<8>(c0), v0.template extract<8>(0)));
                aie::store_v(c1, aie::add(aie::load_v<8>(c1), v0.template extract<8>(1)));
                aie::store_v(c2, aie::add(aie::load_v<8>(c2), v1.template extract<8>(0)));
                aie::store_v(c3, aie::add(aie::load_v<8>(c3), v1.template extract<8>(1)));
            }
        }
    }
}
"""

KERNELS = {"c256": KERNEL_C256, "c256x2": KERNEL_C256_2CH}


# The fixed-point variant: the exact Q8_0 weight scales are kept, but they
# are applied in the integer domain.  The weight block scale becomes an int32
# mantissa relative to one exponent per K chunk and column, the int32 partial
# sum is down-shifted and multiplied in the integer domain, and the float unit
# is only used once per chunk.
#
# A chunk: (MA * KCC/8) tiles of 8x8 int8, then MT f32 row scales
# B chunk: (NA * KCC/8) tiles of 8x8 int8, then NBLK*NT i32 mantissas,
#          then NT f32 column factors
KERNEL_FP256 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern(%(ARGS)s)
{
    constexpr int MT   = %(MT)d;
    constexpr int NT   = %(NT)d;
    constexpr int KCC  = %(KCC)d;
    constexpr int NBLK = KCC / 32;
    constexpr int MA   = MT / 8;
    constexpr int NA   = NT / 8;
    constexpr int SH   = %(SH)d;
    %(ADEFS)s
    const int8_t * a_tiles = at;
    const float  * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = %(BTILES)s;
    const int32_t * bman = (const int32_t *)(%(BTILES)s + KCC * NT);
    const float  * bfac = (const float *)(bman + NBLK * NT);

    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        const int8_t * ai = a_tiles + i * (KCC / 8) * 64;
        const float  * asi = asc + i * 8;
        for (int j = 0; j < NA; j++) {
            const int8_t * bj = b_tiles + j * (KCC / 8) * 64;
            aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
            aie::vector<int32_t, 16> s[4];
            for (int g = 0; g < 4; g++) s[g] = aie::broadcast<int32_t, 16>(0);
            for (int bb = 0; bb < NBLK; bb++) {
                const int8_t * ap = ai + bb * 4 * 64;
                const int8_t * bp = bj + bb * 4 * 64;
                MMUL acc(zero);
                acc.mac(aie::load_v<64>(ap),      aie::load_v<64>(bp));
                acc.mac(aie::load_v<64>(ap + 64),  aie::load_v<64>(bp + 64));
                acc.mac(aie::load_v<64>(ap + 128), aie::load_v<64>(bp + 128));
                acc.mac(aie::load_v<64>(ap + 192), aie::load_v<64>(bp + 192));
                // the down-shift is part of the accumulator read
                aie::vector<int32_t, 64> p = acc.to_vector<int32_t>(SH);
                const int32_t * mp = bman + bb * NT + j * 8;
                aie::vector<int32_t, 8> m8 = aie::load_v<8>(mp);
                aie::vector<int32_t, 16> m16 = aie::broadcast<int32_t, 16>(0);
                m16.insert<8>(0, m8);
                m16.insert<8>(1, m8);
                for (int g = 0; g < 4; g++) {
                    s[g] = aie::add(s[g], aie::mul(p.template extract<16>(g), m16).to_vector<int32_t>());
                }
            }
            aie::vector<float, 8>  w = aie::load_v<8>(bfac + j * 8);
            aie::vector<float, 16> w16 = aie::broadcast<float, 16>(0.0f);
            w16.insert<8>(0, w);
            w16.insert<8>(1, w);
            for (int g = 0; g < 4; g++) {
                const float * ap2 = asi + 2 * g;
                aie::vector<float, 16> a16 = aie::broadcast<float, 16>(0.0f);
                a16.insert<8>(0, aie::broadcast<float, 8>(ap2[0]));
                a16.insert<8>(1, aie::broadcast<float, 8>(ap2[1]));
                aie::vector<float, 16> sc = aie::mul(a16, w16).to_vector<float>();
                aie::vector<float, 16> v =
                    aie::mul(aie::to_float(s[g]), sc).to_vector<float>();
                float * c0 = c + (i * 8 + 2 * g + 0) * NT + j * 8;
                float * c1 = c + (i * 8 + 2 * g + 1) * NT + j * 8;
                aie::store_v(c0, aie::add(aie::load_v<8>(c0), v.template extract<8>(0)));
                aie::store_v(c1, aie::add(aie::load_v<8>(c1), v.template extract<8>(1)));
            }
        }
    }
}
"""

KERNELS["fp256"] = KERNEL_FP256


def chunk_sizes(m_tile, n_tile, k_chunk):
    a_bytes = m_tile * k_chunk + m_tile * 4
    b_bytes = k_chunk * n_tile + n_tile * 4
    return a_bytes, b_bytes


def pack(A_i8, As, B_i8, Bs, M, K, N, m_tile, n_tile, k_chunk):
    """Tile-major packing with the per-chunk (256 element) scales.

    A chunk: int8 tiles of (8 rows x 8 K), then M_TILE f32 row scales
    B chunk: int8 tiles of (8 K x 8 N), then N_TILE f32 column scales
    """
    n_mt = M // m_tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    ma = m_tile // 8
    na = n_tile // 8
    nblk = k_chunk // 32
    a_bytes, b_bytes = chunk_sizes(m_tile, n_tile, k_chunk)

    Ap = np.zeros(n_mt * n_kc * a_bytes, np.int8)
    for mt in range(n_mt):
        for kc in range(n_kc):
            base = (mt * n_kc + kc) * a_bytes
            abuf = Ap[base:base + m_tile * k_chunk].reshape(ma, nblk, 4, 64)
            for im in range(ma):
                for bb in range(nblk):
                    for l in range(4):
                        r0 = mt * m_tile + im * 8
                        c0 = kc * k_chunk + bb * 32 + l * 8
                        abuf[im, bb, l] = A_i8[r0:r0 + 8, c0:c0 + 8].reshape(-1)
            asc = Ap[base + m_tile * k_chunk:base + a_bytes].view(np.float32)
            asc[:] = As[mt * m_tile:(mt + 1) * m_tile, kc]

    Bp = np.zeros(n_nt * n_kc * b_bytes, np.int8)
    for nt in range(n_nt):
        for kc in range(n_kc):
            base = (nt * n_kc + kc) * b_bytes
            bbuf = Bp[base:base + k_chunk * n_tile].reshape(na, nblk, 4, 64)
            for jn in range(na):
                for bb in range(nblk):
                    for l in range(4):
                        r0 = kc * k_chunk + bb * 32 + l * 8
                        c0 = nt * n_tile + jn * 8
                        bbuf[jn, bb, l] = B_i8[r0:r0 + 8, c0:c0 + 8].reshape(-1)
            bsc = Bp[base + k_chunk * n_tile:base + b_bytes].view(np.float32)
            bsc[:] = Bs[kc, nt * n_tile:(nt + 1) * n_tile]

    return Ap, Bp


def ref_c256(A_i8, As, B_i8, Bs):
    """float64 reference: the per-chunk outer-product scales."""
    M, K = A_i8.shape
    N = B_i8.shape[1]
    nkc = As.shape[1]
    C = np.zeros((M, N), dtype=np.float64)
    for c in range(nkc):
        p = A_i8[:, c * 256:(c + 1) * 256].astype(np.int64) @ \
            B_i8[c * 256:(c + 1) * 256, :].astype(np.int64)
        C += p * As[:, c:c + 1].astype(np.float64) * Bs[c:c + 1, :].astype(np.float64)
    return C


def make_data(M, K, N, pattern, seed=7, block_scales=False):
    rng = np.random.default_rng(seed)
    nkc = K // 256
    if block_scales:
        # the fp256 layout keeps the exact Q8_0 per 32 element block scales
        A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
        B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
        if pattern == "realistic":
            As = (rng.uniform(0.002, 0.05, size=(M, nkc)) *
                  np.power(2.0, rng.integers(-3, 4, size=(M, nkc)))).astype(np.float32)
            Bs = (rng.uniform(0.002, 0.05, size=(K // 32, N)) *
                  np.power(2.0, rng.integers(-3, 4, size=(K // 32, N)))).astype(np.float32)
        else:
            As = np.ones((M, nkc), np.float32)
            Bs = np.ones((K // 32, N), np.float32)
        return A_i8, As, B_i8, Bs
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    if pattern == "realistic":
        As = (rng.uniform(0.002, 0.05, size=(M, nkc)) *
              np.power(2.0, rng.integers(-3, 4, size=(M, nkc)))).astype(np.float32)
        Bs = (rng.uniform(0.002, 0.05, size=(nkc, N)) *
              np.power(2.0, rng.integers(-3, 4, size=(nkc, N)))).astype(np.float32)
    elif pattern == "uniform":
        As = np.ones((M, nkc), np.float32)
        Bs = np.ones((nkc, N), np.float32)
    else:
        raise ValueError(pattern)
    return A_i8, As, B_i8, Bs


# --- the fixed point (fp256) layout -----------------------------------------
FP_MB = 8      # the extra mantissa bits
FP_SH = 4      # the down-shift of the int32 partial sum


def chunk_sizes_fp(m_tile, n_tile, k_chunk):
    nblk = k_chunk // 32
    a_bytes = m_tile * k_chunk + m_tile * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4 + n_tile * 4
    return a_bytes, b_bytes


def fixed_point_scales(Bs, nkc, nblk, N):
    """The int32 mantissas and the per column factors of the fixed point
    scale.  Bs is [nkc*nblk, N] (the per 32 element block scales)."""
    mant = np.zeros((nkc, nblk, N), np.int32)
    fac = np.zeros((nkc, N), np.float32)
    for c in range(nkc):
        s = Bs[c * nblk:(c + 1) * nblk, :].astype(np.float64)
        smax = s.max(axis=0)
        e = np.log2(np.maximum(smax, 1e-30))
        m = np.rint(s * np.exp2(FP_MB + FP_SH - e[None, :])).astype(np.int32)
        mant[c] = m
        # the down-shift is compensated by the extra mantissa bits
        fac[c] = np.exp2(e - FP_MB).astype(np.float32)
    return mant, fac


def pack_fp(A_i8, As, B_i8, Bs, M, K, N, m_tile, n_tile, k_chunk):
    """The fp256 packing: the same int8 tiles as the c256 layout, the row
    scales of the chunk and the fixed point mantissas of the weights."""
    n_mt = M // m_tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    ma = m_tile // 8
    na = n_tile // 8
    nblk = k_chunk // 32
    a_bytes, b_bytes = chunk_sizes_fp(m_tile, n_tile, k_chunk)
    mant, fac = fixed_point_scales(Bs, n_kc, nblk, N)

    Ap = np.zeros(n_mt * n_kc * a_bytes, np.int8)
    for mt in range(n_mt):
        for kc in range(n_kc):
            base = (mt * n_kc + kc) * a_bytes
            abuf = Ap[base:base + m_tile * k_chunk].reshape(ma, nblk, 4, 64)
            for im in range(ma):
                for bb in range(nblk):
                    for l in range(4):
                        r0 = mt * m_tile + im * 8
                        c0 = kc * k_chunk + bb * 32 + l * 8
                        abuf[im, bb, l] = A_i8[r0:r0 + 8, c0:c0 + 8].reshape(-1)
            asc = Ap[base + m_tile * k_chunk:base + a_bytes].view(np.float32)
            asc[:] = As[mt * m_tile:(mt + 1) * m_tile, kc]

    Bp = np.zeros(n_nt * n_kc * b_bytes, np.int8)
    for nt in range(n_nt):
        for kc in range(n_kc):
            base = (nt * n_kc + kc) * b_bytes
            bbuf = Bp[base:base + k_chunk * n_tile].reshape(na, nblk, 4, 64)
            for jn in range(na):
                for bb in range(nblk):
                    for l in range(4):
                        r0 = kc * k_chunk + bb * 32 + l * 8
                        c0 = nt * n_tile + jn * 8
                        bbuf[jn, bb, l] = B_i8[r0:r0 + 8, c0:c0 + 8].reshape(-1)
            off = k_chunk * n_tile
            mbuf = Bp[base + off:base + off + nblk * n_tile * 4].view(np.int32)
            for bb in range(nblk):
                mbuf[bb * n_tile:(bb + 1) * n_tile] = mant[kc, bb, nt * n_tile:(nt + 1) * n_tile]
            fbuf = Bp[base + off + nblk * n_tile * 4:base + b_bytes].view(np.float32)
            fbuf[:] = fac[kc, nt * n_tile:(nt + 1) * n_tile]
    return Ap, Bp


def ref_fp(A_i8, As, B_i8, Bs):
    """float64 reference: the exact Q8_0 per 32 element block weight scales
    with the per K chunk activation scale."""
    M, K = A_i8.shape
    N = B_i8.shape[1]
    nblk = 256 // 32          # the 32 element blocks per K chunk
    nkc = K // 256
    C = np.zeros((M, N), dtype=np.float64)
    for c in range(nkc):
        for b in range(nblk):
            sl = slice(c * 256 + b * 32, c * 256 + (b + 1) * 32)
            p = A_i8[:, sl].astype(np.int64) @ B_i8[sl, :].astype(np.int64)
            scale = Bs[c * nblk + b, :].astype(np.float64)
            C += p * As[:, c:c + 1].astype(np.float64) * scale[None, :]
    return C


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------
def build_design(M, K, N, wcols, m_tile, n_tile, k_chunk, variant, layout,
                 kernel_reps=1, place_cols=True, depth=1, drain_wait=True,
                 group_cols=True):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.iron.device.tile import AnyComputeTile, Tile
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config

    n_mt = M // m_tile
    n_nt_c = (N // wcols) // n_tile
    n_kc = K // k_chunk
    fp = variant == "fp256"
    if fp:
        a_bytes, b_bytes = chunk_sizes_fp(m_tile, n_tile, k_chunk)
    else:
        a_bytes, b_bytes = chunk_sizes(m_tile, n_tile, k_chunk)
    c_elems = m_tile * n_tile

    merged = layout == "mrg"
    in_bytes = (a_bytes + b_bytes) if merged else a_bytes
    b_off = a_bytes

    lmem = (in_bytes if merged else (a_bytes + b_bytes)) + c_elems * 4
    stack = 65536 - lmem
    if stack > 16384:
        stack = 16384
    if stack < 4096:
        raise ValueError("no room for the kernel stack")

    src = KERNELS[variant]
    if merged:
        args = "const int8_t * __restrict ab, float * __restrict c"
        btiles = "(ab + %d)" % b_off
    else:
        args = "const int8_t * __restrict at, const int8_t * __restrict bt, float * __restrict c"
        btiles = "bt"
    adefs = "const int8_t * at = ab;" if merged else ""
    src = src % {"ARGS": args, "BTILES": btiles, "ADEFS": adefs,
                 "MT": m_tile, "NT": n_tile, "KCC": k_chunk, "SH": FP_SH}

    kern = ExternalFunction(
        "kern",
        source_string=src,
        arg_types=([np.ndarray[(in_bytes,), np.dtype[np.int8]],
                    np.ndarray[(c_elems,), np.dtype[np.float32]]] if merged else
                   [np.ndarray[(a_bytes,), np.dtype[np.int8]],
                    np.ndarray[(b_bytes,), np.dtype[np.int8]],
                    np.ndarray[(c_elems,), np.dtype[np.float32]]]),
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(in_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]

    def core_fn(*fn_args):
        of_a = fn_args[0]
        of_b = fn_args[1] if not merged else None
        of_c = fn_args[-2]
        kern = fn_args[-1]
        for _ in iron_range(n_mt * n_nt_c):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0.0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                for _r in iron_range(kernel_reps):
                    if merged:
                        kern(elem_a, elem_c)
                    else:
                        elem_b = of_b.acquire(1)
                        kern(elem_a, elem_b, elem_c)
                        of_b.release(1)
                of_a.release(1)
            of_c.release(1)

    fifos = []
    workers = []
    n_dev_cols = 4
    for col in range(wcols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=depth)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=depth)
        if merged:
            # the A and the B chunk share one FIFO: one fill per worker
            fb = None
            fn_args = [fa.cons(), fc.prod(), kern]
        else:
            fb = ObjectFifo(b_ty, name=f"b{col}", depth=depth)
            fn_args = [fa.cons(), fb.cons(), fc.prod(), kern]
        fifos.append((fa, fb, fc))
        if place_cols:
            # NPU1: 4 columns, row 0 = shim, row 1 = memtile, rows 2..5 = cores
            tile = Tile(col=col % n_dev_cols, row=2 + col // n_dev_cols)
        else:
            tile = AnyComputeTile
        workers.append(Worker(core_fn, fn_args,
                              stack_size=stack, tile=tile))

    A_ty = np.ndarray[(n_mt * n_kc * in_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(wcols * n_nt_c * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * in_bytes,),
                                  offset=mt * n_kc * in_bytes,
                                  sizes=[n_kc * in_bytes], strides=[1])
              for mt in range(n_mt)]
    if merged:
        # one buffer per (column, n-tile, m-tile) group, holding the A chunk
        # and the B chunk of every K chunk back to back
        MG = wcols * n_nt_c * n_mt * n_kc * in_bytes
        a_taps = [[TensorAccessPattern(tensor_dims=(MG,),
                                       offset=((col * n_nt_c + nt) * n_mt + mt) * n_kc * in_bytes,
                                       sizes=[n_kc * in_bytes], strides=[1])
                   for mt in range(n_mt)] for col in range(wcols) for nt in range(n_nt_c)]
    b_taps = [[TensorAccessPattern(tensor_dims=(wcols * n_nt_c * n_kc * b_bytes,),
                                   offset=(col * n_nt_c + nt) * n_kc * b_bytes,
                                   sizes=[n_kc * b_bytes], strides=[1])
               for nt in range(n_nt_c)] for col in range(wcols)]
    c_taps = [[[TensorAccessPattern(tensor_dims=(M, N),
                                    offset=mt * m_tile * N + (col * n_nt_c + nt) * n_tile,
                                    sizes=[m_tile, n_tile], strides=[N, 1])
                for nt in range(n_nt_c)] for mt in range(n_mt)] for col in range(wcols)]

    def sequence(A, B, C, *ends):
        a_p = [f[0].prod() for f in fifos]
        b_p = [f[1].prod() for f in fifos if f[1] is not None]
        c_c = [f[2].cons() for f in fifos]
        if group_cols:
            # One task group per (m-tile, n-tile): the fills of every column
            # are issued together and the single wait at the end of the group
            # lets the workers compute at the same time.  With one task group
            # per (m-tile, column, n-tile) the drain of the previous group
            # blocks the fill of the next one and the workers run serially.
            for mt in range(n_mt):
                for nt in range(n_nt_c):
                    tg = iron.TaskGroup()
                    for col in range(wcols):
                        if merged:
                            a_p[col].fill(A, tap=a_taps[col * n_nt_c + nt][mt], group=tg)
                        else:
                            a_p[col].fill(A, tap=a_taps[mt], group=tg)
                            b_p[col].fill(B, tap=b_taps[col][nt], group=tg)
                        c_c[col].drain(C, tap=c_taps[col][mt][nt], group=tg,
                                      wait=(col == wcols - 1))
                    tg.finish()
        else:
            for mt in range(n_mt):
                for col in range(wcols):
                    for nt in range(n_nt_c):
                        tg = iron.TaskGroup()
                        if merged:
                            a_p[col].fill(A, tap=a_taps[col * n_nt_c + nt][mt], group=tg)
                        else:
                            a_p[col].fill(A, tap=a_taps[mt], group=tg)
                            b_p[col].fill(B, tap=b_taps[col][nt], group=tg)
                        last = (mt == n_mt - 1 and col == wcols - 1
                                and nt == n_nt_c - 1)
                        c_c[col].drain(C, tap=c_taps[col][mt][nt], group=tg,
                                      wait=(drain_wait or last))
                        tg.finish()

    prod_cons = []
    for f in fifos:
        prod_cons.append(f[0].prod())
    for f in fifos:
        if f[1] is not None:
            prod_cons.append(f[1].prod())
    for f in fifos:
        prod_cons.append(f[2].cons())
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def run_kernel(wd, Ap, Bp, M, K, N, repeat, wcols, m_tile, n_tile, k_chunk,
               layout, variant="c256"):
    import pyxrt
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(os.path.join(wd, "design.xclbin"))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    gid3, gid1 = kern.group_id(3), kern.group_id(1)

    n_mt = M // m_tile
    n_nt_c = (N // wcols) // n_tile
    n_kc = K // k_chunk
    if variant == "fp256":
        a_bytes, b_bytes = chunk_sizes_fp(m_tile, n_tile, k_chunk)
    else:
        a_bytes, b_bytes = chunk_sizes(m_tile, n_tile, k_chunk)
    merged = layout == "mrg"
    in_bytes = (a_bytes + b_bytes) if merged else a_bytes

    # A is replicated per column, B is N-sliced
    Ap_col = np.zeros((wcols, n_mt * n_kc * in_bytes), np.int8)
    Bp_col = np.zeros((wcols * n_nt_c * n_kc * b_bytes,), np.int8)
    for col in range(wcols):
        if merged:
            for mt in range(n_mt):
                for kc in range(n_kc):
                    src = (mt * n_kc + kc) * a_bytes
                    dst = (mt * n_kc + kc) * in_bytes
                    Ap_col[col][dst:dst + a_bytes] = Ap[src:src + a_bytes]
        else:
            Ap_col[col] = Ap
    for col in range(wcols):
        for nt in range(n_nt_c):
            src = ((col * n_nt_c + nt) * n_kc * b_bytes)
            dst = (col * n_nt_c + nt) * n_kc * b_bytes
            Bp_col[dst:dst + n_kc * b_bytes] = Bp[src:src + n_kc * b_bytes]

    if merged:
        # the A and the B of a worker travel in one buffer: one block per
        # (column, n-tile, m-tile, k-chunk) holding [A chunk | B chunk]
        buf = np.zeros((wcols, n_nt_c, n_mt, n_kc, in_bytes), np.int8)
        for col in range(wcols):
            for nt in range(n_nt_c):
                for mt in range(n_mt):
                    for kc in range(n_kc):
                        s = (mt * n_kc + kc) * a_bytes
                        buf[col, nt, mt, kc, :a_bytes] = Ap[s:s + a_bytes]
                        bs = ((col * n_nt_c + nt) * n_kc + kc) * b_bytes
                        buf[col, nt, mt, kc, a_bytes:] = Bp[bs:bs + b_bytes]
        buf = buf.reshape(-1)
    else:
        buf = None

    a_bo = pyxrt.bo(dev, (buf.nbytes if merged else Ap_col.nbytes),
                    pyxrt.bo.flags.host_only, gid3)
    b_bo = None if merged else pyxrt.bo(dev, Bp_col.nbytes,
                                         pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(dev, M * N * 4, pyxrt.bo.flags.host_only, gid3)
    inst = os.path.join(wd, "design.insts.bin")
    isz = os.path.getsize(inst)
    inst_bo = pyxrt.bo(dev, isz, pyxrt.bo.flags.cacheable, gid1)
    with open(inst, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    if merged:
        a_bo.write(buf.tobytes(), 0)
    else:
        a_bo.write(Ap_col.tobytes(), 0)
        b_bo.write(Bp_col.tobytes(), 0)
    c_bo.write(b"\x00" * (M * N * 4), 0)
    for b in (a_bo, b_bo, c_bo):
        if b is not None:
            b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    times = []
    for _ in range(repeat):
        r = pyxrt.run(kern)
        r.set_arg(0, 3)
        r.set_arg(1, inst_bo)
        r.set_arg(2, isz)
        r.set_arg(3, a_bo)
        r.set_arg(4, b_bo if b_bo is not None else a_bo)
        r.set_arg(5, c_bo)
        t0 = time.perf_counter()
        r.start()
        r.wait()
        times.append((time.perf_counter() - t0) * 1000.0)
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    C = np.frombuffer(c_bo.read(M * N * 4, 0), np.float32).reshape(M, N)
    return times, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--wcols", type=int, default=4)
    ap.add_argument("--m-tile", type=int, default=64)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--variant", default="c256", choices=list(KERNELS.keys()))
    ap.add_argument("--layout", default="mrg", choices=["mrg", "sep"])
    ap.add_argument("--kernel-reps", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--pattern", default="realistic")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--recompile", action="store_true")
    ap.add_argument("--place-cols", dest="place_cols", action="store_true", default=True)
    ap.add_argument("--no-place-cols", dest="place_cols", action="store_false")
    ap.add_argument("--drain-wait", dest="drain_wait", action="store_true", default=True)
    ap.add_argument("--no-drain-wait", dest="drain_wait", action="store_false")
    ap.add_argument("--group-cols", dest="group_cols", action="store_true", default=True)
    ap.add_argument("--no-group-cols", dest="group_cols", action="store_false")
    a = ap.parse_args()

    wd = a.work_dir
    os.makedirs(wd, exist_ok=True)
    xp = os.path.join(wd, "design.xclbin")
    ip = os.path.join(wd, "design.insts.bin")
    pp = os.path.join(wd, "main.pdi")

    compile_s = 0.0
    compile_ok = True
    compile_err = ""
    if a.recompile or not os.path.exists(xp):
        import aie.iron as iron
        from aie.iron import CompileTime, In, Out

        @iron.jit
        def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
            return build_design(a.M, a.K, a.N, a.wcols, a.m_tile, a.n_tile,
                                a.k_chunk, a.variant, a.layout, a.kernel_reps,
                                a.place_cols, 1, a.drain_wait, a.group_cols)

        t0 = time.time()
        try:
            design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        except Exception as ex:            # noqa: BLE001
            compile_ok = False
            compile_err = str(ex)[-4000:]
        compile_s = round(time.time() - t0, 2)
        print(f"compile {'ok' if compile_ok else 'FAILED'} in {compile_s}s")
        if not compile_ok:
            print(compile_err)
            with open(RESULTS, "a") as f:
                f.write(json.dumps({"tag": a.tag, "variant": a.variant,
                                    "compile_ok": False, "compile_s": compile_s,
                                    "error": compile_err}) + "\n")
            sys.exit(1)

    fp = a.variant == "fp256"
    A_i8, As, B_i8, Bs = make_data(a.M, a.K, a.N, a.pattern, block_scales=fp)
    if fp:
        Ap, Bp = pack_fp(A_i8, As, B_i8, Bs, a.M, a.K, a.N, a.m_tile, a.n_tile, a.k_chunk)
        Cref = ref_fp(A_i8, As, B_i8, Bs)
    else:
        Ap, Bp = pack(A_i8, As, B_i8, Bs, a.M, a.K, a.N, a.m_tile, a.n_tile, a.k_chunk)
        Cref = ref_c256(A_i8, As, B_i8, Bs)

    times, C = run_kernel(wd, Ap, Bp, a.M, a.K, a.N, a.repeat, a.wcols,
                          a.m_tile, a.n_tile, a.k_chunk, a.layout, a.variant)
    med = sorted(times)[len(times) // 2]
    d = np.abs(C.astype(np.float64) - Cref)
    refmax = float(np.max(np.abs(Cref)))
    rmse = float(np.sqrt(np.mean(d ** 2)))
    rel = float(np.max(d / np.maximum(np.abs(Cref), 1e-6)))

    rec = {
        "tag": a.tag, "variant": a.variant, "layout": a.layout,
        "pattern": a.pattern,
        "M": a.M, "K": a.K, "N": a.N, "wcols": a.wcols,
        "m_tile": a.m_tile, "n_tile": a.n_tile, "k_chunk": a.k_chunk,
        "kernel_reps": a.kernel_reps, "drain_wait": a.drain_wait,
        "group_cols": a.group_cols, "place_cols": a.place_cols,
        "compile_ok": compile_ok, "compile_s": compile_s,
        "ms_median": round(med, 3),
        "ms_all": [round(t, 3) for t in times],
        "gop_s": round((a.M * a.K * a.N * 2) / max(med, 1e-9) / 1e6, 3),
        "max_abs_err": float(d.max()),
        "ref_max": refmax,
        "rel_err": rel,
        "rmse": rmse,
        "rel_rmse": rmse / max(refmax, 1e-9),
    }
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
