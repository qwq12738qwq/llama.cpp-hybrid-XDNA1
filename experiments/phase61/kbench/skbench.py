#!/usr/bin/env python3
"""Phase 61 standalone kernel benchmark for the SCALE-PRESERVING Q8_0 GEMM.

The production kernel (phase 40M `q8_c4_mmul_v2`, engine `c4_mmul_v2`) keeps
Q8_0's per-32-block scales: for output element C[m,n]

    C[m,n] = sum_b sA[m,b] * sW[b,n] * ( sum_{k in b} A_q[m,k] * W_q[k,n] )

It is numerically correct (PPL 65.21 vs Vulkan 65.29 in phase 60) but slow
(433 ms/GEMM) because the C tile is read-modify-written once per 32-block.

This harness measures candidate kernels for the same math. Every candidate is
compiled with the real aiecc, run on the real NPU with real Q8_0-shaped data
and checked against a float64 numpy reference that uses the same per-block
outer-product scale semantics.

Usage:
  skbench.py --M 512 --K 1024 --N 3584 --variant s2 --work-dir /tmp/sk
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
# kernels
# --------------------------------------------------------------------------

# s0: the production kernel, verbatim (phase 40M p32). 4x8 mmul, per-32-block
# C read-modify-write. Reference point for correctness and speed.
KERNEL_S0 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT = 32, NT = 64, KCC = 256, NBLK = KCC / 32;
    constexpr int MA = MT / 4, NA = NT / 8, KA = KCC / 8, KB = KCC / 8;
    const int8_t * a_tiles = at;
    const float * as = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float * bs = (const float *)(bt + KCC * NT);
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

# s1: 4x8 mmul, float accumulator in registers for the whole K chunk.
# Per 32-block the int32 partial is converted to float, scaled by the
# rank-1 (row x column) block scale and added to the register accumulator.
# The C tile is touched once per (i,j) instead of once per (i,j,block).
KERNEL_S1 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT = 32, NT = 64, KCC = 256, NBLK = KCC / 32;
    constexpr int MA = MT / 4, NA = NT / 8, KA = KCC / 8, KB = KCC / 8;
    const int8_t * a_tiles = at;
    const float * as = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float * bs = (const float *)(bt + KCC * NT);
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        for (int j = 0; j < NA; j++) {
            aie::vector<float, 8> a0 = aie::broadcast<float, 8>(0.0f);
            aie::vector<float, 8> a1 = aie::broadcast<float, 8>(0.0f);
            aie::vector<float, 8> a2 = aie::broadcast<float, 8>(0.0f);
            aie::vector<float, 8> a3 = aie::broadcast<float, 8>(0.0f);
            for (int bb = 0; bb < NBLK; bb++) {
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
                a0 = aie::add(a0, p.template extract<8>(0));
                a1 = aie::add(a1, p.template extract<8>(1));
                a2 = aie::add(a2, p.template extract<8>(2));
                a3 = aie::add(a3, p.template extract<8>(3));
            }
            // the C tile is accumulated across K chunks, so read-modify-write
            // once per (i,j) per chunk
            aie::store_v(c + (i * 4 + 0) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 4 + 0) * NT + j * 8), a0));
            aie::store_v(c + (i * 4 + 1) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 4 + 1) * NT + j * 8), a1));
            aie::store_v(c + (i * 4 + 2) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 4 + 2) * NT + j * 8), a2));
            aie::store_v(c + (i * 4 + 3) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 4 + 3) * NT + j * 8), a3));
        }
    }
}
"""

# s2: native 8x8 mmul (v64acc), float accumulator in registers, 16-lane
# granularity. The 8x8 tile is four 16-lane groups; group q holds rows 2q and
# 2q+1. The per-block scale is a rank-1 outer product: row scale sA[r] times
# column scale sW[l].
KERNEL_S2 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT = 32, NT = 64, KCC = 256, NBLK = KCC / 32;
    constexpr int MA = MT / 8, NA = NT / 8, KA = KCC / 8, KB = KCC / 8;
    const int8_t * a_tiles = at;
    const float * as = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float * bs = (const float *)(bt + KCC * NT);
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        for (int j = 0; j < NA; j++) {
            aie::vector<float, 16> f0 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f1 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f2 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f3 = aie::broadcast<float, 16>(0.0f);
            for (int bb = 0; bb < NBLK; bb++) {
                aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
                MMUL acc(zero);
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 64> av = aie::load_v<64>(a_tiles + (i * KA + bb * 4 + kk) * 64);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
                aie::vector<float, 64> p = aie::to_float(acc.to_vector<int32_t>());
                aie::vector<float, 8> bsv = aie::load_v<8>(bs + bb * NT + j * 8);
                aie::vector<float, 16> w16 = aie::broadcast<float, 16>(0.0f);
                w16.insert<8>(0, bsv);
                w16.insert<8>(1, bsv);
                // group q holds rows 2q and 2q+1
                aie::vector<float, 16> a0 = aie::broadcast<float, 16>(0.0f);
                a0.insert<8>(0, aie::broadcast<float, 8>(as[(i * 8 + 0) * NBLK + bb]));
                a0.insert<8>(1, aie::broadcast<float, 8>(as[(i * 8 + 1) * NBLK + bb]));
                aie::vector<float, 16> a1 = aie::broadcast<float, 16>(0.0f);
                a1.insert<8>(0, aie::broadcast<float, 8>(as[(i * 8 + 2) * NBLK + bb]));
                a1.insert<8>(1, aie::broadcast<float, 8>(as[(i * 8 + 3) * NBLK + bb]));
                aie::vector<float, 16> a2 = aie::broadcast<float, 16>(0.0f);
                a2.insert<8>(0, aie::broadcast<float, 8>(as[(i * 8 + 4) * NBLK + bb]));
                a2.insert<8>(1, aie::broadcast<float, 8>(as[(i * 8 + 5) * NBLK + bb]));
                aie::vector<float, 16> a3 = aie::broadcast<float, 16>(0.0f);
                a3.insert<8>(0, aie::broadcast<float, 8>(as[(i * 8 + 6) * NBLK + bb]));
                a3.insert<8>(1, aie::broadcast<float, 8>(as[(i * 8 + 7) * NBLK + bb]));
                f0 = aie::add(f0, aie::mul(p.template extract<16>(0), aie::mul(a0, w16).to_vector<float>()).to_vector<float>());
                f1 = aie::add(f1, aie::mul(p.template extract<16>(1), aie::mul(a1, w16).to_vector<float>()).to_vector<float>());
                f2 = aie::add(f2, aie::mul(p.template extract<16>(2), aie::mul(a2, w16).to_vector<float>()).to_vector<float>());
                f3 = aie::add(f3, aie::mul(p.template extract<16>(3), aie::mul(a3, w16).to_vector<float>()).to_vector<float>());
            }
            aie::store_v(c + (i * 8 + 0) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 0) * NT + j * 8), f0.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 1) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 1) * NT + j * 8), f0.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 2) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2) * NT + j * 8), f1.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 3) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 3) * NT + j * 8), f1.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 4) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 4) * NT + j * 8), f2.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 5) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 5) * NT + j * 8), f2.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 6) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 6) * NT + j * 8), f3.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 7) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 7) * NT + j * 8), f3.template extract<8>(1)));
        }
    }
}
"""

# s3: 8x8 mmul, float accumulator, but the per-block multiply uses only the
# column scale (the row scale is folded into a per-row constant applied once
# at the end). This is a DIAGNOSTIC: it shows the cost of the rank-1 outer
# product. The host compensates by requantizing the activation to one scale
# per row, so the numerics are NOT the production semantics.
KERNEL_S3 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT = 32, NT = 64, KCC = 256, NBLK = KCC / 32;
    constexpr int MA = MT / 8, NA = NT / 8, KA = KCC / 8, KB = KCC / 8;
    const int8_t * a_tiles = at;
    const float * as = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float * bs = (const float *)(bt + KCC * NT);
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        for (int j = 0; j < NA; j++) {
            aie::vector<float, 16> f0 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f1 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f2 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f3 = aie::broadcast<float, 16>(0.0f);
            float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
            r0 = as[(i * 8 + 0) * NBLK + 0];
            r1 = as[(i * 8 + 1) * NBLK + 0];
            r2 = as[(i * 8 + 2) * NBLK + 0];
            r3 = as[(i * 8 + 3) * NBLK + 0];
            (void)r0; (void)r1; (void)r2; (void)r3;
            for (int bb = 0; bb < NBLK; bb++) {
                aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
                MMUL acc(zero);
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 64> av = aie::load_v<64>(a_tiles + (i * KA + bb * 4 + kk) * 64);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
                aie::vector<float, 64> p = aie::to_float(acc.to_vector<int32_t>());
                aie::vector<float, 8> bsv = aie::load_v<8>(bs + bb * NT + j * 8);
                aie::vector<float, 16> w16 = aie::broadcast<float, 16>(0.0f);
                w16.insert<8>(0, bsv);
                w16.insert<8>(1, bsv);
                f0 = aie::add(f0, aie::mul(p.template extract<16>(0), w16).to_vector<float>());
                f1 = aie::add(f1, aie::mul(p.template extract<16>(1), w16).to_vector<float>());
                f2 = aie::add(f2, aie::mul(p.template extract<16>(2), w16).to_vector<float>());
                f3 = aie::add(f3, aie::mul(p.template extract<16>(3), w16).to_vector<float>());
            }
            aie::store_v(c + (i * 8 + 0) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 0) * NT + j * 8), f0.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 1) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 1) * NT + j * 8), f0.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 2) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2) * NT + j * 8), f1.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 3) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 3) * NT + j * 8), f1.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 4) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 4) * NT + j * 8), f2.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 5) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 5) * NT + j * 8), f2.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 6) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 6) * NT + j * 8), f3.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 7) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 7) * NT + j * 8), f3.template extract<8>(1)));
        }
    }
}
"""

# s4: 8x8 mmul with two independent (i,j) accumulators so the matrix unit
# has two independent dependency chains. Same math as s2.
KERNEL_S4 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT = 32, NT = 64, KCC = 256, NBLK = KCC / 32;
    constexpr int MA = MT / 8, NA = NT / 8, KA = KCC / 8, KB = KCC / 8;
    const int8_t * a_tiles = at;
    const float * as = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float * bs = (const float *)(bt + KCC * NT);
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        for (int j = 0; j + 1 < NA; j += 2) {
            aie::vector<float, 16> f0[4], f1[4];
            for (int q = 0; q < 4; q++) {
                f0[q] = aie::broadcast<float, 16>(0.0f);
                f1[q] = aie::broadcast<float, 16>(0.0f);
            }
            for (int bb = 0; bb < NBLK; bb++) {
                aie::vector<float, 8> bsv0 = aie::load_v<8>(bs + bb * NT + j * 8);
                aie::vector<float, 8> bsv1 = aie::load_v<8>(bs + bb * NT + (j + 1) * 8);
                aie::vector<float, 16> w0 = aie::broadcast<float, 16>(0.0f);
                w0.insert<8>(0, bsv0);
                w0.insert<8>(1, bsv0);
                aie::vector<float, 16> w1 = aie::broadcast<float, 16>(0.0f);
                w1.insert<8>(0, bsv1);
                w1.insert<8>(1, bsv1);
                aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
                MMUL acc0(zero);
                MMUL acc1(zero);
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 64> av = aie::load_v<64>(a_tiles + (i * KA + bb * 4 + kk) * 64);
                    aie::vector<int8_t, 64> bv0 = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    aie::vector<int8_t, 64> bv1 = aie::load_v<64>(b_tiles + ((j + 1) * KB + bb * 4 + kk) * 64);
                    acc0.mac(av, bv0);
                    acc1.mac(av, bv1);
                }
                aie::vector<float, 64> p0 = aie::to_float(acc0.to_vector<int32_t>());
                aie::vector<float, 64> p1 = aie::to_float(acc1.to_vector<int32_t>());
                aie::vector<float, 16> a0 = aie::broadcast<float, 16>(0.0f);
                a0.insert<8>(0, aie::broadcast<float, 8>(as[(i * 8 + 0) * NBLK + bb]));
                a0.insert<8>(1, aie::broadcast<float, 8>(as[(i * 8 + 1) * NBLK + bb]));
                aie::vector<float, 16> a1 = aie::broadcast<float, 16>(0.0f);
                a1.insert<8>(0, aie::broadcast<float, 8>(as[(i * 8 + 2) * NBLK + bb]));
                a1.insert<8>(1, aie::broadcast<float, 8>(as[(i * 8 + 3) * NBLK + bb]));
                aie::vector<float, 16> a2 = aie::broadcast<float, 16>(0.0f);
                a2.insert<8>(0, aie::broadcast<float, 8>(as[(i * 8 + 4) * NBLK + bb]));
                a2.insert<8>(1, aie::broadcast<float, 8>(as[(i * 8 + 5) * NBLK + bb]));
                aie::vector<float, 16> a3 = aie::broadcast<float, 16>(0.0f);
                a3.insert<8>(0, aie::broadcast<float, 8>(as[(i * 8 + 6) * NBLK + bb]));
                a3.insert<8>(1, aie::broadcast<float, 8>(as[(i * 8 + 7) * NBLK + bb]));
                f0[0] = aie::add(f0[0], aie::mul(p0.template extract<16>(0), aie::mul(a0, w0).to_vector<float>()).to_vector<float>());
                f1[0] = aie::add(f1[0], aie::mul(p1.template extract<16>(0), aie::mul(a0, w1).to_vector<float>()).to_vector<float>());
                f0[1] = aie::add(f0[1], aie::mul(p0.template extract<16>(1), aie::mul(a1, w0).to_vector<float>()).to_vector<float>());
                f1[1] = aie::add(f1[1], aie::mul(p1.template extract<16>(1), aie::mul(a1, w1).to_vector<float>()).to_vector<float>());
                f0[2] = aie::add(f0[2], aie::mul(p0.template extract<16>(2), aie::mul(a2, w0).to_vector<float>()).to_vector<float>());
                f1[2] = aie::add(f1[2], aie::mul(p1.template extract<16>(2), aie::mul(a2, w1).to_vector<float>()).to_vector<float>());
                f0[3] = aie::add(f0[3], aie::mul(p0.template extract<16>(3), aie::mul(a3, w0).to_vector<float>()).to_vector<float>());
                f1[3] = aie::add(f1[3], aie::mul(p1.template extract<16>(3), aie::mul(a3, w1).to_vector<float>()).to_vector<float>());
            }
            for (int q = 0; q < 4; q++) {
                aie::store_v(c + (i * 8 + 2 * q + 0) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * q + 0) * NT + j * 8), f0[q].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * q + 1) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * q + 1) * NT + j * 8), f0[q].template extract<8>(1)));
                aie::store_v(c + (i * 8 + 2 * q + 0) * NT + (j + 1) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * q + 0) * NT + (j + 1) * 8), f1[q].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * q + 1) * NT + (j + 1) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * q + 1) * NT + (j + 1) * 8), f1[q].template extract<8>(1)));
            }
        }
    }
}
"""

KERNELS = {"s0": KERNEL_S0, "s1": KERNEL_S1, "s2": KERNEL_S2,
           "s3": KERNEL_S3, "s4": KERNEL_S4}

# s5: 8x8 mmul, 16-lane float accumulator, two independent (i,j) chains, and
# the row scale pre-broadcast on the host into the exact 16-lane pattern the
# accumulator needs. The host stores, per (i, block), the 4 groups of 16
# floats [sA[row] x8, sA[row+1] x8]; the device then spends one 512-bit load
# per group instead of two broadcasts and an insert.
KERNEL_S5 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT = 32, NT = 64, KCC = 256, NBLK = KCC / 32;
    constexpr int MA = MT / 8, NA = NT / 8, KA = KCC / 8, KB = KCC / 8;
    const int8_t * a_tiles = at;
    const float * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float * bsc = (const float *)(bt + KCC * NT);
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        for (int j = 0; j + 1 < NA; j += 2) {
            aie::vector<float, 16> f0[4], f1[4];
            for (int g = 0; g < 4; g++) {
                f0[g] = aie::broadcast<float, 16>(0.0f);
                f1[g] = aie::broadcast<float, 16>(0.0f);
            }
            for (int bb = 0; bb < NBLK; bb++) {
                aie::vector<float, 8> bs0 = aie::load_v<8>(bsc + (bb * NA + j) * 8);
                aie::vector<float, 8> bs1 = aie::load_v<8>(bsc + (bb * NA + j + 1) * 8);
                aie::vector<float, 16> w0 = aie::broadcast<float, 16>(0.0f);
                w0.insert<8>(0, bs0);
                w0.insert<8>(1, bs0);
                aie::vector<float, 16> w1 = aie::broadcast<float, 16>(0.0f);
                w1.insert<8>(0, bs1);
                w1.insert<8>(1, bs1);
                aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
                MMUL acc0(zero);
                MMUL acc1(zero);
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 64> av = aie::load_v<64>(a_tiles + (i * KA + bb * 4 + kk) * 64);
                    aie::vector<int8_t, 64> bv0 = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    aie::vector<int8_t, 64> bv1 = aie::load_v<64>(b_tiles + ((j + 1) * KB + bb * 4 + kk) * 64);
                    acc0.mac(av, bv0);
                    acc1.mac(av, bv1);
                }
                aie::vector<float, 64> p0 = aie::to_float(acc0.to_vector<int32_t>());
                aie::vector<float, 64> p1 = aie::to_float(acc1.to_vector<int32_t>());
                const float * ap = asc + (i * NBLK + bb) * 64;
                for (int g = 0; g < 4; g++) {
                    aie::vector<float, 16> a16 = aie::load_v<16>(ap + g * 16);
                    aie::vector<float, 16> sc0 = aie::mul(a16, w0).to_vector<float>();
                    aie::vector<float, 16> sc1 = aie::mul(a16, w1).to_vector<float>();
                    f0[g] = aie::add(f0[g], aie::mul(p0.template extract<16>(g), sc0).to_vector<float>());
                    f1[g] = aie::add(f1[g], aie::mul(p1.template extract<16>(g), sc1).to_vector<float>());
                }
            }
            for (int g = 0; g < 4; g++) {
                aie::store_v(c + (i * 8 + 2 * g + 0) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 0) * NT + j * 8), f0[g].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * g + 1) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 1) * NT + j * 8), f0[g].template extract<8>(1)));
                aie::store_v(c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8), f1[g].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8), f1[g].template extract<8>(1)));
            }
        }
    }
}
"""
KERNELS["s5"] = KERNEL_S5

# s6: DIAGNOSTIC floor for the 8x8 structure. No scale application at all:
# the int32 partial is accumulated over the whole chunk and C is written once
# per (i,j) per chunk. The numbers are wrong on purpose (no Q8_0 scales); the
# timing is the cost of the mmul chain plus the LMEM loads alone.
KERNEL_S6 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT = 32, NT = 64, KCC = 256, NBLK = KCC / 32;
    constexpr int MA = MT / 8, NA = NT / 8, KA = KCC / 8, KB = KCC / 8;
    const int8_t * a_tiles = at;
    const int8_t * b_tiles = bt;
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        for (int j = 0; j < NA; j++) {
            aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
            MMUL acc(zero);
            for (int bb = 0; bb < NBLK; bb++) {
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 64> av = aie::load_v<64>(a_tiles + (i * KA + bb * 4 + kk) * 64);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
            }
            aie::vector<int32_t, 64> r = acc.to_vector<int32_t>();
            for (int g = 0; g < 4; g++) {
                for (int r2 = 0; r2 < 2; r2++) {
                    aie::vector<int32_t, 8> v = r.template extract<8>(g * 2 + r2);
                    aie::vector<float, 8> f = aie::to_float(v);
                    float * cp = c + (i * 8 + 2 * g + r2) * NT + j * 8;
                    aie::store_v(cp, aie::add(aie::load_v<8>(cp), f));
                }
            }
        }
    }
}
"""
KERNELS["s6"] = KERNEL_S6

# s7: DIAGNOSTIC control. Empty kernel: no arithmetic at all, but the same
# DMA sequence (same number of tasks, same bytes). The wall time is therefore
# the cost of the DMA and the task sequence alone.
KERNEL_S7 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    const int8_t * a_tiles = at;
    const int8_t * b_tiles = bt;
    c[0] = (float)(a_tiles[0] + b_tiles[0]);
}
"""
KERNELS["s7"] = KERNEL_S7

# s8: s4 with the A and B operand tiles preloaded before the mac chain.
# The 8 loads are independent, so the compiler can issue them back to back and
# overlap the LMEM latency instead of exposing it between the four chained
# vmac instructions. Pointer bases are hoisted out of the loops so the inner
# loop carries no index arithmetic.
KERNEL_S8 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT = 32, NT = 64, KCC = 256, NBLK = KCC / 32;
    constexpr int MA = MT / 8, NA = NT / 8, KA = KCC / 8, KB = KCC / 8;
    const int8_t * a_tiles = at;
    const float * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float * bsc = (const float *)(bt + KCC * NT);
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        const int8_t * ai = a_tiles + i * KA * 64;
        const float * asci = asc + i * 8 * NBLK;
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
                aie::store_v(c + (i * 8 + 2 * g + 0) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 0) * NT + j * 8), f0[g].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * g + 1) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 1) * NT + j * 8), f0[g].template extract<8>(1)));
                aie::store_v(c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8), f1[g].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8), f1[g].template extract<8>(1)));
            }
        }
    }
}
"""
KERNELS["s8"] = KERNEL_S8

# s9: REFACTOR. The 8x8 int8 matrix unit needs all four AIE2 accumulator
# registers, and the float scale accumulator needs four more, so the compiler
# spills both to the stack on every block (visible as vlda/vst of the am*/bml*
# registers in the disassembly). The 4x8 unit needs only two accumulator
# registers, which leaves exactly two for the 16-lane float accumulator: the
# int32 partial and the float accumulator then fit in the four accumulators
# with no spilling. The j loop is outermost so that the 16-lane column scale
# is built once per (j, block) instead of once per (i, j, block).
KERNEL_S9 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
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
    const float  * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float  * bsc = (const float *)(bt + KCC * NT);

    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int j = 0; j < NA; j++) {
        const int8_t * bp = b_tiles + j * KB * 64;
        const float  * bsp = bsc + j * 8;
        for (int i = 0; i < MA; i++) {
            const int8_t * ap = a_tiles + i * KA * 32;
            const float  * asp = asc + i * 4 * NBLK;
            aie::vector<float, 16> f0 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f1 = aie::broadcast<float, 16>(0.0f);
            for (int bb = 0; bb < NBLK; bb++) {
                aie::vector<int32_t, 32> zero = aie::broadcast<int32_t, 32>(0);
                MMUL acc(zero);
                const int8_t * apb = ap + bb * 4 * 32;
                const int8_t * bpb = bp + bb * 4 * 64;
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(apb + kk * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bpb + kk * 64);
                    acc.mac(av, bv);
                }
                aie::vector<float, 32> p = aie::to_float(acc.to_vector<int32_t>());
                aie::vector<float, 8> bsv = aie::load_v<8>(bsp + bb * NT);
                aie::vector<float, 16> w16 = aie::broadcast<float, 16>(0.0f);
                w16.insert<8>(0, bsv);
                w16.insert<8>(1, bsv);
                aie::vector<float, 16> a0 = aie::broadcast<float, 16>(0.0f);
                a0.insert<8>(0, aie::broadcast<float, 8>(asp[(0) * NBLK + bb]));
                a0.insert<8>(1, aie::broadcast<float, 8>(asp[(1) * NBLK + bb]));
                aie::vector<float, 16> a1 = aie::broadcast<float, 16>(0.0f);
                a1.insert<8>(0, aie::broadcast<float, 8>(asp[(2) * NBLK + bb]));
                a1.insert<8>(1, aie::broadcast<float, 8>(asp[(3) * NBLK + bb]));
                f0 = aie::add(f0, aie::mul(p.template extract<16>(0), aie::mul(a0, w16).to_vector<float>()).to_vector<float>());
                f1 = aie::add(f1, aie::mul(p.template extract<16>(1), aie::mul(a1, w16).to_vector<float>()).to_vector<float>());
            }
            aie::store_v(c + (i * 4 + 0) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 4 + 0) * NT + j * 8), f0.template extract<8>(0)));
            aie::store_v(c + (i * 4 + 1) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 4 + 1) * NT + j * 8), f0.template extract<8>(1)));
            aie::store_v(c + (i * 4 + 2) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 4 + 2) * NT + j * 8), f1.template extract<8>(0)));
            aie::store_v(c + (i * 4 + 3) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 4 + 3) * NT + j * 8), f1.template extract<8>(1)));
        }
    }
}
"""
KERNELS["s9"] = KERNEL_S9

# s10: REFACTOR. The AIE2 int8 vmac has a ~46-55 cycle accumulator latency
# (measured: kbench/mmulrate.py), so a chain of four dependent vmacs per
# 32-block costs ~4x the latency for every block. This variant gives each of
# the four kk steps of a block its OWN accumulator, so the four vmacs are
# independent, and sums the four int32 partials before the scale is applied.
# The math is unchanged: the int32 partial of a 32-block is the sum of its
# four 8-K partials.
KERNEL_S10 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT   = 32;
    constexpr int NT   = 64;
    constexpr int KCC  = 256;
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
    for (int j = 0; j < NA; j++) {
        const int8_t * bp = b_tiles + j * KB * 64;
        const float  * bsp = bsc + j * 8;
        for (int i = 0; i < MA; i++) {
            const int8_t * ap = a_tiles + i * KA * 64;
            const float  * asp = asc + i * 8 * NBLK;
            aie::vector<float, 16> f0 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f1 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f2 = aie::broadcast<float, 16>(0.0f);
            aie::vector<float, 16> f3 = aie::broadcast<float, 16>(0.0f);
            for (int bb = 0; bb < NBLK; bb++) {
                const int8_t * apb = ap + bb * 4 * 64;
                const int8_t * bpb = bp + bb * 4 * 64;
                aie::vector<int8_t, 64> av0 = aie::load_v<64>(apb);
                aie::vector<int8_t, 64> av1 = aie::load_v<64>(apb + 64);
                aie::vector<int8_t, 64> av2 = aie::load_v<64>(apb + 128);
                aie::vector<int8_t, 64> av3 = aie::load_v<64>(apb + 192);
                aie::vector<int8_t, 64> bv0 = aie::load_v<64>(bpb);
                aie::vector<int8_t, 64> bv1 = aie::load_v<64>(bpb + 64);
                aie::vector<int8_t, 64> bv2 = aie::load_v<64>(bpb + 128);
                aie::vector<int8_t, 64> bv3 = aie::load_v<64>(bpb + 192);
                aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
                MMUL acc0(zero);
                MMUL acc1(zero);
                MMUL acc2(zero);
                MMUL acc3(zero);
                acc0.mac(av0, bv0);
                acc1.mac(av1, bv1);
                acc2.mac(av2, bv2);
                acc3.mac(av3, bv3);
                aie::vector<int32_t, 64> s0 = acc0.to_vector<int32_t>();
                aie::vector<int32_t, 64> s1 = acc1.to_vector<int32_t>();
                aie::vector<int32_t, 64> s2 = acc2.to_vector<int32_t>();
                aie::vector<int32_t, 64> s3 = acc3.to_vector<int32_t>();
                aie::vector<int32_t, 64> sm =
                    aie::add(aie::add(s0, s1), aie::add(s2, s3));
                aie::vector<float, 64> p = aie::to_float(sm);
                aie::vector<float, 8> bsv = aie::load_v<8>(bsp + bb * NT);
                aie::vector<float, 16> w16 = aie::broadcast<float, 16>(0.0f);
                w16.insert<8>(0, bsv);
                w16.insert<8>(1, bsv);
                const float * ap2 = asp + bb;
                aie::vector<float, 16> a0 = aie::broadcast<float, 16>(0.0f);
                a0.insert<8>(0, aie::broadcast<float, 8>(ap2[(0) * NBLK]));
                a0.insert<8>(1, aie::broadcast<float, 8>(ap2[(1) * NBLK]));
                aie::vector<float, 16> a1 = aie::broadcast<float, 16>(0.0f);
                a1.insert<8>(0, aie::broadcast<float, 8>(ap2[(2) * NBLK]));
                a1.insert<8>(1, aie::broadcast<float, 8>(ap2[(3) * NBLK]));
                aie::vector<float, 16> a2 = aie::broadcast<float, 16>(0.0f);
                a2.insert<8>(0, aie::broadcast<float, 8>(ap2[(4) * NBLK]));
                a2.insert<8>(1, aie::broadcast<float, 8>(ap2[(5) * NBLK]));
                aie::vector<float, 16> a3 = aie::broadcast<float, 16>(0.0f);
                a3.insert<8>(0, aie::broadcast<float, 8>(ap2[(6) * NBLK]));
                a3.insert<8>(1, aie::broadcast<float, 8>(ap2[(7) * NBLK]));
                f0 = aie::add(f0, aie::mul(p.template extract<16>(0), aie::mul(a0, w16).to_vector<float>()).to_vector<float>());
                f1 = aie::add(f1, aie::mul(p.template extract<16>(1), aie::mul(a1, w16).to_vector<float>()).to_vector<float>());
                f2 = aie::add(f2, aie::mul(p.template extract<16>(2), aie::mul(a2, w16).to_vector<float>()).to_vector<float>());
                f3 = aie::add(f3, aie::mul(p.template extract<16>(3), aie::mul(a3, w16).to_vector<float>()).to_vector<float>());
            }
            aie::store_v(c + (i * 8 + 0) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 0) * NT + j * 8), f0.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 1) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 1) * NT + j * 8), f0.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 2) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2) * NT + j * 8), f1.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 3) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 3) * NT + j * 8), f1.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 4) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 4) * NT + j * 8), f2.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 5) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 5) * NT + j * 8), f2.template extract<8>(1)));
            aie::store_v(c + (i * 8 + 6) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 6) * NT + j * 8), f3.template extract<8>(0)));
            aie::store_v(c + (i * 8 + 7) * NT + j * 8, aie::add(aie::load_v<8>(c + (i * 8 + 7) * NT + j * 8), f3.template extract<8>(1)));
        }
    }
}
"""
KERNELS["s10"] = KERNEL_S10

# s12: REFACTOR, four independent chains.
#
# Measured facts that drive this structure (kbench/mmulrate.py):
#   * the AIE2 int8 vmac has a ~40 ns accumulator latency, so one chain
#     delivers 12.8 GOP/s and four independent chains deliver 53 GOP/s;
#   * the float multiply-add of the scale application costs ~80 ns per
#     16-lane op on one chain and ~20 ns on four.
# So the kernel must expose ILP. Here the four chains are four different
# output tiles (j, j+1, j+2, j+3) of the same row block i: they share the
# A operand tiles and the A (row) scale, so the loads and the row-scale build
# are amortized over the four chains, and the four vmac chains are
# independent so the matrix latency is hidden.
KERNEL_S12 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT   = 32;
    constexpr int NT   = 64;
    constexpr int KCC  = 256;
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
        const int8_t * ap = a_tiles + i * KA * 64;
        const float  * asp = asc + i * 8 * NBLK;
        for (int j = 0; j + 3 < NA; j += 4) {
            aie::vector<float, 16> f0[4], f1[4], f2[4], f3[4];
            for (int g = 0; g < 4; g++) {
                f0[g] = aie::broadcast<float, 16>(0.0f);
                f1[g] = aie::broadcast<float, 16>(0.0f);
                f2[g] = aie::broadcast<float, 16>(0.0f);
                f3[g] = aie::broadcast<float, 16>(0.0f);
            }
            for (int bb = 0; bb < NBLK; bb++) {
                const int8_t * apb = ap + bb * 4 * 64;
                aie::vector<int8_t, 64> av0 = aie::load_v<64>(apb);
                aie::vector<int8_t, 64> av1 = aie::load_v<64>(apb + 64);
                aie::vector<int8_t, 64> av2 = aie::load_v<64>(apb + 128);
                aie::vector<int8_t, 64> av3 = aie::load_v<64>(apb + 192);
                const int8_t * bpb = b_tiles + bb * 4 * 64;
                aie::vector<int8_t, 64> bv0 = aie::load_v<64>(bpb + (j + 0) * KB * 64);
                aie::vector<int8_t, 64> bv1 = aie::load_v<64>(bpb + (j + 1) * KB * 64);
                aie::vector<int8_t, 64> bv2 = aie::load_v<64>(bpb + (j + 2) * KB * 64);
                aie::vector<int8_t, 64> bv3 = aie::load_v<64>(bpb + (j + 3) * KB * 64);
                aie::vector<int8_t, 64> bw0 = aie::load_v<64>(bpb + (j + 0) * KB * 64 + 64);
                aie::vector<int8_t, 64> bw1 = aie::load_v<64>(bpb + (j + 1) * KB * 64 + 64);
                aie::vector<int8_t, 64> bw2 = aie::load_v<64>(bpb + (j + 2) * KB * 64 + 64);
                aie::vector<int8_t, 64> bw3 = aie::load_v<64>(bpb + (j + 3) * KB * 64 + 64);
                aie::vector<int8_t, 64> bx0 = aie::load_v<64>(bpb + (j + 0) * KB * 64 + 128);
                aie::vector<int8_t, 64> bx1 = aie::load_v<64>(bpb + (j + 1) * KB * 64 + 128);
                aie::vector<int8_t, 64> bx2 = aie::load_v<64>(bpb + (j + 2) * KB * 64 + 128);
                aie::vector<int8_t, 64> bx3 = aie::load_v<64>(bpb + (j + 3) * KB * 64 + 128);
                aie::vector<int8_t, 64> by0 = aie::load_v<64>(bpb + (j + 0) * KB * 64 + 192);
                aie::vector<int8_t, 64> by1 = aie::load_v<64>(bpb + (j + 1) * KB * 64 + 192);
                aie::vector<int8_t, 64> by2 = aie::load_v<64>(bpb + (j + 2) * KB * 64 + 192);
                aie::vector<int8_t, 64> by3 = aie::load_v<64>(bpb + (j + 3) * KB * 64 + 192);

                aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
                MMUL acc0(zero), acc1(zero), acc2(zero), acc3(zero);
                acc0.mac(av0, bv0);
                acc1.mac(av0, bv1);
                acc2.mac(av0, bv2);
                acc3.mac(av0, bv3);
                acc0.mac(av1, bw0);
                acc1.mac(av1, bw1);
                acc2.mac(av1, bw2);
                acc3.mac(av1, bw3);
                acc0.mac(av2, bx0);
                acc1.mac(av2, bx1);
                acc2.mac(av2, bx2);
                acc3.mac(av2, bx3);
                acc0.mac(av3, by0);
                acc1.mac(av3, by1);
                acc2.mac(av3, by2);
                acc3.mac(av3, by3);

                aie::vector<float, 64> p0 = aie::to_float(acc0.to_vector<int32_t>());
                aie::vector<float, 64> p1 = aie::to_float(acc1.to_vector<int32_t>());
                aie::vector<float, 64> p2 = aie::to_float(acc2.to_vector<int32_t>());
                aie::vector<float, 64> p3 = aie::to_float(acc3.to_vector<int32_t>());

                // the row scale is shared by the four chains
                const float * ap2 = asp + bb;
                aie::vector<float, 16> a0 = aie::broadcast<float, 16>(0.0f);
                a0.insert<8>(0, aie::broadcast<float, 8>(ap2[(0) * NBLK]));
                a0.insert<8>(1, aie::broadcast<float, 8>(ap2[(1) * NBLK]));
                aie::vector<float, 16> a1 = aie::broadcast<float, 16>(0.0f);
                a1.insert<8>(0, aie::broadcast<float, 8>(ap2[(2) * NBLK]));
                a1.insert<8>(1, aie::broadcast<float, 8>(ap2[(3) * NBLK]));
                aie::vector<float, 16> a2 = aie::broadcast<float, 16>(0.0f);
                a2.insert<8>(0, aie::broadcast<float, 8>(ap2[(4) * NBLK]));
                a2.insert<8>(1, aie::broadcast<float, 8>(ap2[(5) * NBLK]));
                aie::vector<float, 16> a3 = aie::broadcast<float, 16>(0.0f);
                a3.insert<8>(0, aie::broadcast<float, 8>(ap2[(6) * NBLK]));
                a3.insert<8>(1, aie::broadcast<float, 8>(ap2[(7) * NBLK]));

                const float * bsp = bsc + j * 8;
                for (int g = 0; g < 4; g++) {
                    aie::vector<float, 16> av16 = (g == 0) ? a0 : (g == 1) ? a1 : (g == 2) ? a2 : a3;
                    aie::vector<float, 16> w0 = aie::broadcast<float, 16>(0.0f);
                    aie::vector<float, 16> w1 = aie::broadcast<float, 16>(0.0f);
                    aie::vector<float, 16> w2 = aie::broadcast<float, 16>(0.0f);
                    aie::vector<float, 16> w3 = aie::broadcast<float, 16>(0.0f);
                    aie::vector<float, 8> bsv0 = aie::load_v<8>(bsp + bb * NT);
                    aie::vector<float, 8> bsv1 = aie::load_v<8>(bsp + bb * NT + 8);
                    aie::vector<float, 8> bsv2 = aie::load_v<8>(bsp + bb * NT + 16);
                    aie::vector<float, 8> bsv3 = aie::load_v<8>(bsp + bb * NT + 24);
                    w0.insert<8>(0, bsv0); w0.insert<8>(1, bsv0);
                    w1.insert<8>(0, bsv1); w1.insert<8>(1, bsv1);
                    w2.insert<8>(0, bsv2); w2.insert<8>(1, bsv2);
                    w3.insert<8>(0, bsv3); w3.insert<8>(1, bsv3);
                    f0[g] = aie::add(f0[g], aie::mul(p0.template extract<16>(g), aie::mul(av16, w0).to_vector<float>()).to_vector<float>());
                    f1[g] = aie::add(f1[g], aie::mul(p1.template extract<16>(g), aie::mul(av16, w1).to_vector<float>()).to_vector<float>());
                    f2[g] = aie::add(f2[g], aie::mul(p2.template extract<16>(g), aie::mul(av16, w2).to_vector<float>()).to_vector<float>());
                    f3[g] = aie::add(f3[g], aie::mul(p3.template extract<16>(g), aie::mul(av16, w3).to_vector<float>()).to_vector<float>());
                }
            }
            for (int g = 0; g < 4; g++) {
                aie::store_v(c + (i * 8 + 2 * g + 0) * NT + (j + 0) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 0) * NT + (j + 0) * 8), f0[g].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * g + 1) * NT + (j + 0) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 1) * NT + (j + 0) * 8), f0[g].template extract<8>(1)));
                aie::store_v(c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8), f1[g].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8), f1[g].template extract<8>(1)));
                aie::store_v(c + (i * 8 + 2 * g + 0) * NT + (j + 2) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 0) * NT + (j + 2) * 8), f2[g].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * g + 1) * NT + (j + 2) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 1) * NT + (j + 2) * 8), f2[g].template extract<8>(1)));
                aie::store_v(c + (i * 8 + 2 * g + 0) * NT + (j + 3) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 0) * NT + (j + 3) * 8), f3[g].template extract<8>(0)));
                aie::store_v(c + (i * 8 + 2 * g + 1) * NT + (j + 3) * 8, aie::add(aie::load_v<8>(c + (i * 8 + 2 * g + 1) * NT + (j + 3) * 8), f3[g].template extract<8>(1)));
            }
        }
    }
}
"""
KERNELS["s12"] = KERNEL_S12

# s13: REFACTOR, the structure the measurements point at.
#
# The measured facts (kbench/mmulrate.py):
#   * the float multiply of the scale application costs ~80 ns per 16-lane
#     operation on one dependency chain and ~20 ns on four;
#   * the int8 vmac costs ~40 ns on one chain and ~10 ns on four;
#   * the AIE2 has only four accumulator registers, so an 8x8 int8 partial and
#     a 16-lane float accumulator cannot both live in registers.
#
# So the kernel has to (a) expose ILP and (b) keep the float accumulator out
# of the registers. Both come from the same change: the block loop moves
# inside the column loop.
#
#   for i:                       # row block
#     for bb:                    # 32-block
#        load the four A tiles of (i, bb)   <- once for the whole j loop
#        build the row scale of (i, bb)     <- once for the whole j loop
#        for j:                   # column tile, 8 independent chains
#           the 8 j iterations are independent, so the float multiply and the
#           vmac both get their latency hidden
#
# The float accumulator for the whole tile lives in the kernel's LMEM stack
# (one 512-bit read-modify-write per group per block), which frees the vector
# registers for the partial and the scale.
KERNEL_S13 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT   = 32;
    constexpr int NT   = 64;
    constexpr int KCC  = 256;
    constexpr int NBLK = KCC / 32;
    constexpr int MA   = MT / 8;
    constexpr int NA   = NT / 8;
    constexpr int KA   = KCC / 8;
    constexpr int KB   = KCC / 8;
    const int8_t * a_tiles = at;
    const float  * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float  * bsc = (const float *)(bt + KCC * NT);

    // the float accumulator of the whole output tile, group major:
    // acc[(i*NA + j)*64 + g*16 + t]
    alignas(64) float acc[MA * NA * 64];

    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    {
        aie::vector<float, 16> zero = aie::broadcast<float, 16>(0.0f);
        for (int idx = 0; idx < MA * NA * 4; idx++) {
            aie::store_v(acc + idx * 16, zero);
        }
    }
    for (int i = 0; i < MA; i++) {
        const int8_t * ap = a_tiles + i * KA * 64;
        const float  * asp = asc + i * 8 * NBLK;
        for (int bb = 0; bb < NBLK; bb++) {
            // the A tiles of (i, bb): loaded once, used by every j
            const int8_t * apb = ap + bb * 4 * 64;
            aie::vector<int8_t, 64> av0 = aie::load_v<64>(apb);
            aie::vector<int8_t, 64> av1 = aie::load_v<64>(apb + 64);
            aie::vector<int8_t, 64> av2 = aie::load_v<64>(apb + 128);
            aie::vector<int8_t, 64> av3 = aie::load_v<64>(apb + 192);
            // the row scale of (i, bb): built once, used by every j
            const float * ap2 = asp + bb;
            aie::vector<float, 16> a0 = aie::broadcast<float, 16>(0.0f);
            a0.insert<8>(0, aie::broadcast<float, 8>(ap2[(0) * NBLK]));
            a0.insert<8>(1, aie::broadcast<float, 8>(ap2[(1) * NBLK]));
            aie::vector<float, 16> a1 = aie::broadcast<float, 16>(0.0f);
            a1.insert<8>(0, aie::broadcast<float, 8>(ap2[(2) * NBLK]));
            a1.insert<8>(1, aie::broadcast<float, 8>(ap2[(3) * NBLK]));
            aie::vector<float, 16> a2 = aie::broadcast<float, 16>(0.0f);
            a2.insert<8>(0, aie::broadcast<float, 8>(ap2[(4) * NBLK]));
            a2.insert<8>(1, aie::broadcast<float, 8>(ap2[(5) * NBLK]));
            aie::vector<float, 16> a3 = aie::broadcast<float, 16>(0.0f);
            a3.insert<8>(0, aie::broadcast<float, 8>(ap2[(6) * NBLK]));
            a3.insert<8>(1, aie::broadcast<float, 8>(ap2[(7) * NBLK]));

            for (int j = 0; j < NA; j++) {
                const int8_t * bpb = b_tiles + j * KB * 64 + bb * 4 * 64;
                aie::vector<int8_t, 64> bv0 = aie::load_v<64>(bpb);
                aie::vector<int8_t, 64> bv1 = aie::load_v<64>(bpb + 64);
                aie::vector<int8_t, 64> bv2 = aie::load_v<64>(bpb + 128);
                aie::vector<int8_t, 64> bv3 = aie::load_v<64>(bpb + 192);
                aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
                MMUL mmu(zero);
                mmu.mac(av0, bv0);
                mmu.mac(av1, bv1);
                mmu.mac(av2, bv2);
                mmu.mac(av3, bv3);
                aie::vector<float, 64> p = aie::to_float(mmu.to_vector<int32_t>());

                aie::vector<float, 8> bsv = aie::load_v<8>(bsc + bb * NT + j * 8);
                aie::vector<float, 16> w16 = aie::broadcast<float, 16>(0.0f);
                w16.insert<8>(0, bsv);
                w16.insert<8>(1, bsv);

                float * ac = acc + (i * NA + j) * 64;
                aie::vector<float, 16> s0 = aie::mul(a0, w16).to_vector<float>();
                aie::vector<float, 16> s1 = aie::mul(a1, w16).to_vector<float>();
                aie::vector<float, 16> s2 = aie::mul(a2, w16).to_vector<float>();
                aie::vector<float, 16> s3 = aie::mul(a3, w16).to_vector<float>();
                aie::store_v(ac + 0,  aie::add(aie::load_v<16>(ac + 0),  aie::mul(p.template extract<16>(0), s0).to_vector<float>()));
                aie::store_v(ac + 16, aie::add(aie::load_v<16>(ac + 16), aie::mul(p.template extract<16>(1), s1).to_vector<float>()));
                aie::store_v(ac + 32, aie::add(aie::load_v<16>(ac + 32), aie::mul(p.template extract<16>(2), s2).to_vector<float>()));
                aie::store_v(ac + 48, aie::add(aie::load_v<16>(ac + 48), aie::mul(p.template extract<16>(3), s3).to_vector<float>()));
            }
        }
    }
    // the tile accumulator is added into C
    for (int i = 0; i < MA; i++) {
        for (int j = 0; j < NA; j++) {
            const float * ac = acc + (i * NA + j) * 64;
            for (int r = 0; r < 8; r++) {
                const int g = r / 2;
                const int h = r % 2;
                aie::vector<float, 8> v = aie::load_v<8>(ac + g * 16 + h * 8);
                float * cp = c + (i * 8 + r) * NT + j * 8;
                aie::store_v(cp, aie::add(aie::load_v<8>(cp), v));
            }
        }
    }
}
"""
KERNELS["s13"] = KERNEL_S13


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------
def build_design(M, K, N, wcols, m_tile, n_tile, k_chunk, variant,
                 depth_a=1, depth_b=1, depth_c=1, kernel_reps=1, mmul=4,
                 layout="std"):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config

    n_mt = M // m_tile
    n_nt_c = (N // wcols) // n_tile
    n_kc = K // k_chunk
    nblk = k_chunk // 32
    ma = m_tile // mmul
    na = n_tile // 8

    a_bytes, b_bytes = chunk_sizes(m_tile, n_tile, k_chunk, mmul, layout)
    c_elems = m_tile * n_tile

    # the kernel stack lives in the same 64 KB of core LMEM as the fifos
    stack = 65536 - (a_bytes + b_bytes + c_elems * 4)
    if stack > 16384:
        stack = 16384
    if stack < 4096:
        raise ValueError("no room for the kernel stack")

    l1 = depth_a * a_bytes + depth_b * b_bytes + depth_c * c_elems * 4 + stack
    if l1 > 64 * 1024:
        raise ValueError(f"L1 budget {l1} exceeds 64KB")

    src = KERNELS[variant]
    # the kernel source has the tile shape baked in; patch it. The constants
    # appear both one per line and combined on one line, so normalise the
    # combined form first and then patch each constant.
    combined = [
        (re.compile(r"constexpr\s+int\s+MT\s*=\s*\d+,\s*NT\s*=\s*\d+,\s*KCC\s*=\s*\d+,\s*NBLK\s*=\s*[^;]+;"),
         "constexpr int MT = %d;\\n    constexpr int NT = %d;\\n"
         "    constexpr int KCC = %d;\\n    constexpr int NBLK = KCC / 32;",
         (m_tile, n_tile, k_chunk)),
        (re.compile(r"constexpr\s+int\s+MA\s*=\s*[^,]+,\s*NA\s*=\s*[^,]+,\s*KA\s*=\s*[^,]+,\s*KB\s*=\s*[^;]+;"),
         "constexpr int MA = %d;\\n    constexpr int NA = %d;\\n"
         "    constexpr int KA = KCC / 8;\\n    constexpr int KB = KCC / 8;",
         (m_tile // mmul, n_tile // 8)),
    ]
    for pat, repl, args in combined:
        src = pat.sub(repl % args, src)

    def set_const(src, name, value):
        pat = re.compile(r"constexpr\s+int\s+%s\s*=\s*[^;]+;" % name)
        new, n = pat.subn("constexpr int %s = %d;" % (name, value), src)
        if n != 1:
            raise ValueError("cannot patch %s in kernel %s" % (name, variant))
        return new

    src = set_const(src, "MT", m_tile)
    src = set_const(src, "NT", n_tile)
    src = set_const(src, "KCC", k_chunk)
    src = set_const(src, "MA", m_tile // mmul)
    src = set_const(src, "NA", n_tile // 8)
    if mmul == 8 and "mmul<4, 8, 8" in src:
        src = src.replace("mmul<4, 8, 8", "mmul<8, 8, 8")

    kern = ExternalFunction(
        "kern",
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

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt_c):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0.0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                for _r in iron_range(kernel_reps):
                    kern(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    fifos = []
    workers = []
    for col in range(wcols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=depth_a)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=depth_b)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=depth_c)
        fifos.append((fa, fb, fc))
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern],
                              stack_size=stack))

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(wcols * n_nt_c * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * a_bytes,),
                                  offset=mt * n_kc * a_bytes,
                                  sizes=[n_kc * a_bytes], strides=[1])
              for mt in range(n_mt)]
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
        b_p = [f[1].prod() for f in fifos]
        c_c = [f[2].cons() for f in fifos]
        for mt in range(n_mt):
            for col in range(wcols):
                for nt in range(n_nt_c):
                    tg = iron.TaskGroup()
                    a_p[col].fill(A, tap=a_taps[mt], group=tg)
                    b_p[col].fill(B, tap=b_taps[col][nt], group=tg)
                    c_c[col].drain(C, tap=c_taps[col][mt][nt], group=tg, wait=True)
                    tg.finish()

    prod_cons = []
    for f in fifos:
        prod_cons.append(f[0].prod())
    for f in fifos:
        prod_cons.append(f[1].prod())
    for f in fifos:
        prod_cons.append(f[2].cons())
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


# --------------------------------------------------------------------------
# packing and reference
# --------------------------------------------------------------------------
def chunk_sizes(m_tile, n_tile, k_chunk, mmul, layout):
    nblk = k_chunk // 32
    ma = m_tile // mmul
    na = n_tile // 8
    if layout == "pb":
        # A scale: per (i, block) mmul*8 floats, already in the 16-lane
        # broadcast pattern. B scale: per (block, j) 8 floats.
        a_bytes = m_tile * k_chunk + ma * nblk * mmul * 8 * 4
        b_bytes = k_chunk * n_tile + nblk * na * 8 * 4
    else:
        a_bytes = m_tile * k_chunk + m_tile * nblk * 4
        b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    return a_bytes, b_bytes


def pack(A_i8, As, B_i8, Bs, M, K, N, m_tile, n_tile, k_chunk, mmul, layout="std"):
    """Tile-major packing in the production layout, plus the f32 scale blocks.

    A chunk: int8 tiles of (mmul rows x 8 K), then M_TILE x (KC/32) f32 scales
    B chunk: int8 tiles of (8 K x 8 N),          then (KC/32) x N_TILE f32 scales
    """
    n_mt = M // m_tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    ma = m_tile // mmul
    na = n_tile // 8
    nblk = k_chunk // 32
    ab = mmul * 8

    a_chunk, b_chunk = chunk_sizes(m_tile, n_tile, k_chunk, mmul, layout)

    Ap = np.zeros(n_mt * n_kc * a_chunk, np.int8)
    for mt in range(n_mt):
        for kc in range(n_kc):
            base = (mt * n_kc + kc) * a_chunk
            abuf = Ap[base:base + m_tile * k_chunk].reshape(ma, nblk, 4, ab)
            for im in range(ma):
                for bb in range(nblk):
                    for l in range(4):
                        r0 = mt * m_tile + im * mmul
                        c0 = kc * k_chunk + bb * 32 + l * 8
                        abuf[im, bb, l] = A_i8[r0:r0 + mmul, c0:c0 + 8].reshape(-1)
            asc = Ap[base + m_tile * k_chunk:base + a_chunk].view(np.float32)
            if layout == "pb":
                # per (i, block): mmul*8 floats, grouped in 16-lane blocks of
                # [sA[row] x8, sA[row+1] x8]
                for im in range(ma):
                    for bb in range(nblk):
                        for g in range(mmul * 8 // 16):
                            r = im * mmul + 2 * g
                            v = As[mt * m_tile + r, kc * nblk + bb]
                            asc[(im * nblk + bb) * (mmul * 8) + g * 16: (im * nblk + bb) * (mmul * 8) + g * 16 + 8] = v
                            v2 = As[mt * m_tile + r + 1, kc * nblk + bb]
                            asc[(im * nblk + bb) * (mmul * 8) + g * 16 + 8: (im * nblk + bb) * (mmul * 8) + g * 16 + 16] = v2
            else:
                asc[:] = As[mt * m_tile:(mt + 1) * m_tile, kc * nblk:(kc + 1) * nblk].reshape(-1)

    Bp = np.zeros(n_nt * n_kc * b_chunk, np.int8)
    for nt in range(n_nt):
        for kc in range(n_kc):
            base = (nt * n_kc + kc) * b_chunk
            bbuf = Bp[base:base + k_chunk * n_tile].reshape(na, nblk, 4, 64)
            for jn in range(na):
                for bb in range(nblk):
                    for l in range(4):
                        r0 = kc * k_chunk + bb * 32 + l * 8
                        c0 = nt * n_tile + jn * 8
                        bbuf[jn, bb, l] = B_i8[r0:r0 + 8, c0:c0 + 8].reshape(-1)
            bsc = Bp[base + k_chunk * n_tile:base + b_chunk].view(np.float32)
            if layout == "pb":
                # per (block, j): 8 floats
                for bb in range(nblk):
                    for jn in range(na):
                        bsc[(bb * na + jn) * 8:(bb * na + jn) * 8 + 8] = \
                            Bs[kc * nblk + bb, nt * n_tile + jn * 8: nt * n_tile + jn * 8 + 8]
            else:
                bsc[:] = Bs[kc * nblk:(kc + 1) * nblk, nt * n_tile:(nt + 1) * n_tile].reshape(-1)

    return Ap, Bp


def ref_q8_gemm(A_i8, As, B_i8, Bs):
    """float64 reference with the exact Q8_0 per-block outer-product scales."""
    M, K = A_i8.shape
    N = B_i8.shape[1]
    nb = K // 32
    C = np.zeros((M, N), dtype=np.float64)
    for b in range(nb):
        A_blk = A_i8[:, b * 32:(b + 1) * 32].astype(np.int64)
        B_blk = B_i8[b * 32:(b + 1) * 32, :].astype(np.int64)
        p = A_blk @ B_blk
        scale = As[:, b:b + 1].astype(np.float64) * Bs[b:b + 1, :].astype(np.float64)
        C += p * scale
    return C


def make_data(M, K, N, pattern, seed=7):
    rng = np.random.default_rng(seed)
    nb = K // 32
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    if pattern == "realistic":
        # per-block scales with a wide dynamic range, as in a real Q8_0 tensor
        As = (rng.uniform(0.002, 0.05, size=(M, nb)) *
              np.power(2.0, rng.integers(-3, 4, size=(M, nb)))).astype(np.float32)
        Bs = (rng.uniform(0.002, 0.05, size=(nb, N)) *
              np.power(2.0, rng.integers(-3, 4, size=(nb, N)))).astype(np.float32)
    elif pattern == "uniform":
        As = np.ones((M, nb), np.float32)
        Bs = np.ones((nb, N), np.float32)
    elif pattern == "two_scale":
        As = np.ones((M, nb), np.float32)
        Bs = np.ones((nb, N), np.float32)
        As[:, nb // 2:] = 2.0
        Bs[nb // 2:, :] = 2.0
    elif pattern == "geometric":
        f = np.array([min(2.0 ** i, 65504.0) for i in range(nb)], np.float32)
        As = (np.ones((M, nb), np.float32) * f[None, :])
        Bs = (np.ones((nb, N), np.float32) * f[:, None])
    else:
        raise ValueError(pattern)
    return A_i8, As, B_i8, Bs


def run_kernel(wd, Ap, Bp, M, K, N, repeat, wcols, m_tile, n_tile, k_chunk, mmul,
               layout="std"):
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
    a_bytes, b_bytes = chunk_sizes(m_tile, n_tile, k_chunk, mmul, layout)

    # A is replicated per column group, B is N-sliced
    Ap_col = np.zeros((wcols, n_mt * n_kc * a_bytes), np.int8)
    for col in range(wcols):
        Ap_col[col] = Ap
    Bp_col = np.zeros((wcols * n_nt_c * n_kc * b_bytes,), np.int8)
    for col in range(wcols):
        for nt in range(n_nt_c):
            # production N-partition: column col, tile nt covers the global
            # column range [(col*n_nt_c + nt)*n_tile, +n_tile)
            src = ((col * n_nt_c + nt) * n_kc * b_bytes)
            dst = (col * n_nt_c + nt) * n_kc * b_bytes
            Bp_col[dst:dst + n_kc * b_bytes] = Bp[src:src + n_kc * b_bytes]

    a_bo = pyxrt.bo(dev, Ap_col.nbytes, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(dev, Bp_col.nbytes, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(dev, M * N * 4, pyxrt.bo.flags.host_only, gid3)
    inst = os.path.join(wd, "design.insts.bin")
    isz = os.path.getsize(inst)
    inst_bo = pyxrt.bo(dev, isz, pyxrt.bo.flags.cacheable, gid1)
    with open(inst, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo.write(Ap_col.tobytes(), 0)
    b_bo.write(Bp_col.tobytes(), 0)
    c_bo.write(b"\x00" * (M * N * 4), 0)
    for b in (a_bo, b_bo, c_bo):
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    times = []
    for _ in range(repeat):
        r = pyxrt.run(kern)
        r.set_arg(0, 3)
        r.set_arg(1, inst_bo)
        r.set_arg(2, isz)
        r.set_arg(3, a_bo)
        r.set_arg(4, b_bo)
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
    ap.add_argument("--m-tile", type=int, default=32)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--variant", default="s0", choices=list(KERNELS.keys()))
    ap.add_argument("--mmul", type=int, default=4, choices=[4, 8])
    ap.add_argument("--depth-a", type=int, default=1)
    ap.add_argument("--depth-b", type=int, default=1)
    ap.add_argument("--kernel-reps", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--pattern", default="realistic")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--recompile", action="store_true")
    a = ap.parse_args()

    wd = a.work_dir
    os.makedirs(wd, exist_ok=True)
    xp = os.path.join(wd, "design.xclbin")
    ip = os.path.join(wd, "design.insts.bin")
    pp = os.path.join(wd, "main.pdi")

    layout = "pb" if a.variant in ("s5", "s6") else "std"

    compile_s = 0.0
    compile_ok = True
    compile_err = ""
    if a.recompile or not os.path.exists(xp):
        import aie.iron as iron
        from aie.iron import CompileTime, In, Out

        @iron.jit
        def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
            return build_design(a.M, a.K, a.N, a.wcols, a.m_tile, a.n_tile,
                                a.k_chunk, a.variant, a.depth_a, a.depth_b,
                                1, a.kernel_reps, a.mmul, layout)

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

    A_i8, As, B_i8, Bs = make_data(a.M, a.K, a.N, a.pattern)
    Ap, Bp = pack(A_i8, As, B_i8, Bs, a.M, a.K, a.N, a.m_tile, a.n_tile, a.k_chunk, a.mmul, layout)
    Cref = ref_q8_gemm(A_i8, As, B_i8, Bs)

    times, C = run_kernel(wd, Ap, Bp, a.M, a.K, a.N, a.repeat, a.wcols,
                          a.m_tile, a.n_tile, a.k_chunk, a.mmul, layout)
    med = sorted(times)[len(times) // 2]
    d = np.abs(C.astype(np.float64) - Cref)
    refmax = float(np.max(np.abs(Cref)))
    rmse = float(np.sqrt(np.mean(d ** 2)))
    rel = float(np.max(d / np.maximum(np.abs(Cref), 1e-6)))

    rec = {
        "tag": a.tag, "variant": a.variant, "pattern": a.pattern,
        "M": a.M, "K": a.K, "N": a.N, "wcols": a.wcols,
        "m_tile": a.m_tile, "n_tile": a.n_tile, "k_chunk": a.k_chunk,
        "mmul": a.mmul, "kernel_reps": a.kernel_reps,
        "compile_ok": compile_ok, "compile_s": compile_s,
        "inst_bytes": os.path.getsize(ip),
        "ms_median": round(med, 3),
        "ms_all": [round(t, 3) for t in times],
        "gop_s": round((a.M * a.K * a.N * 2) / max(med, 1e-9) / 1e6, 3),
        "kernel_gop_s": round((a.M * a.K * a.N * 2 * a.kernel_reps) / max(med, 1e-9) / 1e6, 3),
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
