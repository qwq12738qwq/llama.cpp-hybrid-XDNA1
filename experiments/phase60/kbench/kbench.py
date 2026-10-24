#!/usr/bin/env python3
"""Phase 60 kernel benchmark: build, compile, run, verify and time AIE2 int8 GEMM designs.

One process per design so a wedged device cannot poison the next run.

usage: kbench.py --M 512 --K 1024 --N 3584 --wcols 4 --wrows 1 \
                 --m-tile 32 --n-tile 64 --k-chunk 256 --variant v2 --tag name
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.jsonl")


# ---------------------------------------------------------------- kernels
KERNEL_V2 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// v2: baseline. Partial sums are flushed to C per 32-block (bb), so the
// scalar C accumulation runs NBLK*MA*NA*32 times per call.
extern "C" void kern_v2(const int8_t * __restrict at,
                        const int8_t * __restrict bt,
                        int32_t      * __restrict c,
                        int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int bb = 0; bb < nblk; bb++) {
        for (int i = 0; i < ma; i++) {
            for (int j = 0; j < na; j++) {
                MMUL acc;
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(at + ((i * nblk + bb) * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
                alignas(32) int32_t tmp[32];
                aie::store_v(tmp, acc.to_vector<int32_t>());
                for (int rr = 0; rr < 4; rr++)
                    for (int ll = 0; ll < 8; ll++)
                        c[(i * 4 + rr) * nt + j * 8 + ll] += tmp[rr * 8 + ll];
            }
        }
    }
}
"""

# v3: keep the 32-block partial sums inside the MMUL accumulator and flush
# C once per (i,j). Cuts the scalar LMEM read-modify-write count by NBLK.
KERNEL_V3 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v3(const int8_t * __restrict at,
                        const int8_t * __restrict bt,
                        int32_t      * __restrict c,
                        int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < na; j++) {
            MMUL acc;
            for (int bb = 0; bb < nblk; bb++) {
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(at + ((i * nblk + bb) * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
            }
            alignas(32) int32_t tmp[32];
            aie::store_v(tmp, acc.to_vector<int32_t>());
            for (int rr = 0; rr < 4; rr++)
                for (int ll = 0; ll < 8; ll++)
                    c[(i * 4 + rr) * nt + j * 8 + ll] += tmp[rr * 8 + ll];
        }
    }
}
"""

KERNELS = {"v2": KERNEL_V2, "v3": KERNEL_V3}

# v4: 8 accumulators live at once. The A operand is hoisted across the j
# loop (8x fewer A loads) and the C update is done with 8-wide vector
# add/store instead of 32 scalar read-modify-writes.
KERNEL_V4 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v4(const int8_t * __restrict at,
                        const int8_t * __restrict bt,
                        int32_t      * __restrict c,
                        int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    constexpr int NA_MAX = 8;
    for (int i = 0; i < ma; i++) {
        MMUL acc[NA_MAX];
        for (int bb = 0; bb < nblk; bb++) {
            for (int kk = 0; kk < 4; kk++) {
                aie::vector<int8_t, 32> av = aie::load_v<32>(at + ((i * nblk + bb) * 4 + kk) * 32);
                for (int j = 0; j < na; j++) {
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc[j].mac(av, bv);
                }
            }
        }
        for (int j = 0; j < na; j++) {
            alignas(32) int32_t tmp[32];
            aie::store_v(tmp, acc[j].to_vector<int32_t>());
            for (int rr = 0; rr < 4; rr++) {
                int32_t * cp = c + (i * 4 + rr) * nt + j * 8;
                aie::vector<int32_t, 8> cv = aie::load_v<8>(cp);
                aie::vector<int32_t, 8> tv = aie::load_v<8>(tmp + rr * 8);
                aie::store_v(cp, aie::add(cv, tv));
            }
        }
    }
}
"""
KERNELS["v4"] = KERNEL_V4

# v5: native 8x8 int8 matrix unit. A operand is 8 rows x 8 K (64 bytes),
# B operand is 8 K x 8 N (64 bytes). The C update is vectorised.
KERNEL_V5 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v5(const int8_t * __restrict at,
                        const int8_t * __restrict bt,
                        int32_t      * __restrict c,
                        int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    constexpr int NA_MAX = 8;
    for (int i = 0; i < ma; i++) {
        MMUL acc[NA_MAX];
        for (int bb = 0; bb < nblk; bb++) {
            for (int kk = 0; kk < 4; kk++) {
                aie::vector<int8_t, 64> av = aie::load_v<64>(at + ((i * nblk + bb) * 4 + kk) * 64);
                for (int j = 0; j < na; j++) {
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc[j].mac(av, bv);
                }
            }
        }
        for (int j = 0; j < na; j++) {
            alignas(64) int32_t tmp[64];
            aie::store_v(tmp, acc[j].to_vector<int32_t>());
            for (int rr = 0; rr < 8; rr++) {
                int32_t * cp = c + (i * 8 + rr) * nt + j * 8;
                aie::vector<int32_t, 8> cv = aie::load_v<8>(cp);
                aie::vector<int32_t, 8> tv = aie::load_v<8>(tmp + rr * 8);
                aie::store_v(cp, aie::add(cv, tv));
            }
        }
    }
}
"""
KERNELS["v5"] = KERNEL_V5

# v6: native 8x8 matrix unit, 8 accumulators, scalar C update.
# Isolates the effect of the 8x8 unit and the multi-accumulator array.
KERNEL_V6 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v6(const int8_t * __restrict at,
                        const int8_t * __restrict bt,
                        int32_t      * __restrict c,
                        int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    constexpr int NA_MAX = 8;
    for (int i = 0; i < ma; i++) {
        MMUL acc[NA_MAX];
        for (int bb = 0; bb < nblk; bb++) {
            for (int kk = 0; kk < 4; kk++) {
                aie::vector<int8_t, 64> av = aie::load_v<64>(at + ((i * nblk + bb) * 4 + kk) * 64);
                for (int j = 0; j < na; j++) {
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc[j].mac(av, bv);
                }
            }
        }
        for (int j = 0; j < na; j++) {
            alignas(64) int32_t tmp[64];
            aie::store_v(tmp, acc[j].to_vector<int32_t>());
            for (int rr = 0; rr < 8; rr++)
                for (int ll = 0; ll < 8; ll++)
                    c[(i * 8 + rr) * nt + j * 8 + ll] += tmp[rr * 8 + ll];
        }
    }
}
"""
KERNELS["v6"] = KERNEL_V6

# v7: 4x8 unit, single accumulator, vectorised C update.
KERNEL_V7 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v7(const int8_t * __restrict at,
                        const int8_t * __restrict bt,
                        int32_t      * __restrict c,
                        int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < na; j++) {
            MMUL acc;
            for (int bb = 0; bb < nblk; bb++) {
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(at + ((i * nblk + bb) * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
            }
            alignas(32) int32_t tmp[32];
            aie::store_v(tmp, acc.to_vector<int32_t>());
            for (int rr = 0; rr < 4; rr++) {
                int32_t * cp = c + (i * 4 + rr) * nt + j * 8;
                aie::vector<int32_t, 8> cv = aie::load_v<8>(cp);
                aie::vector<int32_t, 8> tv = aie::load_v<8>(tmp + rr * 8);
                aie::store_v(cp, aie::add(cv, tv));
            }
        }
    }
}
"""
KERNELS["v7"] = KERNEL_V7

# v8: 4x8 unit with 8 independent accumulators (one per j). The A operand is
# hoisted out of the j loop so the matrix unit sees independent dependency
# chains. C update stays scalar.
KERNEL_V8 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v8(const int8_t * __restrict at,
                        const int8_t * __restrict bt,
                        int32_t      * __restrict c,
                        int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    constexpr int NA_MAX = 8;
    for (int i = 0; i < ma; i++) {
        MMUL acc[NA_MAX];
        for (int bb = 0; bb < nblk; bb++) {
            for (int kk = 0; kk < 4; kk++) {
                aie::vector<int8_t, 32> av = aie::load_v<32>(at + ((i * nblk + bb) * 4 + kk) * 32);
                for (int j = 0; j < na; j++) {
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc[j].mac(av, bv);
                }
            }
        }
        for (int j = 0; j < na; j++) {
            alignas(32) int32_t tmp[32];
            aie::store_v(tmp, acc[j].to_vector<int32_t>());
            for (int rr = 0; rr < 4; rr++)
                for (int ll = 0; ll < 8; ll++)
                    c[(i * 4 + rr) * nt + j * 8 + ll] += tmp[rr * 8 + ll];
        }
    }
}
"""
KERNELS["v8"] = KERNEL_V8

# v3n: v3 without the C update. Measures the matrix-unit + LMEM-load ceiling
# with the C update removed (one store keeps the loop alive).
KERNEL_V3N = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v3n(const int8_t * __restrict at,
                         const int8_t * __restrict bt,
                         int32_t      * __restrict c,
                         int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    int32_t sink = 0;
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < na; j++) {
            MMUL acc(aie::zeros<int32_t, 32>());
            for (int bb = 0; bb < nblk; bb++) {
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(at + ((i * nblk + bb) * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
            }
            alignas(32) int32_t tmp[32];
            aie::store_v(tmp, acc.to_vector<int32_t>());
            sink += tmp[0];
        }
    }
    c[0] = sink;
}
"""
KERNELS["v3n"] = KERNEL_V3N

# v9: native 8x8 matrix unit, ONE accumulator per (i,j). The v64acc is the
# hardware's native accumulator width, so long chaining does not need the
# compiler's 4-register split that breaks v3.
KERNEL_V9 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v9(const int8_t * __restrict at,
                        const int8_t * __restrict bt,
                        int32_t      * __restrict c,
                        int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < na; j++) {
            MMUL acc(aie::zeros<int32_t, 64>());
            for (int bb = 0; bb < nblk; bb++) {
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 64> av = aie::load_v<64>(at + ((i * nblk + bb) * 4 + kk) * 64);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
            }
            alignas(64) int32_t tmp[64];
            aie::store_v(tmp, acc.to_vector<int32_t>());
            for (int rr = 0; rr < 8; rr++)
                for (int ll = 0; ll < 8; ll++)
                    c[(i * 8 + rr) * nt + j * 8 + ll] += tmp[rr * 8 + ll];
        }
    }
}
"""
KERNELS["v9"] = KERNEL_V9

# v10: 4x8 unit, but each 32-block partial is kept in its own accumulator and
# the partials are summed with aie::add on accumulators. The per-accumulator
# chain stays 4 muls long (the pattern the compiler handles correctly) while
# C is still written only once per (i,j).
KERNEL_V10 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern_v10(const int8_t * __restrict at,
                         const int8_t * __restrict bt,
                         int32_t      * __restrict c,
                         int32_t ma, int32_t na, int32_t nblk, int32_t nt)
{
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < na; j++) {
            MMUL tot(aie::zeros<int32_t, 32>());
            for (int bb = 0; bb < nblk; bb++) {
                MMUL p(aie::zeros<int32_t, 32>());
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(at + ((i * nblk + bb) * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bt + ((j * nblk + bb) * 4 + kk) * 64);
                    p.mac(av, bv);
                }
                tot = aie::add(tot, p);
            }
            alignas(32) int32_t tmp[32];
            aie::store_v(tmp, tot.to_vector<int32_t>());
            for (int rr = 0; rr < 4; rr++)
                for (int ll = 0; ll < 8; ll++)
                    c[(i * 4 + rr) * nt + j * 8 + ll] += tmp[rr * 8 + ll];
        }
    }
}
"""
KERNELS["v10"] = KERNEL_V10


# ---------------------------------------------------------------- design
def build_design(M, K, N, wcols, wrows, m_tile, n_tile, k_chunk, variant,
                 depth_a, depth_b, depth_c, wait_c, kernel_reps=1, mmul=4):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron.device.tile import Tile
    from aie.utils import config

    n_workers = wcols * wrows
    n_mt = M // m_tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    if n_nt % wcols or n_mt % wrows:
        raise ValueError(f"N/n_tile={n_nt} not divisible by wcols={wcols}; "
                         f"M/m_tile={n_mt} not divisible by wrows={wrows}")
    n_per = n_nt // wcols          # n-tiles per column group
    m_per = n_mt // wrows          # m-tiles per row group

    ma = m_tile // mmul
    na = n_tile // 8
    nblk = k_chunk // 32
    a_bytes = m_tile * k_chunk
    b_bytes = k_chunk * n_tile
    c_elems = m_tile * n_tile

    l1 = depth_a * a_bytes + depth_b * b_bytes + depth_c * c_elems * 4 + 0x2000
    if l1 > 64 * 1024:
        raise ValueError(f"L1 budget {l1} exceeds 64KB")

    kern = ExternalFunction(
        "kern_" + variant,
        source_string=KERNELS[variant],
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.int32]],
            np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.int32]]

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(m_per * n_per):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                for _r in iron_range(kernel_reps):
                    kern(elem_a, elem_b, elem_c, ma, na, nblk, n_tile)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    workers = []
    fifos = []
    for w in range(n_workers):
        fa = ObjectFifo(a_ty, name=f"a{w}", depth=depth_a)
        fb = ObjectFifo(b_ty, name=f"b{w}", depth=depth_b)
        fc = ObjectFifo(c_ty, name=f"c{w}", depth=depth_c)
        fifos.append((fa, fb, fc))
        col = w // wrows
        row = 2 + (w % wrows)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern],
                              tile=Tile(col, row), stack_size=8192))

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_nt * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.int32]]

    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * a_bytes,),
                                  offset=mt * n_kc * a_bytes,
                                  sizes=[n_kc * a_bytes], strides=[1])
              for mt in range(n_mt)]
    b_taps = [TensorAccessPattern(tensor_dims=(n_nt * n_kc * b_bytes,),
                                  offset=nt * n_kc * b_bytes,
                                  sizes=[n_kc * b_bytes], strides=[1])
              for nt in range(n_nt)]

    def sequence(A, B, C, *ends):
        a_p = [f[0].prod() for f in fifos]
        b_p = [f[1].prod() for f in fifos]
        c_c = [f[2].cons() for f in fifos]
        for mt in range(n_mt):
            for w in range(n_workers):
                wrow = w % wrows
                wcol = w // wrows
                if not (wrow * m_per <= mt < (wrow + 1) * m_per):
                    continue
                for nl in range(n_per):
                    nt = wcol * n_per + nl
                    tg = iron.TaskGroup()
                    a_p[w].fill(A, tap=a_taps[mt], group=tg)
                    b_p[w].fill(B, tap=b_taps[nt], group=tg)
                    c_tap = TensorAccessPattern(tensor_dims=(M, N),
                                                offset=mt * m_tile * N + nt * n_tile,
                                                sizes=[m_tile, n_tile], strides=[N, 1])
                    c_c[w].drain(C, tap=c_tap, group=tg, wait=wait_c)
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


# ---------------------------------------------------------------- packing
def pack(A, B, M, K, N, m_tile, n_tile, k_chunk, mmul=4):
    """Pack into the layout the mmul kernel expects."""
    n_mt = M // m_tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    ma = m_tile // mmul
    na = n_tile // 8
    nblk = k_chunk // 32
    a_bytes = m_tile * k_chunk
    b_bytes = k_chunk * n_tile
    ab = mmul * 8            # bytes per A block (32 for 4x8, 64 for 8x8)

    Ap = np.zeros(n_mt * n_kc * a_bytes, np.int8)
    for mt in range(n_mt):
        for kc in range(n_kc):
            base = (mt * n_kc + kc) * a_bytes
            for i in range(ma):
                for bb in range(nblk):
                    for l in range(4):
                        off = base + ((i * nblk + bb) * 4 + l) * ab
                        r0 = mt * m_tile + i * mmul
                        c0 = kc * k_chunk + bb * 32 + l * 8
                        Ap[off:off + ab] = A[r0:r0 + mmul, c0:c0 + 8].reshape(-1)

    Bp = np.zeros(n_nt * n_kc * b_bytes, np.int8)
    for nt in range(n_nt):
        for kc in range(n_kc):
            base = (nt * n_kc + kc) * b_bytes
            for j in range(na):
                for bb in range(nblk):
                    for l in range(4):
                        off = base + ((j * nblk + bb) * 4 + l) * 64
                        r0 = kc * k_chunk + bb * 32 + l * 8
                        c0 = nt * n_tile + j * 8
                        Bp[off:off + 64] = B[r0:r0 + 8, c0:c0 + 8].reshape(-1)
    return Ap, Bp


def run_kernel(wd, Ap, Bp, M, N, repeat):
    import pyxrt
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(os.path.join(wd, "design.xclbin"))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    gid3, gid1 = kern.group_id(3), kern.group_id(1)
    a_bo = pyxrt.bo(dev, Ap.nbytes, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(dev, Bp.nbytes, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(dev, M * N * 4, pyxrt.bo.flags.host_only, gid3)
    inst = os.path.join(wd, "design.insts.bin")
    isz = os.path.getsize(inst)
    inst_bo = pyxrt.bo(dev, isz, pyxrt.bo.flags.cacheable, gid1)
    with open(inst, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo.write(Ap.tobytes(), 0)
    b_bo.write(Bp.tobytes(), 0)
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
    C = np.frombuffer(c_bo.read(M * N * 4, 0), np.int32).reshape(M, N)
    return times, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--wcols", type=int, default=4)
    ap.add_argument("--wrows", type=int, default=1)
    ap.add_argument("--m-tile", type=int, default=32)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--variant", default="v2")
    ap.add_argument("--depth-a", type=int, default=2)
    ap.add_argument("--depth-b", type=int, default=2)
    ap.add_argument("--depth-c", type=int, default=1)
    ap.add_argument("--no-wait-c", action="store_true")
    ap.add_argument("--kernel-reps", type=int, default=1)
    ap.add_argument("--mmul", type=int, default=4, choices=[4, 8])
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--recompile", action="store_true")
    a = ap.parse_args()

    wd = a.work_dir
    os.makedirs(wd, exist_ok=True)
    xp = os.path.join(wd, "design.xclbin")
    ip = os.path.join(wd, "design.insts.bin")
    pp = os.path.join(wd, "main.pdi")

    compile_s = 0.0
    if a.recompile or not os.path.exists(xp):
        import aie.iron as iron
        from aie.iron import CompileTime, In, Out

        @iron.jit
        def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
            return build_design(a.M, a.K, a.N, a.wcols, a.wrows, a.m_tile,
                                a.n_tile, a.k_chunk, a.variant, a.depth_a,
                                a.depth_b, a.depth_c, not a.no_wait_c, a.kernel_reps, a.mmul)

        t0 = time.time()
        design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        compile_s = round(time.time() - t0, 2)
        print(f"compiled in {compile_s}s")

    rng = np.random.default_rng(7)
    A = rng.integers(-127, 128, (a.M, a.K), dtype=np.int8)
    B = rng.integers(-127, 128, (a.K, a.N), dtype=np.int8)
    Ap, Bp = pack(A, B, a.M, a.K, a.N, a.m_tile, a.n_tile, a.k_chunk, a.mmul)
    Cref = (A.astype(np.int32) @ B.astype(np.int32))

    times, C = run_kernel(wd, Ap, Bp, a.M, a.N, a.repeat)
    med = sorted(times)[len(times) // 2]
    d = np.abs(C.astype(np.int64) - Cref.astype(np.int64))
    n_workers = a.wcols * a.wrows
    n_mt = a.M // a.m_tile
    n_nt = a.N // a.n_tile
    tasks = 3 * n_mt * n_nt

    rec = {
        "tag": a.tag, "variant": a.variant,
        "M": a.M, "K": a.K, "N": a.N,
        "workers": n_workers, "wcols": a.wcols, "wrows": a.wrows,
        "m_tile": a.m_tile, "n_tile": a.n_tile, "k_chunk": a.k_chunk,
        "depth_a": a.depth_a, "depth_b": a.depth_b, "wait_c": not a.no_wait_c,
        "kernel_reps": a.kernel_reps,
        "dma_tasks": tasks,
        "compile_s": compile_s,
        "inst_bytes": os.path.getsize(ip),
        "ms_median": round(med, 3),
        "ms_all": [round(t, 3) for t in times],
        "gop_s": round((a.M * a.K * a.N * 2) / max(med, 1e-9) / 1e6, 2),
        "kernel_gop_s": round((a.M * a.K * a.N * 2 * a.kernel_reps) / max(med, 1e-9) / 1e6, 2),
        "max_abs_err": int(d.max()),
        "all_zero": bool(np.all(C == 0)),
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
