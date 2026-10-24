#!/usr/bin/env python3
"""Phase 20G: pure int8 GEMM using the AIE2 matrix MAC (aie::mmul<4,8,8>).

Compile-time shape, K-chunked with int32 accumulation in the C tile.

  A packed: per (mt, kc): (MT/4) x (kc/8) blocks of 32B  (4x8 row-major)
  B packed: per (nt, kc): (kc/8) x (NT/8) blocks of 64B  (8x8 row-major)
  C tile : (MT/4) x (NT/8) blocks of 32 int32, written contiguously per tile;
           host re-orders into [M, N].
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


def kernel_src(m_tile, n_tile, k_chunk):
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void i8_mmul(
    const int8_t * __restrict a,
    const int8_t * __restrict b,
    int32_t * __restrict c)
{{
    constexpr int MT = {m_tile}, KC = {k_chunk}, NT = {n_tile};
    constexpr int MA = MT / 4, NA = NT / 8, KA = KC / 8;
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t>;
    for (int im = 0; im < MA; im++) {{
        for (int jn = 0; jn < NA; jn++) {{
            MMUL acc(aie::load_v<32>(c + (im * NA + jn) * 32));
            for (int kk = 0; kk < KA; kk++) {{
                aie::vector<int8_t, 32> av = aie::load_v<32>(a + (im * KA + kk) * 32);
                aie::vector<int8_t, 64> bv = aie::load_v<64>(b + (kk * NA + jn) * 64);
                acc.mac(av, bv);
            }}
            aie::store_v(c + (im * NA + jn) * 32, acc.to_vector<int32_t>());
        }}
    }}
}}
"""


def build(M, K, N, m_tile, n_tile, k_chunk, depth=1):
    assert M % m_tile == 0 and N % n_tile == 0
    assert m_tile % 4 == 0 and n_tile % 8 == 0 and K % k_chunk == 0 and k_chunk % 8 == 0
    n_mt, n_nt, n_kc = M // m_tile, N // n_tile, K // k_chunk
    a_bytes = (m_tile // 4) * (k_chunk // 8) * 32
    b_bytes = (k_chunk // 8) * (n_tile // 8) * 64
    c_elems = m_tile * n_tile
    c_bytes = c_elems * 4
    BD_MAX = 16383 * 4
    for nm, v in (("a_bytes", a_bytes), ("b_bytes", b_bytes)):
        if v > BD_MAX:
            raise ValueError(f"{nm}={v} exceeds BD limit {BD_MAX}")
    stack_bytes = 0x2000
    if stack_bytes + a_bytes + b_bytes + c_bytes > 64 * 1024:
        raise ValueError(f"L1 budget exceeded: {stack_bytes + a_bytes + b_bytes + c_bytes}")

    kern = ExternalFunction(
        "i8_mmul",
        source_string=kernel_src(m_tile, n_tile, k_chunk),
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
    fifo_a = ObjectFifo(a_ty, name="a", depth=depth)
    fifo_b = ObjectFifo(b_ty, name="b", depth=depth)
    fifo_c = ObjectFifo(c_ty, name="c", depth=1)

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt):
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

    worker = Worker(core_fn, [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), kern],
                    stack_size=stack_bytes)

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
    c_taps = [[TensorAccessPattern(tensor_dims=(M * N,),
                                   offset=(mt * n_nt + nt) * c_elems,
                                   sizes=[c_elems], strides=[1])
               for nt in range(n_nt)] for mt in range(n_mt)]

    def sequence(A, B, C, a_p, b_p, c_c):
        for mt in range(n_mt):
            for nt in range(n_nt):
                tg = iron.TaskGroup()
                a_p.fill(A, tap=a_taps[mt], group=tg)
                b_p.fill(B, tap=b_taps[nt], group=tg)
                c_c.drain(C, tap=c_taps[mt][nt], group=tg, wait=True)
                tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty,
                            fifo_a.prod(), fifo_b.prod(), fifo_c.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def pack_A(A, M, K, m_tile, k_chunk):
    n_mt, n_kc = M // m_tile, K // k_chunk
    MA, KA = m_tile // 4, k_chunk // 8
    out = np.empty((n_mt, n_kc, MA, KA, 32), np.int8)
    for mt in range(n_mt):
        for kc in range(n_kc):
            for im in range(MA):
                for kk in range(KA):
                    out[mt, kc, im, kk] = A[mt * m_tile + im * 4:mt * m_tile + im * 4 + 4,
                                            kc * k_chunk + kk * 8:kc * k_chunk + kk * 8 + 8].reshape(-1)
    return out.reshape(-1)


def pack_B(B, K, N, n_tile, k_chunk):
    n_nt, n_kc = N // n_tile, K // k_chunk
    KA, NA = k_chunk // 8, n_tile // 8
    out = np.empty((n_nt, n_kc, KA, NA, 64), np.int8)
    for nt in range(n_nt):
        for kc in range(n_kc):
            for kk in range(KA):
                for jn in range(NA):
                    out[nt, kc, kk, jn] = B[kc * k_chunk + kk * 8:kc * k_chunk + kk * 8 + 8,
                                            nt * n_tile + jn * 8:nt * n_tile + jn * 8 + 8].reshape(-1)
    return out.reshape(-1)


def unpack_C(Cflat, M, N, m_tile, n_tile):
    n_mt, n_nt = M // m_tile, N // n_tile
    MA, NA = m_tile // 4, n_tile // 8
    C = np.empty((M, N), np.int64)
    idx = 0
    for mt in range(n_mt):
        for nt in range(n_nt):
            for im in range(MA):
                for jn in range(NA):
                    C[mt * m_tile + im * 4:mt * m_tile + im * 4 + 4,
                      nt * n_tile + jn * 8:nt * n_tile + jn * 8 + 8] = \
                        Cflat[idx:idx + 32].reshape(4, 8)
                    idx += 32
    return C


@iron.jit
def design(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int],
           N: CompileTime[int], m_tile: CompileTime[int], n_tile: CompileTime[int],
           k_chunk: CompileTime[int], depth: CompileTime[int] = 1):
    return build(M, K, N, m_tile, n_tile, k_chunk, depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, default=16)
    ap.add_argument("--n-tile", type=int, default=32)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")
    d = design.specialize(M=args.M, K=args.K, N=args.N, m_tile=args.m_tile,
                          n_tile=args.n_tile, k_chunk=args.k_chunk, depth=args.depth)
    d.compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"compiled -> {xp}")


if __name__ == "__main__":
    main()
