#!/usr/bin/env python3
"""Phase 20G: Q8_0 scale-aware GEMM on the AIE2 vector MAC, compile-time shape.

Register-blocked over M (RB rows share one B vector load). Per 32-wide Q8 block:
64-lane int8 vector MAC into int32, then the block scales are applied once over
the full 64-lane vector (1 row-scale x 1 col-scale multiply per 64 elements).

  A chunk: [m_tile, k_chunk] int8, then [m_tile, k_chunk/32] f32 row scales
  B chunk: [k_chunk, n_tile] int8, then [k_chunk/32, n_tile] f32 col scales
  C tile : [m_tile, n_tile] f32, row-major, accumulated across K-chunks
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


def kernel_src(m_tile, n_tile, k_chunk, rb):
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void q8_rb(
    const int8_t * __restrict a,
    const int8_t * __restrict b,
    float * __restrict c)
{{
    constexpr int MT = {m_tile}, KC = {k_chunk}, NT = {n_tile};
    constexpr int VEC = 64, RB = {rb}, NB = KC / 32;
    const int8_t * a_data = a;
    const float  * as = (const float *)(a + MT * KC);
    const int8_t * b_data = b;
    const float  * bs = (const float *)(b + KC * NT);
    for (int i = 0; i < MT; i += RB) {{
        aie::vector<float, VEC> cv[RB];
        for (int r = 0; r < RB; r++)
            cv[r] = aie::load_v<VEC>(c + (i + r) * NT);
        for (int kb = 0; kb < NB; kb++) {{
            aie::accum<acc32, VEC> acc[RB];
            for (int r = 0; r < RB; r++)
                acc[r] = aie::zeros<acc32, VEC>();
            for (int l = 0; l < 32; l++) {{
                aie::vector<int8_t, VEC> bv = aie::load_v<VEC>(b_data + (kb * 32 + l) * NT);
                for (int r = 0; r < RB; r++)
                    acc[r] = aie::mac(acc[r], aie::broadcast<int8_t, VEC>(a_data[(i + r) * KC + kb * 32 + l]), bv);
            }}
            aie::vector<float, VEC> bsv = aie::load_v<VEC>(bs + kb * NT);
            for (int r = 0; r < RB; r++) {{
                float asr = as[(i + r) * NB + kb];
                aie::vector<float, VEC> pvf = aie::to_float(acc[r].to_vector<int32_t>());
                aie::vector<float, VEC> t =
                    aie::mul(asr, aie::mul(bsv, pvf).to_vector<float>()).to_vector<float>();
                cv[r] = aie::add(cv[r], t);
            }}
        }}
        for (int r = 0; r < RB; r++)
            aie::store_v(c + (i + r) * NT, cv[r]);
    }}
}}
"""


def build(M, K, N, m_tile, n_tile, k_chunk, rb, depth=1):
    assert K % k_chunk == 0 and M % m_tile == 0 and N % n_tile == 0
    assert m_tile % rb == 0 and n_tile % 64 == 0 and k_chunk % 32 == 0
    n_mt, n_nt, n_kc = M // m_tile, N // n_tile, K // k_chunk
    nb = k_chunk // 32
    a_bytes = m_tile * k_chunk + m_tile * nb * 4
    b_bytes = k_chunk * n_tile + nb * n_tile * 4
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
        "q8_rb",
        source_string=kernel_src(m_tile, n_tile, k_chunk, rb),
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
    fifo_a = ObjectFifo(a_ty, name="a", depth=depth)
    fifo_b = ObjectFifo(b_ty, name="b", depth=depth)
    fifo_c = ObjectFifo(c_ty, name="c", depth=1)

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt):
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

    worker = Worker(core_fn, [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), kern],
                    stack_size=stack_bytes)

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_nt * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * a_bytes,),
                                  offset=mt * n_kc * a_bytes,
                                  sizes=[n_kc * a_bytes], strides=[1])
              for mt in range(n_mt)]
    b_taps = [TensorAccessPattern(tensor_dims=(n_nt * n_kc * b_bytes,),
                                  offset=nt * n_kc * b_bytes,
                                  sizes=[n_kc * b_bytes], strides=[1])
              for nt in range(n_nt)]
    c_taps = [[TensorAccessPattern(tensor_dims=(M, N),
                                   offset=(mt * m_tile) * N + nt * n_tile,
                                   sizes=[m_tile, n_tile], strides=[N, 1])
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


def pack_A(A, As, M, K, m_tile, k_chunk):
    n_mt, n_kc = M // m_tile, K // k_chunk
    nb = k_chunk // 32
    ai = np.empty((n_mt, n_kc, m_tile * k_chunk), np.int8)
    asc = np.empty((n_mt, n_kc, m_tile * nb), np.float32)
    for mt in range(n_mt):
        for kc in range(n_kc):
            ai[mt, kc] = A[mt * m_tile:(mt + 1) * m_tile,
                           kc * k_chunk:(kc + 1) * k_chunk].reshape(-1)
            asc[mt, kc] = As[mt * m_tile:(mt + 1) * m_tile,
                             kc * nb:(kc + 1) * nb].reshape(-1)
    sc = asc.reshape(n_mt, n_kc, -1).view(np.int8)
    return np.concatenate([ai.reshape(n_mt, n_kc, -1), sc], axis=2).reshape(-1)


def pack_B(B, Bs, K, N, n_tile, k_chunk):
    n_nt, n_kc = N // n_tile, K // k_chunk
    nb = k_chunk // 32
    bi = np.empty((n_nt, n_kc, k_chunk * n_tile), np.int8)
    bsc = np.empty((n_nt, n_kc, nb * n_tile), np.float32)
    for nt in range(n_nt):
        for kc in range(n_kc):
            bi[nt, kc] = B[kc * k_chunk:(kc + 1) * k_chunk,
                           nt * n_tile:(nt + 1) * n_tile].reshape(-1)
            bsc[nt, kc] = Bs[kc * nb:(kc + 1) * nb,
                             nt * n_tile:(nt + 1) * n_tile].reshape(-1)
    sc = bsc.reshape(n_nt, n_kc, -1).view(np.int8)
    return np.concatenate([bi.reshape(n_nt, n_kc, -1), sc], axis=2).reshape(-1)


@iron.jit
def design(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int],
           N: CompileTime[int], m_tile: CompileTime[int], n_tile: CompileTime[int],
           k_chunk: CompileTime[int], rb: CompileTime[int], depth: CompileTime[int] = 1):
    return build(M, K, N, m_tile, n_tile, k_chunk, rb, depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, default=32)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--rb", type=int, default=4)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")
    d = design.specialize(M=args.M, K=args.K, N=args.N, m_tile=args.m_tile,
                          n_tile=args.n_tile, k_chunk=args.k_chunk, rb=args.rb,
                          depth=args.depth)
    d.compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"compiled -> {xp}")


if __name__ == "__main__":
    main()
