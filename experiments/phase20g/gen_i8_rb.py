#!/usr/bin/env python3
"""Phase 20G: pure int8 compute-floor kernel.

Compile-time shape (m_tile/n_tile/k_chunk baked into the C++ source) + register
blocking over M (RB rows share one B load) so Peano can actually unroll the K
loop. No scale, no float: int8 x int8 -> int32.

  a chunk: [m_tile, k_chunk] int8
  b chunk: [k_chunk, n_tile] int8
  c tile : [m_tile, n_tile] int32, accumulated across K-chunks
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

extern "C" void i8_rb(
    const int8_t * __restrict a,
    const int8_t * __restrict b,
    int32_t * __restrict c)
{{
    constexpr int MT = {m_tile}, KC = {k_chunk}, NT = {n_tile};
    constexpr int VEC = 64, RB = {rb};
    for (int i = 0; i < MT; i += RB) {{
        aie::accum<acc32, VEC> acc[RB];
        for (int r = 0; r < RB; r++)
            acc[r] = aie::from_vector<acc32>(aie::load_v<VEC>(c + (i + r) * NT));
        for (int l = 0; l < KC; l++) {{
            aie::vector<int8_t, VEC> bv = aie::load_v<VEC>(b + l * NT);
            for (int r = 0; r < RB; r++)
                acc[r] = aie::mac(acc[r], aie::broadcast<int8_t, VEC>(a[(i + r) * KC + l]), bv);
        }}
        for (int r = 0; r < RB; r++)
            aie::store_v(c + (i + r) * NT, acc[r].to_vector<int32_t>());
    }}
}}
"""


def build(M, K, N, m_tile, n_tile, k_chunk, rb, depth=2):
    assert K % k_chunk == 0 and M % m_tile == 0 and N % n_tile == 0
    assert m_tile % rb == 0 and n_tile % 64 == 0
    n_mt, n_nt, n_kc = M // m_tile, N // n_tile, K // k_chunk
    a_bytes = m_tile * k_chunk
    b_bytes = k_chunk * n_tile
    c_elems = m_tile * n_tile
    c_bytes = c_elems * 4
    BD_MAX = 16383 * 4
    for nm, v in (("a_bytes", a_bytes), ("b_bytes", b_bytes)):
        if v > BD_MAX:
            raise ValueError(f"{nm}={v} exceeds BD limit {BD_MAX}")
    stack_bytes = 0x2000
    if stack_bytes + a_bytes + b_bytes + c_bytes > 64 * 1024:
        raise ValueError("L1 budget exceeded")

    kern = ExternalFunction(
        "i8_rb",
        source_string=kernel_src(m_tile, n_tile, k_chunk, rb),
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


def pack_A(A, M, K, m_tile, k_chunk):
    n_mt, n_kc = M // m_tile, K // k_chunk
    out = np.empty((n_mt, n_kc, m_tile * k_chunk), np.int8)
    for mt in range(n_mt):
        for kc in range(n_kc):
            out[mt, kc] = A[mt * m_tile:(mt + 1) * m_tile,
                            kc * k_chunk:(kc + 1) * k_chunk].reshape(-1)
    return out.reshape(-1)


def pack_B(B, K, N, n_tile, k_chunk):
    n_nt, n_kc = N // n_tile, K // k_chunk
    out = np.empty((n_nt, n_kc, k_chunk * n_tile), np.int8)
    for nt in range(n_nt):
        for kc in range(n_kc):
            out[nt, kc] = B[kc * k_chunk:(kc + 1) * k_chunk,
                            nt * n_tile:(nt + 1) * n_tile].reshape(-1)
    return out.reshape(-1)


@iron.jit
def design(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int],
           N: CompileTime[int], m_tile: CompileTime[int], n_tile: CompileTime[int],
           k_chunk: CompileTime[int], rb: CompileTime[int], depth: CompileTime[int] = 2):
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
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")
    t0 = time.time()
    d = design.specialize(M=args.M, K=args.K, N=args.N, m_tile=args.m_tile,
                          n_tile=args.n_tile, k_chunk=args.k_chunk, rb=args.rb,
                          depth=args.depth)
    d.compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"compiled in {time.time() - t0:.1f}s -> {xp}")


if __name__ == "__main__":
    main()
