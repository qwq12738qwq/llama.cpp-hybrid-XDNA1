#!/usr/bin/env python3
"""Phase 20G step 5b: vectorized scale-aware Q8_0 full-K GEMM.

Same math as gen_q8_scale_gemm.py, but the 64-wide inner loop uses the AIE int8
vector MAC (aie::mac) instead of a scalar loop, and one kernel call covers a whole
K-chunk (k_chunk = 256 -> 8 Q8_0 blocks) instead of a single 32-block.

  A chunk: [m_tile, k_chunk] int8 + [m_tile, k_chunk/32] float32
  B chunk: [k_chunk, n_tile] int8 + [k_chunk/32, n_tile] float32
  C tile : [m_tile, n_tile] float32, stays in L1 across all K-chunks

One XRT submit covers the whole grid (single instruction stream). TaskGroup reuse
keeps the number of simultaneously active BDs low.

Usage:
  python gen_q8_scale_vec.py --M 512 --K 1024 --N 1024 --work-dir /tmp/q8v
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

KERNEL_SRC_TMPL = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// a_packed: [m, kc] int8  then  [m, kc/32] float32
// b_packed: [kc, n] int8  then  [kc/32, n] float32
// c:        [m, n] float32, accumulated across K-chunks
extern "C" void q8_chunk_vec(
    const int8_t * __restrict a_packed,
    const int8_t * __restrict b_packed,
    float * __restrict c,
    int32_t m, int32_t kc, int32_t n)
{
    const int32_t nblk = kc / 32;
    const int8_t * a_data = a_packed;
    const float  * a_scales = (const float *)(a_packed + m * kc);
    const int8_t * b_data = b_packed;
    const float  * b_scales = (const float *)(b_packed + kc * n);

    constexpr int32_t VEC = {vec};
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j < n; j += VEC) {
            aie::vector<float, VEC> cv = aie::load_v<VEC>(c + i * n + j);
            for (int32_t kb = 0; kb < nblk; kb++) {
                aie::accum<acc32, VEC> pacc = aie::zeros<acc32, VEC>();
                const int8_t * ar = a_data + i * kc + kb * 32;
                const int8_t * bc = b_data + (kb * 32) * n + j;
                for (int32_t l = 0; l < 32; l++) {
                    aie::vector<int8_t, VEC> av = aie::broadcast<int8_t, VEC>(ar[l]);
                    aie::vector<int8_t, VEC> bv = aie::load_v<VEC>(bc + l * n);
                    pacc = aie::mac(pacc, av, bv);
                }
                aie::vector<int32_t, VEC> pv = pacc.to_vector<int32_t>();
                float as_i = a_scales[i * nblk + kb];
                aie::vector<float, VEC> bsv = aie::load_v<VEC>(b_scales + kb * n + j);
                aie::vector<float, VEC> t1 = aie::mul(as_i, bsv).to_vector<float>();
                aie::vector<float, VEC> t2 = aie::mul(t1, aie::to_float(pv)).to_vector<float>();
                cv = aie::add(cv, t2);
            }
            aie::store_v(c + i * n + j, cv);
        }
    }
}
"""


def build(M, K, N, m_tile, n_tile, k_chunk):
    assert K % k_chunk == 0 and k_chunk % 32 == 0
    assert M % m_tile == 0 and N % n_tile == 0
    n_mt, n_nt, n_kc = M // m_tile, N // n_tile, K // k_chunk
    nblk = k_chunk // 32
    vec_len = 64 if n_tile % 64 == 0 else 32
    assert n_tile % vec_len == 0

    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    c_elems = m_tile * n_tile
    c_bytes = c_elems * 4
    BD_MAX = 16383 * 4
    for nm, v in (("a_bytes", a_bytes), ("b_bytes", b_bytes)):
        if v > BD_MAX:
            raise ValueError(f"{nm}={v} exceeds BD limit {BD_MAX}")

    stack_bytes = 0x2000
    l1_total = stack_bytes + a_bytes + b_bytes + c_bytes
    L1_MAX = 64 * 1024
    if l1_total > L1_MAX:
        raise ValueError(f"L1 budget {l1_total} exceeds {L1_MAX}")

    fused = ExternalFunction(
        "q8_chunk_vec",
        source_string=KERNEL_SRC_TMPL.replace("{vec}", str(vec_len)),
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.float32]],
            np.int32, np.int32, np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]

    fifo_a = ObjectFifo(a_ty, name="a", depth=1)
    fifo_b = ObjectFifo(b_ty, name="b", depth=1)
    fifo_c = ObjectFifo(c_ty, name="c", depth=1)

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt):
            elem_c = of_c.acquire(1)
            for x in iron_range(m_tile * n_tile):
                elem_c[x] = 0.0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c, m_tile, k_chunk, n_tile)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), fused],
        stack_size=stack_bytes,
    )

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_nt * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    # one contiguous tap per M-row covers all its K-chunk tiles
    a_taps = [TensorAccessPattern(
        tensor_dims=(n_mt * n_kc * a_bytes,),
        offset=mt * n_kc * a_bytes, sizes=[n_kc * a_bytes], strides=[1])
        for mt in range(n_mt)]

    # B is stored N-col-major: (nt, kc); one contiguous tap per N-col
    b_taps = [TensorAccessPattern(
        tensor_dims=(n_nt * n_kc * b_bytes,),
        offset=nt * n_kc * b_bytes, sizes=[n_kc * b_bytes], strides=[1])
        for nt in range(n_nt)]

    c_taps = [[TensorAccessPattern(
        tensor_dims=(M, N),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, default=32)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.m_tile, args.n_tile, args.k_chunk)

    print(f"vec: M={args.M} K={args.K} N={args.N} "
          f"m_tile={args.m_tile} n_tile={args.n_tile} k_chunk={args.k_chunk}")
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print(f"  xclbin: {compiled[0]} ({os.path.getsize(compiled[0])} bytes)")
    print(f"  insts:  {compiled[1]} ({os.path.getsize(compiled[1])} bytes)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
