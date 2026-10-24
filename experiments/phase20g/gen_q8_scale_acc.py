#!/usr/bin/env python3
"""Phase 20H: unified scale-aware Q8_0 full-K GEMM generator (vector MAC).

Same kernel as gen_q8_scale_vec.py / gen_q8_scale_multicol.py but a single file
that exposes the candidate matrix axes (tile sizes, K-chunk, columns, buffer
depth, trace) and the matching host packing so a bench runner can compile,
run, verify and trace candidates.

Kernel ABI (one XRT submit covers the whole K):
  a_packed: [m, kc] int8  then  [m, kc/32] f32   (one chunk)
  b_packed: [kc, n] int8  then  [kc/32, n] f32   (one chunk)
  c:        [m, n] f32, accumulated across K-chunks
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
                aie::accum<acc32, VEC> p0 = aie::zeros<acc32, VEC>();
                aie::accum<acc32, VEC> p1 = aie::zeros<acc32, VEC>();
                aie::accum<acc32, VEC> p2 = aie::zeros<acc32, VEC>();
                aie::accum<acc32, VEC> p3 = aie::zeros<acc32, VEC>();
                const int8_t * ar = a_data + i * kc + kb * 32;
                const int8_t * bc = b_data + (kb * 32) * n + j;
                for (int32_t l = 0; l < 32; l += 4) {
                    p0 = aie::mac(p0, aie::broadcast<int8_t, VEC>(ar[l + 0]), aie::load_v<VEC>(bc + (l + 0) * n));
                    p1 = aie::mac(p1, aie::broadcast<int8_t, VEC>(ar[l + 1]), aie::load_v<VEC>(bc + (l + 1) * n));
                    p2 = aie::mac(p2, aie::broadcast<int8_t, VEC>(ar[l + 2]), aie::load_v<VEC>(bc + (l + 2) * n));
                    p3 = aie::mac(p3, aie::broadcast<int8_t, VEC>(ar[l + 3]), aie::load_v<VEC>(bc + (l + 3) * n));
                }
                aie::vector<int32_t, VEC> pv = aie::add(
                    aie::add(p0.to_vector<int32_t>(), p1.to_vector<int32_t>()),
                    aie::add(p2.to_vector<int32_t>(), p3.to_vector<int32_t>()));
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


def pack_A(A, As, M, K, m_tile, k_chunk):
    """A: int8[M,K]; As: f32[M, K/32]  ->  A_ty."""
    nblk = k_chunk // 32
    n_mt = M // m_tile
    n_kc = K // k_chunk
    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    out = np.zeros(n_mt * n_kc * a_bytes, dtype=np.int8)
    for mt in range(n_mt):
        for ch in range(n_kc):
            base = (mt * n_kc + ch) * a_bytes
            out[base:base + m_tile * k_chunk] = A[mt * m_tile:(mt + 1) * m_tile,
                                                   ch * k_chunk:(ch + 1) * k_chunk].reshape(-1)
            sc = As[mt * m_tile:(mt + 1) * m_tile, ch * nblk:(ch + 1) * nblk]
            out[base + m_tile * k_chunk:base + a_bytes] = sc.astype(np.float32).view(np.int8).reshape(-1)
    return out


def pack_B(B, Bs, K, N, n_tile, k_chunk):
    """B: int8[K,N]; Bs: f32[K/32, N]  ->  B_ty (global nt-major)."""
    nblk = k_chunk // 32
    n_nt = N // n_tile
    n_kc = K // k_chunk
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    out = np.zeros(n_nt * n_kc * b_bytes, dtype=np.int8)
    for nt in range(n_nt):
        for ch in range(n_kc):
            base = (nt * n_kc + ch) * b_bytes
            out[base:base + k_chunk * n_tile] = B[ch * k_chunk:(ch + 1) * k_chunk,
                                                   nt * n_tile:(nt + 1) * n_tile].reshape(-1)
            sc = Bs[ch * nblk:(ch + 1) * nblk, nt * n_tile:(nt + 1) * n_tile]
            out[base + k_chunk * n_tile:base + b_bytes] = sc.astype(np.float32).view(np.int8).reshape(-1)
    return out


def build(M, K, N, m_tile, n_tile, k_chunk, n_cols, buffer_depth, trace_size):
    assert K % k_chunk == 0 and k_chunk % 32 == 0
    assert M % m_tile == 0 and N % n_tile == 0 and N % n_cols == 0
    n_mt = M // m_tile
    n_nt_c = (N // n_cols) // n_tile
    assert n_nt_c >= 1
    n_kc = K // k_chunk
    nblk = k_chunk // 32
    vec_len = 64 if n_tile % 64 == 0 else 32
    assert n_tile % vec_len == 0

    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    c_elems = m_tile * n_tile

    kern = ExternalFunction(
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

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt_c):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0.0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c, m_tile, k_chunk, n_tile)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    fifo_a, fifo_b, fifo_c, workers = [], [], [], []
    for col in range(n_cols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=buffer_depth)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=buffer_depth)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=1)
        fifo_a.append(fa)
        fifo_b.append(fb)
        fifo_c.append(fc)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern],
                              stack_size=0x2000, trace=(1 if trace_size > 0 else 0)))

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
        offset=mt * m_tile * N + (col * n_nt_c + nt) * n_tile,
        sizes=[m_tile, n_tile], strides=[N, 1])
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
        prog.enable_trace(trace_size, reuse_output_buffer=True)
    return prog.resolve_program()


@iron.jit
def design(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int],
           N: CompileTime[int], m_tile: CompileTime[int] = 32,
           n_tile: CompileTime[int] = 64, k_chunk: CompileTime[int] = 256,
           n_cols: CompileTime[int] = 1, buffer_depth: CompileTime[int] = 1,
           trace_size: CompileTime[int] = 0):
    return build(M, K, N, m_tile, n_tile, k_chunk, n_cols, buffer_depth, trace_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, default=32)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")
    print(f"perf: M={args.M} K={args.K} N={args.N} mt={args.m_tile} nt={args.n_tile} "
          f"kc={args.k_chunk} cols={args.cols} depth={args.depth}")
    t0 = time.time()
    compiled = design.specialize(
        M=args.M, K=args.K, N=args.N, m_tile=args.m_tile, n_tile=args.n_tile,
        k_chunk=args.k_chunk, n_cols=args.cols, buffer_depth=args.depth,
        trace_size=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s -> {compiled[0]} ({os.path.getsize(compiled[0])} B)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
