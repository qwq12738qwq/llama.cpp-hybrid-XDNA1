#!/usr/bin/env python3
"""Phase 40C: Smallest device-side scale experiment.

Modifies the C4_WA mmul kernel to apply per-block scales on device.
The kernel:
  1. Runs mmul<4,8,8> to get int32 partials
  2. Loads per-block scales from buffer
  3. Applies scales: float(partial) * scale_a * scale_b
  4. Accumulates into float output

This preserves int32 accumulation while adding device-side scaling.
"""
import argparse
import os
import time

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
from aie.iron.controlflow import range_ as iron_range
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config, DefaultNPURuntime, NPUKernel, tensor


KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// Q8_0 C4 mmul with per-block device-side scaling.
// Tile-major operand layout (aie::mmul tile order):
//   A chunk: (16 * KC/8) tiles of 4x8 int8, then 64*(KC/32) f32 scales
//   B chunk: (8  * KC/8) tiles of 8x8 int8, then (KC/32)*64 f32 scales
//   C:      64x64 f32 accumulator, accumulated across K-chunks
extern "C" void q8_c4_device_scale(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c,
    int32_t m, int32_t kc, int32_t n)
{
    constexpr int MT = 64, KC = 256, NT = 64;
    constexpr int NBLK = KC / 32;
    constexpr int KA = KC / 8;
    constexpr int KB = KC / 8;
    const int8_t * a_tiles = at;
    const float  * as = (const float *)(at + MT * KC);
    const int8_t * b_tiles = bt;
    const float  * bs = (const float *)(bt + KC * NT);

    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int bb = 0; bb < NBLK; bb++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 8; j++) {
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


def build(M, K, N, m_tile=64, n_tile=64, k_chunk=256):
    assert M % m_tile == 0 and N % n_tile == 0
    assert k_chunk % 32 == 0
    
    n_mt = M // m_tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    nblk = k_chunk // 32
    
    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4
    c_elems = m_tile * n_tile
    
    kern = ExternalFunction(
        "q8_c4_device_scale",
        source_string=KERNEL_SRC,
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
    fifo_a = ObjectFifo(a_ty, name="a", depth=1)
    fifo_b = ObjectFifo(b_ty, name="b", depth=1)
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
    
    worker = Worker(core_fn, [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), kern], stack_size=0x2000)
    
    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_nt * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]
    
    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * a_bytes,),
                                  offset=mt * n_kc * a_bytes, sizes=[n_kc * a_bytes], strides=[1])
              for mt in range(n_mt)]
    b_taps = [TensorAccessPattern(tensor_dims=(n_nt * n_kc * b_bytes,),
                                  offset=nt * n_kc * b_bytes, sizes=[n_kc * b_bytes], strides=[1])
              for nt in range(n_nt)]
    c_taps = [[TensorAccessPattern(tensor_dims=(M, N),
                                   offset=mt * m_tile * N + nt * n_tile,
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
    
    rt = Runtime(sequence, [A_ty, B_ty, C_ty, fifo_a.prod(), fifo_b.prod(), fifo_c.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit
def design(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], m_tile: CompileTime[int], n_tile: CompileTime[int], k_chunk: CompileTime[int]):
    return build(M, K, N, m_tile, n_tile, k_chunk)


def pack_A(A, As, M, K, m_tile, k_chunk):
    """Pack A in tile-major format for aie::mmul kernel (matches phase39)."""
    n_mt, n_kc = M // m_tile, K // k_chunk
    MA, nb = m_tile // 4, k_chunk // 32
    ai = np.empty((n_mt, n_kc, MA, nb, 4, 32), np.int8)
    asc = np.empty((n_mt, n_kc, m_tile, nb), np.float32)
    for mt in range(n_mt):
        for kc in range(n_kc):
            for bb in range(nb):
                for im in range(MA):
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
    """Pack B in tile-major format for aie::mmul kernel (matches phase39)."""
    n_nt, n_kc = N // n_tile, K // k_chunk
    NA, nb = n_tile // 8, k_chunk // 32
    bi = np.empty((n_nt, n_kc, NA, nb, 4, 64), np.int8)
    bsc = np.empty((n_nt, n_kc, nb, n_tile), np.float32)
    for nt in range(n_nt):
        for kc in range(n_kc):
            for bb in range(nb):
                for kk in range(4):
                    for jn in range(NA):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    
    M, K, N = args.M, args.K, args.N
    mt, nt, kc = 64, 64, 256
    assert M % mt == 0 and N % nt == 0 and K % kc == 0
    
    rng = np.random.default_rng(42)
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    nb = K // 32
    As = np.ones((M, nb), dtype=np.float32)
    Bs = np.ones((nb, N), dtype=np.float32)
    As[:, nb//2:] = 2.0
    Bs[nb//2:, :] = 2.0
    
    C_ref = ref_q8_gemm(A_i8, As, B_i8, Bs)
    print(f"Reference: M={M} K={K} N={N} nb={nb}")
    print(f"  C_ref[0,0]={C_ref[0,0]:.4f} C_ref[M-1,N-1]={C_ref[M-1,N-1]:.4f}")
    
    Ap = pack_A(A_i8, As, M, K, mt, kc)
    Bp = pack_B(B_i8, Bs, K, N, nt, kc)
    
    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")
    
    if not (os.path.exists(xp) and os.path.exists(ip)):
        t0 = time.time()
        compiled = design.specialize(M=M, K=K, N=N, m_tile=mt, n_tile=nt, k_chunk=kc).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"Compiled in {time.time() - t0:.1f}s")
    
    Abuf = np.ascontiguousarray(Ap)
    Bbuf = np.ascontiguousarray(Bp)
    Cout = np.zeros(M * N, np.float32)
    buffers = [tensor(Abuf, dtype=Abuf.dtype), tensor(Bbuf, dtype=Bbuf.dtype),
               tensor(Cout, dtype=Cout.dtype)]
    
    kern = NPUKernel(xp, ip)
    h = DefaultNPURuntime.load(kern)
    DefaultNPURuntime.run(h, buffers)
    
    C = np.asarray(buffers[-1]).astype(np.float64).reshape(M, N)
    diff = np.abs(C - C_ref.astype(np.float64))
    max_err = float(np.max(diff))
    rel_err = float(np.max(diff[np.abs(C_ref) > 1e-6] / np.abs(C_ref[np.abs(C_ref) > 1e-6])))
    bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))
    
    print(f"Device result:")
    print(f"  C_dev[0,0]={C[0,0]:.4f} C_dev[M-1,N-1]={C[M-1,N-1]:.4f}")
    print(f"  max_err={max_err:.4e} rel_err={rel_err:.4e} bad={bad}/{M*N}")


if __name__ == "__main__":
    main()
