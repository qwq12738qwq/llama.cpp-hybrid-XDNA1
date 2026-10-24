#!/usr/bin/env python3
"""Phase 20E: IRON generator for full-K fused Q8_0 GEMM.

Architecture:
  A-tile: [m, K_total] int8 + [m, nb] float32 scales  (one BD, ~34KB)
  B-tile: [k_tile, N_per_col] int8 + [N_per_col] float32 scale (one BD, ~4KB)
  C-tile: [m, N_per_col] float32 (one BD)

Host fills B FIFO nb times per M-row. AIE core iterates K-blocks internally.
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

FUSED_KERNEL_SRC = r"""
#include <stdint.h>

// Packed A: [m, K] int8 + [m, nb] float32
// Packed B: nb blocks of {[32, n] int8 + [n] float32}
// C: [m, n] float32 (accumulator)
extern "C" void fused_mm_block(
    const int8_t * __restrict a_packed,
    const int8_t * __restrict b_all,
    float * __restrict c,
    int32_t m, int32_t K, int32_t n, int32_t block_idx)
{
    const int32_t nb = K / 32;
    const int32_t b_stride = 32 * n + n * 4;
    const int8_t * a_data = a_packed;
    const float  * a_scales = (const float *)(a_packed + m * K);
    const int8_t * b_packed = b_all + block_idx * b_stride;
    const int8_t * b_data = b_packed;
    const float  * b_scales = (const float *)(b_packed + 32 * n);

    for (int32_t i = 0; i < m; i++) {
        const float as_i = a_scales[i * nb + block_idx];
        for (int32_t j = 0; j < n; j++) {
            int32_t psum = 0;
            const int8_t * a_row = a_data + i * K + block_idx * 32;
            const int8_t * b_col = b_data + j;
            for (int32_t l = 0; l < 32; l++) {
                psum += (int32_t)a_row[l] * (int32_t)b_col[l * n];
            }
            c[i * n + j] += as_i * b_scales[j] * (float)psum;
        }
    }
}
"""


def build_fullk(M_total, K_total, N_total, m_tile, n_cols=1):
    k_tile = 32
    nb = K_total // k_tile
    n_per_col = N_total // n_cols
    m, n = m_tile, n_per_col

    # BD limit: 16383 words = 65532 bytes.
    # A packed: [m, K] int8 + [m, nb] float32.  For m=64, K=1024: 73728 > 65532.
    # Use m_tile=63: 64512 < 65532.  For M=512: 8 full tiles + 1 partial (40 rows).
    a_bytes = m * K_total + m * nb * 4
    BD_MAX = 16383 * 4  # 65532 bytes
    if a_bytes > BD_MAX:
        raise ValueError(f"a_bytes={a_bytes} exceeds BD limit {BD_MAX}. "
                         f"Reduce m_tile (current={m_tile})")
    # B: nb K-blocks of {[32, n] int8 + [n] float32}, sent as ONE BD
    b_block_bytes = k_tile * n + n * 4
    b_bytes = nb * b_block_bytes
    if b_bytes > BD_MAX:
        raise ValueError(f"b_bytes={b_bytes} exceeds BD limit {BD_MAX}")
    # C tile: [m, n] float32
    c_elems = m * n

    fused = ExternalFunction(
        "fused_mm_block",
        source_string=FUSED_KERNEL_SRC,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],     # [a_data m*K | a_scales m*nb*4]
            np.ndarray[(b_bytes,), np.dtype[np.int8]],      # [b_data 32*n | b_scale n*4]
            np.ndarray[(c_elems,), np.dtype[np.float32]],   # c
            np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]

    fifo_a = ObjectFifo(a_ty, name="a", depth=1)
    fifo_b = ObjectFifo(b_ty, name="b", depth=1)
    fifo_c = ObjectFifo(c_ty, name="c", depth=1)

    def core_fn(of_a, of_b, of_c, kern, K_val, nb_val):
        for _ in iron_range(M_total // m):
            elem_a = of_a.acquire(1)
            elem_b = of_b.acquire(1)
            elem_c = of_c.acquire(1)
            # zero C using iron_range (generates a loop, not unrolled)
            for i in iron_range(c_elems):
                elem_c[i] = 0.0
            # iterate K-blocks; kernel indexes into the whole-B buffer
            for bl in iron_range(nb_val):
                kern(elem_a, elem_b, elem_c, m, K_val, n, bl)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), fused,
         K_total, nb],
        stack_size=0x2000,
    )

    A_ty = np.ndarray[(M_total * K_total + M_total * nb * 4,), np.dtype[np.int8]]
    n_mtiles = M_total // m
    a_taps_list = [
        TensorAccessPattern(
            tensor_dims=(M_total * K_total + M_total * nb * 4,),
            offset=i * a_bytes, sizes=[a_bytes], strides=[1])
        for i in range(n_mtiles)
    ]

    B_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    b_taps = TensorAccessPattern(
        tensor_dims=(b_bytes,),
        offset=0, sizes=[b_bytes], strides=[1])

    C_ty = np.ndarray[(M_total * N_total,), np.dtype[np.float32]]
    c_taps_list = [
        TensorAccessPattern(
            tensor_dims=(M_total * N_total,),
            offset=i * m * N_total, sizes=[c_elems], strides=[1])
        for i in range(n_mtiles)
    ]

    def sequence(A, B_full, C, a_p, b_p, c_c):
        # Unroll at compile time (M_total//m and nb are small constants)
        for i in range(n_mtiles):
            a_p.fill(A, tap=a_taps_list[i])
            b_p.fill(B_full, tap=b_taps)
            c_c.drain(C, tap=c_taps_list[i], wait=True)

    rt = Runtime(sequence, [A_ty, B_ty, C_ty,
                            fifo_a.prod(), fifo_b.prod(), fifo_c.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--m-tile", type=int, default=56,
                    help="M tile (m*K + m*nb*4 must fit BD limit 65532)")
    ap.add_argument("--n-cols", type=int, default=1)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def fullk_design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build_fullk(args.M, args.K, args.N, args.m_tile, args.n_cols)

    print(f"Building full-K fused: M={args.M} K={args.K} N={args.N} m_tile={args.m_tile}")
    t0 = time.time()
    compiled = fullk_design.specialize(dummy=0).compile(
        xclbin_path=xp, inst_path=ip, pdi_path=pp)
    dt = time.time() - t0
    print(f"Compiled in {dt:.1f}s")
    print(f"  xclbin: {xp}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
