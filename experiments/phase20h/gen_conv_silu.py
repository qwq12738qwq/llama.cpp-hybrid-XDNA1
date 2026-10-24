#!/usr/bin/env python3
"""Phase 20H: XDNA depthwise causal conv1d (SSM_CONV) for Qwen3.5 linear attn.

Matches ggml_ssm_conv layout exactly:
  sx  (src0): [C, Tin] f32   Tin = T + K - 1   (time is the contiguous dim)
  w   (src1): [C, K]   f32                     (tap is the contiguous dim)
  out (dst) : [T, C]   f32                     (channel is the contiguous dim)

  out[c, t] = sum_{k=0..K-1} sx[c, t + k] * w[c, k]

MemTile (L2) staging, two-level ObjectFifo hierarchy:
  A (sx): L3 -> L2 fill -> split  -> per-row L2 -> L1
  W     : L3 -> L2 fill -> forward -> per-col L2 -> L1   (shared by the column)
  O     : L1 -> L3 direct (each (row, col) tile drains its own block; no join,
                           so no T_row*C stride, which the BD unit rejects)

C is split across n_cols columns; T across n_rows rows. Each tile walks channel
sub-blocks of c_sub (inner loop). Shim streams = 2*n_cols + n_rows*n_cols.

Usage:
  python gen_conv_silu.py --C 6144 --T 512 --K 4 --c-sub 32 --n-rows 4 --n-cols 2 --work-dir /tmp/conv
"""
import argparse
import os
import time

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
from aie.iron.controlflow import range_ as iron_range
from aie.iron.runtime.taskgroup import TaskGroup
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config

KERNEL_SRC_TMPL = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// in : [c_sub, win] f32   (win = T_row + K - 1, contiguous per channel)
// w  : [c_sub, K]   f32
// out: [T_row, c_sub] f32  (out[t*c_sub + c] = sum_k in[c*win + t + k] * w[c*K + k])
extern "C" void conv_silu(
    const float * __restrict in,
    const float * __restrict w,
    float * __restrict out,
    int32_t c_sub, int32_t T_row, int32_t win, int32_t K)
{
    constexpr int32_t VEC = {vec};
    for (int32_t c = 0; c < c_sub; c++) {
        const float * ip = in + c * win;
        const float * wp = w + c * K;
        for (int32_t t = 0; t < T_row; t += VEC) {
            aie::accum<accfloat, VEC> acc = aie::zeros<accfloat, VEC>();
            for (int32_t k = 0; k < K; k++) {
                aie::vector<float, VEC> iv;
                for (int32_t i = 0; i < VEC; i++) iv[i] = ip[t + k + i];
                acc = aie::mac(acc, iv, aie::broadcast<float, VEC>(wp[k]));
            }
            aie::vector<float, VEC> ov = acc.to_vector<float>();
            for (int32_t i = 0; i < VEC; i++) out[(t + i) * c_sub + c] = ov[i];
        }
    }
}
"""


def build(C, Tin, K, c_sub, n_rows, n_cols, vec):
    T = Tin - (K - 1)
    assert K >= 1 and K <= 8
    assert C % n_cols == 0
    assert T % n_rows == 0
    C_col = C // n_cols
    T_row = T // n_rows
    win = T_row + K - 1
    assert C_col % c_sub == 0
    n_csub = C_col // c_sub

    VEC = vec
    assert T_row % VEC == 0, f"T_row={T_row} must be a multiple of {VEC}"
    assert c_sub % VEC == 0, f"c_sub={c_sub} must be a multiple of {VEC}"
    n_tiles = n_rows * n_cols

    in_elems = c_sub * win
    w_elems = c_sub * K
    out_elems = T_row * c_sub

    stack_bytes = 0x2000
    l1_per_tile = stack_bytes + in_elems * 4 + w_elems * 4 + out_elems * 4
    L1_MAX = 64 * 1024
    print(f"  per-tile L1: stack={stack_bytes} in={in_elems*4} w={w_elems*4} "
          f"out={out_elems*4} total={l1_per_tile}/{L1_MAX}")
    if l1_per_tile > L1_MAX:
        raise ValueError(f"per-tile L1 {l1_per_tile} exceeds {L1_MAX}")

    a_l2 = n_rows * in_elems * 4
    b_l2 = w_elems * 4
    l2_per_col = a_l2 + b_l2
    L2_MAX = 128 * 1024
    print(f"  per-col L2: A={a_l2} W={b_l2} total={l2_per_col}/{L2_MAX}")
    if l2_per_col > L2_MAX:
        raise ValueError(f"per-col L2 {l2_per_col} exceeds {L2_MAX}")

    n_shim = 2 * n_cols + n_rows * n_cols
    print(f"  shim streams: {n_shim} (2*n_cols + n_rows*n_cols); tiles={n_tiles}")
    if n_shim > 16:
        raise ValueError(f"{n_shim} shim streams exceeds budget")

    fused = ExternalFunction(
        "conv_silu",
        source_string=KERNEL_SRC_TMPL.replace("{vec}", str(VEC)),
        arg_types=[
            np.ndarray[(in_elems,), np.dtype[np.float32]],
            np.ndarray[(w_elems,), np.dtype[np.float32]],
            np.ndarray[(out_elems,), np.dtype[np.float32]],
            np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(in_elems,), np.dtype[np.float32]]
    w_ty = np.ndarray[(w_elems,), np.dtype[np.float32]]
    o_ty = np.ndarray[(out_elems,), np.dtype[np.float32]]
    A_l2_ty = np.ndarray[(n_rows * in_elems,), np.dtype[np.float32]]
    W_l2_ty = np.ndarray[(w_elems,), np.dtype[np.float32]]

    a_l3l2, w_l3l2 = [], []
    A_l2l1, W_l2l1 = [], []
    O_l1l3 = [[None] * n_cols for _ in range(n_rows)]

    for col in range(n_cols):
        a2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=1)
        a_l3l2.append(a2)
        A_l2l1.append(a2.cons().split(
            [row * in_elems for row in range(n_rows)],
            obj_types=[a_ty] * n_rows,
            names=[f"A_L2L1_{col}_{row}" for row in range(n_rows)],
            depths=[1] * n_rows,
        ))

        w2 = ObjectFifo(W_l2_ty, name=f"W_L3L2_{col}", depth=1)
        w_l3l2.append(w2)
        W_l2l1.append(w2.cons().forward(
            obj_type=w_ty, name=f"W_L2L1_{col}", depth=1))

        for row in range(n_rows):
            O_l1l3[row][col] = ObjectFifo(o_ty, name=f"O_{col}_{row}", depth=1)

    def core_fn(of_a, of_w, of_o, kern, c_sub, T_row, win, K, n_csub):
        for _ in iron_range(n_csub):
            ea = of_a.acquire(1)
            ew = of_w.acquire(1)
            eo = of_o.acquire(1)
            kern(ea, ew, eo, c_sub, T_row, win, K)
            of_a.release(1)
            of_w.release(1)
            of_o.release(1)

    workers = Worker.grid(
        n_rows,
        n_cols,
        lambda row, col: Worker(
            core_fn,
            [A_l2l1[col][row].cons(), W_l2l1[col].cons(), O_l1l3[row][col].prod(),
             fused, c_sub, T_row, win, K, n_csub],
            stack_size=stack_bytes,
        ),
    )

    A_ty = np.ndarray[(C * Tin,), np.dtype[np.float32]]
    W_ty = np.ndarray[(C * K,), np.dtype[np.float32]]
    O_ty = np.ndarray[(T * C,), np.dtype[np.float32]]

    def a_tap(col, csub):
        # (n_rows, c_sub, win) block of the (C, Tin) source. Row r starts at
        # time r*T_row (windows overlap by K-1); the channel block is shared.
        return TensorAccessPattern(
            tensor_dims=(C, Tin),
            offset=col * C_col * Tin + csub * c_sub * Tin,
            sizes=[n_rows, c_sub, win],
            strides=[T_row, Tin, 1])

    def w_tap(col, csub):
        return TensorAccessPattern(
            tensor_dims=(C, K),
            offset=col * C_col * K + csub * c_sub * K,
            sizes=[c_sub, K],
            strides=[K, 1])

    def o_tap(row, col, csub):
        return TensorAccessPattern(
            tensor_dims=(T, C),
            offset=row * T_row * C + col * C_col + csub * c_sub,
            sizes=[T_row, c_sub],
            strides=[C, 1])

    def sequence(A, W, O, *fifos):
        a_p = fifos[0:n_cols]
        w_p = fifos[n_cols:2 * n_cols]
        rest = fifos[2 * n_cols:]
        oc = [[rest[r * n_cols + c] for c in range(n_cols)] for r in range(n_rows)]
        for col in range(n_cols):
            for csub in range(n_csub):
                tg = TaskGroup()
                a_p[col].fill(A, tap=a_tap(col, csub), group=tg)
                w_p[col].fill(W, tap=w_tap(col, csub), group=tg)
                for row in range(n_rows):
                    oc[row][col].drain(O, tap=o_tap(row, col, csub), wait=True, group=tg)
                tg.finish()

    prod_cons = []
    for col in range(n_cols):
        prod_cons.append(a_l3l2[col].prod())
    for col in range(n_cols):
        prod_cons.append(w_l3l2[col].prod())
    for row in range(n_rows):
        for col in range(n_cols):
            prod_cons.append(O_l1l3[row][col].cons())
    rt = Runtime(sequence, [A_ty, W_ty, O_ty] + prod_cons)
    flat_workers = [w for row in workers for w in row]
    return iron.Program(iron.get_current_device(), rt,
                        workers=flat_workers).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=6144)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--c-sub", type=int, default=32)
    ap.add_argument("--vec", type=int, default=8)
    ap.add_argument("--n-rows", type=int, default=4)
    ap.add_argument("--n-cols", type=int, default=2)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    Tin = args.T + args.K - 1

    @iron.jit
    def design(A: In, W: In, O: Out, *, dummy: CompileTime[int] = 0):
        return build(args.C, Tin, args.K, args.c_sub, args.n_rows, args.n_cols, args.vec)

    print(f"conv_silu: C={args.C} T={args.T} K={args.K} c_sub={args.c_sub} "
          f"grid={args.n_rows}x{args.n_cols}")
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print(f"  xclbin: {compiled[0]} ({os.path.getsize(compiled[0])} bytes)")
    print(f"  insts:  {compiled[1]} ({os.path.getsize(compiled[1])} bytes)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
