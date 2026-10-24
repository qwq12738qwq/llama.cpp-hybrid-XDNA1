#!/usr/bin/env python3
"""Phase 20I: resident chain microbenchmark.

Question: does keeping intermediate tensors INSIDE the NPU (one staging in, one
staging out) amortise the ~2.2 GB/s host shim-DMA cost, vs N separate
host-staged single-op submissions?

One worker streams the input in chunks; for each chunk it applies N chained
elementwise ops before storing.  N=0 is a pure copy (DMA floor).  Only X and Y
cross the shim, so total shim traffic is fixed at |X|+|Y| regardless of N; the
chain compute is inside the loop.

Op used: v = v*v + 1 (cheap, non-collapsible across N).

Usage:
  python gen_chain.py --S 3145728 --chunk 2048 --ops 1 --work-dir /tmp/c1
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

// x, y: [n] f32 contiguous.  y[i] = f^N(x[i]), f(v)=v*v+1
extern "C" void chain_kernel(const float * __restrict x, float * __restrict y,
                             int32_t n, int32_t N)
{
    constexpr int32_t VEC = {vec};
    const aie::vector<float, VEC> one = aie::broadcast<float, VEC>(1.0f);
    for (int32_t i = 0; i < n; i += VEC) {
        aie::vector<float, VEC> v = aie::load_v<VEC>(x + i);
        for (int32_t k = 0; k < N; k++) {
            v = aie::add(v, one);
        }
        aie::store_v(y + i, v);
    }
}
"""


def build(S, chunk, ops, depth, vec, n_cols):
    assert S % chunk == 0
    n_chunk = S // chunk
    chunk_e = chunk
    assert chunk_e % vec == 0
    assert n_chunk % n_cols == 0
    n_pc = n_chunk // n_cols
    n_shim = 2 * n_cols
    print(f"  shim streams: {n_shim}")
    if n_shim > 16:
        raise ValueError(f"{n_shim} shim streams exceeds budget")

    stack_bytes = 0x2000
    l1 = stack_bytes + depth * 2 * chunk_e * 4
    print(f"  per-tile L1: {l1}/65536")
    if l1 > 64 * 1024:
        raise ValueError(f"L1 {l1} exceeds 65536")

    fused = ExternalFunction(
        "chain_kernel",
        source_string=KERNEL_SRC_TMPL.replace("{vec}", str(vec)),
        arg_types=[
            np.ndarray[(chunk_e,), np.dtype[np.float32]],
            np.ndarray[(chunk_e,), np.dtype[np.float32]],
            np.int32, np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
    )

    x_ty = np.ndarray[(chunk_e,), np.dtype[np.float32]]
    y_ty = np.ndarray[(chunk_e,), np.dtype[np.float32]]
    fx = [ObjectFifo(x_ty, name=f"X{c}", depth=depth) for c in range(n_cols)]
    fy = [ObjectFifo(y_ty, name=f"Y{c}", depth=depth) for c in range(n_cols)]

    def core_fn(of_x, of_y, kern, chunk_e, ops, n_pc):
        for _ in iron_range(n_pc):
            ex = of_x.acquire(1)
            ey = of_y.acquire(1)
            kern(ex, ey, chunk_e, ops)
            of_x.release(1)
            of_y.release(1)

    workers = Worker.grid(
        1, n_cols,
        lambda _r, col: Worker(core_fn, [fx[col].cons(), fy[col].prod(), fused, chunk_e, ops, n_pc],
                               stack_size=stack_bytes),
    )

    X_ty = np.ndarray[(S,), np.dtype[np.float32]]
    Y_ty = np.ndarray[(S,), np.dtype[np.float32]]

    def tap(i):
        return TensorAccessPattern(tensor_dims=(S,), offset=i * chunk_e,
                                   sizes=[chunk_e], strides=[1])

    def sequence(X, Y, *fifos):
        x_p = fifos[0:n_cols]
        y_c = fifos[n_cols:2 * n_cols]
        for col in range(n_cols):
            for j in range(n_pc):
                i = col * n_pc + j
                tg = iron.TaskGroup()
                x_p[col].fill(X, tap=tap(i), group=tg)
                y_c[col].drain(Y, tap=tap(i), wait=True, group=tg)
                tg.finish()

    prods = [fx[c].prod() for c in range(n_cols)]
    cons = [fy[c].cons() for c in range(n_cols)]
    rt = Runtime(sequence, [X_ty, Y_ty] + prods + cons)
    flat = [w for row in workers for w in row]
    return iron.Program(iron.get_current_device(), rt, workers=flat).resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=6144 * 512)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--ops", type=int, default=1)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--vec", type=int, default=8)
    ap.add_argument("--n-cols", type=int, default=1)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(X: In, Y: Out, *, dummy: CompileTime[int] = 0):
        return build(args.S, args.chunk, args.ops, args.depth, args.vec, args.n_cols)

    print(f"chain: S={args.S} ({args.S*4/1e6:.2f} MB) chunk={args.chunk} ops={args.ops} depth={args.depth}")
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print(f"  xclbin: {compiled[0]} ({os.path.getsize(compiled[0])} bytes)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
