#!/usr/bin/env python3
"""Phase 63: a real multi-tile XDNA GEMM design (MemTile relay).

Phase 62 measured two things about the production design:

  * the shim NoC DMA budget is the number of shim-attached ObjectFifos
    (2 channels each), so the 4 worker design of phases 60/61 is at the limit
    and an 8 or 16 worker design does not compile,
  * the workers of a design never run concurrently, because the runtime
    sequence waits for every drain.

This generator builds the same GEMM with a different data flow: the operands
are relayed through the MemTile of the column (the `delegate_tile` of the
ObjectFifo), the workers of a column split the work, and the outputs are
gathered back through the MemTile.  That is the topology the compiler itself
asked for ("fan traffic through a MemTile").

The knobs:
  --cols        the number of columns (the shims)
  --rows        the number of compute tiles per column
  --split m|n   how the workers of a column split the output:
                  m: the rows (each worker gets its own activation slice,
                     the weights are broadcast)
                  n: the columns (each worker gets its own weight slice,
                     the activation is broadcast)
  --merge       pack the A and the B chunk into one buffer (one fill per
                worker instead of two, which halves the shim channel use)

The object sizes:
  A chunk (per worker): wm*KCC + wm*4      (the int8 tiles + the row scales)
  B chunk (per worker): KCC*wn + wn*4      (the int8 tiles + the col scales)
  C tile  (per worker): wm*wn floats

usage:
  gen_mtile.py --M 512 --K 1024 --N 3584 --cols 4 --rows 4 --split m \
      --m-tile 64 --n-tile 64 --k-chunk 256 --work-dir DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.iron.device.tile import Tile
    from aie.iron.dataflow.objectfifo import ObjectFifoLink
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False


# The phase 62 kernel: one scale per K chunk, two independent (i,j) chains.
# MT is the rows per worker.
KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void q8_c4_mmul_v4(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int MT   = %(MT)d;
    constexpr int NT   = %(NT)d;
    constexpr int KCC  = %(KCC)d;
    constexpr int NBLK = KCC / 32;
    constexpr int MA   = MT / 8;
    constexpr int NA   = NT / 8;
    const int8_t * a_tiles = at;
    const float  * asc = (const float *)(at + MT * KCC);
    const int8_t * b_tiles = bt;
    const float  * bsc = (const float *)(bt + KCC * NT);

    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    for (int i = 0; i < MA; i++) {
        const int8_t * ai = a_tiles + i * (KCC / 8) * 64;
        const float  * asi = asc + i * 8;
        for (int j = 0; j + 1 < NA; j += 2) {
            const int8_t * bj0 = b_tiles + j * (KCC / 8) * 64;
            const int8_t * bj1 = b_tiles + (j + 1) * (KCC / 8) * 64;
            aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
            MMUL acc0(zero);
            MMUL acc1(zero);
            for (int bb = 0; bb < NBLK; bb++) {
                const int8_t * ap = ai + bb * 4 * 64;
                aie::vector<int8_t, 64> a0 = aie::load_v<64>(ap);
                aie::vector<int8_t, 64> a1 = aie::load_v<64>(ap + 64);
                aie::vector<int8_t, 64> a2 = aie::load_v<64>(ap + 128);
                aie::vector<int8_t, 64> a3 = aie::load_v<64>(ap + 192);
                const int8_t * bp0 = bj0 + bb * 4 * 64;
                const int8_t * bp1 = bj1 + bb * 4 * 64;
                acc0.mac(a0, aie::load_v<64>(bp0));
                acc0.mac(a1, aie::load_v<64>(bp0 + 64));
                acc0.mac(a2, aie::load_v<64>(bp0 + 128));
                acc0.mac(a3, aie::load_v<64>(bp0 + 192));
                acc1.mac(a0, aie::load_v<64>(bp1));
                acc1.mac(a1, aie::load_v<64>(bp1 + 64));
                acc1.mac(a2, aie::load_v<64>(bp1 + 128));
                acc1.mac(a3, aie::load_v<64>(bp1 + 192));
            }
            aie::vector<float, 64> p0 = aie::to_float(acc0.to_vector<int32_t>());
            aie::vector<float, 64> p1 = aie::to_float(acc1.to_vector<int32_t>());
            aie::vector<float, 8>  w0 = aie::load_v<8>(bsc + j * 8);
            aie::vector<float, 8>  w1 = aie::load_v<8>(bsc + (j + 1) * 8);
            aie::vector<float, 16> w16a = aie::broadcast<float, 16>(0.0f);
            w16a.insert<8>(0, w0);
            w16a.insert<8>(1, w0);
            aie::vector<float, 16> w16b = aie::broadcast<float, 16>(0.0f);
            w16b.insert<8>(0, w1);
            w16b.insert<8>(1, w1);
            for (int g = 0; g < 4; g++) {
                const float * ap2 = asi + 2 * g;
                aie::vector<float, 16> a16 = aie::broadcast<float, 16>(0.0f);
                a16.insert<8>(0, aie::broadcast<float, 8>(ap2[0]));
                a16.insert<8>(1, aie::broadcast<float, 8>(ap2[1]));
                aie::vector<float, 16> sca = aie::mul(a16, w16a).to_vector<float>();
                aie::vector<float, 16> scb = aie::mul(a16, w16b).to_vector<float>();
                aie::vector<float, 16> v0 =
                    aie::mul(p0.template extract<16>(g), sca).to_vector<float>();
                aie::vector<float, 16> v1 =
                    aie::mul(p1.template extract<16>(g), scb).to_vector<float>();
                float * c0 = c + (i * 8 + 2 * g + 0) * NT + j * 8;
                float * c1 = c + (i * 8 + 2 * g + 1) * NT + j * 8;
                float * c2 = c + (i * 8 + 2 * g + 0) * NT + (j + 1) * 8;
                float * c3 = c + (i * 8 + 2 * g + 1) * NT + (j + 1) * 8;
                aie::store_v(c0, aie::add(aie::load_v<8>(c0), v0.template extract<8>(0)));
                aie::store_v(c1, aie::add(aie::load_v<8>(c1), v0.template extract<8>(1)));
                aie::store_v(c2, aie::add(aie::load_v<8>(c2), v1.template extract<8>(0)));
                aie::store_v(c3, aie::add(aie::load_v<8>(c3), v1.template extract<8>(1)));
            }
        }
    }
}
"""


def build(M, K, N, cols, rows, m_tile, n_tile, k_chunk, split, merge,
          fifo_depth=1, stack_size=0x2000):
    """Build the design and return the resolved IRON program."""
    n_mt = M // m_tile          # the m-tiles of the whole output
    n_nt = N // n_tile          # the n-tiles of the whole output
    n_kc = K // k_chunk
    if n_nt % cols:
        raise ValueError(f"the {n_nt} n-tiles do not divide over {cols} columns")

    # the work of one worker
    if split == "m":
        wm, wn = m_tile // rows, n_tile
    else:
        wm, wn = m_tile, n_tile // rows

    a_bytes = wm * k_chunk + wm * 4
    b_bytes = k_chunk * wn + wn * 4
    c_piece = wm * wn
    if merge:
        in_bytes = a_bytes + b_bytes
        lmem = in_bytes + c_piece * 4
    else:
        in_bytes = a_bytes
        lmem = in_bytes + b_bytes + c_piece * 4
    if lmem > 64 * 1024:
        raise ValueError(f"the worker needs {lmem} bytes of LMEM, the limit is 65536")
    # the core LMEM holds the operand buffers, the C buffer and the stack:
    # the double buffered variant needs a smaller stack
    stack = min(stack_size if stack_size else 16384, 65536 - lmem)
    if stack < 4096:
        raise ValueError("no room for the kernel stack")

    src = KERNEL_SRC % {"MT": wm, "NT": wn, "KCC": k_chunk}
    kern = ExternalFunction(
        "q8_c4_mmul_v4",
        source_string=src,
        arg_types=([np.ndarray[(in_bytes,), np.dtype[np.int8]],
                    np.ndarray[(c_piece,), np.dtype[np.float32]]] if merge else
                   [np.ndarray[(a_bytes,), np.dtype[np.int8]],
                    np.ndarray[(b_bytes,), np.dtype[np.int8]],
                    np.ndarray[(c_piece,), np.dtype[np.float32]]]),
        include_dirs=[config.cxx_header_path()],
    )

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_piece,), np.dtype[np.float32]]
    # the gathered C tile of a column
    c_grp_ty = np.ndarray[(rows * c_piece,), np.dtype[np.float32]]
    # the operand objects: the activation travels as a group of `rows` per
    # worker chunks, the weights as one chunk that the column shares
    a_grp_ty = np.ndarray[(rows * a_bytes,), np.dtype[np.int8]]
    a_obj, b_obj = a_grp_ty, b_ty
    # the object that travels on the wire: the whole group when it is split
    a_grp_ty = np.ndarray[(rows * a_bytes,), np.dtype[np.int8]]
    b_grp_ty = np.ndarray[(rows * b_bytes,), np.dtype[np.int8]]
    c_grp_ty = np.ndarray[(rows * c_piece,), np.dtype[np.float32]]

    shims = [Tile(c, 0) for c in range(cols)]
    mems = [Tile(c, 1) for c in range(cols)]
    cores = [[Tile(c, 2 + r) for r in range(rows)] for c in range(cols)]

    # the groups a worker of a column handles
    n_nt_col = n_nt // cols
    if split == "m":
        groups_per_worker = n_mt * n_nt_col
    else:
        groups_per_worker = n_mt * n_nt_col // rows

    def core_fn(*args):
        if merge:
            of_in, of_c, kern = args
        else:
            of_a, of_b, of_c, kern = args
        for _ in iron_range(groups_per_worker):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_piece):
                elem_c[x] = 0.0
            for _kc in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                if merge:
                    kern(elem_a, elem_c)
                else:
                    elem_b = of_b.acquire(1)
                    kern(elem_a, elem_b, elem_c)
                    of_b.release(1)
                of_a.release(1)
            of_c.release(1)

    workers = []
    fifos = []
    for c in range(cols):
        # The host fills the operands into the shim, they are forwarded
        # through the MemTile of the column and the workers of the column
        # read them from the MemTile L2: one host transfer feeds the whole
        # column instead of one transfer per worker.
        # The activation is split between the workers of the column (each of
        # them computes its own rows) and the weights are broadcast (all the
        # workers of the column use the same weights).  The host packs the
        # activation group as the concatenation of the per worker chunks, so
        # the activation is transferred once, without replication.
        fa = ObjectFifo(a_grp_ty, name=f"a{c}", depth=fifo_depth)
        fb = ObjectFifo(b_obj, name=f"b{c}", depth=fifo_depth)
        b_fwd = fb.cons().forward(tile=mems[c], obj_type=b_ty, depth=fifo_depth)
        # the C output: the workers write their piece into the MemTile L2 and
        # the MemTile gathers the pieces into one tile that the shim drains:
        # one drain per column instead of one per worker
        c_fifos = [ObjectFifo(c_ty, name=f"c{c}_{r}", depth=fifo_depth,
                              delegate_tile=mems[c]) for r in range(rows)]
        fc = ObjectFifo(c_grp_ty, name=f"c{c}", depth=fifo_depth)
        _ = ObjectFifoLink([c_fifos[r].cons() for r in range(rows)], fc.prod(),
                           tile=mems[c],
                           src_offsets=[r * c_piece for r in range(rows)])
        c_handles = [fc.cons()]

        # the activation is split by the MemTile, the weights are forwarded
        # to it and broadcast to the workers
        a_subs = fa.cons().split([i * a_bytes for i in range(rows)],
                                  tile=mems[c], obj_types=[a_ty] * rows)
        a_handles = [s.cons() for s in a_subs]
        b_handles = [b_fwd.cons() for _ in range(rows)]

        for r in range(rows):
            if merge:
                fn_args = [a_handles[r], c_fifos[r].prod(), kern]
            else:
                fn_args = [a_handles[r], b_handles[r], c_fifos[r].prod(), kern]
            workers.append(Worker(core_fn, fn_args, stack_size=stack,
                                  tile=cores[c][r]))
        fifos.append((fa, fb, c_handles))

    # the host side buffers
    A_ty = np.ndarray[(n_mt * n_kc * rows * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(cols * n_nt_col * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    def sequence(A, B, C, *ends):
        # the ends are, per column: the A producer, the B producer and the
        # C consumer (the gathered tile)
        #
        # Phase 62: one task group per (m-tile, column, n-tile) with a wait
        # on the drain serialises the workers: the fill of the next group is
        # only issued after the drain of the previous one has completed, which
        # is after the worker has finished computing it.  One task group per
        # (m-tile, n-tile) with the fills of all the columns issued together
        # lets the workers compute at the same time.
        k = 0
        cols_end = []
        for c in range(cols):
            a_p = ends[k]; k += 1
            b_p = ends[k]; k += 1
            c_c = ends[k]; k += 1
            cols_end.append((a_p, b_p, c_c))
        for mt in range(n_mt):
            for nt in range(n_nt_col):
                tg = iron.TaskGroup()
                for ci, (a_p, b_p, c_c) in enumerate(cols_end):
                    a_p.fill(A, tap=None, group=tg)
                    if not merge:
                        b_p.fill(B, tap=None, group=tg)
                    # only the last drain waits, the others overlap
                    c_c.drain(C, tap=None, group=tg,
                              wait=(ci == cols - 1))
                tg.finish()

    prod_cons = []
    for f in fifos:
        prod_cons.append(f[0].prod())
        prod_cons.append(f[1].prod())
        for fc in f[2]:
            prod_cons.append(fc)
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--split", default="m", choices=["m", "n"])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--stack", type=int, default=0,
                    help="the core stack size in bytes (0 = the default 16 KB)")
    ap.add_argument("--depth", type=int, default=1,
                    help="the depth of the operand FIFOs: 2 double buffers the "
                         "transfer against the compute")
    ap.add_argument("--m-tile", type=int, default=64)
    ap.add_argument("--n-tile", type=int, default=64)
    ap.add_argument("--k-chunk", type=int, default=256)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.makedirs(a.work_dir, exist_ok=True)
    xp = os.path.join(a.work_dir, "design.xclbin")
    ip = os.path.join(a.work_dir, "design.insts.bin")
    pp = os.path.join(a.work_dir, "main.pdi")

    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(a.M, a.K, a.N, a.cols, a.rows, a.m_tile, a.n_tile,
                     a.k_chunk, a.split, a.merge, fifo_depth=a.depth,
                     stack_size=a.stack)

    t0 = time.time()
    err = ""
    ok = True
    try:
        design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    except Exception as ex:            # noqa: BLE001
        ok = False
        err = str(ex)[-8000:]
    dt = round(time.time() - t0, 2)
    rec = {"tag": a.tag, "cols": a.cols, "rows": a.rows, "split": a.split,
           "merge": a.merge, "workers": a.cols * a.rows,
           "m_tile": a.m_tile, "n_tile": a.n_tile, "k_chunk": a.k_chunk,
           "compile_ok": ok, "compile_s": dt, "error": err}
    with open(os.path.join(a.work_dir, "compile.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"compile {'ok' if ok else 'FAILED'} in {dt}s  "
          f"({a.cols} cols x {a.rows} rows = {a.cols * a.rows} workers, "
          f"split {a.split}{', merged' if a.merge else ''})")
    if not ok:
        print(err)
        sys.exit(1)
    print("SUCCESS")


if __name__ == "__main__":
    main()
