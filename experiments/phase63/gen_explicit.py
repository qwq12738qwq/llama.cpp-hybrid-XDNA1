#!/usr/bin/env python3
"""Phase 63: the explicit low level data path (aie.flow + TileDma + locks).

STATUS: the explicit path does not compile as written, and the reason is a
property of the IRON API rather than of the hardware (see "the blocker"
below).  The file is kept because the parts that do work are the ones the
ObjectFifo model cannot express, and because the blocker is the concrete
thing to fix to go further.

The blocker
-----------

The shim's L2 buffer is not an `aie.buffer`: in MLIR-AIE it is an
`aie.shim_dma_allocation`, and only the `Flow(shim_symbol=...)` path of IRON
creates one.  The IRON `Buffer` always lowers to `aie.buffer`, which is not
valid in a ShimNOCTile, so a shim side buffer descriptor cannot be built:

    "Operand 0 of operation aie.dma_bd must be a Value (is not a Value)"

Going further needs the allocation's memref to be handed to the TileDma,
which the IRON API does not expose today.  The MemTile side of the same
program (the S2MM channels that receive the operands, the core side that
reads them out of the L2, and the lock protocol) is implemented below and is
the part that the ObjectFifo model cannot express.


The ObjectFifo model of the phase 63 design expresses the MemTile relay, but
two things it cannot express are the reason the low level path is worth
writing down:

  * a host filled ObjectFifo cannot live in the MemTile L2 (the compiler:
    "aie.objectfifo.allocate op objectfifo has no shared memory access to
    delegate tile's memory module"), so the 1 row per column topologies fail,
  * the broadcast of the MemTile L2 to the workers is a property of the
    objectfifo machinery; with explicit buffer descriptors it is a plain
    "every core reads the same L2 region" and the lock protocol is visible.

This generator builds the same data path with the explicit primitives:

    host -> the shim L2 allocation -> (the shim's TileDma) -> the MemTile L2
          -> (the cores' TileDma) -> the core LMEM -> the kernel
          -> (the cores' TileDma) -> the MemTile L2 -> (the shim's TileDma)
          -> the host

The locks are the classic two lock protocol per buffer: a "free" lock that
the producer waits for and a "ready" lock that the consumer waits for.

usage:
  gen_explicit.py --cols 1 --rows 2 --m-tile 64 --n-tile 64 --k-chunk 256 \
      --work-dir DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.iron.device.tile import Tile
    from aie.iron.buffer import Buffer
    from aie.iron.lock import Lock
    from aie.iron.dataflow.flow import Flow
    from aie.iron.dataflow.tile_dma import TileDma, DmaChannel, Bd, Acquire, Release
    from aie.dialects._aie_enum_gen import DMAChannelDir, AIETileType
    from aie.utils import config
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False


KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void kern(const int8_t * __restrict at,
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
        for (int j = 0; j < NA; j++) {
            const int8_t * bj = b_tiles + j * (KCC / 8) * 64;
            aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
            MMUL acc(zero);
            for (int bb = 0; bb < NBLK; bb++) {
                const int8_t * ap = ai + bb * 4 * 64;
                const int8_t * bp = bj + bb * 4 * 64;
                acc.mac(aie::load_v<64>(ap),      aie::load_v<64>(bp));
                acc.mac(aie::load_v<64>(ap + 64),  aie::load_v<64>(bp + 64));
                acc.mac(aie::load_v<64>(ap + 128), aie::load_v<64>(bp + 128));
                acc.mac(aie::load_v<64>(ap + 192), aie::load_v<64>(bp + 192));
            }
            aie::vector<float, 64> p = aie::to_float(acc.to_vector<int32_t>());
            aie::vector<float, 8>  w = aie::load_v<8>(bsc + j * 8);
            aie::vector<float, 16> w16 = aie::broadcast<float, 16>(0.0f);
            w16.insert<8>(0, w);
            w16.insert<8>(1, w);
            for (int g = 0; g < 4; g++) {
                const float * ap2 = asi + 2 * g;
                aie::vector<float, 16> a16 = aie::broadcast<float, 16>(0.0f);
                a16.insert<8>(0, aie::broadcast<float, 8>(ap2[0]));
                a16.insert<8>(1, aie::broadcast<float, 8>(ap2[1]));
                aie::vector<float, 16> sc = aie::mul(a16, w16).to_vector<float>();
                aie::vector<float, 16> v =
                    aie::mul(p.template extract<16>(g), sc).to_vector<float>();
                float * c0 = c + (i * 8 + 2 * g + 0) * NT + j * 8;
                float * c1 = c + (i * 8 + 2 * g + 1) * NT + j * 8;
                aie::store_v(c0, aie::add(aie::load_v<8>(c0), v.template extract<8>(0)));
                aie::store_v(c1, aie::add(aie::load_v<8>(c1), v.template extract<8>(1)));
            }
        }
    }
}
"""


def build(M, K, N, cols, rows, m_tile, n_tile, k_chunk, stack_size=0x2000):
    n_mt = M // m_tile
    n_nt = N // n_tile
    n_kc = K // k_chunk
    wm, wn = m_tile, n_tile // rows

    a_bytes = wm * k_chunk + wm * 4
    b_bytes = k_chunk * wn + wn * 4
    c_piece = wm * wn
    lmem = a_bytes + b_bytes + c_piece * 4
    if lmem > 64 * 1024:
        raise ValueError(f"the worker needs {lmem} bytes of LMEM")
    stack = min(stack_size, 65536 - lmem)
    if stack < 4096:
        raise ValueError("no room for the kernel stack")

    src = KERNEL_SRC % {"MT": wm, "NT": wn, "KCC": k_chunk}
    kern = ExternalFunction(
        "kern",
        source_string=src,
        arg_types=[np.ndarray[(a_bytes,), np.dtype[np.int8]],
                   np.ndarray[(b_bytes,), np.dtype[np.int8]],
                   np.ndarray[(c_piece,), np.dtype[np.float32]]],
        include_dirs=[config.cxx_header_path()],
    )

    shims = [Tile(c, 0).with_type(AIETileType.ShimNOCTile) for c in range(cols)]
    mems = [Tile(c, 1).with_type(AIETileType.MemTile) for c in range(cols)]
    cores = [[Tile(c, 2 + r).with_type(AIETileType.CoreTile) for r in range(rows)]
             for c in range(cols)]

    flows = []
    tile_dmas = []
    locks = []
    workers = []

    def core_fn(buf_a, buf_b, buf_c, lock_ready, lock_free, kern):
        # the compute waits for the data of the chunk and releases the slot
        # when it is done with it
        lock_ready.acquire(1)
        kern(buf_a, buf_b, buf_c)
        lock_free.release(1)

    for c in range(cols):
        # the MemTile L2 holds the operands; the cores read them from there
        a_mem = Buffer(type=np.ndarray[(a_bytes,), np.dtype[np.int8]], tile=mems[c])
        b_mem = Buffer(type=np.ndarray[(b_bytes,), np.dtype[np.int8]], tile=mems[c])
        c_mem = Buffer(type=np.ndarray[(c_piece * rows,), np.dtype[np.float32]],
                       tile=mems[c])
        # the shim side: the host data lands in the shim L2 and the shim's DMA
        # moves it into the MemTile L2
        a_shim = Buffer(type=np.ndarray[(a_bytes,), np.dtype[np.int8]],
                        tile=shims[c], name=f"a{c}")
        b_shim = Buffer(type=np.ndarray[(b_bytes,), np.dtype[np.int8]],
                        tile=shims[c], name=f"b{c}")

        # the locks: one "free" and one "ready" per buffer and per column
        lk_a_free = Lock(mems[c], init=1)
        lk_a_ready = Lock(mems[c], init=0)
        lk_b_free = Lock(mems[c], init=1)
        lk_b_ready = Lock(mems[c], init=0)
        lk_c_free = Lock(mems[c], init=rows)
        lk_c_ready = Lock(mems[c], init=0)
        locks += [lk_a_free, lk_a_ready, lk_b_free, lk_b_ready,
                  lk_c_free, lk_c_ready]

        # the MemTile's DMA: the stream from the shim -> the MemTile L2
        tile_dmas.append(TileDma(mems[c], [
            DmaChannel(DMAChannelDir.S2MM, 0, [
                Bd(a_mem, acquires=[], releases=[]),
            ]),
            DmaChannel(DMAChannelDir.S2MM, 1, [
                Bd(b_mem, acquires=[], releases=[]),
            ]),
        ]))
        # the shim's DMA: the L2 allocation -> the MemTile L2
        tile_dmas.append(TileDma(shims[c], [
            DmaChannel(DMAChannelDir.MM2S, 0, [
                Bd(a_shim, acquires=[Acquire(lk_a_free, 1)],
                   releases=[Release(lk_a_ready, 1)]),
            ]),
            DmaChannel(DMAChannelDir.MM2S, 1, [
                Bd(b_shim, acquires=[Acquire(lk_b_free, 1)],
                   releases=[Release(lk_b_ready, 1)]),
            ]),
        ]))
        # the shim -> the MemTile routes
        flows.append(Flow(shims[c], mems[c], src_channel=0, dst_channel=0,
                          shim_symbol=f"a{c}"))
        flows.append(Flow(shims[c], mems[c], src_channel=1, dst_channel=1,
                          shim_symbol=f"b{c}"))

        for r in range(rows):
            a_core = Buffer(type=np.ndarray[(a_bytes,), np.dtype[np.int8]], tile=cores[c][r])
            b_core = Buffer(type=np.ndarray[(b_bytes,), np.dtype[np.int8]], tile=cores[c][r])
            c_core = Buffer(type=np.ndarray[(c_piece,), np.dtype[np.float32]], tile=cores[c][r])
            # the core reads the operands from the MemTile L2: the same
            # MemTile region, read by every core of the column
            tile_dmas.append(TileDma(cores[c][r], [
                DmaChannel(DMAChannelDir.S2MM, 0, [
                    Bd(a_core, acquires=[Acquire(lk_a_ready, 1)],
                       releases=[Release(lk_a_free, 0)]),
                ]),
                DmaChannel(DMAChannelDir.S2MM, 1, [
                    Bd(b_core, acquires=[Acquire(lk_b_ready, 1)],
                       releases=[Release(lk_b_free, 0)]),
                ]),
            ]))
            flows.append(Flow(mems[c], cores[c][r], src_channel=0, dst_channel=0))
            flows.append(Flow(mems[c], cores[c][r], src_channel=1, dst_channel=1))
            workers.append(Worker(core_fn, [a_core, b_core, c_core,
                                            lk_a_ready, lk_a_free, kern],
                                  stack_size=stack, tile=cores[c][r]))

    # the runtime sequence: the host fills the shim L2 allocations
    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(cols * (n_nt // cols) * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    def sequence(A, B, C, *ends):
        # the host fills the shim L2 allocations; the shim's DMA program then
        # moves them into the MemTile L2
        from aie.dialects.aiex import shim_dma_single_bd_task
        from aie.helpers.taplib import TensorAccessPattern
        n_nt_col = n_nt // cols
        for c in range(cols):
            for mt in range(n_mt):
                for nt in range(n_nt_col):
                    for kc in range(n_kc):
                        at = TensorAccessPattern(
                            tensor_dims=(n_mt * n_kc * a_bytes,),
                            offset=(mt * n_kc + kc) * a_bytes,
                            sizes=[a_bytes], strides=[1])
                        shim_dma_single_bd_task(f"a{c}", A, tap=at)
                        bt = TensorAccessPattern(
                            tensor_dims=(cols * n_nt_col * n_kc * b_bytes,),
                            offset=((c * n_nt_col + nt) * n_kc + kc) * b_bytes,
                            sizes=[b_bytes], strides=[1])
                        shim_dma_single_bd_task(f"b{c}", B, tap=bt)

    prod_cons = []
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    for fl in flows:
        rt.add_flow(fl)
    for td in tile_dmas:
        rt.add_tile_dma(td)
    for lk in locks:
        rt.add_lock(lk)
    return prog.resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--rows", type=int, default=2)
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
        return build(a.M, a.K, a.N, a.cols, a.rows, a.m_tile, a.n_tile, a.k_chunk)

    t0 = time.time()
    err = ""
    ok = True
    try:
        design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    except Exception as ex:            # noqa: BLE001
        ok = False
        err = str(ex)[-6000:]
    dt = round(time.time() - t0, 2)
    print(f"compile {'ok' if ok else 'FAILED'} in {dt}s")
    if not ok:
        print(err)
        sys.exit(1)
    print("SUCCESS")


if __name__ == "__main__":
    main()
