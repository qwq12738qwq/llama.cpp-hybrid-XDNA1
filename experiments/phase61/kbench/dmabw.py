#!/usr/bin/env python3
"""Phase 61B: peak streaming DMA bandwidth of the XDNA (RyzenAI NPU1).

This is the number that decides whether the NPU can take any *memory bound*
operator (GLU, SILU, MUL, CONCAT, norms) off the GPU: an elementwise op on
the device is only as fast as the DMA that feeds it.

The design is a single worker that copies three big buffers through the
device: the runtime sequence fills A and B from host memory and drains C back,
and the kernel does nothing but touch one element. The wall time of the run is
therefore the DMA time.

usage: dmabw.py [--mb 64] [--cols 1]
"""
import argparse
import os
import time

import numpy as np

KERNEL = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void touch(const int8_t * __restrict at,
                      const int8_t * __restrict bt,
                      float        * __restrict c)
{
    c[0] = (float)(at[0] + bt[0]);
}
"""


def build(mb, n_cols, elem_kb):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config

    nbytes = mb * 1024 * 1024
    elem = elem_kb * 1024
    assert nbytes % elem == 0
    n_elem = nbytes // elem
    a_ty = np.ndarray[(elem,), np.dtype[np.int8]]
    b_ty = np.ndarray[(elem,), np.dtype[np.int8]]
    c_ty = np.ndarray[(elem // 4,), np.dtype[np.float32]]

    kern = ExternalFunction(
        "touch",
        source_string=KERNEL,
        arg_types=[
            np.ndarray[(elem,), np.dtype[np.int8]],
            np.ndarray[(elem,), np.dtype[np.int8]],
            np.ndarray[(elem // 4,), np.dtype[np.float32]],
        ],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_elem):
            elem_a = of_a.acquire(1)
            elem_b = of_b.acquire(1)
            elem_c = of_c.acquire(1)
            kern(elem_a, elem_b, elem_c)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    fifos = []
    workers = []
    for col in range(n_cols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=1)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=1)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=1)
        fifos.append((fa, fb, fc))
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern],
                              stack_size=4096))

    A_ty = np.ndarray[(nbytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(nbytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(nbytes // 4,), np.dtype[np.float32]]

    def sequence(A, B, C, *ends):
        for col in range(n_cols):
            for e in range(n_elem):
                a_tap = TensorAccessPattern(tensor_dims=(nbytes,), offset=e * elem,
                                            sizes=[elem], strides=[1])
                b_tap = TensorAccessPattern(tensor_dims=(nbytes,), offset=e * elem,
                                            sizes=[elem], strides=[1])
                c_tap = TensorAccessPattern(tensor_dims=(nbytes // 4,), offset=e * (elem // 4),
                                            sizes=[elem // 4], strides=[1])
                tg = iron.TaskGroup()
                fifos[col][0].prod().fill(A, tap=a_tap, group=tg)
                fifos[col][1].prod().fill(B, tap=b_tap, group=tg)
                fifos[col][2].cons().drain(C, tap=c_tap, group=tg, wait=True)
                tg.finish()

    prod_cons = []
    for f in fifos:
        prod_cons.append(f[0].prod())
    for f in fifos:
        prod_cons.append(f[1].prod())
    for f in fifos:
        prod_cons.append(f[2].cons())
    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=32)
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--elem-kb", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--work-dir", default="/tmp/dmabw")
    a = ap.parse_args()

    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    os.makedirs(a.work_dir, exist_ok=True)
    xp = os.path.join(a.work_dir, "design.xclbin")
    ip = os.path.join(a.work_dir, "design.insts.bin")
    pp = os.path.join(a.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(a.mb, a.cols, a.elem_kb)

    t0 = time.time()
    design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print("compiled in %.1fs" % (time.time() - t0))

    import pyxrt
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xp)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    gid3, gid1 = kern.group_id(3), kern.group_id(1)

    nbytes = a.mb * 1024 * 1024
    a_bo = pyxrt.bo(dev, nbytes, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(dev, nbytes, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(dev, nbytes, pyxrt.bo.flags.host_only, gid3)
    inst_bo = pyxrt.bo(dev, os.path.getsize(ip), pyxrt.bo.flags.cacheable, gid1)
    with open(ip, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    a_bo.write(np.zeros(nbytes, np.int8).tobytes(), 0)
    b_bo.write(np.zeros(nbytes, np.int8).tobytes(), 0)
    c_bo.write(np.zeros(nbytes, np.int8).tobytes(), 0)
    for b in (a_bo, b_bo, c_bo):
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    times = []
    for _ in range(a.repeat):
        r = pyxrt.run(kern)
        r.set_arg(0, 3)
        r.set_arg(1, inst_bo)
        r.set_arg(2, os.path.getsize(ip))
        r.set_arg(3, a_bo)
        r.set_arg(4, b_bo)
        r.set_arg(5, c_bo)
        t0 = time.perf_counter()
        r.start()
        r.wait()
        times.append((time.perf_counter() - t0) * 1000.0)
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    med = sorted(times)[len(times) // 2]
    traffic = 3 * nbytes * a.cols
    print("mb=%d cols=%d  median=%.2f ms  traffic=%.1f MB  bandwidth=%.2f GB/s"
          % (a.mb, a.cols, med, traffic / 1e6, traffic / (med / 1000.0) / 1e9))
    print("  all runs: " + " ".join("%.1f" % t for t in times))


if __name__ == "__main__":
    main()
