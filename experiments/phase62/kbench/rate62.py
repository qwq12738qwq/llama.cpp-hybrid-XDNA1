#!/usr/bin/env python3
"""Phase 62: the AIE2 integer path vs the float path.

Phase 61 measured the float scale application (k7/k8/k9) and concluded that
the Q8_0 per-32-block scale is at the arithmetic balance point of the AIE2.
The hypothesis this benchmark tests is different:

    the float ops are not throughput bound, they are LATENCY bound
    (a float vmac.f costs ~80 ns of latency on the AIE2 float matrix unit),
    and the int32 vector path has a much shorter latency.

If that is true, a fixed-point (integer) scale application removes the
bottleneck without touching the quantization semantics beyond the rounding
of the fixed-point mantissa.

The kernels below all run the same shape: N iterations of an op on 16-lane
vectors, with the operands either in registers (pure op cost) or loaded from
LMEM each iteration (comparable with the phase 61 numbers).

usage: rate62.py [--iters 4096] [--work-dir DIR]
"""
import argparse
import os
import time

import numpy as np

# ---------------------------------------------------------------- int32 path
# int32 vector multiply, one chain: acc = acc + x*y
I1 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<int32_t, 16> x = aie::load_v<16>((const int32_t *)at);
    aie::vector<int32_t, 16> y = aie::load_v<16>((const int32_t *)bt);
    aie::vector<int32_t, 16> acc = aie::broadcast<int32_t, 16>(0);
    for (int i = 0; i < N; i++) {
        acc = aie::add(acc, aie::mul(x, y).to_vector<int32_t>());
    }
    aie::store_v((int32_t *)c, acc);
}
"""

# four independent int32 multiply chains
I2 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<int32_t, 16> x = aie::load_v<16>((const int32_t *)at);
    aie::vector<int32_t, 16> y = aie::load_v<16>((const int32_t *)bt);
    aie::vector<int32_t, 16> a0 = aie::broadcast<int32_t, 16>(0);
    aie::vector<int32_t, 16> a1 = aie::broadcast<int32_t, 16>(0);
    aie::vector<int32_t, 16> a2 = aie::broadcast<int32_t, 16>(0);
    aie::vector<int32_t, 16> a3 = aie::broadcast<int32_t, 16>(0);
    for (int i = 0; i < N; i++) {
        a0 = aie::add(a0, aie::mul(x, y).to_vector<int32_t>());
        a1 = aie::add(a1, aie::mul(x, y).to_vector<int32_t>());
        a2 = aie::add(a2, aie::mul(x, y).to_vector<int32_t>());
        a3 = aie::add(a3, aie::mul(x, y).to_vector<int32_t>());
    }
    aie::store_v((int32_t *)c, a0);
    c[16] = a1[0] + a2[0] + a3[0];
}
"""

# the real inner loop shape of the fixed-point kernel:
# down-shift the int32 partial, multiply by the fixed-point scale, accumulate
I3 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<int32_t, 16> x = aie::load_v<16>((const int32_t *)at);
    aie::vector<int32_t, 16> y = aie::load_v<16>((const int32_t *)bt);
    aie::vector<int32_t, 16> acc = aie::broadcast<int32_t, 16>(0);
    for (int i = 0; i < N; i++) {
        aie::vector<int32_t, 16> p = aie::downshift(x, 4);
        acc = aie::add(acc, aie::mul(p, y).to_vector<int32_t>());
    }
    aie::store_v((int32_t *)c, acc);
}
"""

# int32 down-shift only
I4 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<int32_t, 16> x = aie::load_v<16>((const int32_t *)at);
    aie::vector<int32_t, 16> acc = aie::broadcast<int32_t, 16>(0);
    for (int i = 0; i < N; i++) {
        acc = aie::add(acc, aie::downshift(x, 4));
    }
    aie::store_v((int32_t *)c, acc);
}
"""

# ---------------------------------------------------------------- float path
# clone of the phase 61 k7: one float multiply-add chain with LMEM loads
F1 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<float, 16> acc = aie::broadcast<float, 16>(0.0f);
    for (int i = 0; i < N; i++) {
        aie::vector<float, 16> x = aie::load_v<16>((const float *)(at + i * 64));
        aie::vector<float, 16> y = aie::load_v<16>((const float *)(bt + i * 64));
        acc = aie::add(acc, aie::mul(x, y).to_vector<float>());
    }
    aie::store_v(c, acc);
}
"""

# clone of the phase 61 k8: four float multiply-add chains
F2 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<float, 16> a0 = aie::broadcast<float, 16>(0.0f);
    aie::vector<float, 16> a1 = aie::broadcast<float, 16>(0.0f);
    aie::vector<float, 16> a2 = aie::broadcast<float, 16>(0.0f);
    aie::vector<float, 16> a3 = aie::broadcast<float, 16>(0.0f);
    for (int i = 0; i < N; i++) {
        aie::vector<float, 16> x = aie::load_v<16>((const float *)(at + i * 64));
        aie::vector<float, 16> y = aie::load_v<16>((const float *)(bt + i * 64));
        a0 = aie::add(a0, aie::mul(x, y).to_vector<float>());
        a1 = aie::add(a1, aie::mul(x, y).to_vector<float>());
        a2 = aie::add(a2, aie::mul(x, y).to_vector<float>());
        a3 = aie::add(a3, aie::mul(x, y).to_vector<float>());
    }
    aie::store_v(c, a0);
    c[1] = a1[0] + a2[0] + a3[0];
}
"""

# clone of the phase 61 k9: int32 -> float conversion
F3 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<float, 16> acc = aie::broadcast<float, 16>(0.0f);
    for (int i = 0; i < N; i++) {
        aie::vector<int32_t, 16> v = aie::load_v<16>((const int32_t *)(at + i * 64));
        aie::vector<float, 16> f = aie::to_float(v);
        acc = aie::add(acc, f);
    }
    aie::store_v(c, acc);
}
"""

# four independent conversion chains (is the conversion latency or throughput bound?)
F4 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<float, 16> a0 = aie::broadcast<float, 16>(0.0f);
    aie::vector<float, 16> a1 = aie::broadcast<float, 16>(0.0f);
    aie::vector<float, 16> a2 = aie::broadcast<float, 16>(0.0f);
    aie::vector<float, 16> a3 = aie::broadcast<float, 16>(0.0f);
    for (int i = 0; i < N; i++) {
        aie::vector<int32_t, 16> v = aie::load_v<16>((const int32_t *)(at + i * 64));
        aie::vector<float, 16> f = aie::to_float(v);
        a0 = aie::add(a0, f);
        a1 = aie::add(a1, f);
        a2 = aie::add(a2, f);
        a3 = aie::add(a3, f);
    }
    aie::store_v(c, a0);
    c[1] = a1[0] + a2[0] + a3[0];
}
"""

# the phase 61 inner-loop float shape: build the scale product, then apply it
# (two dependent float ops, exactly what the Q8_0 kernel does per block)
F5 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    aie::vector<float, 16> acc = aie::broadcast<float, 16>(0.0f);
    aie::vector<float, 16> p = aie::load_v<16>((const float *)at);
    aie::vector<float, 16> w = aie::load_v<16>((const float *)bt);
    for (int i = 0; i < N; i++) {
        aie::vector<float, 16> sc = aie::mul(w, w).to_vector<float>();
        acc = aie::add(acc, aie::mul(p, sc).to_vector<float>());
    }
    aie::store_v(c, acc);
}
"""


def build(iters, src, wdir):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config

    a_bytes = 64 * 64
    b_bytes = 64 * 64
    c_bytes = 4
    kern = ExternalFunction(
        "kern",
        source_string=src,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_bytes,), np.dtype[np.float32]],
        ],
        include_dirs=[config.cxx_header_path()],
    )
    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_bytes,), np.dtype[np.float32]]

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(1):
            elem_a = of_a.acquire(1)
            elem_b = of_b.acquire(1)
            elem_c = of_c.acquire(1)
            kern(elem_a, elem_b, elem_c)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    fa = ObjectFifo(a_ty, name="a", depth=1)
    fb = ObjectFifo(b_ty, name="b", depth=1)
    fc = ObjectFifo(c_ty, name="c", depth=1)
    workers = [Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern], stack_size=4096)]

    A_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(c_bytes,), np.dtype[np.float32]]
    at = TensorAccessPattern(tensor_dims=(a_bytes,), offset=0, sizes=[a_bytes], strides=[1])
    bt = TensorAccessPattern(tensor_dims=(b_bytes,), offset=0, sizes=[b_bytes], strides=[1])
    ct = TensorAccessPattern(tensor_dims=(c_bytes,), offset=0, sizes=[c_bytes], strides=[1])

    def sequence(A, B, C, *ends):
        tg = iron.TaskGroup()
        fa.prod().fill(A, tap=at, group=tg)
        fb.prod().fill(B, tap=bt, group=tg)
        fc.cons().drain(C, tap=ct, group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, fa.prod(), fb.prod(), fc.cons()])
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


def run(wdir, iters, repeat=5):
    import pyxrt
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(os.path.join(wdir, "design.xclbin"))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    gid3, gid1 = kern.group_id(3), kern.group_id(1)
    a_bo = pyxrt.bo(dev, 64 * 64, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(dev, 64 * 64, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(dev, 4, pyxrt.bo.flags.host_only, gid3)
    ip = os.path.join(wdir, "design.insts.bin")
    inst_bo = pyxrt.bo(dev, os.path.getsize(ip), pyxrt.bo.flags.cacheable, gid1)
    with open(ip, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo.write(np.ones(64 * 64, np.int8).tobytes(), 0)
    b_bo.write(np.ones(64 * 64, np.int8).tobytes(), 0)
    c_bo.write(np.zeros(4, np.int8).tobytes(), 0)
    for b in (a_bo, b_bo, c_bo):
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    times = []
    for _ in range(repeat):
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
    return sorted(times)[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4096)
    ap.add_argument("--work-dir", default="/tmp/rate62")
    a = ap.parse_args()

    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    # name, source, number of independent chains, ops per chain per iteration
    specs = (
        ("i1_int32_mul_1", I1, 1, 2),
        ("i2_int32_mul_4", I2, 4, 2),
        ("i3_int_shift_mul_1", I3, 1, 3),
        ("i4_int_shift_1", I4, 1, 2),
        ("f1_float_mul_1", F1, 1, 2),
        ("f2_float_mul_4", F2, 4, 2),
        ("f3_i32_to_f32_1", F3, 1, 2),
        ("f4_i32_to_f32_4", F4, 4, 2),
        ("f5_float_2dep_1", F5, 1, 3),
    )
    for name, src, nch, nops in specs:
        wdir = os.path.join(a.work_dir, name)
        os.makedirs(wdir, exist_ok=True)
        xp = os.path.join(wdir, "design.xclbin")
        ip = os.path.join(wdir, "design.insts.bin")
        pp = os.path.join(wdir, "main.pdi")

        src_i = src % {"N": a.iters}

        @iron.jit
        def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
            return build(a.iters, src_i, wdir)

        t0 = time.time()
        design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        ct = time.time() - t0
        ms = run(wdir, a.iters)
        ns = (ms * 1e6) / (a.iters * nch)
        print("%-20s compile %5.1fs  run %8.3f ms  %8.2f ns per 16-lane chain-iter "
              "(%d ops/iter)" % (name, ct, ms, ns, nops))


if __name__ == "__main__":
    main()
