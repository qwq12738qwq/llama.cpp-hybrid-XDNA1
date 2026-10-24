#!/usr/bin/env python3
"""Phase 61: what does the AIE2 int8 matrix unit actually deliver?

Three kernels, same data volume, to separate the three possible limits:

  k1  one serial vmac chain          -> the vmac LATENCY
  k2  four independent vmac chains    -> the vmac THROUGHPUT
  k3  a vmac chain with an LMEM load  -> the LOAD + vmac rate

The reference is the arithmetic peak: one 8x8x8 int8 vmac is 512 MACs, and
the phase 60 report puts the AIE2.0 int8 peak at 256 MAC/cycle/core, i.e.
512 GOP/s per core, so a vmac should retire every 2 cycles.

usage: mmulrate.py [--iters 4096] [--m 512 --k 1024 --n 3584]
"""
import argparse
import os
import time

import numpy as np

K1 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    aie::vector<int8_t, 64> a = aie::load_v<64>(at);
    aie::vector<int8_t, 64> b = aie::load_v<64>(bt);
    aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
    MMUL acc(zero);
    for (int i = 0; i < N; i++) {
        acc.mac(a, b);
    }
    alignas(64) int32_t tmp[64];
    aie::store_v(tmp, acc.to_vector<int32_t>());
    c[0] = (float) tmp[0];
}
"""

K2 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    aie::vector<int8_t, 64> a = aie::load_v<64>(at);
    aie::vector<int8_t, 64> b = aie::load_v<64>(bt);
    aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
    MMUL acc0(zero);
    MMUL acc1(zero);
    MMUL acc2(zero);
    MMUL acc3(zero);
    for (int i = 0; i < N; i++) {
        acc0.mac(a, b);
        acc1.mac(a, b);
        acc2.mac(a, b);
        acc3.mac(a, b);
    }
    alignas(64) int32_t tmp[64];
    aie::store_v(tmp, acc0.to_vector<int32_t>());
    c[0] = (float) tmp[0] + (float) acc1.to_vector<int32_t>()[0];
}
"""

K3 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
    MMUL acc(zero);
    for (int i = 0; i < N; i++) {
        aie::vector<int8_t, 64> a = aie::load_v<64>(at + (i & 63) * 64);
        aie::vector<int8_t, 64> b = aie::load_v<64>(bt + (i & 63) * 64);
        acc.mac(a, b);
    }
    alignas(64) int32_t tmp[64];
    aie::store_v(tmp, acc.to_vector<int32_t>());
    c[0] = (float) tmp[0];
}
"""


K4 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;
    aie::vector<int8_t, 64> a = aie::load_v<64>(at);
    aie::vector<int8_t, 64> b = aie::load_v<64>(bt);
    aie::vector<int32_t, 64> zero = aie::broadcast<int32_t, 64>(0);
    MMUL acc0(zero); MMUL acc1(zero); MMUL acc2(zero); MMUL acc3(zero);
    MMUL acc4(zero); MMUL acc5(zero); MMUL acc6(zero); MMUL acc7(zero);
    for (int i = 0; i < N; i++) {
        acc0.mac(a, b); acc1.mac(a, b); acc2.mac(a, b); acc3.mac(a, b);
        acc4.mac(a, b); acc5.mac(a, b); acc6.mac(a, b); acc7.mac(a, b);
    }
    alignas(64) int32_t tmp[64];
    aie::store_v(tmp, acc0.to_vector<int32_t>());
    c[0] = (float) tmp[0];
}
"""

# 4x8 unit: 32 int32 accumulator = two 512-bit accumulator registers, so
# several independent chains fit without spilling.
K5 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    aie::vector<int8_t, 32> a = aie::load_v<32>(at);
    aie::vector<int8_t, 64> b = aie::load_v<64>(bt);
    aie::vector<int32_t, 32> zero = aie::broadcast<int32_t, 32>(0);
    MMUL acc0(zero);
    MMUL acc1(zero);
    for (int i = 0; i < N; i++) {
        acc0.mac(a, b);
        acc1.mac(a, b);
    }
    alignas(32) int32_t tmp[32];
    aie::store_v(tmp, acc0.to_vector<int32_t>());
    c[0] = (float) tmp[0];
}
"""

K6 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void kern(const int8_t * __restrict at,
                     const int8_t * __restrict bt,
                     float        * __restrict c)
{
    constexpr int N = %(N)d;
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    aie::vector<int8_t, 32> a = aie::load_v<32>(at);
    aie::vector<int8_t, 64> b = aie::load_v<64>(bt);
    aie::vector<int32_t, 32> zero = aie::broadcast<int32_t, 32>(0);
    MMUL acc0(zero); MMUL acc1(zero); MMUL acc2(zero); MMUL acc3(zero);
    for (int i = 0; i < N; i++) {
        acc0.mac(a, b); acc1.mac(a, b); acc2.mac(a, b); acc3.mac(a, b);
    }
    alignas(32) int32_t tmp[32];
    aie::store_v(tmp, acc0.to_vector<int32_t>());
    c[0] = (float) tmp[0];
}
"""


# The scale application: a float multiply and an add on a 16-lane vector.
# The compiler turns aie::mul on float vectors into vmac.f, i.e. the FLOAT
# MATRIX unit, which is the same hardware as the int8 vmac.
K7 = r"""
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

K8 = r"""
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

# the int32 -> float conversion alone
K9 = r"""
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
    ap.add_argument("--work-dir", default="/tmp/mmulrate")
    a = ap.parse_args()

    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    results = {}
    specs = (("k1_serial_8x8", K1, 1, 8), ("k2_4chain_8x8", K2, 4, 8),
             ("k4_8chain_8x8", K4, 8, 8), ("k5_2chain_4x8", K5, 2, 4),
             ("k6_4chain_4x8", K6, 4, 4), ("k3_loads", K3, 1, 8),
             ("k7_float_1chain", K7, 1, 0), ("k8_float_4chain", K8, 4, 0),
             ("k9_i32_to_f32", K9, 1, 0))
    for name, src, nch, unit in specs:
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
        if unit == 0:
            # float / conversion kernels: report ns per 16-lane operation
            print("%-16s compile %4.1fs  run %8.3f ms  %8.2f ns per 16-lane op"
                  % (name, ct, ms, (ms * 1e6) / (a.iters * nch)))
            results[name] = (ms, 0)
            continue
        macs = a.iters * nch * (512 if unit == 8 else 256)
        results[name] = (ms, macs)
        print("%-16s compile %4.1fs  run %8.3f ms  %10.1f GOP/s  %8.2f ns/vmac"
              % (name, ct, ms, macs / (ms / 1000.0) / 1e9,
                 (ms * 1e6) / (a.iters * nch)))

    print()
    print("reference: AIE2 int8 peak (phase 60) = 512 GOP/s/core = 2048 for 4 cores")
    for k, (ms, macs) in results.items():
        print("  %-16s %6.2f GOP/s/core  %6.2f%% of the 4-core peak" %
              (k, macs / (ms / 1000.0) / 1e9 / 4.0,
               100.0 * (macs / (ms / 1000.0) / 1e9) / 2048.0))


if __name__ == "__main__":
    main()
