#!/usr/bin/env python3
"""Phase 50: build kernel variants from the audited phase40 generator.

Variants (each changes exactly ONE variable vs production c4_mmul_v2):
  dma_only : core acquires/releases A/B (identical DMA traffic) but never
             calls the compute kernel; C stays zero. Isolates DMA+setup time.
  trace    : production compute + AIE event trace (INSTR0/1, LOCK_STALL,
             STREAM_STALL) on column 0.

Usage (ironenv python):
  python gen_variant.py --variant dma_only --M 512 --K 1024 --N 3584 --work-dir /tmp/p50_dma
  python gen_variant.py --variant trace --M 512 --K 1024 --N 3584 --trace-size 4096 --work-dir /tmp/p50_trace
"""
import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))


def load_gen():
    spec = importlib.util.spec_from_file_location(
        "gen_q8_c4_mmul_v2",
        os.path.join(os.path.dirname(__file__), "..", "phase40", "gen_q8_c4_mmul_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Phase 50 candidate: C-resident loop interchange of the audited P32 kernel.
# Same tile-major layout, same MMUL<4,8,8> MACs, same per-32 scale semantics;
# only the loop order changes: bb is innermost and each (i,j) output block
# (4 rows x 8 cols = 32 floats) is accumulated in vector registers across the
# 8 bb blocks, touching C memory once per (i,j) per K-chunk instead of once
# per (bb,i,j) (8x fewer C read-modify-writes, register-resident hot loop).
# FP addition order differs from P32 across bb (non-associative), so outputs
# may differ in the last ulp; the correctness gate (max_err < 1.0, and PP
# token check) decides acceptance.
KERNEL_SRC_CREG = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void q8_c4_mmul_v2(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float * __restrict c)
{
    constexpr int MT = 32;
    constexpr int NT = 64;
    constexpr int KC = 256;
    constexpr int NBLK = KC / 32;
    constexpr int MA = MT / 4;
    constexpr int NA = NT / 8;
    constexpr int KA = KC / 8;
    constexpr int KB = KC / 8;
    const int8_t * a_tiles = at;
    const float * as = (const float *)(at + MT * KC);
    const int8_t * b_tiles = bt;
    const float * bs = (const float *)(bt + KC * NT);
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
#pragma clang loop unroll_count(1)
    for (int i = 0; i < MA; i++) {
#pragma clang loop unroll_count(1)
        for (int j = 0; j < NA; j++) {
            aie::vector<float, 8> r0 = aie::zeros<float, 8>();
            aie::vector<float, 8> r1 = aie::zeros<float, 8>();
            aie::vector<float, 8> r2 = aie::zeros<float, 8>();
            aie::vector<float, 8> r3 = aie::zeros<float, 8>();
#pragma clang loop unroll_count(1)
            for (int bb = 0; bb < NBLK; bb++) {
                aie::vector<int32_t, 32> zero = aie::broadcast<int32_t, 32>(0);
                MMUL acc(zero);
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(a_tiles + (i * KA + bb * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
                aie::vector<float, 32> p = aie::to_float(acc.to_vector<int32_t>());
                aie::vector<float, 8> bsv = aie::load_v<8>(bs + bb * NT + j * 8);
                aie::vector<float, 8> s0 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 0) * NBLK + bb]), bsv);
                aie::vector<float, 8> s1 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 1) * NBLK + bb]), bsv);
                aie::vector<float, 8> s2 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 2) * NBLK + bb]), bsv);
                aie::vector<float, 8> s3 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 3) * NBLK + bb]), bsv);
                aie::vector<float, 32> scale = aie::broadcast<float, 32>(0.0f);
                scale.insert<8>(0, s0);
                scale.insert<8>(1, s1);
                scale.insert<8>(2, s2);
                scale.insert<8>(3, s3);
                p = aie::mul(p, scale);
                r0 = aie::add(r0, p.template extract<8>(0));
                r1 = aie::add(r1, p.template extract<8>(1));
                r2 = aie::add(r2, p.template extract<8>(2));
                r3 = aie::add(r3, p.template extract<8>(3));
            }
            float * c0 = c + (i * 4 + 0) * NT + j * 8;
            float * c1 = c + (i * 4 + 1) * NT + j * 8;
            float * c2 = c + (i * 4 + 2) * NT + j * 8;
            float * c3 = c + (i * 4 + 3) * NT + j * 8;
            aie::store_v(c0, aie::add(aie::load_v<8>(c0), r0));
            aie::store_v(c1, aie::add(aie::load_v<8>(c1), r1));
            aie::store_v(c2, aie::add(aie::load_v<8>(c2), r2));
            aie::store_v(c3, aie::add(aie::load_v<8>(c3), r3));
        }
    }
}
"""


# Phase 50 candidate s1: per-row 8-float ops, no vector<float,32>, no insert-based
# scale vector. a-scales hoisted per (i,bb), one to_float(32) per block (the API
# needs >=16-wide input; result rows extracted as 8-wide aliases).
KERNEL_SRC_S1 = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

// s9: exact int32 -> fp32 via the +BIAS bit trick with a store/load reinterpret.
// |x| < 2^24 is guaranteed (<= 1024 int8 products). BIAS = 0x4B400000 keeps
// (v + BIAS) in [2^23, 2^24) where the fp32 spacing is 1.0, so
// float_bits(v + BIAS) - 12582912.0f == (float)v exactly for both halves:
// hi = x >> 16 (|hi| <= 256) and lo = x - hi*65536 (0..65535).
// Constants are created once and passed in, so they stay in registers.
__aie_inline aie::vector<float, 16> p50_conv16(
    aie::vector<int32_t, 16> x,
    aie::vector<int32_t, 16> bi,
    aie::vector<float, 16> bf,
    int32_t * tmp)
{
    aie::vector<int32_t, 16> hi = aie::downshift(x, 16);
    aie::vector<int32_t, 16> hs = aie::upshift(hi, 16);
    aie::vector<int32_t, 16> lo = aie::sub(x, hs);
    aie::store_v(tmp, aie::add(hi, bi));
    aie::store_v(tmp + 16, aie::add(lo, bi));
    aie::vector<float, 16> fhi = aie::load_v<16>((const float *)tmp);
    aie::vector<float, 16> flo = aie::load_v<16>((const float *)(tmp + 16));
    fhi = aie::sub(fhi, bf);
    flo = aie::sub(flo, bf);
    return aie::add(aie::mul(fhi, aie::broadcast<float, 16>(65536.0f)).to_vector<float>(), flo);
}

extern "C" void q8_c4_mmul_v2(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float * __restrict c)
{
    constexpr int MT = 32;
    constexpr int NT = 64;
    constexpr int KC = 256;
    constexpr int NBLK = KC / 32;
    constexpr int MA = MT / 4;
    constexpr int NA = NT / 8;
    constexpr int KA = KC / 8;
    constexpr int KB = KC / 8;
    const int8_t * a_tiles = at;
    const float * as = (const float *)(at + MT * KC);
    const int8_t * b_tiles = bt;
    const float * bs = (const float *)(bt + KC * NT);
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    const aie::vector<int32_t, 16> bi = aie::broadcast<int32_t, 16>(12582912);
    const aie::vector<float, 16> bf = aie::broadcast<float, 16>(12582912.0f);
    const aie::vector<float, 16> k64 = aie::broadcast<float, 16>(65536.0f);
    alignas(64) int32_t tmpbuf[32];
    int32_t * __restrict tmp = tmpbuf;
#pragma clang loop unroll_count(1)
    for (int bb = 0; bb < NBLK; bb++) {
        const float * asp = as + bb;
        const float * bsp = bs + bb * NT;
#pragma clang loop unroll_count(1)
        for (int i = 0; i < MA; i++) {
            float a0 = asp[(i * 4 + 0) * NBLK];
            float a1 = asp[(i * 4 + 1) * NBLK];
            float a2 = asp[(i * 4 + 2) * NBLK];
            float a3 = asp[(i * 4 + 3) * NBLK];
            const int8_t * ak = a_tiles + (i * KA + bb * 4) * 32;
            float * c0 = c + (i * 4 + 0) * NT;
            float * c1 = c + (i * 4 + 1) * NT;
            float * c2 = c + (i * 4 + 2) * NT;
            float * c3 = c + (i * 4 + 3) * NT;
#pragma clang loop unroll_count(1)
            for (int j = 0; j < NA; j++) {
                aie::vector<int32_t, 32> zero = aie::broadcast<int32_t, 32>(0);
                MMUL acc(zero);
                const int8_t * bk = b_tiles + (j * KB + bb * 4) * 64;
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(ak + kk * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bk + kk * 64);
                    acc.mac(av, bv);
                }
                aie::vector<int32_t, 32> vi = acc.to_vector<int32_t>();
                aie::vector<float, 8> bsv = aie::load_v<8>(bsp + j * 8);
                aie::vector<float, 16> f01 = p50_conv16(vi.template extract<16>(0), bi, bf, tmp);
                aie::vector<float, 16> f23 = p50_conv16(vi.template extract<16>(1), bi, bf, tmp);
                float * q0 = c0 + j * 8;
                float * q1 = c1 + j * 8;
                float * q2 = c2 + j * 8;
                float * q3 = c3 + j * 8;
                aie::store_v(q0, aie::add(aie::load_v<8>(q0),
                    aie::mul(f01.template extract<8>(0),
                        aie::mul(aie::broadcast<float, 8>(a0), bsv).to_vector<float>()).to_vector<float>()));
                aie::store_v(q1, aie::add(aie::load_v<8>(q1),
                    aie::mul(f01.template extract<8>(1),
                        aie::mul(aie::broadcast<float, 8>(a1), bsv).to_vector<float>()).to_vector<float>()));
                aie::store_v(q2, aie::add(aie::load_v<8>(q2),
                    aie::mul(f23.template extract<8>(0),
                        aie::mul(aie::broadcast<float, 8>(a2), bsv).to_vector<float>()).to_vector<float>()));
                aie::store_v(q3, aie::add(aie::load_v<8>(q3),
                    aie::mul(f23.template extract<8>(1),
                        aie::mul(aie::broadcast<float, 8>(a3), bsv).to_vector<float>()).to_vector<float>()));
            }
        }
    }
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["dma_only", "trace", "mmul_only", "scale_only", "orig", "creg", "s1", "d0", "d1", "d2", "d3", "accfloat"], required=True)
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--trace-size", type=int, default=8192)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    gen = load_gen()
    src = open(gen.__file__).read()
    if args.variant == "dma_only":
        needle = "                kern(elem_a, elem_b, elem_c)"
        assert src.count(needle) == 1, f"needle count={src.count(needle)}"
        src = src.replace(needle, "                pass  # PHASE50 dma_only: skip compute, keep DMA traffic identical")
        print("[gen_variant] dma_only: compute call removed, DMA acquire/release unchanged")
    elif args.variant in ("mmul_only", "scale_only"):
        pass  # patched on modu["KERNEL_SRC_P32"] after exec (keeps TRACE source intact)
    elif args.variant == "creg":
        pass  # full replacement source below
    elif args.variant == "s1":
        pass  # full replacement source below
    elif args.variant == "orig":
        pass  # kernel_variant="original" selected at build call below
    modu = {}
    exec(compile(src, gen.__file__, "exec"), modu)
    if args.variant == "creg":
        modu["KERNEL_SRC_P32"] = KERNEL_SRC_CREG
        print("[gen_variant] creg: C-resident loop-interchanged kernel installed")
    if args.variant.startswith("d"):
        modu["KERNEL_SRC_P32"] = KERNEL_SRC_S1.replace("STAGE_SEL", args.variant[1:])
        print(f"[gen_variant] diagnostic stage {args.variant[1:]} installed")
    if args.variant == "s1":
        modu["KERNEL_SRC_P32"] = KERNEL_SRC_S1
        print("[gen_variant] s1: restructured kernel installed")
    if args.variant == "accfloat":
        p32 = modu["KERNEL_SRC_P32"]
        old = "aie::vector<float, 32> p = aie::to_float(acc.to_vector<int32_t>());"
        assert p32.count(old) == 1
        p32 = p32.replace(old, "aie::vector<float, 32> p = aie::to_float(acc.to_accum());")
        modu["KERNEL_SRC_P32"] = p32
        print("[gen_variant] accfloat: to_float direct on MMUL accumulator")
    if args.variant in ("mmul_only", "scale_only"):
        p32 = modu["KERNEL_SRC_P32"]
        if args.variant == "mmul_only":
            # keep MAC loop live via int32 store sink; drop float scale+accumulate
            old_p = "aie::vector<float, 32> p = aie::to_float(acc.to_vector<int32_t>());"
            assert p32.count(old_p) == 1
            p32 = p32.replace(old_p, "alignas(32) int32_t sink[32];\n"
                                     "                aie::store_v(sink, acc.to_vector<int32_t>());")
            old_store = """                for (int rr = 0; rr < 4; rr++) {
                    float * cp = c + (i * 4 + rr) * NT + j * 8;
                    aie::store_v(cp, aie::add(aie::load_v<8>(cp), p.template extract<8>(rr)));
                }"""
            assert p32.count(old_store) == 1
            p32 = p32.replace(old_store, "                (void)sink[0];  // PHASE50 mmul_only: no C update")
            for li in ["                aie::vector<float, 8> bsv = aie::load_v<8>(bs + bb * NT + j * 8);\n",
                       "                aie::vector<float, 8> s0 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 0) * NBLK + bb]), bsv);\n",
                       "                aie::vector<float, 8> s1 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 1) * NBLK + bb]), bsv);\n",
                       "                aie::vector<float, 8> s2 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 2) * NBLK + bb]), bsv);\n",
                       "                aie::vector<float, 8> s3 = aie::mul(aie::broadcast<float, 8>(as[(i * 4 + 3) * NBLK + bb]), bsv);\n",
                       "                aie::vector<float, 32> scale = aie::broadcast<float, 32>(0.0f);\n",
                       "                scale.insert<8>(0, s0);\n",
                       "                scale.insert<8>(1, s1);\n",
                       "                scale.insert<8>(2, s2);\n",
                       "                scale.insert<8>(3, s3);\n",
                       "                p = aie::mul(p, scale);\n"]:
                assert p32.count(li) == 1, f"missing: {li[:70]}"
                p32 = p32.replace(li, "")
            print("[gen_variant] mmul_only: MAC loop kept, C update + scale lines removed")
        else:
            old_mac = """                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(a_tiles + (i * KA + bb * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }"""
            assert p32.count(old_mac) == 1
            p32 = p32.replace(old_mac, "                (void)a_tiles; (void)b_tiles;  // PHASE50 scale_only: skip MACs, acc stays zero")
            print("[gen_variant] scale_only: MAC loop removed, float path kept")
        modu["KERNEL_SRC_P32"] = p32
    build = modu["build"]

    import aie.iron as iron
    from aie.iron import CompileTime, In, Out
    from aie.iron import ExternalFunction as _EF

    extra_flags = [f for f in os.environ.get("P50_KFLAGS", "").split(",") if f]

    class _EF2(_EF):
        def __init__(self, name, **kw):
            kw.setdefault("compile_flags", [])
            kw["compile_flags"] = list(kw["compile_flags"]) + extra_flags
            super().__init__(name, **kw)

    modu["ExternalFunction"] = _EF2
    if extra_flags:
        print(f"[gen_variant] kernel compile flags: {extra_flags}")

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    ts = args.trace_size if args.variant == "trace" else 0
    kv = "original" if args.variant == "orig" else "p32"

    # NB: the iron JIT cache key does not cover source_string kernel bodies,
    # so every variant needs a unique nonce to force a real recompile.
    nonce = int(os.environ.get("P50_NONCE", "0"))

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.cols, trace_size=ts, kernel_variant=kv,
                     stack_size=int(os.environ.get("P50_STACK", "0x1000"), 16),
                     fifo_depth=int(os.environ.get("P50_ADEPTH", "2")),
                     b_fifo_depth=int(os.environ.get("P50_BDEPTH", "2")),
                     c_fifo_depth=int(os.environ.get("P50_CDEPTH", "1")))

    t0 = time.time()
    compiled = design.specialize(dummy=nonce).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"[gen_variant] compiled in {time.time()-t0:.1f}s -> {compiled[0]}")


if __name__ == "__main__":
    main()

