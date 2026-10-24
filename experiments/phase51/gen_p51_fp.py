#!/usr/bin/env python3
"""Phase 51 Candidate A: fixed-point integer accumulation kernel (int64)."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase50"))

import numpy as np

M_TILE, N_TILE, KC, NBLK = 32, 64, 256, 8
MA, NA = M_TILE // 4, N_TILE // 8  # 8, 8
KA, KB = KC // 8, KC // 8

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void q8_c4_mmul_v2_fp(
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
    const int32_t * as = (const int32_t *)(at + MT * KC);
    const int8_t * b_tiles = bt;
    const int32_t * bs = (const int32_t *)(bt + KC * NT);
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int bb = 0; bb < NBLK; bb++) {
        const int32_t * bsp = bs + bb * NA * 16;
        for (int i = 0; i < MA; i++) {
            const int32_t * asp = as + (bb * MA + i) * 32;
            const int8_t * ak = a_tiles + (i * KA + bb * 4) * 32;
            float * cb = c + i * NA * 32;
            for (int j = 0; j < NA; j++) {
                aie::vector<int32_t, 32> zero32 = aie::broadcast<int32_t, 32>(0);
                MMUL acc(zero32);
                const int8_t * bk = b_tiles + (j * KB + bb * 4) * 64;
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(ak + kk * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(bk + kk * 64);
                    acc.mac(av, bv);
                }
                aie::vector<int32_t, 32> dot = acc.to_vector<int32_t>();
                // 4 chunks of 8 lanes
                aie::vector<int64_t, 8> acc0 = aie::broadcast<int64_t, 8>(0);
                aie::vector<int64_t, 8> acc1 = aie::broadcast<int64_t, 8>(0);
                aie::vector<int64_t, 8> acc2 = aie::broadcast<int64_t, 8>(0);
                aie::vector<int64_t, 8> acc3 = aie::broadcast<int64_t, 8>(0);
                // load scales
                aie::vector<int32_t, 8> as0 = aie::load_v<8>(asp);       // a scale row0 broadcast
                aie::vector<int32_t, 8> as1 = aie::load_v<8>(asp + 8);
                aie::vector<int32_t, 8> as2 = aie::load_v<8>(asp + 16);
                aie::vector<int32_t, 8> as3 = aie::load_v<8>(asp + 24);
                aie::vector<int32_t, 8> bs0 = aie::load_v<8>(bsp + j * 16);
                aie::vector<int32_t, 8> bs1 = aie::load_v<8>(bsp + j * 16 + 8);
                // dot chunks
                aie::vector<int32_t, 8> d0 = dot.template extract<8>(0);
                aie::vector<int32_t, 8> d1 = dot.template extract<8>(1);
                aie::vector<int32_t, 8> d2 = dot.template extract<8>(2);
                aie::vector<int32_t, 8> d3 = dot.template extract<8>(3);
                // convert dot to int64
                aie::vector<int64_t, 8> d0l = aie::to_vector<int64_t>(d0);
                aie::vector<int64_t, 8> d1l = aie::to_vector<int64_t>(d1);
                aie::vector<int64_t, 8> d2l = aie::to_vector<int64_t>(d2);
                aie::vector<int64_t, 8> d3l = aie::to_vector<int64_t>(d3);
                // scale products (int32 x int32 -> int64), as int64 fixed Q32.32? We store scales as Q16.16;
                // product of two Q16.16 is Q32.32. Multiply dot (int32) by scale product (Q32.32) -> Q32.32.
                // To keep magnitude safe, instead treat scale product as Q16.16 (single combined scale).
                // Here we approximate: scale_combined = (as * bs) >> 16 (Q16.16). Compute on host? 
                // For kernel simplicity, compute product as int64 and shift.
                aie::vector<int64_t, 8> sp0 = aie::mul(as0, bs0);
                aie::vector<int64_t, 8> sp1 = aie::mul(as1, bs1);
                aie::vector<int64_t, 8> sp2 = aie::mul(as2, bs0);
                aie::vector<int64_t, 8> sp3 = aie::mul(as3, bs1);
                // shift right 16 to get Q16.16 combined scale
                // (aie may not have shift on int64 vec; use mul by 1/65536 via float? fallback: keep Q32.32)
                // Accumulate: acc += dot * sp >> 16
                acc0 = aie::add(acc0, aie::mul(d0l, sp0));  // Q? magnitude large
                acc1 = aie::add(acc1, aie::mul(d1l, sp1));
                acc2 = aie::add(acc2, aie::mul(d2l, sp2));
                acc3 = aie::add(acc3, aie::mul(d3l, sp3));
                float * q = cb + j * 32;
                // convert accumulated int64 -> float, divide by 2^32 (since sp is Q32.32 and dot int32)
                // Actually we accumulated dot(int32)*sp(Q32.32) = Q32.32 with value ~5e5*1e12=5e17.
                // Need to divide by 2^32 at end. to_float then /4.29e9.
                aie::vector<float,8> f0 = aie::to_float(acc0);
                aie::vector<float,8> f1 = aie::to_float(acc1);
                aie::vector<float,8> f2 = aie::to_float(acc2);
                aie::vector<float,8> f3 = aie::to_float(acc3);
                // divide by 2^32
                aie::vector<float,8> sc = aie::broadcast<float,8>(1.0f/4294967296.0f);
                f0 = aie::mul(f0, sc).to_vector<float>();
                f1 = aie::mul(f1, sc).to_vector<float>();
                f2 = aie::mul(f2, sc).to_vector<float>();
                f3 = aie::mul(f3, sc).to_vector<float>();
                aie::store_v(q, aie::add(aie::load_v<8>(q), f0));
                aie::store_v(q+8, aie::add(aie::load_v<8>(q+8), f1));
                aie::store_v(q+16, aie::add(aie::load_v<8>(q+16), f2));
                aie::store_v(q+24, aie::add(aie::load_v<8>(q+24), f3));
            }
        }
    }
}
"""

def build(M, K, N, n_cols, fifo_depth=2, b_fifo_depth=2, c_fifo_depth=1, stack_size=0x800):
    import aie.iron as iron
    from aie.iron import ObjectFifo, Runtime, Worker, ExternalFunction
    from aie.iron.controlflow import range_ as iron_range
    from aie.helpers.taplib import TensorAccessPattern
    from aie.utils import config

    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    n_kc = K // KC

    a_bytes = M_TILE * KC + NBLK * MA * 32 * 4  # int8 tiles + int32 scales (same size as float? 4 bytes)
    # scales are int32 now, 4 bytes each; same count NBLK*MA*32, so same bytes
    b_bytes = KC * N_TILE + NBLK * NA * 16 * 4
    c_elems = M_TILE * N_TILE

    kern = ExternalFunction("q8_c4_mmul_v2_fp", source_string=KERNEL_SRC,
        arg_types=[np.ndarray[(a_bytes,), np.dtype[np.int8]],
                   np.ndarray[(b_bytes,), np.dtype[np.int8]],
                   np.ndarray[(c_elems,), np.dtype[np.float32]]],
        include_dirs=[config.cxx_header_path()])

    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt_c):
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

    fifo_a, fifo_b, fifo_c = [], [], []
    workers = []
    for col in range(n_cols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=fifo_depth)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=b_fifo_depth if b_fifo_depth else fifo_depth)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=c_fifo_depth)
        fifo_a.append(fa); fifo_b.append(fb); fifo_c.append(fc)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern], stack_size=stack_size))

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_cols * n_nt_c * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * a_bytes,), offset=mt * n_kc * a_bytes, sizes=[n_kc * a_bytes], strides=[1]) for mt in range(n_mt)]
    b_taps = [[TensorAccessPattern(tensor_dims=(n_cols * n_nt_c * n_kc * b_bytes,), offset=(col * n_nt_c + nt) * n_kc * b_bytes, sizes=[n_kc * b_bytes], strides=[1]) for nt in range(n_nt_c)] for col in range(n_cols)]

    def _c_group(mt, col, nt):
        return (mt * n_cols * n_nt_c + col * n_nt_c + nt) * c_elems
    c_taps = [[[TensorAccessPattern(tensor_dims=(M * N,), offset=_c_group(mt, col, nt), sizes=[c_elems], strides=[1]) for nt in range(n_nt_c)] for mt in range(n_mt)] for col in range(n_cols)]

    def sequence(A, B, C, *fifos):
        a_p = fifos[0:n_cols]; b_p = fifos[n_cols:2*n_cols]; c_c = fifos[2*n_cols:3*n_cols]
        for mt in range(n_mt):
            for col in range(n_cols):
                for nt in range(n_nt_c):
                    tg = iron.TaskGroup() if False else None
                    import aie.iron as _iron
                    tg = _iron.TaskGroup()
                    a_p[col].fill(A, tap=a_taps[mt], group=tg)
                    b_p[col].fill(B, tap=b_taps[col][nt], group=tg)
                    c_c[col].drain(C, tap=c_taps[col][mt][nt], group=tg, wait=True)
                    tg.finish()

    prod_cons = []
    for col in range(n_cols):
        prod_cons.append(fifo_a[col].prod())
    for col in range(n_cols):
        prod_cons.append(fifo_b[col].prod())
    for col in range(n_cols):
        prod_cons.append(fifo_c[col].cons())

    rt = Runtime(sequence, [A_ty, B_ty, C_ty] + prod_cons)
    prog = iron.Program(iron.get_current_device(), rt, workers=workers)
    return prog.resolve_program()


def pack_A_fp(A, As, M, K):
    # A: int8 (M,K); As: float (M, K//32)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
    from gen_q8_c4_mmul_v2 import pack_A
    nb = KC // 32
    n_mt, n_kc = M // M_TILE, K // KC
    MA = M_TILE // 4
    a_chunk = M_TILE * KC
    Ap = pack_A(A, As, M, K, M_TILE, KC)  # int8 tiles + float scales (old layout)
    # Re-extract scales and convert to fixed int32
    new_a_chunk = a_chunk + NBLK * MA * 32 * 4
    out = np.zeros(n_mt * n_kc * new_a_chunk, np.int8)
    for mt in range(n_mt):
        for kc in range(n_kc):
            off = (mt * n_kc + kc) * (a_chunk + M_TILE * nb * 4)
            sc = Ap[off + a_chunk:off + a_chunk + M_TILE * nb * 4].view(np.float32).reshape(M_TILE, nb)
            # convert to fixed int32 broadcast layout (same as s12)
            bc = np.empty((NBLK, MA, 32), np.int32)
            for bb in range(NBLK):
                for i in range(MA):
                    for r in range(4):
                        val = sc[i * 4 + r, bb] * 65536.0
                        bc[bb, i, r * 8:(r + 1) * 8] = np.int32(np.round(val))
            o = (mt * n_kc + kc) * new_a_chunk
            out[o:o + a_chunk] = Ap[off:off + a_chunk]
            out[o + a_chunk:o + a_chunk + NBLK * MA * 32 * 4] = bc.reshape(-1).view(np.int8)
    return out


def pack_B_fp(B, Bs, K, N):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
    from gen_q8_c4_mmul_v2 import pack_B
    nb = KC // 32
    n_nt, n_kc = N // N_TILE, K // KC
    NA = N_TILE // 8
    b_chunk = KC * N_TILE
    Bp = pack_B(B, Bs, K, N, N_TILE, KC)
    new_b_chunk = b_chunk + NBLK * NA * 16 * 4
    out = np.zeros(n_nt * n_kc * new_b_chunk, np.int8)
    for nt in range(n_nt):
        for kc in range(n_kc):
            off = (nt * n_kc + kc) * (b_chunk + NBLK * N_TILE * 4)
            sc = Bp[off + b_chunk:off + b_chunk + NBLK * N_TILE * 4].view(np.float32).reshape(NBLK, N_TILE)
            bc = np.empty((NBLK, NA, 16), np.int32)
            for bb in range(NBLK):
                for j in range(NA):
                    col_scale = sc[bb, j * 8:j * 8 + 8]
                    fixed = np.round(col_scale * 65536.0).astype(np.int32)
                    bc[bb, j, 0:8] = fixed
                    bc[bb, j, 8:16] = fixed
            o = (nt * n_kc + kc) * new_b_chunk
            out[o:o + b_chunk] = Bp[off:off + b_chunk]
            out[o + b_chunk:o + b_chunk + NBLK * NA * 16 * 4] = bc.reshape(-1).view(np.int8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(args.M, args.K, args.N, args.cols,
                     fifo_depth=int(os.environ.get("P51_ADEPTH", "2")),
                     b_fifo_depth=int(os.environ.get("P51_BDEPTH", "2")),
                     c_fifo_depth=int(os.environ.get("P51_CDEPTH", "1")),
                     stack_size=int(os.environ.get("P51_STACK", "0x800"), 16))

    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"[gen_p51_fp] compiled in {time.time()-t0:.1f}s -> {compiled[0]}")
    print(f"xclbin size: {os.path.getsize(xp)}")

    # Quick correctness check
    from phase40.ref_q8_gemm import ref_q8_gemm  # may not exist; fallback
    print("COMPILE DONE")


if __name__ == "__main__":
    main()
