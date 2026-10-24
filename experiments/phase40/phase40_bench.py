#!/usr/bin/env python3
"""Phase 40: Q8_0 C4 Device-Side Scale Experiment.

Compares Q8SCALEV-C1, Q8MMUL-C1, Q8MMUL-C4 kernels with a CPU reference.
Fixes the C4 packing issue from Phase 39.

Usage:
  python phase40_bench.py --M 512 --K 1024 --N 3584 --iters 3
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "/workspace/llama.cpp/experiments/phase20g")
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase39")

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Runtime, Worker, ExternalFunction
from aie.iron.controlflow import range_ as iron_range
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config

# Import reference from Phase 40B
sys.path.insert(0, "/workspace/llama.cpp/experiments/phase40")
from ref_q8_gemm import q8_0_gemm_ref as ref_q8_0_gemm

# Import pack functions from phase39
from bench_q8_c4 import pack_A_vec, pack_B_vec, run_kernel_wa, pack_A_mmul, pack_B_mmul

M_TILE = 64
N_TILE = 64
KC = 256
NBLK = KC // 32
KA = KC // 8
KB = KC // 8

KERNEL_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void q8_mmul_scale(
    const int8_t * __restrict at,
    const int8_t * __restrict bt,
    float        * __restrict c)
{
    constexpr int KC    = 256;
    constexpr int NBLK  = KC / 32;
    constexpr int KA    = KC / 8;
    constexpr int KB    = KC / 8;
    const int8_t * a_tiles = at;
    const float  * as = (const float *)(at + 64 * KC);
    const int8_t * b_tiles = bt;
    const float  * bs = (const float *)(bt + KC * 64);

    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, accauto>;
    for (int bb = 0; bb < NBLK; bb++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 8; j++) {
                MMUL acc;
                for (int kk = 0; kk < 4; kk++) {
                    aie::vector<int8_t, 32> av = aie::load_v<32>(a_tiles + (i * KA + bb * 4 + kk) * 32);
                    aie::vector<int8_t, 64> bv = aie::load_v<64>(b_tiles + (j * KB + bb * 4 + kk) * 64);
                    acc.mac(av, bv);
                }
                alignas(32) int32_t tmp[32];
                aie::store_v(tmp, acc.to_vector<int32_t>());
                aie::vector<float, 8> bsv = aie::load_v<8>(bs + bb * 64 + j * 8);
                for (int rr = 0; rr < 4; rr++) {
                    int row = i * 4 + rr;
                    float asi = as[row * NBLK + bb];
                    aie::vector<float, 8> pvf = aie::to_float(aie::load_v<8>(tmp + rr * 8));
                    aie::vector<float, 8> f = aie::mul(asi, aie::mul(bsv, pvf).to_vector<float>()).to_vector<float>();
                    float * cp = c + row * 64 + j * 8;
                    aie::store_v(cp, aie::add(aie::load_v<8>(cp), f));
                }
            }
        }
    }
}
"""


def gen_data(M, K, N, seed=42, uniform=False):
    rng = np.random.default_rng(seed)
    A = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    nb = K // 32
    if uniform:
        As = np.ones((M, nb), np.float32)
        Bs = np.ones((nb, N), np.float32)
    else:
        As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
        Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)
    return A, As, B, Bs


def pack_A_mmul(A, As, M, K, m_tile, k_chunk):
    n_mt, n_kc = M // m_tile, K // k_chunk
    nb = k_chunk // 32
    ai = np.empty((n_mt, n_kc, m_tile // 4, nb, 4, 32), np.int8)
    asc = np.empty((n_mt, n_kc, m_tile, nb), np.float32)
    for mt in range(n_mt):
        for kc in range(n_kc):
            for bb in range(nb):
                for im in range(m_tile // 4):
                    for kk in range(4):
                        r0 = mt * m_tile + im * 4
                        c0 = kc * k_chunk + bb * 32 + kk * 8
                        ai[mt, kc, im, bb, kk] = A[r0:r0 + 4, c0:c0 + 8].reshape(-1)
            asc[mt, kc] = As[mt * m_tile:(mt + 1) * m_tile,
                             kc * nb:(kc + 1) * nb]
    int8 = ai.reshape(n_mt, n_kc, -1)
    sc = asc.reshape(n_mt, n_kc, -1).view(np.int8)
    out = np.concatenate([int8, sc], axis=2)
    return out.reshape(-1)


def pack_B_mmul(B, Bs, K, N, n_tile, k_chunk):
    n_nt, n_kc = N // n_tile, K // k_chunk
    nb = k_chunk // 32
    bi = np.empty((n_nt, n_kc, n_tile // 8, nb, 4, 64), np.int8)
    bsc = np.empty((n_nt, n_kc, nb, n_tile), np.float32)
    for nt in range(n_nt):
        for kc in range(n_kc):
            for bb in range(nb):
                for kk in range(4):
                    for jn in range(n_tile // 8):
                        r0 = kc * k_chunk + bb * 32 + kk * 8
                        c0 = nt * n_tile + jn * 8
                        bi[nt, kc, jn, bb, kk] = B[r0:r0 + 8, c0:c0 + 8].reshape(-1)
                bsc[nt, kc, bb] = Bs[kc * nb + bb,
                                     nt * n_tile:(nt + 1) * n_tile]
    int8 = bi.reshape(n_nt, n_kc, -1)
    sc = bsc.reshape(n_nt, n_kc, -1).view(np.int8)
    return np.concatenate([int8, sc], axis=2).reshape(-1)


def pack_B_mmul_c4(B, Bs, K, N, n_tile, k_chunk, n_cols):
    """Pack B for C4: reorder tiles by (col, nt) for multi-worker access."""
    n_nt, n_kc = N // n_tile, K // k_chunk
    nb = k_chunk // 32
    n_nt_c = n_nt // n_cols

    full_B = pack_B_mmul(B, Bs, K, N, n_tile, k_chunk)

    b_bytes = k_chunk * n_tile + nb * n_tile * 4
    full_reshaped = full_B.reshape(n_nt, n_kc, b_bytes)

    reordered = np.empty((n_cols, n_nt_c, n_kc, b_bytes), dtype=np.int8)
    for col in range(n_cols):
        for nt_c in range(n_nt_c):
            src_nt = col * n_nt_c + nt_c
            reordered[col, nt_c] = full_reshaped[src_nt]

    return reordered.reshape(-1)


def make_c1_design(M, K, N):
    n_mt, n_nt, n_kc = M // M_TILE, N // N_TILE, K // KC
    a_bytes = M_TILE * KC + M_TILE * NBLK * 4
    b_bytes = KC * N_TILE + NBLK * N_TILE * 4
    c_elems = M_TILE * N_TILE

    kern = ExternalFunction(
        "q8_mmul_scale",
        source_string=KERNEL_SRC,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.float32]],
        ],
        include_dirs=[config.cxx_header_path()],
    )
    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]
    fifo_a = ObjectFifo(a_ty, name="a", depth=1)
    fifo_b = ObjectFifo(b_ty, name="b", depth=1)
    fifo_c = ObjectFifo(c_ty, name="c", depth=1)

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0.0
            for _ch in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(core_fn, [fifo_a.cons(), fifo_b.cons(), fifo_c.prod(), kern], stack_size=0x1000)
    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_nt * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]
    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * a_bytes,),
                                  offset=mt * n_kc * a_bytes, sizes=[n_kc * a_bytes], strides=[1])
              for mt in range(n_mt)]
    b_taps = [TensorAccessPattern(tensor_dims=(n_nt * n_kc * b_bytes,),
                                  offset=nt * n_kc * b_bytes, sizes=[n_kc * b_bytes], strides=[1])
              for nt in range(n_nt)]
    c_taps = [[TensorAccessPattern(tensor_dims=(M, N),
                                   offset=(mt * M_TILE) * N + nt * N_TILE,
                                   sizes=[M_TILE, N_TILE], strides=[N, 1])
               for nt in range(n_nt)] for mt in range(n_mt)]
    def sequence(A, B, C, a_p, b_p, c_c):
        for mt in range(n_mt):
            for nt in range(n_nt):
                tg = iron.TaskGroup()
                a_p.fill(A, tap=a_taps[mt], group=tg)
                b_p.fill(B, tap=b_taps[nt], group=tg)
                c_c.drain(C, tap=c_taps[mt][nt], group=tg, wait=True)
                tg.finish()
    rt = Runtime(sequence, [A_ty, B_ty, C_ty, fifo_a.prod(), fifo_b.prod(), fifo_c.cons()])
    return iron.Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def make_c4_design(M, K, N, n_cols):
    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    n_kc = K // KC
    a_bytes = M_TILE * KC + M_TILE * NBLK * 4
    b_bytes = KC * N_TILE + NBLK * N_TILE * 4
    c_elems = M_TILE * N_TILE

    kern = ExternalFunction(
        "q8_mmul_scale",
        source_string=KERNEL_SRC,
        arg_types=[
            np.ndarray[(a_bytes,), np.dtype[np.int8]],
            np.ndarray[(b_bytes,), np.dtype[np.int8]],
            np.ndarray[(c_elems,), np.dtype[np.float32]],
        ],
        include_dirs=[config.cxx_header_path()],
    )
    a_ty = np.ndarray[(a_bytes,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_bytes,), np.dtype[np.int8]]
    c_ty = np.ndarray[(c_elems,), np.dtype[np.float32]]
    fifo_a, fifo_b, fifo_c = [], [], []
    workers = []

    def core_fn(of_a, of_b, of_c, kern):
        for _ in iron_range(n_mt * n_nt_c):
            elem_c = of_c.acquire(1)
            for x in iron_range(c_elems):
                elem_c[x] = 0.0
            for _ch in iron_range(n_kc):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kern(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    for col in range(n_cols):
        fa = ObjectFifo(a_ty, name=f"a{col}", depth=1)
        fb = ObjectFifo(b_ty, name=f"b{col}", depth=1)
        fc = ObjectFifo(c_ty, name=f"c{col}", depth=1)
        fifo_a.append(fa)
        fifo_b.append(fb)
        fifo_c.append(fc)
        workers.append(Worker(core_fn, [fa.cons(), fb.cons(), fc.prod(), kern], stack_size=0x1000))

    A_ty = np.ndarray[(n_mt * n_kc * a_bytes,), np.dtype[np.int8]]
    B_ty = np.ndarray[(n_cols * n_nt_c * n_kc * b_bytes,), np.dtype[np.int8]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]
    a_taps = [TensorAccessPattern(tensor_dims=(n_mt * n_kc * a_bytes,),
                                  offset=mt * n_kc * a_bytes, sizes=[n_kc * a_bytes], strides=[1])
              for mt in range(n_mt)]
    b_taps = [[TensorAccessPattern(tensor_dims=(n_cols * n_nt_c * n_kc * b_bytes,),
                                   offset=(col * n_nt_c + nt) * n_kc * b_bytes,
                                   sizes=[n_kc * b_bytes], strides=[1])
               for nt in range(n_nt_c)] for col in range(n_cols)]
    c_taps = [[[TensorAccessPattern(tensor_dims=(M, N),
                                     offset=mt * M_TILE * N + (col * n_nt_c + nt) * N_TILE,
                                     sizes=[M_TILE, N_TILE], strides=[N, 1])
                 for nt in range(n_nt_c)] for mt in range(n_mt)] for col in range(n_cols)]
    def sequence(A, B, C, *fifos):
        a_p = fifos[0:n_cols]
        b_p = fifos[n_cols:2 * n_cols]
        c_c = fifos[2 * n_cols:3 * n_cols]
        for mt in range(n_mt):
            for col in range(n_cols):
                for nt in range(n_nt_c):
                    tg = iron.TaskGroup()
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
    return iron.Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def run_kernel_generic(xclbin_path, A_buf, B_buf, M, K, N):
    import pyxrt
    xclbin = pyxrt.xclbin(xclbin_path)
    uuid = xclbin.get_uuid()
    device = pyxrt.device(0)
    device.register_xclbin(xclbin)
    hwctx = pyxrt.hw_context(device, uuid)
    kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
    gid1 = kernel.group_id(1)
    gid3 = kernel.group_id(3)
    inst_bytes = os.path.getsize(xclbin_path.replace(".xclbin", ".insts.bin"))
    with open(xclbin_path.replace(".xclbin", ".insts.bin"), "rb") as f:
        insts = np.frombuffer(f.read(), dtype=np.uint32)
    inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
    inst_map = np.frombuffer(inst_bo.map(), dtype=np.uint32)
    inst_map[:] = insts
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo = pyxrt.bo(device, A_buf.nbytes, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(device, B_buf.nbytes, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(device, M * N * 4, pyxrt.bo.flags.host_only, gid3)
    c_map = np.frombuffer(c_bo.map(), dtype=np.float32)
    c_map[:] = 0
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo.write(np.ascontiguousarray(A_buf).tobytes(), 0)
    b_bo.write(np.ascontiguousarray(B_buf).tobytes(), 0)
    a_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    b_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run = pyxrt.run(kernel)
    run.set_arg(0, np.uint64(3))
    run.set_arg(1, inst_bo)
    run.set_arg(2, np.uint32(inst_bytes))
    run.set_arg(3, a_bo)
    run.set_arg(4, b_bo)
    run.set_arg(5, c_bo)
    t0 = time.perf_counter()
    run.start()
    run.wait()
    t1 = time.perf_counter()
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    c_raw = np.frombuffer(c_bo.map(), dtype=np.float32)
    C = c_raw.reshape(M, N).copy()
    return C, (t1 - t0) * 1000.0


def run_correctness_tests(M, K, N, iters=3):
    print(f"\n{'='*70}")
    print(f"[phase40] Correctness tests M={M} K={K} N={N}")
    print(f"{'='*70}")

    nb = K // 32
    if N % N_TILE != 0 or M % M_TILE != 0 or K % KC != 0:
        print(f"  SKIPPED: tile alignment required")
        return

    cache_dir = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

    for test_name, uniform, scale_desc in [
        ("uniform", True, "all scales = 1"),
        ("two-scale", False, "first half = 1, second half = 2"),
        ("non-uniform", False, "[1, 2, 4, 8, 16, ...]"),
    ]:
        print(f"\n--- Test: {test_name} ({scale_desc}) ---")
        A, As, B, Bs = gen_data(M, K, N, seed=42, uniform=uniform)

        if test_name == "two-scale":
            As = np.ones((M, nb), np.float32)
            As[:, nb//2:] = 2.0
            Bs = np.ones((nb, N), np.float32)
            Bs[nb//2:, :] = 2.0

        if test_name == "non-uniform":
            rng = np.random.default_rng(42)
            As = np.power(2.0, rng.integers(0, 8, size=(M, nb))).astype(np.float32)
            Bs = np.power(2.0, rng.integers(0, 8, size=(nb, N))).astype(np.float32)

        C_ref = ref_q8_0_gemm(A, As, B, Bs)

        results = []

        # 1. Q8SCALEV-C1
        q8v_xc = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.xclbin"
        q8v_ins = f"{cache_dir}/M{M}_K{K}_N{N}_C1_Q8SCALEV/design.insts.bin"
        if os.path.exists(q8v_xc) and os.path.exists(q8v_ins):
            print(f"  running Q8SCALEV-C1...")
            m_tile, n_tile, k_chunk = 32, 64, 256
            from bench_q8_c4 import pack_A_vec, pack_B_vec
            Ap = pack_A_vec(A, As, M, K, m_tile, k_chunk)
            Bp = pack_B_vec(B, Bs, K, N, n_tile, k_chunk)
            times = []
            for it in range(iters):
                try:
                    C_dev, t = run_kernel_generic(q8v_xc, Ap, Bp, M, K, N)
                    times.append(t)
                except Exception as e:
                    print(f"    iter {it} failed: {e}")
                    break
            if times:
                dev_ms = np.median(times)
                diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
                max_err = float(np.max(diff))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mask = np.abs(C_ref) > 1e-6
                rel_err = float(np.max(diff[mask] / np.abs(C_ref[mask]))) if np.any(mask) else 0.0
                mismatches = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))
                gops = M * N * K / (dev_ms / 1000.0) / 1e9
                results.append(("Q8SCALEV-C1", 1, dev_ms, gops, max_err, rmse, rel_err, mismatches))
                print(f"    device={dev_ms:.2f}ms max_err={max_err:.3e} rmse={rmse:.3e} bad={mismatches}")
        else:
            print(f"  Q8SCALEV-C1 kernel not found")

        # 2. C4_WA int8 reference
        wa_xc = f"{cache_dir}/M{M}_K{K}_N{N}_C4_WA/design.xclbin"
        wa_ins = f"{cache_dir}/M{M}_K{K}_N{N}_C4_WA/design.insts.bin"
        if os.path.exists(wa_xc) and os.path.exists(wa_ins):
            print(f"  running C4_WA int8 reference...")
            A_i8 = A.astype(np.int8)
            B_i8 = B.astype(np.int8)
            B_i8_row = np.ascontiguousarray(B_i8.T)
            from bench_q8_c4 import run_kernel_wa
            times = []
            for it in range(iters):
                try:
                    C_i32, t = run_kernel_wa(wa_xc, wa_ins, A_i8, B_i8_row, M, K, N)
                    times.append(t)
                except Exception as e:
                    print(f"    iter {it} failed: {e}")
                    break
            if times:
                dev_ms = np.median(times)
                gops = M * N * K / (dev_ms / 1000.0) / 1e9
                results.append(("C4_WA-I8", 4, dev_ms, gops, 0.0, 0.0, 0.0, 0))
                print(f"    device={dev_ms:.2f}ms gops={gops:.1f}")

        # 3. C1 device-scale (our new kernel)
        ds_xc = f"{cache_dir}/M{M}_K{K}_N{N}_C1_DEVICE_SCALE/design.xclbin"
        ds_ins = f"{cache_dir}/M{M}_K{K}_N{N}_C1_DEVICE_SCALE/design.insts.bin"
        if os.path.exists(ds_xc) and os.path.exists(ds_ins):
            print(f"  running C1 device-scale...")
            m_tile, n_tile, k_chunk = 64, 64, 256
            Ap = pack_A_mmul(A, As, M, K, m_tile, k_chunk)
            Bp = pack_B_mmul(B, Bs, K, N, n_tile, k_chunk)
            times = []
            for it in range(iters):
                try:
                    C_dev, t = run_kernel_generic(ds_xc, Ap, Bp, M, K, N)
                    times.append(t)
                except Exception as e:
                    print(f"    iter {it} failed: {e}")
                    break
            if times:
                dev_ms = np.median(times)
                diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
                max_err = float(np.max(diff))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mask = np.abs(C_ref) > 1e-6
                rel_err = float(np.max(diff[mask] / np.abs(C_ref[mask]))) if np.any(mask) else 0.0
                mismatches = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))
                gops = M * N * K / (dev_ms / 1000.0) / 1e9
                results.append(("C1-DEVICE-SCALE", 1, dev_ms, gops, max_err, rmse, rel_err, mismatches))
                print(f"    device={dev_ms:.2f}ms max_err={max_err:.3e} rmse={rmse:.3e} bad={mismatches}")
        else:
            print(f"  C1 device-scale kernel not found")

        print(f"\n  {'Kernel':<20} {'Cols':>5} {'Dev ms':>9} {'GOPS':>8} {'MaxErr':>12} {'RMSE':>12} {'RelErr':>12} {'Bad':>8}")
        for name, cols, dev_ms, gops, max_err, rmse, rel_err, bad in results:
            print(f"  {name:<20} {cols:>5} {dev_ms:>9.2f} {gops:>8.1f} {max_err:>12.3e} {rmse:>12.3e} {rel_err:>12.3e} {bad:>8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    run_correctness_tests(args.M, args.K, args.N, args.iters)
