#!/usr/bin/env python3
"""Phase 50 Stage 1: trustworthy perf decomposition for c4_mmul_v2.

Single process holds ONE XRT device/hw_context across all iters (no
context-creation bombing). Splits every stage separately:

  pack_ms (host Q8 repack+tile-major, numpy reference path)
  bo_write_ms (host memcpy into BO maps)
  sync_to_ms (BO sync TO_DEVICE, A+B+C)
  submit_ms (run ctor + set_arg + start)
  wait_ms (run.wait() ONLY - device execution + DMA wait, no sync)
  sync_from_ms (C sync FROM_DEVICE)
  copyback_ms (BO map -> numpy)
  correctness max_err vs ref_q8_gemm

Also reports DMA byte accounting (unique vs actual transferred given the
per-(mt,col,nt) task-group sequence) and effective GFLOP/s.

Usage (ironenv python):
  python microbench_stage1.py --M 512 --K 1024 --N 3584 --iters 5 --pattern realistic
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase40"))
from gen_q8_c4_mmul_v2 import pack_A, pack_B, ref_q8_gemm, M_TILE, N_TILE, KC, NBLK

import pyxrt

CACHE = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"


def dma_accounting(M, K, N, n_cols=4):
    m, n, kc = M_TILE, N_TILE, KC
    nblk = NBLK
    n_mt = M // m
    n_nt_c = (N // n_cols) // n
    n_kc = K // kc
    a_chunk = m * kc + m * nblk * 4
    b_chunk = kc * n + nblk * n * 4
    a_unique = n_mt * n_kc * a_chunk
    b_unique = n_cols * n_nt_c * n_kc * b_chunk
    c_bytes = M * N * 4
    groups = n_mt * n_cols * n_nt_c
    # sequence() fills A(mt,all-kc) + B(col,nt,all-kc) per (mt,col,nt) group
    a_actual = groups * (n_kc * a_chunk)
    b_actual = groups * (n_kc * b_chunk)
    return {
        "n_mt": n_mt, "n_nt_c": n_nt_c, "n_kc": n_kc, "groups": groups,
        "a_unique": a_unique, "b_unique": b_unique, "c_bytes": c_bytes,
        "a_actual": a_actual, "b_actual": b_actual,
        "dma_unique": a_unique + b_unique + c_bytes,
        "dma_actual": a_actual + b_actual + c_bytes,
        "redundancy": (a_actual + b_actual + c_bytes) / (a_unique + b_unique + c_bytes),
        "a_redundancy": a_actual / a_unique,
        "b_redundancy": b_actual / b_unique,
    }


def run_once_timed(kernel, inst_bo, inst_bytes, a_bo, b_bo, c_bo, C_out):
    t = {}
    c_map = np.frombuffer(c_bo.map(), dtype=np.float32)
    c_map[:] = 0
    t0 = time.perf_counter_ns()
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t1 = time.perf_counter_ns()
    run = pyxrt.run(kernel)
    run.set_arg(0, np.uint64(3))
    run.set_arg(1, inst_bo)
    run.set_arg(2, np.uint32(inst_bytes))
    run.set_arg(3, a_bo)
    run.set_arg(4, b_bo)
    run.set_arg(5, c_bo)
    t2 = time.perf_counter_ns()
    run.start()
    t3 = time.perf_counter_ns()
    run.wait()
    t4 = time.perf_counter_ns()
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    t5 = time.perf_counter_ns()
    C_out[:] = np.frombuffer(c_bo.map(), dtype=np.float32)
    t6 = time.perf_counter_ns()
    return {
        "c_zero_sync_ms": (t1 - t0) / 1e6,
        "run_ctor_arg_ms": (t2 - t1) / 1e6,
        "submit_ms": (t3 - t2) / 1e6,
        "wait_ms": (t4 - t3) / 1e6,
        "sync_from_ms": (t5 - t4) / 1e6,
        "copyback_ms": (t6 - t5) / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--pattern", type=str, default="realistic",
                    choices=["uniform", "two_scale", "geometric", "realistic"])
    ap.add_argument("--cache-dir", type=str, default=CACHE)
    ap.add_argument("--xclbin", type=str, default=None,
                    help="override xclbin path (variant testing); insts = same dir/design.insts.bin")
    ap.add_argument("--expect-zeros", action="store_true",
                    help="dma_only variant: expect C==0, check that instead of ref")
    ap.add_argument("--repack", type=str, default=None, choices=["s8", "s10"],
                    help="repack scale sections for host-pre-broadcast kernel layout")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    M, K, N, n_cols = args.M, args.K, args.N, args.cols
    nb = K // 32
    flops = 2.0 * M * K * N

    dma = dma_accounting(M, K, N, n_cols)
    print(f"[stage1] M={M} K={K} N={N} cols={n_cols} iters={args.iters} pattern={args.pattern}")
    print(f"[stage1] DMA unique={dma['dma_unique']/1e6:.2f}MB actual={dma['dma_actual']/1e6:.2f}MB "
          f"redundancy={dma['redundancy']:.2f}x (A {dma['a_redundancy']:.1f}x B {dma['b_redundancy']:.1f}x) "
          f"groups={dma['groups']}")
    print(f"[stage1] FLOPs={flops/1e9:.3f} GFLOP")

    rng = np.random.default_rng(42)
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    if args.pattern == "realistic":
        As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
        Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)
    elif args.pattern == "uniform":
        As = np.ones((M, nb), dtype=np.float32)
        Bs = np.ones((nb, N), dtype=np.float32)
    elif args.pattern == "geometric":
        factors = np.array([min(2.0 ** i, 65504.0) for i in range(nb)], dtype=np.float32)
        As = np.ones((M, nb), dtype=np.float32) * factors[np.newaxis, :]
        Bs = np.ones((nb, N), dtype=np.float32) * factors[:, np.newaxis]
    else:
        As = np.ones((M, nb), dtype=np.float32)
        Bs = np.ones((nb, N), dtype=np.float32)
        As[:, nb // 2:] = 2.0
        Bs[nb // 2:, :] = 2.0

    t0 = time.perf_counter()
    C_ref = ref_q8_gemm(A_i8, As, B_i8, Bs)
    t_ref = (time.perf_counter() - t0) * 1000
    print(f"[stage1] numpy ref_q8_gemm: {t_ref:.1f}ms C[0,0]={C_ref[0,0]:.4f}")

    t0 = time.perf_counter()
    Ap = pack_A(A_i8, As, M, K, M_TILE, KC)
    t_pack_a = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    Bp = pack_B(B_i8, Bs, K, N, N_TILE, KC)
    t_pack_b = (time.perf_counter() - t0) * 1000
    if args.repack == "s10":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import gen_s10
        Ap = gen_s10.pack_A_s10(A_i8, As, M, K)
        Bp = gen_s10.pack_B_s10(B_i8, Bs, K, N)
        t_pack_a = t_pack_b = 0.0
        print(f"[stage1] repack=s10: pack_A={Ap.nbytes/1e6:.2f}MB pack_B={Bp.nbytes/1e6:.2f}MB")
    elif args.repack == "s8":
        nb = K // 32
        n_kc = K // KC
        MA32, NA8 = M_TILE // 4, N_TILE // 8
        a_chunk = M_TILE * KC
        b_chunk = KC * N_TILE
        n_mt, n_nt = M // M_TILE, N // N_TILE
        for mt in range(n_mt):
            for kc in range(n_kc):
                off = (mt * n_kc + kc) * (a_chunk + M_TILE * nb * 4)
                sc = Ap[off + a_chunk:].view(np.float32).reshape(M_TILE, nb)
                out = np.empty((NBLK, MA32, 32), np.float32)
                for bb in range(NBLK):
                    for i in range(MA32):
                        for r in range(4):
                            out[bb, i, r * 8:(r + 1) * 8] = sc[i * 4 + r, bb]
                Ap[off + a_chunk:] = out.reshape(-1).view(np.int8)
        for nt in range(n_nt):
            for kc in range(n_kc):
                off = (nt * n_kc + kc) * (b_chunk + NBLK * N_TILE * 4)
                sc = Bp[off + b_chunk:].view(np.float32).reshape(NBLK, N_TILE)
                out = np.empty((NBLK, NA8, 16), np.float32)
                for bb in range(NBLK):
                    for j in range(NA8):
                        out[bb, j, 0:8] = sc[bb, j * 8:j * 8 + 8]
                        out[bb, j, 8:16] = sc[bb, j * 8:j * 8 + 8]
                Bp[off + b_chunk:] = out.reshape(-1).view(np.int8)
        print("[stage1] repack=s8: host-pre-broadcast scales installed")
    print(f"[stage1] pack_A={t_pack_a:.1f}ms ({Ap.nbytes/1e6:.2f}MB) pack_B={t_pack_b:.1f}ms ({Bp.nbytes/1e6:.2f}MB)")

    if args.xclbin:
        xp = args.xclbin
        ip = os.path.join(os.path.dirname(xp), "design.insts.bin")
    else:
        xp = os.path.join(args.cache_dir, f"M{M}_K{K}_N{N}_C{n_cols}_Q8SCALEV_MMUL", "design.xclbin")
        ip = xp.replace(".xclbin", ".insts.bin")
    if not os.path.exists(xp):
        print(f"[stage1] ERROR missing xclbin {xp}")
        sys.exit(2)
    print(f"[stage1] using cached {xp}")

    xclbin = pyxrt.xclbin(xp)
    device = pyxrt.device(0)
    device.register_xclbin(xclbin)
    hwctx = pyxrt.hw_context(device, xclbin.get_uuid())
    kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
    gid1 = kernel.group_id(1)
    gid3 = kernel.group_id(3)
    inst_bytes = os.path.getsize(ip)
    with open(ip, "rb") as f:
        insts = np.frombuffer(f.read(), dtype=np.uint32)
    inst_bo = pyxrt.bo(device, inst_bytes, pyxrt.bo.flags.cacheable, gid1)
    np.frombuffer(inst_bo.map(), dtype=np.uint32)[:] = insts
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    t0 = time.perf_counter()
    a_bo = pyxrt.bo(device, Ap.nbytes, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(device, Bp.nbytes, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(device, M * N * 4, pyxrt.bo.flags.host_only, gid3)
    np.frombuffer(a_bo.map(), dtype=np.uint8)[:] = np.frombuffer(np.ascontiguousarray(Ap), dtype=np.uint8)
    t_w0 = (time.perf_counter() - t0) * 1000
    np.frombuffer(b_bo.map(), dtype=np.uint8)[:] = np.frombuffer(np.ascontiguousarray(Bp), dtype=np.uint8)
    t_w1 = (time.perf_counter() - t0) * 1000 - t_w0
    t0 = time.perf_counter()
    a_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    b_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t_sync_to = (time.perf_counter() - t0) * 1000
    print(f"[stage1] BO alloc+write A={t_w0:.1f}ms B={t_w1:.1f}ms sync_TO(A+B)={t_sync_to:.1f}ms")

    C_out = np.zeros(M * N, dtype=np.float32)
    for it in range(args.warmup):
        run_once_timed(kernel, inst_bo, inst_bytes, a_bo, b_bo, c_bo, C_out)
    print(f"[stage1] warmup {args.warmup} done")

    rows = []
    for it in range(args.iters):
        seg = run_once_timed(kernel, inst_bo, inst_bytes, a_bo, b_bo, c_bo, C_out)
        C_dev = C_out.reshape(M, N)
        if args.expect_zeros:
            seg["max_err"] = float(np.max(np.abs(C_dev)))
            seg["zeros_ok"] = seg["max_err"] == 0.0
        else:
            diff = np.abs(C_dev.astype(np.float64) - C_ref.astype(np.float64))
            seg["max_err"] = float(np.max(diff))
        seg["gflops_wait"] = (flops / 1e9) / (seg["wait_ms"] / 1e3)
        rows.append(seg)
        print(f"[stage1] iter={it} " + " ".join(f"{k}={v:.3f}" for k, v in seg.items() if k != "max_err") +
              f" max_err={seg['max_err']:.4e} gflops(wait)={seg['gflops_wait']:.2f}")

    waits = sorted(r["wait_ms"] for r in rows)
    med = waits[len(waits) // 2]
    print(f"[stage1] WAIT median={med:.1f}ms min={min(waits):.1f} max={max(waits):.1f} "
          f"GFLOP/s(wait)={(flops/1e9)/(med/1e3):.2f} DMA_BW(actual)={dma['dma_actual']/1e6/(med/1e3):.1f}MB/s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"M": M, "K": K, "N": N, "flops": flops, "dma": dma,
                       "pack_a_ms": t_pack_a, "pack_b_ms": t_pack_b,
                       "bo_write_a_ms": t_w0, "bo_write_b_ms": t_w1, "sync_to_ms": t_sync_to,
                       "rows": rows}, f, indent=1)
        print(f"[stage1] wrote {args.out}")


if __name__ == "__main__":
    main()
