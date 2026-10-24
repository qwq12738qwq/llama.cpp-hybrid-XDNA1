#!/usr/bin/env python3
"""Benchmark/reference harness for the mmul-layout XDNA full-K Q8_0 GEMM.

mmul layout A: [m/4][kc/32][4][4x8] int8, then [m][kc/32] float32
mmul layout B: [kc/32][n/8][4][8x8] int8, then [kc/32][n] float32
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time

import numpy as np
import pyxrt

HERE = os.path.dirname(os.path.abspath(__file__))
IRONENV = os.environ.get("IRONENV", "/Storage/Cache/ironenv-v143/bin/python")
RESULTS = os.path.join(HERE, "results", "fullk_bench.jsonl")


def compile_design(wd, a, recompile=False):
    xclbin = os.path.join(wd, "design.xclbin")
    if os.path.exists(xclbin) and not recompile:
        return 0.0
    cmd = [IRONENV, os.path.join(HERE, "gen_fullk_mmul.py"),
           "--M", str(a.M), "--K", str(a.K), "--N", str(a.N),
           "--m-tile", str(a.m_tile), "--n-tile", str(a.n_tile),
           "--k-chunk", str(a.k_chunk), "--work-dir", wd]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("compile failed:\n" + p.stdout + p.stderr)
    return round(time.time() - t0, 2)


def tile_placements(wd):
    found = set()
    for e in glob.glob(os.path.join(wd, "**", "elfs_main_core_*"), recursive=True):
        m = re.match(r"elfs_main_core_(\d+)_(\d+)$", os.path.basename(e))
        if m:
            found.add((int(m.group(1)), int(m.group(2))))
    return sorted(found)


def make_data(M, K, N, m_tile, n_tile, k_chunk, seed=11):
    n_kc, n_nt, n_mt = K // k_chunk, N // n_tile, M // m_tile
    nblk, nb_total = k_chunk // 32, K // 32
    n_t = n_tile // 8
    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4

    rng = np.random.default_rng(seed)
    A = rng.integers(-127, 128, (M, K), dtype=np.int8)
    B = rng.integers(-127, 128, (K, N), dtype=np.int8)
    a_sc = np.maximum(np.abs(A).reshape(M, nb_total, 32).max(2).astype(np.float32) / 127.0, 1e-12)
    b_sc = np.zeros((nb_total, N), np.float32)
    for kb in range(nb_total):
        b_sc[kb] = np.maximum(np.abs(B[kb * 32:(kb + 1) * 32]).max(0).astype(np.float32) / 127.0, 1e-12)

    A_pack = np.zeros(n_mt * n_kc * a_bytes, np.int8)
    for mt in range(n_mt):
        for kc in range(n_kc):
            base = (mt * n_kc + kc) * a_bytes
            for mi in range(m_tile // 4):
                for kb in range(nblk):
                    for l in range(4):
                        off = base + ((mi * nblk + kb) * 4 + l) * 32
                        r0 = mt * m_tile + mi * 4
                        c0 = kc * k_chunk + kb * 32 + l * 8
                        A_pack[off:off + 32] = A[r0:r0 + 4, c0:c0 + 8].reshape(-1)
            A_pack[base + m_tile * k_chunk:base + a_bytes] = \
                a_sc[mt * m_tile:(mt + 1) * m_tile,
                     kc * nblk:(kc + 1) * nblk].reshape(-1).view(np.int8)

    B_pack = np.zeros(n_kc * n_nt * b_bytes, np.int8)
    for kc in range(n_kc):
        for nt in range(n_nt):
            base = (kc * n_nt + nt) * b_bytes
            for kb in range(nblk):
                for ni in range(n_t):
                    for l in range(4):
                        off = base + (((kb * n_t) + ni) * 4 + l) * 64
                        r0 = kc * k_chunk + kb * 32 + l * 8
                        c0 = nt * n_tile + ni * 8
                        B_pack[off:off + 64] = B[r0:r0 + 8, c0:c0 + 8].reshape(-1)
            B_pack[base + k_chunk * n_tile:base + b_bytes] = \
                b_sc[kc * nblk:(kc + 1) * nblk,
                     nt * n_tile:(nt + 1) * n_tile].reshape(-1).view(np.int8)

    C_ref = np.zeros((M, N), np.float64)
    for kb in range(nb_total):
        p = A[:, kb * 32:(kb + 1) * 32].astype(np.int32) @ B[kb * 32:(kb + 1) * 32].astype(np.int32)
        C_ref += (a_sc[:, kb][:, None].astype(np.float64) * b_sc[kb][None, :].astype(np.float64)
                  * p.astype(np.float64))
    return A_pack, B_pack, C_ref


def run_kernel(wd, A_pack, B_pack, M, N, repeat):
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(os.path.join(wd, "design.xclbin"))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    gid3, gid1 = kern.group_id(3), kern.group_id(1)
    a_bo = pyxrt.bo(dev, A_pack.nbytes, pyxrt.bo.flags.host_only, gid3)
    b_bo = pyxrt.bo(dev, B_pack.nbytes, pyxrt.bo.flags.host_only, gid3)
    c_bo = pyxrt.bo(dev, M * N * 4, pyxrt.bo.flags.host_only, gid3)
    inst = os.path.join(wd, "design.insts.bin")
    isz = os.path.getsize(inst)
    inst_bo = pyxrt.bo(dev, isz, pyxrt.bo.flags.cacheable, gid1)
    with open(inst, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    a_bo.write(A_pack.tobytes(), 0)
    b_bo.write(B_pack.tobytes(), 0)
    c_bo.write(b"\x00" * (M * N * 4), 0)
    for b in (a_bo, b_bo, c_bo):
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    times = []
    for _ in range(repeat):
        r = pyxrt.run(kern)
        r.set_arg(0, 3)
        r.set_arg(1, inst_bo)
        r.set_arg(2, isz)
        r.set_arg(3, a_bo)
        r.set_arg(4, b_bo)
        r.set_arg(5, c_bo)
        t0 = time.perf_counter()
        r.start()
        r.wait()
        times.append((time.perf_counter() - t0) * 1000.0)
    c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    C = np.frombuffer(c_bo.read(M * N * 4, 0), np.float32).reshape(M, N)
    return times, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, required=True)
    ap.add_argument("--n-tile", type=int, required=True)
    ap.add_argument("--k-chunk", type=int, required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--recompile", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cps = compile_design(a.work_dir, a, a.recompile)
    A_pack, B_pack, C_ref = make_data(a.M, a.K, a.N, a.m_tile, a.n_tile, a.k_chunk)
    times, C = run_kernel(a.work_dir, A_pack, B_pack, a.M, a.N, a.repeat)
    med = sorted(times)[len(times) // 2]
    macs = a.M * a.K * a.N
    d64 = np.abs(C.astype(np.float64) - C_ref)
    maxv = float(np.max(np.abs(C_ref)))

    rec = {
        "tag": a.tag, "design": "mmul",
        "M": a.M, "K": a.K, "N": a.N,
        "m_tile": a.m_tile, "n_tile": a.n_tile, "k_chunk": a.k_chunk,
        "xrt_submits": 1,
        "compute_tiles": len(tile_placements(a.work_dir)),
        "tiles": [f"({x},{y})" for x, y in tile_placements(a.work_dir)],
        "compile_s": cps,
        "ms_median": round(med, 4),
        "ms_all": [round(t, 4) for t in times],
        "macs": macs,
        "mmac_s": round(macs / max(med, 1e-9) / 1e3, 2),
        "ns_per_mac": round(med * 1e6 / macs, 4),
        "max_rel_err": float(d64.max() / maxv),
        "max_abs_err": float(d64.max()),
        "all_zero": bool(np.all(C == 0)),
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
