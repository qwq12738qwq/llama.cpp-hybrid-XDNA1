#!/usr/bin/env python3
"""Benchmark/reference harness for the MemTile-staged full-K Q8_0 GEMM.

A layout (column-major by (col, kc, row)) so each column's chunk is contiguous:
    A[(col * n_kc + kc) * n_rows + row] = chunk of M-block (col*n_rows+row)
B and C layouts are unchanged from the per-tile harness: tile t of the grid
handles M-block t = col * n_rows + row.
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
    cmd = [IRONENV, os.path.join(HERE, "gen_fullk_memtile.py"),
           "--M", str(a.M), "--K", str(a.K), "--N", str(a.N),
           "--m-tile", str(a.m_tile), "--n-tile", str(a.n_tile),
           "--k-chunk", str(a.k_chunk),
           "--n-rows", str(a.n_rows), "--n-cols", str(a.n_cols),
           "--work-dir", wd]
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


def make_data(M, K, N, m_tile, n_tile, k_chunk, n_rows, n_cols, seed=11):
    n_tiles = n_rows * n_cols
    n_kc, n_nt, n_mt = K // k_chunk, N // n_tile, M // m_tile
    assert n_mt == n_tiles
    nblk, nb_total = k_chunk // 32, K // 32
    a_bytes = m_tile * k_chunk + m_tile * nblk * 4
    b_bytes = k_chunk * n_tile + nblk * n_tile * 4

    rng = np.random.default_rng(seed)
    A = rng.integers(-127, 128, (M, K), dtype=np.int8)
    B = rng.integers(-127, 128, (K, N), dtype=np.int8)
    a_sc = np.maximum(np.abs(A).reshape(M, nb_total, 32).max(2).astype(np.float32) / 127.0, 1e-12)
    b_sc = np.zeros((nb_total, N), np.float32)
    for kb in range(nb_total):
        b_sc[kb] = np.maximum(np.abs(B[kb * 32:(kb + 1) * 32]).max(0).astype(np.float32) / 127.0, 1e-12)

    A_pack = np.zeros(n_tiles * n_kc * a_bytes, np.int8)
    for col in range(n_cols):
        for kc in range(n_kc):
            for row in range(n_rows):
                t = col * n_rows + row
                base = ((col * n_kc + kc) * n_rows + row) * a_bytes
                A_pack[base:base + m_tile * k_chunk] = \
                    A[t * m_tile:(t + 1) * m_tile,
                      kc * k_chunk:(kc + 1) * k_chunk].reshape(-1)
                A_pack[base + m_tile * k_chunk:base + a_bytes] = \
                    a_sc[t * m_tile:(t + 1) * m_tile,
                         kc * nblk:(kc + 1) * nblk].reshape(-1).view(np.int8)

    B_pack = np.zeros(n_kc * n_nt * b_bytes, np.int8)
    for nt in range(n_nt):
        for kc in range(n_kc):
            base = (nt * n_kc + kc) * b_bytes
            B_pack[base:base + k_chunk * n_tile] = \
                B[kc * k_chunk:(kc + 1) * k_chunk,
                  nt * n_tile:(nt + 1) * n_tile].reshape(-1)
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
    ap.add_argument("--n-rows", type=int, required=True)
    ap.add_argument("--n-cols", type=int, required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--recompile", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cps = compile_design(a.work_dir, a, a.recompile)
    A_pack, B_pack, C_ref = make_data(a.M, a.K, a.N, a.m_tile, a.n_tile,
                                      a.k_chunk, a.n_rows, a.n_cols)
    times, C = run_kernel(a.work_dir, A_pack, B_pack, a.M, a.N, a.repeat)
    med = sorted(times)[len(times) // 2]
    macs = a.M * a.K * a.N
    d64 = np.abs(C.astype(np.float64) - C_ref)
    maxv = float(np.max(np.abs(C_ref)))

    # Analytic DMA traffic. Per column: A/B L3L2 fills; per tile: L2->L1 of A+B
    # and L1->L2->L3 of C. B is replicated once per column (a shim stream can
    # only reach its own column), so B's host-side reuse factor is n_cols.
    n_tiles = a.n_rows * a.n_cols
    n_kc, n_nt = a.K // a.k_chunk, a.N // a.n_tile
    nblk = a.k_chunk // 32
    a_bytes = a.m_tile * a.k_chunk + a.m_tile * nblk * 4
    b_bytes = a.k_chunk * a.n_tile + nblk * a.n_tile * 4
    c_bytes = a.m_tile * a.n_tile * 4
    dma = {
        "a_host_l2": a.n_cols * n_kc * a.n_rows * a_bytes,
        "a_l2_l1": n_tiles * n_kc * a_bytes,
        "b_host_l2": a.n_cols * n_kc * b_bytes,
        "b_l2_l1": n_tiles * n_kc * b_bytes,
        "c_l1_l2": n_tiles * n_nt * c_bytes,
        "c_l2_l3": n_tiles * n_nt * c_bytes,
    }
    dma["total"] = sum(dma.values())

    rec = {
        "tag": a.tag, "design": "memtile",
        "M": a.M, "K": a.K, "N": a.N,
        "m_tile": a.m_tile, "n_tile": a.n_tile, "k_chunk": a.k_chunk,
        "grid": f"{a.n_rows}x{a.n_cols}",
        "xrt_submits": 1,
        "compute_tiles": len(tile_placements(a.work_dir)),
        "tiles": [f"({x},{y})" for x, y in tile_placements(a.work_dir)],
        "compile_s": cps,
        "ms_median": round(med, 4),
        "ms_all": [round(t, 4) for t in times],
        "macs": macs,
        "mmac_s": round(macs / max(med, 1e-9) / 1e3, 2),
        "ns_per_mac": round(med * 1e6 / macs, 4),
        "dma_bytes": dma,
        "b_bytes_per_mac": round((dma["b_host_l2"] + dma["b_l2_l1"]) / macs, 6),
        "b_host_bytes_per_mac": round(dma["b_host_l2"] / macs, 6),
        "b_reuse_host": round(a.M / a.n_cols, 2),
        "b_reuse_ideal": a.M,
        "total_dma_per_mac": round(dma["total"] / macs, 6),
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
