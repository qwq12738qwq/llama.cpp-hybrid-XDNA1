#!/usr/bin/env python3
"""Unified XDNA full-K Q8_0 GEMM benchmark harness.

One JSON record per run: shape, compile plan (L1/BD budget), compute-tile
placement parsed from `main_core_<col>_<row>.elf`, latency, MAC throughput,
DMA bytes and float64 correctness.
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
GENS = {"tiled": "gen_fullk_tiled.py", "multi": "gen_fullk_multi.py", "vec": "gen_fullk_vec.py", "multivec": "gen_fullk_multi_vec.py"}


def compile_design(design, wd, args, recompile=False):
    xclbin = os.path.join(wd, "design.xclbin")
    if os.path.exists(xclbin) and not recompile:
        return {"compiled": False, "compile_s": 0.0, "log": ""}
    cmd = [IRONENV, os.path.join(HERE, GENS[design]),
           "--M", str(args.M), "--K", str(args.K), "--N", str(args.N),
           "--m-tile", str(args.m_tile), "--n-tile", str(args.n_tile),
           "--k-chunk", str(args.k_chunk), "--work-dir", wd]
    if design in ("multi", "multivec"):
        cmd += ["--n-tiles", str(args.n_tiles)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    if p.returncode != 0:
        raise SystemExit(f"compile failed:\n{log}")
    return {"compiled": True, "compile_s": round(time.time() - t0, 2), "log": log}


def parse_plan(log):
    plan = {}
    m = re.search(r"total=(\d+)/(\d+)", log)
    if m:
        plan["l1_used"], plan["l1_limit"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"BD:\s*(.+)", log)
    if m:
        plan["bd"] = m.group(1).strip()
    return plan


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
            off = (mt * n_kc + kc) * a_bytes
            r, c = slice(mt * m_tile, (mt + 1) * m_tile), slice(kc * k_chunk, (kc + 1) * k_chunk)
            A_pack[off:off + m_tile * k_chunk] = A[r, c].reshape(-1)
            kb = slice(kc * nblk, (kc + 1) * nblk)
            A_pack[off + m_tile * k_chunk:off + a_bytes] = a_sc[r, kb].reshape(-1).view(np.int8)

    B_pack = np.zeros(n_kc * n_nt * b_bytes, np.int8)
    for kc in range(n_kc):
        for nt in range(n_nt):
            off = (kc * n_nt + nt) * b_bytes
            r, c = slice(kc * k_chunk, (kc + 1) * k_chunk), slice(nt * n_tile, (nt + 1) * n_tile)
            B_pack[off:off + k_chunk * n_tile] = B[r, c].reshape(-1)
            kb = slice(kc * nblk, (kc + 1) * nblk)
            B_pack[off + k_chunk * n_tile:off + b_bytes] = b_sc[kb, c].reshape(-1).view(np.int8)

    C_ref = np.zeros((M, N), np.float64)
    for kb in range(nb_total):
        p = A[:, kb * 32:(kb + 1) * 32].astype(np.int32) @ B[kb * 32:(kb + 1) * 32].astype(np.int32)
        C_ref += (a_sc[:, kb][:, None].astype(np.float64) * b_sc[kb][None, :].astype(np.float64)
                  * p.astype(np.float64))

    meta = {"n_mt": n_mt, "n_nt": n_nt, "n_kc": n_kc,
            "a_bytes": a_bytes, "b_bytes": b_bytes}
    return A_pack, B_pack, C_ref, meta


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
    ap.add_argument("--design", choices=sorted(GENS), default="multi")
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m-tile", type=int, required=True)
    ap.add_argument("--n-tile", type=int, required=True)
    ap.add_argument("--k-chunk", type=int, required=True)
    ap.add_argument("--n-tiles", type=int, default=1)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--recompile", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    c = compile_design(args.design, args.work_dir, args, args.recompile)
    A_pack, B_pack, C_ref, meta = make_data(
        args.M, args.K, args.N, args.m_tile, args.n_tile, args.k_chunk)
    times, C = run_kernel(args.work_dir, A_pack, B_pack, args.M, args.N, args.repeat)

    times_sorted = sorted(times)
    med = times_sorted[len(times_sorted) // 2]
    macs = args.M * args.K * args.N
    d64 = np.abs(C.astype(np.float64) - C_ref)
    maxv = float(np.max(np.abs(C_ref)))
    n_tiles_used = meta["n_mt"]
    dma_src = (meta["n_mt"] * meta["n_kc"] * meta["a_bytes"]
               + meta["n_kc"] * meta["n_nt"] * meta["b_bytes"]
               + meta["n_mt"] * meta["n_nt"] * args.m_tile * args.n_tile * 4)
    dma_dst = (meta["n_mt"] * meta["n_kc"] * meta["a_bytes"]
               + meta["n_mt"] * meta["n_nt"] * meta["n_kc"] * meta["b_bytes"]
               + meta["n_mt"] * meta["n_nt"] * args.m_tile * args.n_tile * 4)

    rec = {
        "tag": args.tag, "design": args.design,
        "M": args.M, "K": args.K, "N": args.N,
        "m_tile": args.m_tile, "n_tile": args.n_tile, "k_chunk": args.k_chunk,
        "grid": f"{meta['n_mt']}x{meta['n_nt']}x{meta['n_kc']}",
        "xrt_submits": 1,
        "compute_tiles": n_tiles_used,
        "tiles": [f"({c0},{r0})" for c0, r0 in tile_placements(args.work_dir)],
        "plan": parse_plan(c["log"]),
        "compile_s": c["compile_s"],
        "ms_median": round(med, 4),
        "ms_min": round(min(times), 4),
        "ms_all": [round(t, 4) for t in times],
        "macs": macs,
        "mmac_s": round(macs / max(med, 1e-9) / 1e3, 2),
        "ns_per_mac": round(med * 1e6 / macs, 4),
        "dma_src_bytes": dma_src,
        "dma_dst_bytes": dma_dst,
        "b_bytes_per_mac": round(dma_dst * 0 + meta["n_kc"] * meta["b_bytes"] * meta["n_mt"] / macs, 6),
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
