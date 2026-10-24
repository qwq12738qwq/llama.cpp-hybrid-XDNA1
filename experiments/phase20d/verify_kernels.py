#!/usr/bin/env python3
"""Phase 20D: numeric correctness check for cached XDNA int8 GEMM kernels.

Runs each cached K=32 int8 kernel on the NPU via pyxrt and compares against a
numpy int8 x int8 -> int32 reference. Integer accumulation is exact, so a
difference means a real kernel or data-layout bug, not fp rounding.

Usage:
    /Storage/Cache/ironenv-v143/bin/python verify_kernels.py --group new
    /Storage/Cache/ironenv-v143/bin/python verify_kernels.py --group all --log out.jsonl
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pyxrt

KC = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"
TILE_M, TILE_K, TILE_N = 64, 32, 64
K = 32
PROD_N = [512, 768, 1024, 1536, 2048, 2304, 3072, 3584, 4096, 4608, 5120, 6144]
KERNEL_NAME = "MLIR_AIE"


def wd(M, N, C):
    return os.path.join(KC, f"M{M}_K{K}_N{N}_C{C}_Tm{TILE_M}_Tk{TILE_K}_Tn{TILE_N}")


def has_artifacts(M, N, C):
    d = wd(M, N, C)
    return (os.path.exists(os.path.join(d, "design.xclbin")) and
            os.path.exists(os.path.join(d, "design.insts.bin")))


def feasible(N, C):
    return N % (TILE_N * C) == 0


def run_kernel(M, N, C, A, B, dev):
    d = wd(M, N, C)
    xb = pyxrt.xclbin(os.path.join(d, "design.xclbin"))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, KERNEL_NAME)

    in_sz, wt_sz, out_sz = M * K, K * N, M * N * 4
    gid3 = kern.group_id(3)
    gid1 = kern.group_id(1)
    in_bo = pyxrt.bo(dev, in_sz, pyxrt.bo.flags.host_only, gid3)
    wt_bo = pyxrt.bo(dev, wt_sz, pyxrt.bo.flags.host_only, gid3)
    out_bo = pyxrt.bo(dev, out_sz, pyxrt.bo.flags.host_only, gid3)

    inst_path = os.path.join(d, "design.insts.bin")
    isz = os.path.getsize(inst_path)
    inst_bo = pyxrt.bo(dev, isz, pyxrt.bo.flags.cacheable, gid1)
    with open(inst_path, "rb") as f:
        inst_bo.write(f.read(), 0)
    inst_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    in_bo.write(A.tobytes(), 0)
    wt_bo.write(B.tobytes(), 0)
    out_bo.write(b"\x00" * out_sz, 0)
    for bo in (in_bo, wt_bo, out_bo):
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run = pyxrt.run(kern)
    run.set_arg(0, 3)
    run.set_arg(1, inst_bo)
    run.set_arg(2, isz)
    run.set_arg(3, in_bo)
    run.set_arg(4, wt_bo)
    run.set_arg(5, out_bo)
    t0 = time.time()
    run.start()
    run.wait()
    dt = time.time() - t0

    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    raw = out_bo.read(out_sz, 0)
    out = np.frombuffer(raw, dtype=np.int32).reshape(M, N).copy()
    return out, dt


def check(M, N, C, dev, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    ref = A.astype(np.int32) @ B.astype(np.int32)
    got, dt = run_kernel(M, N, C, A, B, dev)
    diff = np.abs(ref.astype(np.int64) - got.astype(np.int64))
    return {
        "M": M, "K": K, "N": N, "n_cols": C,
        "bit_exact": bool(np.array_equal(ref, got)),
        "max_abs": int(diff.max()),
        "mismatch_elems": int((diff > 0).sum()),
        "total_elems": int(ref.size),
        "kernel_ms": round(dt * 1000, 3),
    }


def targets(group):
    out = []
    for C in (1, 2, 3, 4):
        for N in PROD_N:
            if not feasible(N, C):
                continue
            for M in (512, 448):
                if has_artifacts(M, N, C):
                    out.append((M, N, C))
    if group == "new":
        # the multi-column coverage added in this pass
        out = [(M, N, C) for (M, N, C) in out if C >= 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="new", choices=["new", "all"])
    ap.add_argument("--limit", type=int, default=0, help="stop after N kernels (debug)")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    todo = targets(args.group)
    if args.limit:
        todo = todo[:args.limit]
    print(f"verifying {len(todo)} kernel(s), group={args.group}")

    dev = pyxrt.device(0)
    results = []
    n_pass = 0
    t_start = time.time()
    for i, (M, N, C) in enumerate(todo, 1):
        name = f"M{M}_K32_N{N}_C{C}"
        try:
            r = check(M, N, C, dev)
        except Exception as exc:
            r = {"M": M, "K": K, "N": N, "n_cols": C, "error": f"{type(exc).__name__}: {exc}"}
        ok = r.get("bit_exact", False)
        if ok:
            n_pass += 1
        tag = "PASS" if ok else ("ERROR" if "error" in r else "MISMATCH")
        extra = "" if ok else (f"  max_abs={r.get('max_abs')} mismatch={r.get('mismatch_elems')}/{r.get('total_elems')}"
                              if "error" not in r else f"  {r['error']}")
        print(f"[{i}/{len(todo)}] {name:24s} {tag:8s} {r.get('kernel_ms', '-')} ms{extra}", flush=True)
        results.append(r)

    print(f"\n=== {n_pass}/{len(todo)} bit-exact in {time.time()-t_start:.1f}s ===")
    if args.log:
        with open(args.log, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"log: {args.log}")
    return 0 if n_pass == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
