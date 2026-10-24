#!/usr/bin/env python3
"""Phase 20D: generate missing multi-column XDNA GEMM kernels.

Drives experiments/npu-int8-gemm/gen_xdna_gemm.py (IRON + Peano, no Vitis) to emit
design.xclbin / design.insts.bin / main.pdi into the production kernel cache.

The production kernel uses K=32 with per-block Q8_0 scales; the host side chunks
over the full K dimension (see ggml/src/ggml-xdna/ggml-xdna.cpp).

Usage:
    /Storage/Cache/ironenv-v143/bin/python gen_missing_kernels.py --group m512_c2c3
    /Storage/Cache/ironenv-v143/bin/python gen_missing_kernels.py --group all --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time

IRON_PY = "/Storage/Cache/ironenv-v143/bin/python"
GEN = "/workspace/llama.cpp/experiments/npu-int8-gemm/gen_xdna_gemm.py"
KC = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache"

TILE_M, TILE_K, TILE_N = 64, 32, 64
K = 32

# Production N whitelist for Qwen3.5-0.8B PP, see is_mul_mat_supported() in ggml-xdna.cpp
PROD_N = [512, 768, 1024, 1536, 2048, 2304, 3072, 3584, 4096, 4608, 5120, 6144]


def wd(M, N, C):
    return os.path.join(KC, f"M{M}_K{K}_N{N}_C{C}_Tm{TILE_M}_Tk{TILE_K}_Tn{TILE_N}")


def has_artifacts(M, N, C):
    d = wd(M, N, C)
    return (os.path.exists(os.path.join(d, "design.xclbin")) and
            os.path.exists(os.path.join(d, "design.insts.bin")))


def feasible(N, C):
    # gen_xdna_gemm.py requires N % (tile_n * n_cols) == 0
    return N % (TILE_N * C) == 0


def plan(group):
    want = []
    if group in ("m512_c2c3", "all"):
        for C in (2, 3):
            for N in PROD_N:
                if feasible(N, C):
                    want.append((512, N, C))
    if group in ("m448_all", "all"):
        for C in (1, 2, 3, 4):
            for N in PROD_N:
                if feasible(N, C):
                    want.append((448, N, C))
    if group in ("m512_rest", "all"):
        for C in (1, 4):
            for N in PROD_N:
                if feasible(N, C):
                    want.append((512, N, C))
    return list(dict.fromkeys(want))


def build(M, N, C):
    out = wd(M, N, C)
    os.makedirs(out, exist_ok=True)
    cmd = [
        IRON_PY, GEN,
        "--M", str(M), "--K", str(K), "--N", str(N), "--cols", str(C),
        "--tile-m", str(TILE_M), "--tile-k", str(TILE_K), "--tile-n", str(TILE_N),
        "--work-dir", out,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = (proc.returncode == 0 and has_artifacts(M, N, C))
    return ok, dt, proc.stdout, proc.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all",
                    choices=["m512_c2c3", "m448_all", "m512_rest", "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=None, help="append JSON lines of per-kernel results")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if artifacts already exist (overwrites production kernels)")
    args = ap.parse_args()

    todo = plan(args.group)
    if not args.force:
        skipped = [t for t in todo if has_artifacts(*t)]
        todo = [t for t in todo if not has_artifacts(*t)]
        if skipped:
            print(f"skipping {len(skipped)} shape(s) that already have artifacts (use --force to rebuild)")
    print(f"group={args.group}  kernels to build: {len(todo)}")
    if args.dry_run:
        for M, N, C in todo:
            print(f"  M{M}_K{K}_N{N}_C{C}_Tm{TILE_M}_Tk{TILE_K}_Tn{TILE_N}")
        return 0

    results = []
    n_ok = 0
    t_start = time.time()
    for i, (M, N, C) in enumerate(todo, 1):
        name = f"M{M}_K{K}_N{N}_C{C}"
        print(f"[{i}/{len(todo)}] {name} ...", flush=True)
        ok, dt, out, err = build(M, N, C)
        status = "OK" if ok else "FAIL"
        if ok:
            n_ok += 1
        print(f"    {status}  {dt:.1f}s", flush=True)
        if not ok:
            tail = (err or out).strip().splitlines()[-5:]
            for line in tail:
                print(f"    | {line}", flush=True)
        results.append({
            "M": M, "K": K, "N": N, "n_cols": C,
            "status": status, "seconds": round(dt, 2),
            "work_dir": wd(M, N, C),
        })

    elapsed = time.time() - t_start
    print(f"\n=== {n_ok}/{len(todo)} built in {elapsed:.1f}s ===")
    if args.log:
        with open(args.log, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"log: {args.log}")
    return 0 if n_ok == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
