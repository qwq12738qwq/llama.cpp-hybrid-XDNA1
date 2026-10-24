#!/usr/bin/env python3
"""Phase 43: Compile down projection kernel (K=3584, N=1024).

Uses the v1 single-column topology since the shape is new and
the 1x4 baseline is the validated production kernel.

Usage (inside npu-dev container):
  python3 experiments/phase43/compile_down.py --work-dir /tmp/q8c4mmul_down
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase40'))
from gen_q8_c4_mmul_v2 import build

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out
    HAVE_AIE = True
except ImportError:
    HAVE_AIE = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=3584)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    if not HAVE_AIE:
        print("ERROR: aie.iron not available. Run inside npu-dev container.")
        sys.exit(1)

    M, K, N = args.M, args.K, args.N
    n_cols = args.cols

    print(f"[down compile] M={M} K={K} N={N} cols={n_cols}")
    print(f"  n_kc={K//256}  n_mt={M//32}  n_nt_c={(N//n_cols)//64}")

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(M, K, N, n_cols)

    print("  Compiling...")
    t0 = time.time()
    try:
        compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
        print(f"  Compiled in {time.time()-t0:.1f}s -> {compiled[0]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("SUCCESS")


if __name__ == "__main__":
    main()
