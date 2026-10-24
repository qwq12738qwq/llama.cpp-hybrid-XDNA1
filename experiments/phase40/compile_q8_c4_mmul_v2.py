#!/usr/bin/env python3
"""Phase 40M: Compile the mmul-based Q8SCALEV-C4 kernel for production shapes.

Run inside the npu-dev container:
  python experiments/phase40/compile_q8_c4_mmul_v2.py --work-dir /tmp/q8c4mmul_prod
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from gen_q8_c4_mmul_v2 import build

try:
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out
    HAVE_AIE = True
except ImportError:
    print("ERROR: aie.iron not available. Run inside npu-dev container.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    n_cols = args.cols

    print(f"Phase 40M: Compiling Q8SCALEV-C4 mmul v2")
    print(f"  M={M} K={K} N={N} cols={n_cols}")

    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(M, K, N, n_cols)

    print("Compiling...")
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"Compiled in {time.time() - t0:.1f}s")
    print(f"  xclbin: {compiled[0]} ({os.path.getsize(compiled[0])} bytes)")
    print(f"  insts:  {compiled[1]} ({os.path.getsize(compiled[1])} bytes)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
