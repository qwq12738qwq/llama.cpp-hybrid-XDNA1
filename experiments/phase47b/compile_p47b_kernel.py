#!/usr/bin/env python3
"""Phase 47B: compile a rows x cols Q8_0 scale-aware mmul kernel for real FFN shapes.

The production Q8SCALEV-C4 kernel is hard-wired to 1 row x 4 cols (4 AIE tiles).
Phase 43 reported that 2x4/3x4 fail aiecc with a ShimNOCTile budget error, so
this script checks whether 4x4 (16 tiles) is reachable on this device, which is
the only topology that showed ~900 GOPS device time in the Phase 36/47A runs.

Usage:
  python compile_p47b_kernel.py --M 512 --K 1024 --N 3584 --rows 4 --cols 4 \
      --work-dir /tmp/p47b/M512_K1024_N3584_4x4
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase40"))
from gen_q8_c4_mmul_v2_v4 import build  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, In, Out  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--fifo-depth", type=int, default=1)
    ap.add_argument("--work-dir", type=str, required=True)
    args = ap.parse_args()

    M, K, N = args.M, args.K, args.N
    os.makedirs(args.work_dir, exist_ok=True)
    xp = os.path.join(args.work_dir, "design.xclbin")
    ip = os.path.join(args.work_dir, "design.insts.bin")
    pp = os.path.join(args.work_dir, "main.pdi")

    @iron.jit
    def design(A: In, B: In, C: Out, *, dummy: CompileTime[int] = 0):
        return build(M, K, N, args.cols, n_rows=args.rows, fifo_depth=args.fifo_depth)

    print(f"[p47b] compiling M={M} K={K} N={N} rows={args.rows} cols={args.cols} "
          f"fifo_depth={args.fifo_depth} -> {args.work_dir}", flush=True)
    t0 = time.time()
    compiled = design.specialize(dummy=0).compile(xclbin_path=xp, inst_path=ip, pdi_path=pp)
    print(f"[p47b] SUCCESS in {time.time() - t0:.1f}s xclbin={os.path.getsize(compiled[0])} "
          f"insts={os.path.getsize(compiled[1])}")


if __name__ == "__main__":
    main()
